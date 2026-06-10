"""Data acquisition for the equity-index panel.

Daily OHLC bars are pulled via yfinance. Daily-OHLC realised volatility is a
range proxy and is used here because it is free; intraday 5-minute data is not
freely redistributable. The legacy Oxford-Man 5-minute RV panel is treated as
a frozen, out-of-sample validation target where the date ranges overlap (a
later phase; the loader here is still a stub).

Missing-data policy lives here on purpose so it is enforced once: no
backward-fill is ever permitted, because back-filling leaks future
information into a forecasting target, and no forward-fill of OHLC is applied
either. Each index is processed on its own trading calendar; non-trading days
(rows with no close) are dropped rather than filled. Aligning a covariate such
as VIX onto an index calendar is a features concern, not a data-loading one,
so it is deliberately not done here.

Caching. Every raw pull is written to a gitignored ``data/raw`` directory as
parquet keyed by ticker and date range, and read back on later runs. This
freezes the panel after the first pull so the study is reproducible. An
``end`` of ``None`` is treated as a single stable cache key ("open"), so the
first pull is what gets frozen for that key.

Reproducibility. Per-ticker pulls run sequentially, which honours
``conformal_rv.N_JOBS = 1``; no threading is introduced. Loading draws no
random numbers, so ``conformal_rv.SEED`` does not enter here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

_LOGGER = logging.getLogger(__name__)

# The working equity-index panel. VIX is intentionally not a member: it is a
# market-vol covariate, loaded separately (see ``VIX`` / ``load_vix``) so it is
# never accidentally treated as a forecasting target.
INDICES: list[str] = [
    "^GSPC",
    "^GSPTSE",
    "^FTSE",
    "^GDAXI",
    "^STOXX50E",
    "^N225",
    "^HSI",
    "^AXJO",
]
VIX: str = "^VIX"

# Anchored to the repo root, not the process CWD, so the frozen cache is found
# regardless of where a run is launched from.
_DEFAULT_CACHE_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "raw"

_OHLC_COLUMNS: list[str] = ["open", "high", "low", "close"]


def load_index_ohlc(
    tickers: str | Sequence[str],
    start: str = "2000-01-01",
    end: str | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Return daily OHLC frames keyed by ticker.

    Each frame is indexed by date with columns ``open/high/low/close`` (plus
    ``volume`` when the source provides it). Tickers are loaded one at a time
    on their own calendars; a per-ticker failure falls back to the Yahoo chart
    JSON endpoint before giving up. Pulls are cached and frozen (see module
    docstring), so repeat calls are offline.
    """
    ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
    base = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    # Sequential by design: honours N_JOBS = 1 and keeps the per-ticker source
    # log readable rather than interleaved.
    return {ticker: _load_one(ticker, start, end, base) for ticker in ticker_list}


def load_vix(
    start: str = "2000-01-01",
    end: str | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return the VIX daily frame, the market-vol covariate.

    A thin wrapper over :func:`load_index_ohlc` so VIX shares the same caching
    and missing-data policy, while staying out of the index panel.
    """
    return load_index_ohlc(VIX, start=start, end=end, cache_dir=cache_dir)[VIX]


def load_oxford_man_rv(symbol: str) -> pd.DataFrame:
    """Return the frozen Oxford-Man 5-minute RV series for validation.

    Read-only validation anchor for a later phase; not implemented in the
    data-loading phase (1a). It is recorded here so the module's eventual
    surface is visible in the frozen design.
    """
    raise NotImplementedError("Oxford-Man validation loader is a later phase")


def _load_one(
    ticker: str, start: str, end: str | None, cache_dir: Path
) -> pd.DataFrame:
    """Load one ticker: cache, then yfinance, then chart-JSON fallback."""
    path = _cache_path(ticker, start, end, cache_dir)
    if path.exists():
        # Cached frames are already normalised; return them verbatim so the
        # frozen panel is bit-stable across runs.
        _LOGGER.info("loaded %s from cache (%s)", ticker, path.name)
        return pd.read_parquet(path)

    frame = _pull_yfinance(ticker, start, end)
    source = "yfinance"
    if frame is None or frame.empty:
        # yfinance is flaky on some indices and some IP ranges; the browser-
        # impersonating chart endpoint is the documented fallback.
        frame = _fetch_chart_json(ticker, start, end)
        source = "chart-json-fallback"

    frame = _normalise_ohlc(frame)
    _LOGGER.info("loaded %s from %s (%d rows)", ticker, source, len(frame))

    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def _pull_yfinance(ticker: str, start: str, end: str | None) -> pd.DataFrame | None:
    """Pull raw daily bars via yfinance, or ``None`` on empty/error.

    Returning ``None`` rather than raising lets the caller fall through to the
    chart-JSON endpoint without conflating "no data" with "hard failure".
    """
    # Lazy import: keeps module import (and the fast test subset) light, and
    # means a yfinance import problem only bites when a live pull is attempted.
    import yfinance as yf

    try:
        raw = yf.Ticker(ticker).history(
            start=start, end=end, auto_adjust=False, actions=False
        )
    except Exception as exc:  # any failure here should trigger the fallback
        _LOGGER.warning("yfinance failed for %s: %s", ticker, exc)
        return None
    if raw is None or raw.empty:
        return None
    result: pd.DataFrame = raw
    return result


def _fetch_chart_json(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Fetch daily bars from the Yahoo chart JSON endpoint.

    Used only when yfinance yields nothing. The endpoint rejects default
    Python user agents, so the request is made with curl_cffi impersonating a
    browser. Imported lazily so curl_cffi is a fallback-only dependency.
    """
    from curl_cffi import requests as cffi_requests

    period1 = int(pd.Timestamp(start).timestamp())
    period2 = (
        int(pd.Timestamp(end).timestamp())
        if end is not None
        else int(pd.Timestamp.now(tz="UTC").timestamp())
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": period1, "period2": period2, "interval": "1d"}
    # curl_cffi's request methods are untyped; treat the response as Any so the
    # untyped .raise_for_status()/.json() calls do not trip --strict.
    response: Any = cffi_requests.get(url, params=params, impersonate="chrome")
    response.raise_for_status()

    result = response.json()["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(result["timestamp"], unit="s", utc=True),
    )
    return frame


def _normalise_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise a raw frame to the OHLC contract, no filling applied.

    Lower-cases columns, keeps OHLC (and volume if present), drops rows with no
    close (non-trading days), sorts ascending, strips any timezone and snaps to
    midnight dates so the two sources align, and de-duplicates dates. Internal
    NaNs other than a missing close are left in place so bad bars propagate.
    """
    out = frame.copy()
    # Wrap in Index so the assignment matches the typed columns setter.
    out.columns = pd.Index([str(column).lower() for column in out.columns])

    keep = list(_OHLC_COLUMNS)
    if "volume" in out.columns:
        keep.append("volume")
    out = out.loc[:, keep]

    # A missing close means the market did not trade that day: drop, never fill.
    out = out[out["close"].notna()]
    out = out.sort_index()

    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out.index = out.index.normalize()

    # Belt and braces against a source returning a date twice; keep the last.
    out = out[~out.index.duplicated(keep="last")]
    return out


def _cache_path(ticker: str, start: str, end: str | None, cache_dir: Path) -> Path:
    """Parquet cache path keyed by ticker and date range.

    ``end=None`` maps to a single stable key so the first pull is what gets
    frozen, rather than the key drifting with the current date.
    """
    safe_ticker = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    end_key = end if end is not None else "open"
    return cache_dir / f"{safe_ticker}__{start}__{end_key}.parquet"
