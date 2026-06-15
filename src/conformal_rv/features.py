"""Feature construction for the realised-volatility panel.

The output is a tidy long panel: one row per (index, date), with every
covariate constructed so that a row dated ``t`` depends only on information
available at the close of ``t``. The target for a horizon ``h`` is realised
vol at ``t + h``; nothing in a feature row may touch data dated after ``t``.

Leakage discipline lives here because it is the whole point of the study. Two
rules are enforced and not merely documented:

- No backward-fill, anywhere. Back-filling a feature would carry a future
  value into the past and invalidate every coverage claim.
- Any alignment of an off-calendar covariate (VIX, or another index's RV) onto
  an index's own trading calendar is forward-fill only, and each forward-fill
  carries a one-line justification at the call site.

The HAR cascade (Corsi 2009) of daily, weekly and monthly RV averages is the
regressor block the HAR base model consumes; ``har_features`` builds it here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# HAR cascade windows in trading days: daily, weekly, monthly.
HAR_WINDOWS: tuple[int, int, int] = (1, 5, 22)

# Region of each index. Used as a static categorical feature and, later, to
# reason about which indices share trading hours.
REGION: dict[str, str] = {
    "^GSPC": "AMER",
    "^GSPTSE": "AMER",
    "^FTSE": "EMEA",
    "^GDAXI": "EMEA",
    "^STOXX50E": "EMEA",
    "^N225": "APAC",
    "^HSI": "APAC",
    "^AXJO": "APAC",
}

# Column order of the long panel, grouped static / known / unknown.
_FEATURE_COLUMNS: tuple[str, ...] = (
    # Static
    "index",
    "date",
    "region",
    # Time-varying KNOWN (calendar; future values genuinely known)
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "is_month_end",
    "is_quarter_end",
    # Time-varying UNKNOWN (known only up to t; every value dated <= t - 1)
    "lag_log_rv",
    "lag_return",
    "lag_range",
    "vix_level",
    "cross_index_avg_rv",
)


def forward_fill_feature(series: pd.Series[float], reason: str) -> pd.Series[float]:
    """Forward-fill a single feature, requiring an explicit justification.

    The ``reason`` is mandatory so that every fill is auditable rather than
    silently applied. Only forward-fill is performed: back-fill is never
    permitted, because it would move future information into the past.
    """
    if not reason.strip():
        raise ValueError("a non-empty justification is required for any forward-fill")
    return series.ffill()


def har_features(log_rv: pd.Series[float]) -> pd.DataFrame:
    """Build the Corsi (2009) HAR cascade for one index, from log realised vol.

    Three regressors at the forecast origin ``t``: the daily component is
    log-RV at ``t``; the weekly and monthly components are trailing means over
    ``t-4..t`` and ``t-21..t``. All three are realised and known at ``t``, and
    the target sits at ``t + h`` with ``h >= 1``, so conditioning on log-RV
    through ``t`` is information at ``t``, not lookahead.

    No fill is applied: the leading rows without a full month are NaN, for the
    caller to drop. Back-fill is never used (it would import a future value).
    """
    weekly, monthly = HAR_WINDOWS[1], HAR_WINDOWS[2]
    return pd.DataFrame(
        {
            "har_daily": log_rv,
            "har_weekly": log_rv.rolling(weekly).mean(),
            "har_monthly": log_rv.rolling(monthly).mean(),
        }
    )


def build_features(
    rv_by_index: dict[str, pd.Series[float]],
    ohlc_by_index: dict[str, pd.DataFrame],
    vix: pd.Series[float],
) -> pd.DataFrame:
    """Build the tidy long feature panel across all indices.

    ``rv_by_index`` and ``ohlc_by_index`` are keyed by ticker; ``vix`` is the
    market-vol covariate on its own calendar. Each index is processed on its
    own trading calendar (the dates of its RV series), and every unknown
    covariate is lagged so the row dated ``t`` sees only data dated ``<= t-1``.
    """
    frames: list[pd.DataFrame] = []

    for name, rv in rv_by_index.items():
        rv = rv.sort_index()
        dates = pd.DatetimeIndex(rv.index, name="date")
        ohlc = ohlc_by_index[name].sort_index()

        # Index-specific intraday quantities, on the index's own calendar.
        log_close: pd.Series[float] = np.log(ohlc["close"])
        daily_return: pd.Series[float] = log_close.diff()
        log_high: pd.Series[float] = np.log(ohlc["high"])
        log_low: pd.Series[float] = np.log(ohlc["low"])
        daily_range: pd.Series[float] = log_high - log_low

        frame = _calendar_features(dates)

        # --- Static group ---
        frame["index"] = name
        frame["region"] = REGION[name]

        # --- Time-varying UNKNOWN group: every value dated <= t - 1 ---
        # Lag by one trading day so a row dated t cannot see same-day RV.
        frame["lag_log_rv"] = rv.shift(1)
        frame["lag_return"] = daily_return.reindex(dates).shift(1)
        frame["lag_range"] = daily_range.reindex(dates).shift(1)

        # VIX is forward-filled onto this index's calendar (it trades on US
        # hours, so on a non-US trading day we carry the last known close
        # forward, never a future one). It enters at t, not lagged: the VIX
        # close at t is known at t and the target is at t + h with h >= 1, so it
        # sits in the information set rather than being lookahead.
        frame["vix_level"] = _align_forward(
            vix, dates, reason="carry last known VIX onto this index's calendar"
        )

        # Cross-index average RV is the leakage-sensitive one (see helper).
        frame["cross_index_avg_rv"] = _cross_index_average_rv(rv_by_index, name, dates)

        frames.append(frame.reset_index())

    panel = pd.concat(frames, ignore_index=True)
    panel["index"] = panel["index"].astype("category")
    panel["region"] = panel["region"].astype("category")
    panel = panel.loc[:, list(_FEATURE_COLUMNS)]
    return panel.sort_values(["index", "date"], kind="stable").reset_index(drop=True)


def _calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Known calendar features, a deterministic function of the date alone.

    These are the only covariates whose future values are genuinely known, so
    they are taken at ``t`` itself rather than lagged.
    """
    iso_week = dates.isocalendar().week.to_numpy()
    return pd.DataFrame(
        {
            "day_of_week": dates.dayofweek,
            "day_of_month": dates.day,
            "week_of_year": iso_week,
            "month": dates.month,
            "is_month_end": dates.is_month_end,
            "is_quarter_end": dates.is_quarter_end,
        },
        index=dates,
    )


def _align_forward(
    series: pd.Series[float], target_dates: pd.DatetimeIndex, reason: str
) -> pd.Series[float]:
    """Align an off-calendar series onto ``target_dates``, forward-fill only.

    The series is reindexed onto the union of its own and the target dates,
    forward-filled (so each target date inherits the last known value at or
    before it), then restricted to the target dates. A missing leading stretch
    stays missing: there is nothing earlier to carry forward, and back-fill is
    forbidden.
    """
    combined = series.index.union(target_dates)
    filled = forward_fill_feature(series.reindex(combined), reason)
    return filled.reindex(target_dates)


def _cross_index_average_rv(
    rv_by_index: dict[str, pd.Series[float]],
    target: str,
    target_dates: pd.DatetimeIndex,
) -> pd.Series[float]:
    """Mean of the *other* indices' RV, dated ``<= t - 1``, on ``target``'s axis.

    The peers' values are taken strictly before ``t``. This is deliberate: the
    indices span time zones, so a peer that closes earlier on the same calendar
    date would otherwise leak same-day information. Forward-filling each peer
    then shifting one row guarantees every contribution is dated ``<= t - 1``.
    """
    others = [name for name in rv_by_index if name != target]
    if not others:
        # A single-index panel has no cross-sectional peer set.
        return pd.Series(np.nan, index=target_dates, dtype=float)

    # Union calendar across peers; forward-fill each so it carries its most
    # recent realised vol (forward only, never back).
    wide = pd.concat({name: rv_by_index[name] for name in others}, axis=1).sort_index()
    wide_filled = wide.ffill()
    peer_mean: pd.Series[float] = wide_filled.mean(axis=1)

    # Shift one row on the union axis so date t sees only peers dated <= t - 1.
    peer_mean_lagged = peer_mean.shift(1)
    return _align_forward(
        peer_mean_lagged,
        target_dates,
        reason="carry last known peer-average RV onto this index's calendar",
    )
