"""Unit tests for the data-loading module.

Fast tests exercise the offline paths only: the cache read, the normalisation
contract, and the module constants. Anything that hits the network is marked
``slow`` and runs only under ``--slow``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_rv import data


def test_index_constants() -> None:
    assert data.INDICES == [
        "^GSPC",
        "^GSPTSE",
        "^FTSE",
        "^GDAXI",
        "^STOXX50E",
        "^N225",
        "^HSI",
        "^AXJO",
    ]
    # VIX is a covariate, not a panel member, so it must stay out of INDICES.
    assert data.VIX == "^VIX"
    assert data.VIX not in data.INDICES


def test_loaders_are_present() -> None:
    assert callable(data.load_index_ohlc)
    assert callable(data.load_vix)
    assert callable(data.load_oxford_man_rv)


def test_oxford_man_loader_is_a_later_phase() -> None:
    with pytest.raises(NotImplementedError):
        data.load_oxford_man_rv("SPX2.rv")


def test_normalise_drops_non_trading_days_and_lowercases() -> None:
    # Mixed-case columns, unsorted index, and a NaN close that must be dropped.
    raw = pd.DataFrame(
        {
            "Open": [3.0, 2.0, 1.0],
            "High": [3.0, 2.0, 1.0],
            "Low": [3.0, 2.0, 1.0],
            "Close": [3.0, np.nan, 1.0],
        },
        index=pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-01"]),
    )
    out = data._normalise_ohlc(raw)

    assert list(out.columns) == ["open", "high", "low", "close"]
    assert out.index.is_monotonic_increasing
    # The 2020-01-02 row had no close, so it is dropped as a non-trading day.
    assert len(out) == 2
    assert pd.Timestamp("2020-01-02") not in out.index


def test_normalise_does_not_fill_internal_nans() -> None:
    # A NaN high (close present) must survive normalisation: no ffill/bfill.
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [np.nan, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )
    out = data._normalise_ohlc(raw)
    assert len(out) == 2
    assert np.isnan(out["high"].iloc[0])


def test_cache_read_avoids_network(tmp_path) -> None:
    # A frame already present in the cache must be returned verbatim without any
    # network access (this test has no network and would fail if one were made).
    ticker = "^GSPC"
    start, end = "2000-01-01", None
    frame = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.5],
            "low": [0.9],
            "close": [1.2],
        },
        index=pd.to_datetime(["2020-01-02"]),
    )
    path = data._cache_path(ticker, start, end, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)

    out = data.load_index_ohlc(ticker, start=start, end=end, cache_dir=tmp_path)
    assert set(out) == {ticker}
    pd.testing.assert_frame_equal(out[ticker], frame)


@pytest.mark.slow
def test_live_pull_smoke(tmp_path) -> None:
    # Real network pull, kept behind --slow. A short window keeps it cheap.
    out = data.load_index_ohlc(
        "^GSPC", start="2020-01-01", end="2020-02-01", cache_dir=tmp_path
    )
    frame = out["^GSPC"]
    assert not frame.empty
    assert {"open", "high", "low", "close"}.issubset(frame.columns)
    assert frame.index.is_monotonic_increasing
    # No-fill policy: every retained row is a real trading day with a close.
    assert frame["close"].notna().all()
