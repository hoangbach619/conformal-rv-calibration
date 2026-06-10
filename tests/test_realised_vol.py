"""Unit tests for the realised-volatility estimators.

All fast tests run on synthetic OHLC with hand-checkable properties; none of
them touch the network. Estimator outputs are daily realised *variances*.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from conformal_rv import realised_vol as rv


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLC frame from (open, high, low, close) tuples."""
    index = pd.date_range("2020-01-01", periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close"])


def test_parkinson_known_value() -> None:
    # high/low chosen so log(high) - log(low) == 1 exactly, hence the estimator
    # collapses to 1 / (4 ln 2) independent of open/close.
    ohlc = _frame([(1.0, math.e, 1.0, 1.0)])
    assert rv.parkinson(ohlc).iloc[0] == pytest.approx(1.0 / (4.0 * math.log(2.0)))


def test_rogers_satchell_known_value() -> None:
    # open == close (log 0), high = e**0.1, low = e**-0.1, so RS reduces to
    # 0.1**2 + 0.1**2 = 0.02.
    ohlc = _frame([(1.0, math.exp(0.1), math.exp(-0.1), 1.0)])
    assert rv.rogers_satchell(ohlc).iloc[0] == pytest.approx(0.02)


def test_garman_klass_known_value() -> None:
    # With close == open the Garman-Klass close-open term vanishes, leaving
    # 0.5 * (log high - log low)**2 = 0.5 * 0.2**2 = 0.02.
    ohlc = _frame([(1.0, math.exp(0.1), math.exp(-0.1), 1.0)])
    assert rv.garman_klass(ohlc).iloc[0] == pytest.approx(0.02)


def test_single_day_estimators_are_non_negative() -> None:
    # Valid bars (high >= max(open, close), low <= min(open, close)) must give
    # non-negative variances for every single-bar estimator.
    ohlc = _frame(
        [
            (100.0, 102.0, 99.0, 101.0),
            (101.0, 103.5, 100.5, 100.7),
            (100.7, 101.2, 98.0, 98.5),
            (98.5, 99.0, 96.0, 97.0),
        ]
    )
    for estimator in (rv.parkinson, rv.garman_klass, rv.rogers_satchell):
        values = estimator(ohlc)
        assert (values >= 0.0).all()


def test_yang_zhang_is_windowed_and_positive() -> None:
    window = 21
    # Build a zero-overnight-gap series: each open equals the previous close,
    # so the overnight variance term is exactly zero and Yang-Zhang stays
    # strictly positive from the intraday terms alone.
    closes = [100.0 + i for i in range(30)]
    rows: list[tuple[float, float, float, float]] = []
    prev_close = closes[0]
    for close in closes:
        open_ = prev_close  # zero overnight gap by construction
        high = max(open_, close) + 1.0
        low = min(open_, close) - 1.0
        rows.append((open_, high, low, close))
        prev_close = close
    ohlc = _frame(rows)

    yz = rv.yang_zhang(ohlc, window=window)
    # Undefined until a full window (plus the one-day overnight lag) accrues;
    # the leading NaNs are left in place, not filled.
    assert yz.iloc[:window].isna().all()
    assert yz.iloc[window:].notna().all()
    assert (yz.dropna() > 0.0).all()


def test_rolling_estimator_matches_manual_mean() -> None:
    ohlc = _frame([(100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i) for i in range(10)])
    expected = rv.parkinson(ohlc).rolling(5).mean()
    pd.testing.assert_series_equal(rv.rolling_parkinson(ohlc, window=5), expected)


def test_nan_inputs_propagate_not_filled() -> None:
    # A NaN high on the second bar must yield a NaN there and nowhere else; the
    # series length is unchanged (no row is dropped or filled).
    ohlc = _frame([(1.0, 2.0, 0.5, 1.5), (1.0, math.nan, 0.5, 1.0)])
    out = rv.parkinson(ohlc)
    assert len(out) == 2
    assert not math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[1])


def test_to_log_rv_inverts() -> None:
    variance = pd.Series([0.01, 0.04, 0.25])
    log_rv = rv.to_log_rv(variance)
    # log rv = 0.5 log(variance), so exp(2 * log rv) recovers the variance.
    assert np.allclose(np.exp(2.0 * log_rv.to_numpy()), variance.to_numpy())


def test_to_log_rv_propagates_nan() -> None:
    variance = pd.Series([0.04, math.nan, 0.09])
    log_rv = rv.to_log_rv(variance)
    assert math.isnan(log_rv.iloc[1])
    assert not math.isnan(log_rv.iloc[0])
