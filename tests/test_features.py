"""Unit tests for the feature-construction module.

The headline tests are about leakage: no future value may reach a past feature
row, and no gap is ever back-filled. None touch the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conformal_rv import features

_TICKERS = ("^GSPC", "^FTSE")  # one AMER, one EMEA, so cross-index RV is exercised


def _synthetic_panel(
    seed: int = 0, n: int = 40
) -> tuple[dict[str, pd.Series[float]], dict[str, pd.DataFrame], pd.Series[float]]:
    """Build a small synthetic panel of RV, OHLC and VIX on a shared calendar."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)

    rv_by_index: dict[str, pd.Series[float]] = {}
    ohlc_by_index: dict[str, pd.DataFrame] = {}
    for ticker in _TICKERS:
        close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
        high = close + rng.uniform(0.5, 2.0, n)
        low = close - rng.uniform(0.5, 2.0, n)
        open_ = close + rng.normal(0.0, 0.5, n)
        ohlc_by_index[ticker] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close}, index=dates
        )
        rv_by_index[ticker] = pd.Series(
            np.log(rng.uniform(0.01, 0.5, n)), index=dates, name="log_rv"
        )

    vix = pd.Series(15.0 + rng.uniform(0.0, 5.0, n), index=dates, name="close")
    return rv_by_index, ohlc_by_index, vix


def test_no_future_value_reaches_an_earlier_row() -> None:
    rv, ohlc, vix = _synthetic_panel()
    base = features.build_features(rv, ohlc, vix)

    # Fix a reference date t and perturb only strictly future values (dates > t):
    # every RV series and VIX itself. Rows dated <= t must be untouched.
    cutoff = rv["^GSPC"].index[25]
    rv_perturbed = {name: series.copy() for name, series in rv.items()}
    for series in rv_perturbed.values():
        series[series.index > cutoff] += 10.0
    vix_perturbed = vix.copy()
    vix_perturbed[vix_perturbed.index > cutoff] += 10.0
    perturbed = features.build_features(rv_perturbed, ohlc, vix_perturbed)

    # Rows dated at or before t are byte-for-byte equal. This holds even though
    # vix_level is the contemporaneous VIX at t: a future VIX (t' > t) cannot
    # reach a row dated <= t.
    before_base = base[base["date"] <= cutoff].reset_index(drop=True)
    before_perturbed = perturbed[perturbed["date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_base, before_perturbed)

    # Sanity: the future perturbation did change later rows, so the test is not
    # vacuously comparing two identical panels.
    assert not base.equals(perturbed)


def test_leading_gap_is_forward_filled_not_back_filled() -> None:
    rv, ohlc, vix = _synthetic_panel()
    vix_gapped = vix.copy()
    vix_gapped.iloc[:5] = np.nan  # leading gap: no VIX for the first five days

    panel = features.build_features(rv, ohlc, vix_gapped)
    gspc = panel[panel["index"] == "^GSPC"].sort_values("date")

    # With no earlier VIX to carry forward and back-fill forbidden, the earliest
    # vix_level entries must stay NaN rather than borrowing the first real value.
    assert gspc["vix_level"].iloc[:3].isna().all()
    # And the first available value, once it appears, is finite (forward-fill
    # did eventually carry a real observation).
    assert gspc["vix_level"].notna().any()


def test_calendar_features_depend_only_on_the_date() -> None:
    rv1, ohlc1, vix1 = _synthetic_panel(seed=1)
    rv2, ohlc2, vix2 = _synthetic_panel(seed=2)
    panel1 = features.build_features(rv1, ohlc1, vix1)
    panel2 = features.build_features(rv2, ohlc2, vix2)

    calendar_columns = [
        "index",
        "date",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "is_month_end",
        "is_quarter_end",
    ]
    # Different RV/OHLC/VIX values, identical dates -> identical calendar block.
    pd.testing.assert_frame_equal(panel1[calendar_columns], panel2[calendar_columns])

    # And the calendar columns match a direct computation from the date.
    gspc = panel1[panel1["index"] == "^GSPC"].sort_values("date")
    direct = pd.DatetimeIndex(gspc["date"])
    assert (gspc["day_of_week"].to_numpy() == direct.dayofweek.to_numpy()).all()
    assert (gspc["month"].to_numpy() == direct.month.to_numpy()).all()


def test_har_features_cascade_components() -> None:
    log_rv = pd.Series(
        np.arange(30, dtype=float), index=pd.bdate_range("2020-01-01", periods=30)
    )
    cascade = features.har_features(log_rv)

    assert list(cascade.columns) == ["har_daily", "har_weekly", "har_monthly"]
    # Daily is log-RV at t; weekly/monthly are trailing means over t-4..t, t-21..t.
    assert cascade["har_daily"].iloc[25] == log_rv.iloc[25]
    assert cascade["har_weekly"].iloc[25] == log_rv.iloc[21:26].mean()
    assert cascade["har_monthly"].iloc[25] == log_rv.iloc[4:26].mean()
    # No back-fill: incomplete leading windows stay NaN.
    assert cascade["har_weekly"].iloc[:4].isna().all()
    assert cascade["har_monthly"].iloc[:21].isna().all()
