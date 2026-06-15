"""Unit tests for the quantile AR(1) baseline.

The single-regressor design means coefficients are recovered directly (no
collinearity), so recovery is checked at the coefficient level. Monotonicity,
no-lookahead and determinism mirror the QR-HAR tests. None touch the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from conformal_rv.models import baseline_ar

_Z = {0.1: float(norm.ppf(0.1)), 0.5: 0.0, 0.9: float(norm.ppf(0.9))}


def _simulate_ar1(
    n: int, intercept: float, slope: float, sigma: float, seed: int
) -> pd.Series[float]:
    """Simulate log-RV from a stationary AR(1) with location-shift noise."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(n - 1):
        values[t + 1] = intercept + slope * values[t] + rng.normal(0.0, sigma)
    return pd.Series(values, index=pd.bdate_range("2000-01-03", periods=n))


def test_design_is_a_single_regressor() -> None:
    log_rv = _simulate_ar1(50, 0.0, 0.5, 0.1, seed=0)
    design = baseline_ar.ar1_features(log_rv)
    assert list(design.columns) == ["lag1_log_rv"]


def test_fit_recovers_true_conditional_quantiles() -> None:
    intercept, slope, sigma = 0.0, 0.5, 0.1
    log_rv = _simulate_ar1(10000, intercept, slope, sigma, seed=42)

    design = baseline_ar.ar1_features(log_rv)
    model = baseline_ar.QRARBaseline().fit(design, log_rv.shift(-1))

    # Single regressor, so each quantile's [intercept, slope] is recovered
    # directly: the slope is shared, the intercept shifts by sigma * z_tau.
    for tau in model.quantiles:
        coefficients = model.coefficients(tau)
        assert coefficients[1] == pytest.approx(slope, abs=0.05)
        assert coefficients[0] == pytest.approx(intercept + sigma * _Z[tau], abs=0.05)


def test_quantiles_are_monotone_after_rearrangement() -> None:
    log_rv = _simulate_ar1(2000, 0.0, 0.5, sigma=0.15, seed=3)
    design = baseline_ar.ar1_features(log_rv)
    model = baseline_ar.QRARBaseline().fit(design, log_rv.shift(-1))

    predictions = model.predict_quantiles(design.dropna())
    assert (np.diff(predictions, axis=1) >= 0.0).all()


def test_predict_quantiles_have_no_lookahead() -> None:
    log_rv = _simulate_ar1(200, 0.0, 0.5, 0.1, seed=1)
    design = baseline_ar.ar1_features(log_rv)
    model = baseline_ar.QRARBaseline().fit(design.iloc[:80], log_rv.shift(-1).iloc[:80])

    complete = design.dropna()
    position = 100
    batch = model.predict_quantiles(complete)
    single = model.predict_quantiles(complete.iloc[[position]])
    assert np.allclose(batch[position], single[0])


def test_fit_is_deterministic() -> None:
    log_rv = _simulate_ar1(500, 0.0, 0.5, 0.1, seed=7)
    design = baseline_ar.ar1_features(log_rv)
    target = log_rv.shift(-1)

    first = baseline_ar.QRARBaseline().fit(design, target)
    second = baseline_ar.QRARBaseline().fit(design, target)

    complete = design.dropna()
    for tau in first.quantiles:
        assert np.array_equal(first.coefficients(tau), second.coefficients(tau))
    assert np.array_equal(
        first.predict_quantiles(complete), second.predict_quantiles(complete)
    )
