"""Unit tests for the scoring rule and PIT calibration diagnostics.

Pinball loss is hand-checked; the PIT is uniform (high KS p-value) on calibrated
data and clearly non-uniform on a miscalibrated stream. None touch the network.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from conformal_rv.metrics import scoring


def test_horizons_are_pre_registered() -> None:
    assert scoring.HORIZONS == (1, 5, 10, 22)


def test_pinball_loss_hand_checked() -> None:
    lower = np.array([0.0, 0.0])
    upper = np.array([2.0, 2.0])
    y = np.array([1.0, 3.0])
    # alpha = 0.2 -> tau_low = 0.1, tau_high = 0.9.
    # point 0 (y=1): low 0.1*(1-0)=0.1; high (1-2)<0 -> (0.9-1)*(-1)=0.1.
    # point 1 (y=3): low 0.1*(3-0)=0.3; high 0.9*(3-2)=0.9.
    # mean over the four evaluations [0.1, 0.3, 0.1, 0.9] = 0.35.
    assert scoring.pinball_loss(lower, upper, y, alpha=0.2) == pytest.approx(0.35)


def test_pit_values_interpolate_the_predictive_cdf() -> None:
    quantile_forecasts = np.array([[0.0, 10.0, 20.0]])
    levels = np.array([0.1, 0.5, 0.9])
    # y = 5 is halfway from 0 to 10, so the CDF is 0.1 + 0.5 * (0.5 - 0.1) = 0.3.
    pit = scoring.pit_values(quantile_forecasts, np.array([5.0]), levels)
    assert pit[0] == pytest.approx(0.3)


def test_pit_is_uniform_under_correct_calibration() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.normal(0.0, 1.0, n)
    levels = np.linspace(0.001, 0.999, 999)
    # Correct predictive distribution: the true N(0, 1) quantiles for every row.
    forecasts = np.tile(norm.ppf(levels), (n, 1))

    pit = scoring.pit_values(forecasts, y, levels)
    _, p_value = scoring.ks_uniformity(pit)
    assert p_value > 0.05  # uniformity is not rejected


def test_pit_is_non_uniform_under_miscalibration() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    # Outcomes are twice as dispersed as the predictive distribution claims.
    y = rng.normal(0.0, 2.0, n)
    levels = np.linspace(0.001, 0.999, 999)
    forecasts = np.tile(norm.ppf(levels), (n, 1))

    pit = scoring.pit_values(forecasts, y, levels)
    statistic, p_value = scoring.ks_uniformity(pit)
    assert statistic > 0.1
    assert p_value < 0.01  # uniformity is clearly rejected
