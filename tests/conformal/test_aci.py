"""Unit tests for Adaptive Conformal Inference.

The headline test drives a distribution shift under which static CQR
under-covers and checks that ACI restores coverage. The rest pin the alpha_t
adaptation direction, the saturation cases, and determinism. None touch the
network.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_rv.conformal import aci
from conformal_rv.conformal.cqr import conformalise_cqr, conformity_scores


def test_aci_restores_coverage_under_shift_better_than_cqr() -> None:
    rng = np.random.default_rng(0)
    n_cal, n_test = 1000, 1500
    # Calibration is N(0, 1); the test stream is wider, N(0, 2), so a band tuned
    # on calibration under-covers the test until ACI adapts.
    cal_y = rng.normal(0.0, 1.0, n_cal)
    test_y = rng.normal(0.0, 2.0, n_test)
    cal_lower, cal_upper = np.full(n_cal, -1.0), np.full(n_cal, 1.0)
    test_lower, test_upper = np.full(n_test, -1.0), np.full(n_test, 1.0)

    static = conformalise_cqr(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )
    adaptive = aci.conformalise_aci(
        cal_lower,
        cal_upper,
        cal_y,
        test_lower,
        test_upper,
        test_y,
        alpha=0.2,
        gamma=0.05,
    )

    # Static CQR clearly under-covers; ACI lands much closer to the 0.80 nominal.
    assert static.coverage < 0.7
    assert abs(adaptive.conformal.coverage - 0.8) < abs(static.coverage - 0.8)
    assert adaptive.conformal.coverage > 0.72


def test_alpha_trajectory_follows_the_update_rule() -> None:
    # Controlled streams isolate the update direction from the window dynamics.
    cal_y = np.zeros(100)
    cal_lower, cal_upper = np.full(100, -0.01), np.full(100, 0.01)
    n_test = 25

    # Every test point sits outside a pinhole band, so err = 1 every step and
    # alpha_{t+1} = alpha_t + gamma(alpha - 1) drives alpha_t strictly down (the
    # adaptation that widens the interval to chase the missed coverage).
    under = aci.conformalise_aci(
        cal_lower,
        cal_upper,
        cal_y,
        np.full(n_test, -0.01),
        np.full(n_test, 0.01),
        np.full(n_test, 100.0),
        alpha=0.2,
        gamma=0.02,
    )
    assert under.alpha_trajectory[5] < under.alpha_trajectory[0]
    assert np.all(np.diff(under.alpha_trajectory[:6]) < 0.0)

    # Every test point sits inside a huge band, so err = 0 every step and
    # alpha_{t+1} = alpha_t + gamma * alpha drives alpha_t strictly up.
    over = aci.conformalise_aci(
        cal_lower,
        cal_upper,
        cal_y,
        np.full(n_test, -100.0),
        np.full(n_test, 100.0),
        np.zeros(n_test),
        alpha=0.2,
        gamma=0.02,
    )
    assert over.alpha_trajectory[5] > over.alpha_trajectory[0]
    assert np.all(np.diff(over.alpha_trajectory) > 0.0)


def test_radius_saturates_at_the_calibration_extremes() -> None:
    # The shared quantile clamps rather than going to +/-inf: under sustained
    # miscoverage the radius saturates at the widest and narrowest the
    # calibration scores can express.
    rng = np.random.default_rng(0)
    cal_y = rng.normal(0.0, 1.0, 500)
    cal_lower, cal_upper = np.full(500, -1.0), np.full(500, 1.0)
    reference = np.sort(conformity_scores(cal_lower, cal_upper, cal_y))
    widest, narrowest = float(reference[-1]), float(reference[0])
    n_test = 50

    # Sustained under-coverage drives alpha_t past 0; the radius clamps at the
    # widest reference score (interval upper = test_upper + radius).
    under = aci.conformalise_aci(
        cal_lower,
        cal_upper,
        cal_y,
        np.full(n_test, -1.0),
        np.full(n_test, 1.0),
        np.full(n_test, 50.0),
        alpha=0.2,
        gamma=0.5,
    )
    radius_under = under.conformal.upper - 1.0
    assert under.alpha_trajectory.min() <= 0.0
    assert radius_under.max() == pytest.approx(widest)
    assert np.all(radius_under <= widest + 1e-9)

    # Sustained over-coverage drives alpha_t past 1; the radius clamps at the
    # narrowest reference score.
    over = aci.conformalise_aci(
        cal_lower,
        cal_upper,
        cal_y,
        np.full(n_test, -100.0),
        np.full(n_test, 100.0),
        np.zeros(n_test),
        alpha=0.2,
        gamma=0.5,
    )
    radius_over = over.conformal.upper - 100.0
    assert over.alpha_trajectory.max() >= 1.0
    assert radius_over.min() == pytest.approx(narrowest)
    assert np.all(radius_over >= narrowest - 1e-9)


def test_conformalise_aci_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    cal_y = rng.normal(0.0, 1.0, 300)
    test_y = rng.normal(0.0, 1.5, 200)
    cal_lower, cal_upper = np.full(300, -1.0), np.full(300, 1.0)
    test_lower, test_upper = np.full(200, -1.0), np.full(200, 1.0)

    first = aci.conformalise_aci(
        cal_lower,
        cal_upper,
        cal_y,
        test_lower,
        test_upper,
        test_y,
        alpha=0.2,
        gamma=0.05,
    )
    second = aci.conformalise_aci(
        cal_lower,
        cal_upper,
        cal_y,
        test_lower,
        test_upper,
        test_y,
        alpha=0.2,
        gamma=0.05,
    )

    assert np.array_equal(first.conformal.lower, second.conformal.lower)
    assert np.array_equal(first.conformal.upper, second.conformal.upper)
    assert np.array_equal(first.alpha_trajectory, second.alpha_trajectory)
