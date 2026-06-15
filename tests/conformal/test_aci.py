"""Unit tests for Adaptive Conformal Inference.

The headline test drives a distribution shift under which static CQR
under-covers and checks that ACI restores coverage. The rest pin the alpha_t
adaptation direction, the saturation cases, and determinism. None touch the
network.
"""

from __future__ import annotations

import numpy as np

from conformal_rv.conformal import aci
from conformal_rv.conformal.cqr import conformalise_cqr


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


def test_saturation_covers_all_then_empties() -> None:
    cal_y = np.zeros(50)
    cal_lower, cal_upper = np.full(50, -0.01), np.full(50, 0.01)

    # Every test point is far outside a pinhole band: sustained under-coverage
    # drives alpha_t <= 0, where the correction is +inf and the interval covers
    # all of R (so those steps are covered).
    n_test = 30
    far_y = np.full(n_test, 100.0)
    covers_all = aci.conformalise_aci(
        cal_lower,
        cal_upper,
        cal_y,
        np.full(n_test, -0.01),
        np.full(n_test, 0.01),
        far_y,
        alpha=0.2,
        gamma=0.5,
    )
    assert covers_all.alpha_trajectory.min() <= 0.0
    infinite = np.isposinf(covers_all.conformal.upper)
    assert infinite.any()
    assert covers_all.conformal.covered[infinite].all()

    # Every test point sits inside a huge band: sustained over-coverage drives
    # alpha_t >= 1, where the correction is -inf and the interval is empty.
    inside_y = np.zeros(n_test)
    empties = aci.conformalise_aci(
        cal_lower,
        cal_upper,
        cal_y,
        np.full(n_test, -100.0),
        np.full(n_test, 100.0),
        inside_y,
        alpha=0.2,
        gamma=0.5,
    )
    assert empties.alpha_trajectory.max() >= 1.0
    empty = empties.conformal.lower > empties.conformal.upper
    assert empty.any()
    assert not empties.conformal.covered[empty].any()


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
