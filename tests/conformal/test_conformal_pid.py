"""Unit tests for Conformal PID control.

The headline tests show PID restoring coverage under shift and the integral term
removing a persistent bias that a proportional-only controller leaves. The rest
pin the anti-windup saturation, the scorecaster hook, and determinism. None
touch the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_rv.conformal import conformal_pid
from conformal_rv.conformal.cqr import (
    conformal_correction,
    conformalise_cqr,
    conformity_scores,
)


def test_pid_restores_coverage_under_shift_better_than_cqr() -> None:
    rng = np.random.default_rng(0)
    n_cal, n_test = 1000, 1500
    cal_y = rng.normal(0.0, 1.0, n_cal)
    test_y = rng.normal(0.0, 2.0, n_test)  # wider than calibration
    cal_lower, cal_upper = np.full(n_cal, -1.0), np.full(n_cal, 1.0)
    test_lower, test_upper = np.full(n_test, -1.0), np.full(n_test, 1.0)

    static = conformalise_cqr(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )
    pid = conformal_pid.conformalise_pid(
        cal_lower,
        cal_upper,
        cal_y,
        test_lower,
        test_upper,
        test_y,
        alpha=0.2,
        k_p=0.2,
        k_i=2.0,
        integral_scale=5.0,
    )

    assert static.coverage < 0.6
    assert abs(pid.conformal.coverage - 0.8) < abs(static.coverage - 0.8)
    assert pid.conformal.coverage > 0.7


def test_integral_term_removes_persistent_bias() -> None:
    rng = np.random.default_rng(1)
    n_cal, n_test = 1000, 1500
    cal_y = rng.normal(0.0, 1.0, n_cal)
    # A constant mean offset: the calibration-tuned band is systematically off.
    test_y = rng.normal(2.0, 1.0, n_test)
    cal_lower, cal_upper = np.full(n_cal, -1.0), np.full(n_cal, 1.0)
    test_lower, test_upper = np.full(n_test, -1.0), np.full(n_test, 1.0)

    proportional_only = conformal_pid.conformalise_pid(
        cal_lower,
        cal_upper,
        cal_y,
        test_lower,
        test_upper,
        test_y,
        alpha=0.2,
        k_p=0.2,
        k_i=0.0,
    )
    full_pid = conformal_pid.conformalise_pid(
        cal_lower,
        cal_upper,
        cal_y,
        test_lower,
        test_upper,
        test_y,
        alpha=0.2,
        k_p=0.2,
        k_i=2.0,
        integral_scale=5.0,
    )

    # Proportional-only is memoryless and leaves a clear steady-state bias; the
    # integrator drives coverage back to nominal.
    assert proportional_only.conformal.coverage < 0.6
    assert abs(full_pid.conformal.coverage - 0.8) < abs(
        proportional_only.conformal.coverage - 0.8
    )
    assert full_pid.conformal.coverage > 0.7


def test_integral_term_is_anti_windup_bounded() -> None:
    rng = np.random.default_rng(2)
    n_cal, n_test = 200, 400
    cal_y = rng.normal(0.0, 1.0, n_cal)
    cal_lower, cal_upper = np.full(n_cal, -0.01), np.full(n_cal, 0.01)
    # Permanent under-coverage would wind up a pure integrator without bound.
    test_y = np.full(n_test, 50.0)
    test_lower, test_upper = np.full(n_test, -0.01), np.full(n_test, 0.01)
    k_p, k_i, scale = 0.2, 1.5, 3.0

    result = conformal_pid.conformalise_pid(
        cal_lower,
        cal_upper,
        cal_y,
        test_lower,
        test_upper,
        test_y,
        alpha=0.2,
        k_p=k_p,
        k_i=k_i,
        integral_scale=scale,
    )
    base = conformal_correction(conformity_scores(cal_lower, cal_upper, cal_y), 0.2)

    assert np.isfinite(result.q_trajectory).all()
    # The integral term is capped at k_i and the proportional term at 0.8 * k_p.
    assert result.q_trajectory.max() <= base + 0.8 * k_p + k_i + 1e-9


def test_scorecaster_length_is_checked() -> None:
    cal_y = np.zeros(20)
    cal_lower, cal_upper = np.full(20, -1.0), np.full(20, 1.0)
    test_y = np.zeros(10)
    test_lower, test_upper = np.full(10, -1.0), np.full(10, 1.0)
    with pytest.raises(ValueError, match="scorecaster"):
        conformal_pid.conformalise_pid(
            cal_lower,
            cal_upper,
            cal_y,
            test_lower,
            test_upper,
            test_y,
            alpha=0.2,
            scorecaster=np.zeros(5),
        )


def test_conformalise_pid_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    cal_y = rng.normal(0.0, 1.0, 300)
    test_y = rng.normal(0.0, 1.5, 200)
    cal_lower, cal_upper = np.full(300, -1.0), np.full(300, 1.0)
    test_lower, test_upper = np.full(200, -1.0), np.full(200, 1.0)

    first = conformal_pid.conformalise_pid(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )
    second = conformal_pid.conformalise_pid(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )

    assert np.array_equal(first.q_trajectory, second.q_trajectory)
    assert np.array_equal(first.conformal.covered, second.conformal.covered)
