"""Unit tests for the shared online-conformal quantile helper.

The fixed-reference, smooth, clamped quantile is the machinery ACI, AgACI and
DtACI all rest on, so that a single fixed-gamma ACI is exactly one DtACI expert.
None touch the network.
"""

from __future__ import annotations

import numpy as np

from conformal_rv.conformal import _online, aci


def test_reference_radius_clamps_at_the_edges() -> None:
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # alpha_t <= 0 asks for a level >= 1: the widest reference radius (the max).
    assert _online.reference_radius(scores, -0.1) == 4.0
    # alpha_t >= 1 asks for a level <= 0: the narrowest (the min).
    assert _online.reference_radius(scores, 1.1) == 0.0
    # In range, the radius is the linearly-interpolated quantile of the level.
    assert _online.reference_radius(scores, 0.5) == 2.0  # level 0.5 -> midpoint


def test_reference_radius_interpolates_between_scores() -> None:
    scores = np.array([0.0, 10.0])
    # level = 1 - 0.25 = 0.75; position 0.75 * (2 - 1) = 0.75 between 0 and 10.
    assert _online.reference_radius(scores, 0.25) == 7.5


def test_calibration_reference_is_the_sorted_scores() -> None:
    cal_lower, cal_upper = np.full(4, -1.0), np.full(4, 1.0)
    cal_y = np.array([0.0, 5.0, -3.0, 0.5])
    # score = max(-1 - y, y - 1): 0 -> -1, 5 -> 4, -3 -> 2, 0.5 -> -0.5.
    reference = _online.calibration_reference(cal_lower, cal_upper, cal_y)
    assert np.allclose(reference, np.sort([-1.0, 4.0, 2.0, -0.5]))


def test_single_aci_uses_the_shared_reference_radius() -> None:
    # The harmonisation claim: each ACI interval's radius is exactly the shared
    # clamped quantile at the level ACI reports, so a single fixed-gamma ACI is
    # one DtACI expert reading the same fixed reference.
    rng = np.random.default_rng(0)
    n_cal, n_test = 400, 100
    cal_lower, cal_upper = np.full(n_cal, -1.0), np.full(n_cal, 1.0)
    cal_y = rng.normal(0.0, 1.0, n_cal)
    test_lower, test_upper = np.full(n_test, -1.0), np.full(n_test, 1.0)
    test_y = rng.normal(0.0, 1.5, n_test)

    result = aci.conformalise_aci(
        cal_lower,
        cal_upper,
        cal_y,
        test_lower,
        test_upper,
        test_y,
        alpha=0.2,
        gamma=0.05,
    )
    reference = _online.calibration_reference(cal_lower, cal_upper, cal_y)

    # upper = test_upper + radius, so the radius is recoverable per step.
    radii = result.conformal.upper - test_upper
    expected = np.array(
        [_online.reference_radius(reference, a) for a in result.alpha_trajectory]
    )
    assert np.allclose(radii, expected)
