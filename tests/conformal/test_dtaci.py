"""Unit tests for Dynamically-tuned Adaptive Conformal Inference.

DtACI is checked for restoring coverage under shift, for re-tuning toward the
faster gammas after a regime break (a controlled comparison against a calm
continuation), for the clamped radius edge cases, and for determinism. None
touch the network.
"""

from __future__ import annotations

import numpy as np

from conformal_rv.conformal import dtaci
from conformal_rv.conformal.cqr import conformalise_cqr

_N_CAL = 1000


def _band(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full(n, -1.0), np.full(n, 1.0)


def test_dtaci_restores_coverage_under_shift_better_than_cqr() -> None:
    rng = np.random.default_rng(1)
    cal_y = rng.normal(0.0, 1.0, _N_CAL)
    test_y = rng.normal(0.0, 2.0, 1000)  # wider than calibration
    cal_lower, cal_upper = _band(_N_CAL)
    test_lower, test_upper = _band(1000)

    static = conformalise_cqr(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )
    result = dtaci.conformalise_dtaci(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )

    assert static.coverage < 0.6
    assert abs(result.conformal.coverage - 0.8) < abs(static.coverage - 0.8)
    assert result.conformal.coverage > 0.72


def test_dtaci_upweights_faster_gammas_after_a_break() -> None:
    # Controlled comparison: an identical calm prefix, then one stream breaks to
    # a wider regime and one stays calm. Sharing the prefix fixes the weights at
    # the break point, so any difference afterwards is the break's doing. The
    # break must pull weight onto the faster gammas. Averaged over seeds to tame
    # the stochastic weight path.
    calm, tail, window = 300, 200, 120
    cal_lower, cal_upper = _band(_N_CAL)
    test_lower, test_upper = _band(calm + tail)

    broken_fast_mass = []
    calm_fast_mass = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        cal_y = rng.normal(0.0, 1.0, _N_CAL)
        prefix = rng.normal(0.0, 1.0, calm)
        broken = np.concatenate([prefix, rng.normal(0.0, 3.0, tail)])
        stayed = np.concatenate([prefix, rng.normal(0.0, 1.0, tail)])

        after_break = dtaci.conformalise_dtaci(
            cal_lower, cal_upper, cal_y, test_lower, test_upper, broken, alpha=0.2
        )
        after_calm = dtaci.conformalise_dtaci(
            cal_lower, cal_upper, cal_y, test_lower, test_upper, stayed, alpha=0.2
        )
        # Mass on the faster half of the gammas, just after the break point.
        transient = slice(calm, calm + window)
        broken_fast_mass.append(
            after_break.weight_trajectory[transient, 4:].sum(axis=1).mean()
        )
        calm_fast_mass.append(
            after_calm.weight_trajectory[transient, 4:].sum(axis=1).mean()
        )

    assert np.mean(broken_fast_mass) > np.mean(calm_fast_mass)


def test_reference_radius_clamps_at_the_edges() -> None:
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    n = scores.shape[0]
    # alpha_t <= 0 asks for a level >= 1: the widest reference radius (the max).
    assert dtaci._reference_radius(scores, n, -0.1) == 4.0
    # alpha_t >= 1 asks for a level <= 0: the narrowest (the min).
    assert dtaci._reference_radius(scores, n, 1.1) == 0.0
    # In range, the radius is the linearly-interpolated quantile of the level.
    assert dtaci._reference_radius(scores, n, 0.5) == 2.0  # level 0.5 -> midpoint


def test_conformalise_dtaci_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    cal_y = rng.normal(0.0, 1.0, 400)
    test_y = rng.normal(0.0, 2.0, 300)
    cal_lower, cal_upper = _band(400)
    test_lower, test_upper = _band(300)

    first = dtaci.conformalise_dtaci(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )
    second = dtaci.conformalise_dtaci(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )

    assert np.array_equal(first.conformal.lower, second.conformal.lower)
    assert np.array_equal(first.conformal.upper, second.conformal.upper)
    assert np.array_equal(first.weight_trajectory, second.weight_trajectory)
