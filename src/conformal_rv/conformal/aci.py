"""Adaptive Conformal Inference (Gibbs and Candes 2021).

A named baseline to beat. ACI adjusts the effective miscoverage level alpha_t
with a single fixed learning rate gamma, using the online coverage error. Its
known weakness through regime breaks is exactly what the primary endpoint
probes.

Method. Starting from ``alpha_t = alpha``, at each test step t the correction
``Q_t`` is the finite-sample ``(1 - alpha_t)`` conformal quantile of every
conformity score seen so far -- the calibration scores plus the revealed test
scores from strictly earlier steps (an expanding window). The interval is
``[lower_t - Q_t, upper_t + Q_t]``; with ``err_t = 1`` when ``y_t`` falls
outside it, the level updates as

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t).

So under-coverage (``err_t = 1``) lowers ``alpha_t``, which raises ``Q_t`` and
widens the interval; sustained over-coverage raises ``alpha_t`` and tightens
it. The saturation cases are handled explicitly: ``alpha_t <= 0`` gives a
correction of ``+inf`` (the interval covers all of R) and ``alpha_t >= 1``
gives ``-inf`` (an empty interval).

Reproducibility. Given the data and gamma the run is deterministic: no random
numbers (so ``conformal_rv.SEED`` does not enter) and no parallel work (so
``conformal_rv.N_JOBS = 1`` holds by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conformal_rv.conformal.cqr import (
    ConformalResult,
    conformal_correction,
    conformity_scores,
)


@dataclass(frozen=True)
class ACIResult:
    """An ACI run: the calibrated interval and the ``alpha_t`` trajectory.

    ``alpha_trajectory`` is the level used at each test step, length equal to
    the test stream, so the adaptation can be inspected against coverage.
    """

    conformal: ConformalResult
    alpha_trajectory: np.ndarray


def _aci_correction(scores: np.ndarray, alpha_t: float) -> float:
    """Conformal correction at the current level, with the saturation cases.

    ``alpha_t <= 0`` cannot be met by any finite quantile, so the interval
    covers everything (``+inf``); ``alpha_t >= 1`` asks for a degenerate level,
    so the interval is empty (``-inf``). In between, the finite-sample CQR
    quantile applies.
    """
    if alpha_t <= 0.0:
        return float("inf")
    if alpha_t >= 1.0:
        return float("-inf")
    return conformal_correction(scores, alpha_t)


def conformalise_aci(
    cal_lower: np.ndarray,
    cal_upper: np.ndarray,
    cal_y: np.ndarray,
    test_lower: np.ndarray,
    test_upper: np.ndarray,
    test_y: np.ndarray,
    alpha: float,
    gamma: float,
) -> ACIResult:
    """Run ACI over one index and one horizon's calibration and test streams."""
    test_lower = np.asarray(test_lower, dtype=float)
    test_upper = np.asarray(test_upper, dtype=float)
    test_y = np.asarray(test_y, dtype=float)

    calibration_scores = conformity_scores(cal_lower, cal_upper, cal_y)
    # Test scores are precomputed for convenience but only consulted for steps
    # strictly before t (``test_scores[:t]``), so the window never sees t.
    test_scores = conformity_scores(test_lower, test_upper, test_y)

    steps = int(test_y.shape[0])
    lowers = np.empty(steps, dtype=float)
    uppers = np.empty(steps, dtype=float)
    covered = np.empty(steps, dtype=bool)
    alpha_trajectory = np.empty(steps, dtype=float)

    alpha_t: float = float(alpha)
    for t in range(steps):
        alpha_trajectory[t] = alpha_t
        seen = np.concatenate([calibration_scores, test_scores[:t]])
        correction = _aci_correction(seen, alpha_t)

        lower_t = float(test_lower[t]) - correction
        upper_t = float(test_upper[t]) + correction
        lowers[t] = lower_t
        uppers[t] = upper_t

        is_covered = bool(lower_t <= float(test_y[t]) <= upper_t)
        covered[t] = is_covered
        error = 0.0 if is_covered else 1.0
        alpha_t = alpha_t + gamma * (alpha - error)

    conformal = ConformalResult(lower=lowers, upper=uppers, y=test_y, covered=covered)
    return ACIResult(conformal=conformal, alpha_trajectory=alpha_trajectory)
