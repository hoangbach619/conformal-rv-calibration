"""Aggregated Adaptive Conformal Inference (Zaffran et al. 2022).

A named baseline to beat. AgACI runs a grid of ACI experts at different
learning rates and aggregates their interval bounds online, so no single gamma
has to be chosen. It is the strongest of the pre-2023 baselines in this study.

Shared family. Each expert is the ACI of aci.py, which reads its radius off the
same fixed calibration reference with the same smooth, clamped quantile (see
_online.py) shared by the whole online family. The expert bounds are therefore
always finite (the quantile saturates at the calibration extremes rather than
collapsing to infinity), so the aggregation needs no special-casing. CQR alone
keeps the discrete order statistic for its finite-sample guarantee.

Method. For a fixed log-spaced grid of gammas, the lower and upper bounds are
aggregated separately by exponentially weighted averaging: each expert's weight
decays with its pinball loss at the matching quantile level (``alpha/2`` for the
lower bound, ``1 - alpha/2`` for the upper), with the standard known-horizon
rate ``eta = sqrt(8 ln K / T)``. Predicting the weighted-average bound and
updating weights only on past losses keeps the scheme online and, by the
exponential-weights guarantee, no worse than the best single-gamma ACI up to the
aggregation regret.

The aggregation here is EWA / Hedge, a variant of Zaffran et al. (2022), who
use Bernstein Online Aggregation (BOA).

Reproducibility. Deterministic given the data and grid: no random numbers (so
``conformal_rv.SEED`` does not enter) and no parallel work (so
``conformal_rv.N_JOBS = 1`` holds by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conformal_rv.conformal.aci import conformalise_aci
from conformal_rv.conformal.cqr import ConformalResult

# Fixed log-spaced grid of learning rates spanning slow to fast adaptation.
DEFAULT_GAMMAS: tuple[float, ...] = tuple(
    float(gamma) for gamma in np.geomspace(0.001, 0.5, 8)
)


@dataclass(frozen=True)
class AgACIResult:
    """An AgACI run: the aggregated interval and the final expert weights.

    ``weights_lower`` and ``weights_upper`` are the per-expert weights after the
    last step, aligned with ``gammas``; they sum to one and reveal which
    learning rates the aggregation settled on for each bound.
    """

    conformal: ConformalResult
    gammas: np.ndarray
    weights_lower: np.ndarray
    weights_upper: np.ndarray


def _pinball_loss(y: float, predictions: np.ndarray, tau: float) -> np.ndarray:
    """Pinball loss at level ``tau`` of each expert's bound."""
    residual = y - predictions
    losses: np.ndarray = np.where(
        residual >= 0.0, tau * residual, (tau - 1.0) * residual
    )
    return losses


def _aggregate_bounds(
    bounds: np.ndarray, y: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exponentially weighted online aggregation of one bound across experts.

    The expert bounds are finite (the shared quantile saturates at the
    calibration extremes), so the aggregate is the plain weighted average.
    """
    steps, experts = bounds.shape
    eta = float(np.sqrt(8.0 * np.log(experts) / steps))

    weights: np.ndarray = np.full(experts, 1.0 / experts)
    aggregate = np.empty(steps, dtype=float)
    for t in range(steps):
        aggregate[t] = float(np.dot(weights, bounds[t]))

        # Update on the loss revealed at t, shifting by the minimum first to
        # avoid underflow (a constant factor cancels in the renormalisation).
        losses = _pinball_loss(float(y[t]), bounds[t], tau)
        weights = weights * np.exp(-eta * (losses - losses.min()))
        weights = weights / float(weights.sum())
    return aggregate, weights


def conformalise_agaci(
    cal_lower: np.ndarray,
    cal_upper: np.ndarray,
    cal_y: np.ndarray,
    test_lower: np.ndarray,
    test_upper: np.ndarray,
    test_y: np.ndarray,
    alpha: float,
    gammas: tuple[float, ...] = DEFAULT_GAMMAS,
) -> AgACIResult:
    """Run AgACI over one index and one horizon's calibration and test streams."""
    test_y = np.asarray(test_y, dtype=float)

    lower_experts: list[np.ndarray] = []
    upper_experts: list[np.ndarray] = []
    for gamma in gammas:
        expert = conformalise_aci(
            cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha, gamma
        )
        lower_experts.append(expert.conformal.lower)
        upper_experts.append(expert.conformal.upper)

    lower_matrix = np.column_stack(lower_experts)
    upper_matrix = np.column_stack(upper_experts)

    # Each bound is the quantile at its own level: alpha/2 below, 1 - alpha/2
    # above, so its pinball loss scores the right tail of the interval.
    aggregate_lower, weights_lower = _aggregate_bounds(
        lower_matrix, test_y, alpha / 2.0
    )
    aggregate_upper, weights_upper = _aggregate_bounds(
        upper_matrix, test_y, 1.0 - alpha / 2.0
    )

    covered: np.ndarray = (test_y >= aggregate_lower) & (test_y <= aggregate_upper)
    conformal = ConformalResult(
        lower=aggregate_lower, upper=aggregate_upper, y=test_y, covered=covered
    )
    return AgACIResult(
        conformal=conformal,
        gammas=np.asarray(gammas, dtype=float),
        weights_lower=weights_lower,
        weights_upper=weights_upper,
    )
