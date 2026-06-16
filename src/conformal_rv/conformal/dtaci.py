"""Dynamically-tuned Adaptive Conformal Inference (Gibbs and Candes 2024).

Primary online correction under test. DtACI runs K ACI experts over a fixed
gamma grid and aggregates them in alpha-space, so the effective learning rate
auto-tunes to the regime without a hand-set gamma.

Shared family. Every expert reads its radius off the same fixed calibration
reference with the same smooth, clamped quantile (see _online.py), so a DtACI
expert is exactly a single fixed-gamma ACI and the methods differ only in their
adaptation. CQR alone keeps the discrete order statistic for its finite-sample
guarantee. Using the fixed reference makes the level alpha the sole adaptation
knob: an expanding window would absorb the shift and leave nothing for gamma to
tune.

Scheme (following the 2024 paper, not AgACI's plain EWA). Each expert i keeps
its own level alpha_t^i and updates it on *its own* coverage error with gamma_i.
The output level is the weighted average alpha_bar = sum_i p_i alpha_t^i, and the
prediction interval uses the conformal quantile at ``1 - alpha_bar``. The
weights are exponential in each expert's recent pinball loss, but -- and this is
the difference from plain EWA -- they are mixed back toward uniform every step:

    w_bar_i ∝ w_i * exp(-eta * loss_i),   w_i <- (1 - sigma) * w_bar_i + sigma/K.

The ``sigma/K`` floor keeps every expert alive, so when the regime shifts the
weights can re-tune toward the now-better gammas. Plain EWA has no floor: its
weights collapse onto the historically best expert and never recover after a
shift. ``eta`` and ``sigma`` follow the paper's localisation length I (the
scale of a stationary stretch): ``sigma = 1/(2I)`` and a dimensionless
``eta = sqrt(8 ln K / I)`` applied to the pinball losses scaled by a single,
global loss scale (the spread of the reference scores), so the weights track
cumulative performance rather than per-step noise.

Reproducibility. Deterministic given the data and grid: no random numbers (so
``conformal_rv.SEED`` does not enter) and no parallel work (so
``conformal_rv.N_JOBS = 1`` holds by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conformal_rv.conformal._online import calibration_reference, reference_radius
from conformal_rv.conformal.cqr import ConformalResult, conformity_scores

# Fixed log-spaced grid of learning rates spanning slow to fast adaptation.
DEFAULT_GAMMAS: tuple[float, ...] = tuple(
    float(gamma) for gamma in np.geomspace(0.001, 0.5, 8)
)

# Localisation length I: the scale of a stationary stretch the method should
# re-tune within. Sets the meta rate and the re-exploration floor.
_LOCALISATION = 100


@dataclass(frozen=True)
class DtACIResult:
    """A DtACI run: the aggregated interval and the expert weight trajectory.

    ``weight_trajectory`` is ``(steps, K)``, the weights used to form the output
    at each step, aligned with ``gammas``; each row sums to one.
    """

    conformal: ConformalResult
    gammas: np.ndarray
    weight_trajectory: np.ndarray


def _pinball(score: float, radius: float, tau: float) -> float:
    """Pinball loss at level ``tau`` of a radius against the realised score."""
    residual = score - radius
    return tau * residual if residual >= 0.0 else (tau - 1.0) * residual


def conformalise_dtaci(
    cal_lower: np.ndarray,
    cal_upper: np.ndarray,
    cal_y: np.ndarray,
    test_lower: np.ndarray,
    test_upper: np.ndarray,
    test_y: np.ndarray,
    alpha: float,
    gammas: tuple[float, ...] = DEFAULT_GAMMAS,
) -> DtACIResult:
    """Run DtACI over one index and one horizon's calibration and test streams."""
    test_lower = np.asarray(test_lower, dtype=float)
    test_upper = np.asarray(test_upper, dtype=float)
    test_y = np.asarray(test_y, dtype=float)

    # Fixed reference: the calibration scores, sorted once. Every expert reads
    # its radius off this same reference, so only the level alpha differs.
    sorted_calibration = calibration_reference(cal_lower, cal_upper, cal_y)
    test_scores = conformity_scores(test_lower, test_upper, test_y)

    gamma_array = np.asarray(gammas, dtype=float)
    experts = gamma_array.shape[0]
    localisation = float(_LOCALISATION)
    sigma = 1.0 / (2.0 * localisation)
    eta = float(np.sqrt(8.0 * np.log(experts) / localisation))
    tau = 1.0 - alpha
    # A single, global loss scale (the spread of the reference scores) makes eta
    # dimensionless. Crucially it is constant across steps: normalising each step
    # by its own range instead would discard the loss magnitude and leave the
    # weights chasing per-step noise rather than cumulative performance.
    loss_scale = float(np.std(sorted_calibration)) or 1.0

    weights: np.ndarray = np.full(experts, 1.0 / experts)
    alpha_experts: np.ndarray = np.full(experts, float(alpha))

    steps = int(test_y.shape[0])
    lowers = np.empty(steps, dtype=float)
    uppers = np.empty(steps, dtype=float)
    covered = np.empty(steps, dtype=bool)
    weight_trajectory = np.empty((steps, experts), dtype=float)

    for t in range(steps):
        weight_trajectory[t] = weights

        alpha_bar = float(np.dot(weights, alpha_experts))
        q_bar = reference_radius(sorted_calibration, alpha_bar)
        lowers[t] = float(test_lower[t]) - q_bar
        uppers[t] = float(test_upper[t]) + q_bar
        score = float(test_scores[t])
        covered[t] = bool(score <= q_bar)

        losses = np.empty(experts, dtype=float)
        errors = np.empty(experts, dtype=float)
        for i in range(experts):
            radius = reference_radius(sorted_calibration, float(alpha_experts[i]))
            losses[i] = _pinball(score, radius, tau)
            errors[i] = 1.0 if score > radius else 0.0

        # Exponential weights on the globally-scaled losses, shifted by the
        # per-step minimum only for numerical stability (a constant factor that
        # cancels in the renormalisation), then mixed back toward uniform by the
        # sigma floor -- the re-exploration that is the DtACI difference.
        scaled = losses / loss_scale
        decayed = weights * np.exp(-eta * (scaled - scaled.min()))
        decayed = decayed / decayed.sum()
        weights = (1.0 - sigma) * decayed + sigma / experts

        # Each expert moves its own level on its own coverage error.
        alpha_experts = alpha_experts + gamma_array * (alpha - errors)

    conformal = ConformalResult(lower=lowers, upper=uppers, y=test_y, covered=covered)
    return DtACIResult(
        conformal=conformal,
        gammas=gamma_array,
        weight_trajectory=weight_trajectory,
    )
