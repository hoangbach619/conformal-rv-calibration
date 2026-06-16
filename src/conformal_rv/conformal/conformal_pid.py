"""Conformal PID control (Angelopoulos, Candes and Tibshirani 2023).

Primary online correction under test. It treats coverage tracking as a control
problem and builds the radius q_t directly from three terms:

- Proportional: a memoryless response k_p * (err_{t-1} - alpha) to the most
  recent coverage error (err = 1 when the outcome falls outside).
- Integral: k_i * sat(sum_{s < t}(err_s - alpha)), a saturating function of the
  running error sum. This is the term that removes a *systematic* coverage
  bias: a proportional-only controller settles at a non-zero steady-state error
  when the base interval is offset, and the integrator drives that error to
  zero. The saturation is anti-windup, bounding the integral contribution.
- Scorecaster (the derivative-like forecast term): optional and OFF by default.
  A clear hook is left (the ``scorecaster`` argument, a per-step additive
  forecast); omitting it is a deliberate simplification, since the study does
  not train a separate score-forecasting model.

The radius centres on the split-conformal correction from the calibration set,
so q_t is the CQR radius nudged online; the interval is
``[base_lower_t - q_t, base_upper_t + q_t]``, exactly as cqr applies it.

Gains are problem-scale dependent (they act in the units of the conformity
score), so they are explicit arguments and reported with results rather than
hidden. Defaults are deliberately gentle.

Reproducibility. Deterministic given the data and gains: no random numbers (so
``conformal_rv.SEED`` does not enter) and no parallel work (so
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
class PIDResult:
    """A Conformal PID run: the calibrated interval and the radius trajectory.

    ``q_trajectory`` is the controlled radius at each test step, so the
    proportional/integral response can be inspected against coverage.
    """

    conformal: ConformalResult
    q_trajectory: np.ndarray


def _saturate(value: float, scale: float) -> float:
    """Smooth anti-windup saturation: near-linear for small inputs, bounded.

    ``tanh(value / scale)`` is approximately ``value / scale`` while the integral
    is within ``scale`` and tends to +/-1 beyond, so the integral term cannot
    wind up without bound.
    """
    return float(np.tanh(value / scale))


def conformalise_pid(
    cal_lower: np.ndarray,
    cal_upper: np.ndarray,
    cal_y: np.ndarray,
    test_lower: np.ndarray,
    test_upper: np.ndarray,
    test_y: np.ndarray,
    alpha: float,
    k_p: float = 0.1,
    k_i: float = 0.1,
    integral_scale: float = 1.0,
    scorecaster: np.ndarray | None = None,
) -> PIDResult:
    """Run Conformal PID over one index and one horizon's streams.

    With ``k_i = 0`` the controller is proportional-only and leaves a steady
    coverage bias whenever the base interval is offset; the default ``k_i > 0``
    integrates that error away. ``scorecaster``, when given, is added to the
    radius at each step and must match the test length.
    """
    test_lower = np.asarray(test_lower, dtype=float)
    test_upper = np.asarray(test_upper, dtype=float)
    test_y = np.asarray(test_y, dtype=float)
    test_scores = conformity_scores(test_lower, test_upper, test_y)

    # Centre the radius on the split-conformal correction, so the controller
    # perturbs the CQR baseline rather than starting from nothing.
    base_radius = conformal_correction(
        conformity_scores(cal_lower, cal_upper, cal_y), alpha
    )

    steps = int(test_y.shape[0])
    forecast = (
        np.zeros(steps, dtype=float)
        if scorecaster is None
        else np.asarray(scorecaster, dtype=float)
    )
    if forecast.shape[0] != steps:
        raise ValueError("scorecaster must match the test length")

    lowers = np.empty(steps, dtype=float)
    uppers = np.empty(steps, dtype=float)
    covered = np.empty(steps, dtype=bool)
    q_trajectory = np.empty(steps, dtype=float)

    integral = 0.0
    last_error = 0.0
    for t in range(steps):
        proportional = k_p * last_error
        integral_term = k_i * _saturate(integral, integral_scale)
        q_t = base_radius + proportional + integral_term + float(forecast[t])
        q_trajectory[t] = q_t

        lowers[t] = float(test_lower[t]) - q_t
        uppers[t] = float(test_upper[t]) + q_t
        is_covered = bool(float(test_scores[t]) <= q_t)
        covered[t] = is_covered

        # Error signal feeding both terms; the integral accumulates it, the
        # proportional term sees only the latest value.
        error = (0.0 if is_covered else 1.0) - alpha
        integral += error
        last_error = error

    conformal = ConformalResult(lower=lowers, upper=uppers, y=test_y, covered=covered)
    return PIDResult(conformal=conformal, q_trajectory=q_trajectory)
