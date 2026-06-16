"""Proper scoring rules and PIT calibration diagnostics.

Pinball (quantile) loss is the proper scoring rule for the interval, evaluated
at its two quantile levels (``alpha/2`` and ``1 - alpha/2``) and reported at
horizons 1, 5, 10 and 22 trading days.

PIT values are tested for uniformity with a Kolmogorov-Smirnov statistic
(Diebold, Gunther and Tay, 1998): under correct calibration the probability
integral transform of the outcomes through the predictive CDF is uniform, so KS
against the uniform is a single sharp summary of distributional miscalibration.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kstest

# Forecast horizons in trading days at which pinball loss is reported.
HORIZONS: tuple[int, int, int, int] = (1, 5, 10, 22)


def _pinball(targets: np.ndarray, forecast: np.ndarray, tau: float) -> np.ndarray:
    """Per-point pinball loss of a quantile forecast at level ``tau``."""
    residual = targets - forecast
    loss: np.ndarray = np.where(residual >= 0.0, tau * residual, (tau - 1.0) * residual)
    return loss


def pinball_loss(
    lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float
) -> float:
    """Average pinball loss at the two interval quantiles.

    ``lower`` is the ``alpha/2`` quantile forecast and ``upper`` the
    ``1 - alpha/2`` quantile forecast. The loss is averaged over both quantiles
    and all points, so it is the proper scoring rule for the interval.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    y = np.asarray(y, dtype=float)

    loss_low = _pinball(y, lower, alpha / 2.0)
    loss_high = _pinball(y, upper, 1.0 - alpha / 2.0)
    return float(np.mean(np.concatenate([loss_low, loss_high])))


def pit_values(
    quantile_forecasts: np.ndarray, y: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    """Probability integral transform of ``y`` through the predicted CDF.

    ``quantile_forecasts`` is ``(n, Q)`` predicted quantile values at the
    ascending ``levels`` ``(Q,)``; the predictive CDF of each row is the
    piecewise-linear interpolation through ``(value, level)``, and the PIT is
    that CDF evaluated at the outcome. Outcomes beyond the predicted quantiles
    clamp to the extreme levels.
    """
    forecasts = np.asarray(quantile_forecasts, dtype=float)
    targets = np.asarray(y, dtype=float)
    grid = np.asarray(levels, dtype=float)

    pit = np.empty(targets.shape[0], dtype=float)
    for i in range(targets.shape[0]):
        pit[i] = float(np.interp(targets[i], forecasts[i], grid))
    return pit


def ks_uniformity(pit: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and p-value of the PIT against the uniform."""
    result = kstest(np.asarray(pit, dtype=float), "uniform")
    return float(result.statistic), float(result.pvalue)
