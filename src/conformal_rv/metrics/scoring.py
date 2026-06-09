"""Proper scoring rules and PIT calibration diagnostics.

Pinball (quantile) loss is reported at horizons 1, 5, 10 and 22 trading days.
PIT values are tested for uniformity with a Kolmogorov-Smirnov statistic;
under correct calibration the PIT is uniform, so KS is a single sharp summary
of distributional miscalibration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Forecast horizons in trading days at which pinball loss is reported.
HORIZONS: tuple[int, int, int, int] = (1, 5, 10, 22)


def pinball_loss(
    predicted_quantile: "np.ndarray", targets: "np.ndarray", quantile: float
) -> float:
    """Average pinball loss at one quantile level."""
    raise NotImplementedError


def pit_values(
    cdf_at_target: "np.ndarray",
) -> "np.ndarray":
    """Probability integral transform values for the realised targets."""
    raise NotImplementedError


def pit_ks_statistic(pit: "np.ndarray") -> tuple[float, float]:
    """KS statistic and p-value of the PIT against the uniform."""
    raise NotImplementedError
