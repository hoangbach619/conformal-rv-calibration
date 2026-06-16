"""Shared machinery for the online conformal family (ACI, AgACI, DtACI).

These three methods read their interval radius off the *same* fixed calibration
reference using the *same* smooth, interpolated quantile clamped to ``[0, 1]``.
Sharing this means a single fixed-gamma ACI run is exactly one DtACI expert, so
a comparison between the methods isolates the adaptation mechanism (a single
gamma, EWA aggregation, or dynamic tuning) rather than a difference in the score
window or the quantile estimator.

CQR alone keeps the discrete order-statistic quantile (see cqr.py): its
finite-sample coverage guarantee depends on that exact rank, so it is
deliberately not replaced by this smooth, clamped variant.
"""

from __future__ import annotations

import numpy as np

from conformal_rv.conformal.cqr import conformity_scores


def calibration_reference(
    cal_lower: np.ndarray, cal_upper: np.ndarray, cal_y: np.ndarray
) -> np.ndarray:
    """The sorted conformity scores used as the fixed quantile reference."""
    return np.sort(conformity_scores(cal_lower, cal_upper, cal_y))


def reference_radius(sorted_reference: np.ndarray, alpha_t: float) -> float:
    """Smooth, clamped ``(1 - alpha_t)`` quantile of the fixed reference scores.

    The quantile is linearly interpolated and the level is clamped to
    ``[0, 1]``: ``alpha_t <= 0`` returns the widest reference radius (the largest
    score) and ``alpha_t >= 1`` the narrowest (the smallest). Interpolating keeps
    the radius a smooth function of the level, so an adaptation that overshoots
    gets a bounded radius, and clamping means the interval saturates at the
    widest and narrowest the calibration scores can express rather than
    collapsing to an infinite or an empty set.
    """
    n = sorted_reference.shape[0]
    level = 1.0 - alpha_t
    if level <= 0.0:
        return float(sorted_reference[0])
    if level >= 1.0:
        return float(sorted_reference[-1])
    position = level * (n - 1)
    lower = int(np.floor(position))
    if lower + 1 >= n:
        return float(sorted_reference[lower])
    fraction = position - lower
    step = sorted_reference[lower + 1] - sorted_reference[lower]
    return float(sorted_reference[lower] + fraction * step)
