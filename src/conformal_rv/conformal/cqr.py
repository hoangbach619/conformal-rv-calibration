"""Conformalised Quantile Regression (Romano, Patterson, Candes 2019).

Base interval constructor for the study. It takes the lower/upper conditional
quantiles from a quantile model and calibrates them on a held-out set so the
marginal coverage is exact in the exchangeable case. Everything online is a
correction on top of this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def conformity_scores(
    calibration_intervals: "np.ndarray", calibration_targets: "np.ndarray"
) -> "np.ndarray":
    """CQR nonconformity scores: signed distance outside the quantile band."""
    raise NotImplementedError


def calibrate(scores: "np.ndarray", alpha: float) -> float:
    """Return the finite-sample-adjusted (1 - alpha) quantile of the scores."""
    raise NotImplementedError


def apply(test_intervals: "np.ndarray", offset: float) -> "np.ndarray":
    """Widen test quantile bands by the calibrated offset."""
    raise NotImplementedError
