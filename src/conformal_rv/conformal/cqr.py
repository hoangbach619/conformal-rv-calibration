"""Conformalised Quantile Regression (Romano, Patterson and Candes 2019).

Base interval constructor for the study. It takes the lower and upper
conditional quantiles from a quantile model and calibrates them on a held-out
set so the marginal coverage is exact in the exchangeable case. Every online
method is a correction layered on top of this.

Method. On the calibration set the nonconformity score is

    E_i = max(lower_i - y_i, y_i - upper_i),

the signed distance by which y_i falls outside the raw quantile band (negative
when it is comfortably inside). The correction Q is the
``ceil((n+1)(1-alpha))/n`` empirical quantile of the E_i -- the finite-sample
adjusted (1-alpha) quantile -- and the calibrated test interval is
``[lower - Q, upper + Q]``. Q can be negative: when the raw band over-covers,
every E_i is negative and the correction tightens the interval, which is
correct.

Assumptions and caveats. CQR's guarantee assumes the calibration and test
scores are exchangeable. Here they are not exactly: the target is a windowed,
overlapping h-step quantity and volatility is autocorrelated, so the
calibration scores are dependent and the coverage is only approximate. That
gap is the whole point of the study -- CQR is the baseline the online methods
are meant to improve through a regime shift, not a method expected to hold
conditional coverage on its own.

Reproducibility. The construction is deterministic: no random numbers (so
``conformal_rv.SEED`` does not enter) and no parallel work (so
``conformal_rv.N_JOBS = 1`` holds by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalResult:
    """Calibrated interval and its realised coverage, per test point.

    General enough for the online methods to reuse: ``lower`` and ``upper`` are
    the calibrated bounds, ``y`` the realised outcomes, and ``covered`` the
    per-point hit flags (lower <= y <= upper).
    """

    lower: np.ndarray
    upper: np.ndarray
    y: np.ndarray
    covered: np.ndarray

    @property
    def coverage(self) -> float:
        """Marginal coverage: the fraction of test points inside the interval."""
        return float(self.covered.mean())


def conformity_scores(
    lower: np.ndarray, upper: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """CQR nonconformity scores ``max(lower - y, y - upper)``, one per point.

    Negative where the outcome sits inside the raw band, positive where it
    falls outside; the magnitude is the distance to the nearer bound.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    y = np.asarray(y, dtype=float)
    scores: np.ndarray = np.maximum(lower - y, y - upper)
    return scores


def conformal_correction(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample adjusted ``(1 - alpha)`` quantile of the scores.

    The correction is the ``k``-th smallest score with
    ``k = ceil((n + 1)(1 - alpha))``. When ``k > n`` -- too few calibration
    points for a finite correction at this level -- it is ``+inf``, so the
    interval conservatively covers everything. The value may be negative when
    the raw band over-covers, which tightens the interval.
    """
    values = np.asarray(scores, dtype=float)
    n = values.shape[0]
    if n == 0:
        raise ValueError("CQR needs at least one calibration score")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(values)[rank - 1])


def conformalise_cqr(
    cal_lower: np.ndarray,
    cal_upper: np.ndarray,
    cal_y: np.ndarray,
    test_lower: np.ndarray,
    test_upper: np.ndarray,
    test_y: np.ndarray,
    alpha: float,
) -> ConformalResult:
    """Calibrate a quantile band by CQR for one index and one horizon.

    The caller passes the lower and upper quantile forecasts and realised
    outcomes for the calibration and test windows of a single series. For the
    80% interval ``alpha`` is 0.20.
    """
    correction = conformal_correction(
        conformity_scores(cal_lower, cal_upper, cal_y), alpha
    )

    test_lower = np.asarray(test_lower, dtype=float)
    test_upper = np.asarray(test_upper, dtype=float)
    test_y = np.asarray(test_y, dtype=float)

    lower: np.ndarray = test_lower - correction
    upper: np.ndarray = test_upper + correction
    covered: np.ndarray = (test_y >= lower) & (test_y <= upper)
    return ConformalResult(lower=lower, upper=upper, y=test_y, covered=covered)
