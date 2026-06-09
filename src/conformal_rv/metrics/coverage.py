"""Coverage metrics, marginal and regime-conditional.

The primary endpoint is the regime-conditional coverage gap: empirical
coverage in calm periods minus empirical coverage in the 60-trading-day
post-break window. Marginal coverage is secondary. The split between the two
is the entire point of the study, so it lives in a dedicated function rather
than being computed ad hoc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def marginal_coverage(
    intervals: "np.ndarray", targets: "np.ndarray"
) -> float:
    """Fraction of targets falling inside their interval."""
    raise NotImplementedError


def conditional_coverage(
    intervals: "np.ndarray", targets: "np.ndarray", regime_mask: "np.ndarray"
) -> float:
    """Empirical coverage restricted to the observations where mask is true."""
    raise NotImplementedError


def coverage_gap(
    intervals: "np.ndarray",
    targets: "np.ndarray",
    calm_mask: "np.ndarray",
    post_break_mask: "np.ndarray",
) -> float:
    """Primary endpoint: calm coverage minus post-break coverage."""
    raise NotImplementedError
