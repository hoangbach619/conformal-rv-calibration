"""Multiple-testing control across the method-by-regime comparisons.

The study runs many tests (several methods, several break windows, several
estimators). Benjamini-Hochberg controls the false-discovery rate at 0.10.
The effective number of tests is reported alongside, because correlated tests
make the nominal count misleading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Pre-registered false-discovery-rate level for Benjamini-Hochberg.
FDR_LEVEL: float = 0.10


def benjamini_hochberg(
    p_values: np.ndarray, fdr_level: float = FDR_LEVEL
) -> np.ndarray:
    """Return the boolean reject/accept vector under BH at the given FDR."""
    raise NotImplementedError


def effective_number_of_tests(p_values: np.ndarray) -> float:
    """Estimate the effective number of independent tests in the family."""
    raise NotImplementedError
