"""Multiple-testing control across the method-by-regime comparisons.

The study runs many tests (several methods, several break windows, several
estimators). Benjamini-Hochberg controls the false-discovery rate at 0.10
across the family of index-by-horizon-by-method tests. The effective number of
tests is reported alongside, because correlated tests make the nominal count
misleading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pre-registered false-discovery-rate level for Benjamini-Hochberg.
FDR_LEVEL: float = 0.10


@dataclass(frozen=True)
class BenjaminiHochbergResult:
    """A Benjamini-Hochberg run: the rejection mask and the adjusted p-values.

    ``rejected[i]`` and ``adjusted_pvalues[i]`` align with the input p-values;
    the adjusted (BH q-) values are monotone non-decreasing in the ranked
    p-values and clamped to one.
    """

    rejected: np.ndarray
    adjusted_pvalues: np.ndarray


def benjamini_hochberg(
    pvalues: np.ndarray, fdr: float = FDR_LEVEL
) -> BenjaminiHochbergResult:
    """Benjamini-Hochberg step-up procedure at false-discovery rate ``fdr``.

    Reject every hypothesis whose p-value is at most the largest ranked p-value
    ``p_(k)`` satisfying ``p_(k) <= (k / m) * fdr``; ties are handled by the
    threshold comparison. The adjusted p-values are the running minimum of
    ``(m / rank) * p_(rank)`` from the largest rank down, clamped to one.
    """
    p = np.asarray(pvalues, dtype=float)
    m = int(p.shape[0])
    if m == 0:
        raise ValueError("need at least one p-value")

    order = np.argsort(p, kind="stable")
    ranked = p[order]
    ranks = np.arange(1, m + 1)

    below = ranked <= (ranks / m) * fdr
    threshold = (
        float(ranked[int(np.nonzero(below)[0].max())])
        if bool(below.any())
        else float("-inf")
    )
    rejected: np.ndarray = p <= threshold

    # Adjusted (q-) values: enforce monotonicity by a right-to-left running min.
    inflated = (m / ranks) * ranked
    adjusted_sorted: np.ndarray = np.minimum.accumulate(inflated[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return BenjaminiHochbergResult(rejected=rejected, adjusted_pvalues=adjusted)


def effective_number_of_tests(p_values: np.ndarray) -> float:
    """Estimate the effective number of independent tests in the family.

    Not implemented in this phase: a faithful estimate needs the correlation
    structure of the test statistics, not the p-values alone, so it lands with
    the analysis that has that structure to hand.
    """
    raise NotImplementedError("effective number of tests is a later phase")
