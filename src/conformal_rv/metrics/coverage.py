"""Coverage metrics, marginal and regime-conditional.

The primary endpoint is the regime-conditional coverage gap: empirical coverage
in calm periods minus empirical coverage in the 60-trading-day post-break
window. Marginal coverage is secondary. The split between the two is the entire
point of the study, so it lives in a dedicated function rather than being
computed ad hoc.

Regime labelling. Each test point carries a label: the sentinel ``CALM`` for a
calm point, or the name of the break whose 60-day post-break window it falls
in. Coverage is then reported calm, pooled post-break, and per break.

All functions read a ``ConformalResult`` (the per-point ``covered`` flags),
which every method -- CQR and the online corrections -- returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conformal_rv.conformal.cqr import ConformalResult

# Label marking a calm test point; any other label is a break name.
CALM = "calm"


def _coverage_over(covered: np.ndarray, mask: np.ndarray) -> float:
    """Coverage over the masked points, or NaN when the mask is empty."""
    if not bool(mask.any()):
        return float("nan")
    return float(covered[mask].mean())


def empirical_coverage(result: ConformalResult) -> float:
    """Marginal coverage: the fraction of test points inside their interval."""
    return result.coverage


@dataclass(frozen=True)
class RegimeCoverage:
    """Coverage split by regime.

    ``calm`` and ``post_break`` are the pooled calm and post-break coverages,
    ``by_break`` the coverage within each named break's window, and ``gap`` the
    primary endpoint, calm minus post-break.
    """

    calm: float
    post_break: float
    by_break: dict[str, float]
    gap: float


def regime_conditional_coverage(
    result: ConformalResult, regime: np.ndarray
) -> RegimeCoverage:
    """Coverage in calm periods versus the post-break windows.

    ``regime`` labels each test point ``CALM`` or with a break name (see the
    module docstring). Returns calm and pooled post-break coverage, the coverage
    within each break, and the calm-minus-post-break gap.
    """
    labels = np.asarray(regime)
    covered = result.covered
    if labels.shape[0] != covered.shape[0]:
        raise ValueError("regime labels must match the number of test points")

    calm_mask = labels == CALM
    calm = _coverage_over(covered, calm_mask)
    post_break = _coverage_over(covered, ~calm_mask)
    by_break = {
        str(name): _coverage_over(covered, labels == name)
        for name in np.unique(labels[~calm_mask])
    }
    return RegimeCoverage(
        calm=calm, post_break=post_break, by_break=by_break, gap=calm - post_break
    )


@dataclass(frozen=True)
class DecayCurve:
    """Coverage bucketed by days since the most recent break onset.

    ``coverage[i]`` and ``counts[i]`` describe the points whose days-since-break
    fall in ``[bucket_edges[i], bucket_edges[i + 1])``; coverage is NaN where a
    bucket is empty.
    """

    bucket_edges: np.ndarray
    coverage: np.ndarray
    counts: np.ndarray


def coverage_decay_curve(
    result: ConformalResult, days_since_break: np.ndarray, buckets: np.ndarray
) -> DecayCurve:
    """Coverage as a function of days since the most recent break onset.

    Bucketing days since the break shows whether and when coverage degrades and
    then recovers. ``buckets`` are the bin edges (length B + 1 for B buckets).
    """
    days = np.asarray(days_since_break, dtype=float)
    edges = np.asarray(buckets, dtype=float)
    covered = result.covered

    n_buckets = edges.shape[0] - 1
    coverage = np.empty(n_buckets, dtype=float)
    counts = np.empty(n_buckets, dtype=int)
    for b in range(n_buckets):
        mask = (days >= edges[b]) & (days < edges[b + 1])
        counts[b] = int(mask.sum())
        coverage[b] = _coverage_over(covered, mask)
    return DecayCurve(bucket_edges=edges, coverage=coverage, counts=counts)
