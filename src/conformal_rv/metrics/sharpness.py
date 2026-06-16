"""Interval sharpness, the width side of the coverage-width trade-off.

A method that restores coverage by inflating widths without bound is not
useful. Width is therefore summarised calm versus post-break so the width cost
of any coverage restoration is legible; in particular the H3 calm-width ratio
against the split-conformal/CQR base is the ratio of the two calm medians.

Widths are read from a ``ConformalResult`` and split by the same regime labels
as coverage.py (the ``CALM`` sentinel against break names).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conformal_rv.conformal.cqr import ConformalResult
from conformal_rv.metrics.coverage import CALM


def interval_width(result: ConformalResult) -> np.ndarray:
    """Per-point interval width, ``upper - lower``."""
    width: np.ndarray = result.upper - result.lower
    return width


def _median(values: np.ndarray) -> float:
    """Median of the values, or NaN when there are none."""
    return float(np.median(values)) if values.size else float("nan")


def _mean(values: np.ndarray) -> float:
    """Mean of the values, or NaN when there are none."""
    return float(np.mean(values)) if values.size else float("nan")


@dataclass(frozen=True)
class WidthSummary:
    """Median and mean interval width, calm versus post-break."""

    calm_median: float
    calm_mean: float
    post_break_median: float
    post_break_mean: float


def summarise_width(result: ConformalResult, regime: np.ndarray) -> WidthSummary:
    """Median and mean interval width in calm versus post-break periods.

    The calm median is the quantity the H3 width ratio divides between a method
    and the CQR base.
    """
    width = interval_width(result)
    labels = np.asarray(regime)
    if labels.shape[0] != width.shape[0]:
        raise ValueError("regime labels must match the number of test points")

    calm_width = width[labels == CALM]
    post_break_width = width[labels != CALM]
    return WidthSummary(
        calm_median=_median(calm_width),
        calm_mean=_mean(calm_width),
        post_break_median=_median(post_break_width),
        post_break_mean=_mean(post_break_width),
    )
