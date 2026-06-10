"""Interval sharpness, the width side of the coverage-width trade-off.

A method that restores coverage by inflating widths without bound is not
useful. Width cost is therefore reported relative to split-conformal/CQR so
the trade-off is legible (see H3 in the pre-registration).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def mean_width(intervals: np.ndarray) -> float:
    """Average interval width."""
    raise NotImplementedError


def relative_width(intervals: np.ndarray, reference_intervals: np.ndarray) -> float:
    """Mean width relative to a reference method, the H3 quantity."""
    raise NotImplementedError
