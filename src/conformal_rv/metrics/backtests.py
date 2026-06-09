"""Interval-coverage backtests.

Kupiec POF tests the unconditional hit rate. Christoffersen adds an
independence test (hits should not cluster) and a joint conditional-coverage
test. Both are reported because a method can pass the unconditional test
while failing badly on clustering through a break, which is the failure mode
this study cares about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def kupiec_pof(
    hits: "np.ndarray", alpha: float
) -> tuple[float, float]:
    """Kupiec proportion-of-failures test. Returns statistic and p-value."""
    raise NotImplementedError


def christoffersen_independence(
    hits: "np.ndarray",
) -> tuple[float, float]:
    """Christoffersen independence test on the hit sequence."""
    raise NotImplementedError


def christoffersen_conditional_coverage(
    hits: "np.ndarray", alpha: float
) -> tuple[float, float]:
    """Christoffersen joint conditional-coverage test."""
    raise NotImplementedError
