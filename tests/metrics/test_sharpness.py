"""Unit tests for the sharpness metrics, on hand-checkable intervals."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_rv.conformal.cqr import ConformalResult
from conformal_rv.metrics import sharpness


def _result(lower: np.ndarray, upper: np.ndarray) -> ConformalResult:
    n = lower.shape[0]
    return ConformalResult(
        lower=lower, upper=upper, y=np.zeros(n), covered=np.zeros(n, dtype=bool)
    )


def test_interval_width_is_upper_minus_lower() -> None:
    result = _result(np.array([0.0, -1.0, 2.0]), np.array([1.0, 1.0, 5.0]))
    assert np.allclose(sharpness.interval_width(result), [1.0, 2.0, 3.0])


def test_summarise_width_splits_calm_and_post_break() -> None:
    # Lower at 0, so width == upper. Calm widths [1, 1, 3]; post-break [2, 4, 6].
    lower = np.zeros(6)
    upper = np.array([1.0, 1.0, 3.0, 2.0, 4.0, 6.0])
    regime = np.array(["calm", "calm", "calm", "A", "A", "A"])

    summary = sharpness.summarise_width(_result(lower, upper), regime)

    assert summary.calm_median == 1.0
    assert summary.calm_mean == pytest.approx(5 / 3)
    assert summary.post_break_median == 4.0
    assert summary.post_break_mean == 4.0
