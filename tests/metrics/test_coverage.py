"""Unit tests for the coverage metrics.

Coverage on a known covered fraction, regime-conditional coverage on labelled
windows, and a decay curve on a constructed degrade-then-recover stream. None
touch the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_rv.conformal.cqr import ConformalResult
from conformal_rv.metrics import coverage


def _result(covered: list[bool]) -> ConformalResult:
    """A ConformalResult carrying only the per-point covered flags."""
    zeros = np.zeros(len(covered))
    return ConformalResult(
        lower=zeros, upper=zeros, y=zeros, covered=np.asarray(covered, dtype=bool)
    )


def test_empirical_coverage_is_the_covered_fraction() -> None:
    result = _result([True] * 8 + [False] * 2)
    assert coverage.empirical_coverage(result) == 0.8


def test_regime_conditional_coverage_splits_calm_and_breaks() -> None:
    # 5 calm (all covered), break A 3 points (2 covered), break B 2 points (1).
    covered = [True, True, True, True, True, True, True, False, True, False]
    regime = np.array(["calm"] * 5 + ["A"] * 3 + ["B"] * 2)
    result = _result(covered)

    rc = coverage.regime_conditional_coverage(result, regime)

    assert rc.calm == 1.0
    assert rc.by_break["A"] == pytest.approx(2 / 3)
    assert rc.by_break["B"] == 0.5
    # Pooled post-break: 3 of the 5 break points are covered.
    assert rc.post_break == pytest.approx(0.6)
    assert rc.gap == pytest.approx(1.0 - 0.6)


def test_regime_length_mismatch_is_rejected() -> None:
    result = _result([True, False, True])
    with pytest.raises(ValueError, match="match the number"):
        coverage.regime_conditional_coverage(result, np.array(["calm", "A"]))


def test_coverage_decay_curve_shows_degrade_then_recover() -> None:
    # Five points per bucket; coverage rises 0.2 -> 0.4 -> 0.8 -> 1.0 with days
    # since the break (degraded just after, recovering as days pass).
    days = np.repeat([2, 7, 12, 17], 5)
    covered = [
        True,
        False,
        False,
        False,
        False,  # bucket [0, 5):  0.2
        True,
        True,
        False,
        False,
        False,  # bucket [5, 10):  0.4
        True,
        True,
        True,
        True,
        False,  # bucket [10, 15):   0.8
        True,
        True,
        True,
        True,
        True,  # bucket [15, 20):    1.0
    ]
    buckets = np.array([0, 5, 10, 15, 20, 25])  # last bucket has no points
    result = _result(covered)

    curve = coverage.coverage_decay_curve(result, days, buckets)

    assert np.allclose(curve.coverage[:4], [0.2, 0.4, 0.8, 1.0])
    assert np.all(np.diff(curve.coverage[:4]) > 0.0)  # monotone recovery
    assert np.isnan(curve.coverage[4])  # empty bucket
    assert curve.counts.tolist() == [5, 5, 5, 5, 0]
