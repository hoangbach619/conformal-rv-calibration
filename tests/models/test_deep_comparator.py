"""Tests for the deep comparator stub."""

from __future__ import annotations

from conformal_rv.models import deep_comparator


def test_quantile_levels_are_stored() -> None:
    model = deep_comparator.DeepComparator(
        lower_quantile=0.05, upper_quantile=0.95
    )
    assert model.lower_quantile == 0.05
    assert model.upper_quantile == 0.95
