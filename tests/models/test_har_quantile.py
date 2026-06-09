"""Tests for the quantile-HAR model stub."""

from __future__ import annotations

from conformal_rv.models import har_quantile


def test_quantile_levels_are_stored() -> None:
    model = har_quantile.QuantileHARModel(lower_quantile=0.05, upper_quantile=0.95)
    assert model.lower_quantile == 0.05
    assert model.upper_quantile == 0.95
