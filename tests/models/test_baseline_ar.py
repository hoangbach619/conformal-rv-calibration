"""Tests for the AR baseline stub."""

from __future__ import annotations

from conformal_rv.models import baseline_ar


def test_order_is_stored() -> None:
    model = baseline_ar.ARBaseline(order=3)
    assert model.order == 3
