"""Tests for the HAR-RV model stub."""

from __future__ import annotations

from conformal_rv.models import har


def test_har_model_has_fit_predict() -> None:
    model = har.HARModel()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
