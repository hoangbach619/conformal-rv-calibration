"""Tests for the sharpness metric stubs."""

from __future__ import annotations

from conformal_rv.metrics import sharpness


def test_sharpness_api_is_present() -> None:
    assert callable(sharpness.mean_width)
    assert callable(sharpness.relative_width)
