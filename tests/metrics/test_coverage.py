"""Tests for the coverage metric stubs."""

from __future__ import annotations

from conformal_rv.metrics import coverage


def test_coverage_api_is_present() -> None:
    assert callable(coverage.marginal_coverage)
    assert callable(coverage.conditional_coverage)
    assert callable(coverage.coverage_gap)
