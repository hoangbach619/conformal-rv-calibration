"""Tests for the multiplicity-control stubs."""

from __future__ import annotations

from conformal_rv.metrics import multiplicity


def test_fdr_level_is_pre_registered() -> None:
    assert multiplicity.FDR_LEVEL == 0.10


def test_multiplicity_api_is_present() -> None:
    assert callable(multiplicity.benjamini_hochberg)
    assert callable(multiplicity.effective_number_of_tests)
