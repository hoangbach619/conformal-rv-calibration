"""Tests for the scoring and PIT stubs."""

from __future__ import annotations

from conformal_rv.metrics import scoring


def test_horizons_are_pre_registered() -> None:
    assert scoring.HORIZONS == (1, 5, 10, 22)


def test_scoring_api_is_present() -> None:
    assert callable(scoring.pinball_loss)
    assert callable(scoring.pit_values)
    assert callable(scoring.pit_ks_statistic)
