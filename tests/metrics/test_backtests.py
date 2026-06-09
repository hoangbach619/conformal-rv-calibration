"""Tests for the coverage backtest stubs."""

from __future__ import annotations

from conformal_rv.metrics import backtests


def test_backtest_api_is_present() -> None:
    assert callable(backtests.kupiec_pof)
    assert callable(backtests.christoffersen_independence)
    assert callable(backtests.christoffersen_conditional_coverage)
