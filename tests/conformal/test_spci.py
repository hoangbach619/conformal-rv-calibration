"""Tests for the SPCI comparator stub."""

from __future__ import annotations

from conformal_rv.conformal import spci


def test_spci_stores_window() -> None:
    method = spci.SPCI(alpha=0.1, window=100)
    assert method.window == 100
