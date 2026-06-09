"""Tests for the AgACI baseline stub."""

from __future__ import annotations

from conformal_rv.conformal import agaci


def test_agaci_stores_gamma_grid() -> None:
    method = agaci.AgACI(alpha=0.1, gammas=(0.001, 0.01, 0.1))
    assert method.gammas == (0.001, 0.01, 0.1)
