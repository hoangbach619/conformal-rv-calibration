"""Tests for the DtACI stub."""

from __future__ import annotations

from conformal_rv.conformal import dtaci


def test_dtaci_stores_gamma_grid() -> None:
    method = dtaci.DtACI(alpha=0.1, gammas=(0.001, 0.01, 0.1))
    assert method.gammas == (0.001, 0.01, 0.1)
