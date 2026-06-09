"""Tests for the ACI baseline stub."""

from __future__ import annotations

from conformal_rv.conformal import aci


def test_aci_stores_alpha_and_gamma() -> None:
    method = aci.ACI(alpha=0.1, gamma=0.01)
    assert method.alpha == 0.1
    assert method.gamma == 0.01
