"""Tests for the Conformal PID stub."""

from __future__ import annotations

from conformal_rv.conformal import conformal_pid


def test_pid_stores_gains() -> None:
    method = conformal_pid.ConformalPID(alpha=0.1, k_p=0.1, k_i=0.01)
    assert method.k_p == 0.1
    assert method.k_i == 0.01
