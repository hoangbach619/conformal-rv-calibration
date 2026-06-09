"""Tests for the realised-volatility estimator stubs."""

from __future__ import annotations

from conformal_rv import realised_vol


def test_estimators_are_present() -> None:
    for name in (
        "parkinson",
        "garman_klass",
        "rogers_satchell",
        "yang_zhang",
        "log_rv",
    ):
        assert callable(getattr(realised_vol, name))
