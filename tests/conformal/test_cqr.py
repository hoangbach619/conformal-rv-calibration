"""Tests for the CQR base constructor stub."""

from __future__ import annotations

from conformal_rv.conformal import cqr


def test_cqr_api_is_present() -> None:
    assert callable(cqr.conformity_scores)
    assert callable(cqr.calibrate)
    assert callable(cqr.apply)
