"""Tests for the feature-construction stubs."""

from __future__ import annotations

from conformal_rv import features


def test_har_windows_are_daily_weekly_monthly() -> None:
    assert features.HAR_WINDOWS == (1, 5, 22)


def test_feature_builders_are_present() -> None:
    assert callable(features.har_features)
    assert callable(features.forward_fill_feature)
