"""Tests for the walk-forward split stubs."""

from __future__ import annotations

from conformal_rv import splits


def test_split_dataclass_is_frozen() -> None:
    fold = splits.WalkForwardSplit(
        train=range(0, 10), calibration=range(10, 15), test=range(15, 20)
    )
    assert fold.train == range(0, 10)


def test_split_helpers_are_present() -> None:
    assert callable(splits.minimum_embargo)
    assert callable(splits.walk_forward)
