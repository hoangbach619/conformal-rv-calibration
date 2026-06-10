"""Unit tests for the walk-forward split construction.

The invariants under test are the embargo gaps, intra-block disjointness, and
non-overlapping test windows across consecutive blocks.
"""

from __future__ import annotations

import itertools

import pandas as pd

from conformal_rv import splits


def _axis(n: int) -> pd.DatetimeIndex:
    """A clean trading-date axis of business days."""
    return pd.bdate_range("2000-01-03", periods=n)


def _pos(axis: pd.DatetimeIndex, timestamp: pd.Timestamp) -> int:
    """Integer position of a date on the axis, for trading-day arithmetic."""
    return int(axis.get_loc(timestamp))


def test_minimum_embargo_is_horizon_plus_lookback() -> None:
    assert splits.minimum_embargo(22, 22) == 44
    assert splits.EMBARGO == 44


def test_blocks_respect_embargo_and_are_disjoint() -> None:
    axis = _axis(3000)
    blocks = splits.walk_forward_splits(axis)
    assert blocks  # the axis is long enough for at least one fold

    for block in blocks:
        max_train = _pos(axis, block.train[-1])
        min_calibration = _pos(axis, block.calibration[0])
        max_calibration = _pos(axis, block.calibration[-1])
        min_test = _pos(axis, block.test[0])

        # (max train date) + embargo <= (min calibration date), and likewise
        # between calibration and test, in trading-day count.
        assert max_train + splits.EMBARGO <= min_calibration
        assert max_calibration + splits.EMBARGO <= min_test

        # The embargo gap is at least 44 trading days on each side.
        assert min_calibration - max_train - 1 >= splits.EMBARGO
        assert min_test - max_calibration - 1 >= splits.EMBARGO

        # No date appears in more than one of train/calibration/test.
        train_set = set(block.train)
        calibration_set = set(block.calibration)
        test_set = set(block.test)
        assert train_set.isdisjoint(calibration_set)
        assert calibration_set.isdisjoint(test_set)
        assert train_set.isdisjoint(test_set)


def test_consecutive_test_windows_do_not_overlap() -> None:
    axis = _axis(3000)
    blocks = splits.walk_forward_splits(axis)
    assert len(blocks) >= 2  # need at least two folds to compare

    for earlier, later in itertools.pairwise(blocks):
        assert _pos(axis, earlier.test[-1]) < _pos(axis, later.test[0])


def test_block_row_masks_select_the_block_dates() -> None:
    axis = _axis(2600)
    block = splits.walk_forward_splits(axis)[0]
    panel = pd.DataFrame({"date": axis, "value": range(len(axis))})

    masks = splits.block_row_masks(panel, block)

    assert panel.loc[masks.train, "date"].tolist() == list(block.train)
    assert panel.loc[masks.test, "date"].tolist() == list(block.test)
    assert int(masks.calibration.sum()) == len(block.calibration)
    # The three masks never select the same row twice.
    assert not (masks.train & masks.calibration).any()
    assert not (masks.calibration & masks.test).any()
    assert not (masks.train & masks.test).any()
