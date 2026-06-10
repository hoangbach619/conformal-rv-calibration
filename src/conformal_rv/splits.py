"""Walk-forward split construction with embargoes.

Each block is ordered train, embargo, calibration, embargo, test. Two
embargoes appear because conformal calibration needs its own clean separation
from both the training fit and the test evaluation.

Why the embargo is 44 trading days: it covers the 22-day maximum forecast
horizon plus the ~22-day maximum estimation and feature look-back (the monthly
HAR term and the Yang-Zhang window). With that gap, no calibration or test
feature window can overlap a training target window, so the target cannot leak
across a fold boundary. ``minimum_embargo`` encodes this in one place.

Blocks are built on the trading-date axis; a panel row belongs to whichever
block date set contains its date (see ``block_row_masks``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Leakage budget, in trading days. The embargo must cover both.
MAX_HORIZON: int = 22
MAX_LOOKBACK: int = 22


def minimum_embargo(max_horizon: int, max_lookback: int) -> int:
    """Smallest admissible embargo length in trading days.

    Encodes the leakage constraint in one place so the split logic, the
    defaults and the tests cannot drift apart.
    """
    return max_horizon + max_lookback


# Default embargo, named so callers and tests agree on the 44 = 22 + 22 figure.
EMBARGO: int = minimum_embargo(MAX_HORIZON, MAX_LOOKBACK)


@dataclass(frozen=True)
class WalkForwardBlock:
    """The train, calibration and test date sets for one fold.

    Each field is the set of trading dates assigned to that part of the fold.
    Frozen so a block cannot be mutated after its embargo has been validated.
    """

    train: pd.DatetimeIndex
    calibration: pd.DatetimeIndex
    test: pd.DatetimeIndex


@dataclass(frozen=True)
class BlockRowMasks:
    """Boolean row masks selecting a block's train/calibration/test panel rows."""

    train: pd.Series[bool]
    calibration: pd.Series[bool]
    test: pd.Series[bool]


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train: int = 2000,
    embargo: int = EMBARGO,
    calibration: int = 250,
    test: int = 250,
    step: int = 250,
) -> list[WalkForwardBlock]:
    """Ordered walk-forward blocks honouring the embargo on both sides.

    Positions are taken on the ordered trading-date axis. Each fold spans
    ``train + embargo + calibration + embargo + test`` dates; the start rolls
    forward by ``step``. With ``step >= test`` the test windows do not overlap.
    """
    axis = pd.DatetimeIndex(dates)
    fold_span = train + embargo + calibration + embargo + test

    blocks: list[WalkForwardBlock] = []
    start = 0
    while start + fold_span <= len(axis):
        train_end = start + train
        calibration_start = train_end + embargo
        calibration_end = calibration_start + calibration
        test_start = calibration_end + embargo
        test_end = test_start + test

        blocks.append(
            WalkForwardBlock(
                train=axis[start:train_end],
                calibration=axis[calibration_start:calibration_end],
                test=axis[test_start:test_end],
            )
        )
        start += step

    return blocks


def block_row_masks(
    panel: pd.DataFrame, block: WalkForwardBlock, date_column: str = "date"
) -> BlockRowMasks:
    """Row masks selecting the panel rows that fall in each part of ``block``.

    A row is assigned by its date, so the masks work for the long multi-index
    panel where many rows share a date.
    """
    panel_dates = panel[date_column]
    return BlockRowMasks(
        train=panel_dates.isin(block.train),
        calibration=panel_dates.isin(block.calibration),
        test=panel_dates.isin(block.test),
    )
