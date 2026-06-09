"""Walk-forward split construction with embargoes.

The split order is train, embargo, calibration, embargo, test. Two embargoes
appear because conformal calibration needs its own clean separation from both
the training fit and the test evaluation. Each embargo must be at least the
maximum forecast horizon plus the maximum feature look-back, otherwise
overlapping windows leak the target across the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardSplit:
    """Index ranges for one walk-forward fold.

    Stored as half-open integer positions into the ordered time index. Frozen
    so a fold cannot be mutated after the embargo has been validated.
    """

    train: range
    calibration: range
    test: range


def minimum_embargo(max_horizon: int, max_lookback: int) -> int:
    """Smallest admissible embargo length in trading days.

    Encodes the leakage constraint in one place so splits and tests agree.
    """
    raise NotImplementedError


def walk_forward(
    n_obs: int,
    *,
    train_size: int,
    calibration_size: int,
    test_size: int,
    embargo: int,
) -> list[WalkForwardSplit]:
    """Generate ordered walk-forward folds honouring the embargo on both sides."""
    raise NotImplementedError
