"""Feature construction for HAR-style models.

The HAR design (Corsi 2009) cascades daily, weekly and monthly averages of
realised volatility. Feature look-back is recorded here because the embargo
in splits.py must cover the maximum look-back plus the maximum forecast
horizon to prevent leakage across the walk-forward boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# HAR cascade windows in trading days: daily, weekly, monthly.
HAR_WINDOWS: tuple[int, int, int] = (1, 5, 22)


def har_features(log_rv: "pd.Series") -> "pd.DataFrame":
    """Build the daily/weekly/monthly HAR regressors from log realised vol."""
    raise NotImplementedError


def forward_fill_feature(series: "pd.Series", reason: str) -> "pd.Series":
    """Forward-fill a single feature, requiring an explicit justification.

    The ``reason`` argument is mandatory so that every fill is auditable; it
    is recorded rather than silently applied. Backward-fill is never allowed
    (see data.py).
    """
    raise NotImplementedError
