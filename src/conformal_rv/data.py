"""Data acquisition for the equity-index panel.

Daily OHLC bars are pulled via yfinance. Daily-OHLC realised volatility is a
range proxy and is used here because it is free; intraday 5-minute data is not
freely redistributable. The legacy Oxford-Man 5-minute RV panel is treated as
a frozen, out-of-sample validation target where the date ranges overlap.

Missing-data policy lives here on purpose so it is enforced once: no
backward-fill is ever permitted, because back-filling leaks future
information into a forecasting target. Forward-fill is allowed per feature
only with an explicit justification at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def load_index_ohlc(ticker: str, start: str, end: str) -> "pd.DataFrame":
    """Return a daily OHLC frame for one equity index.

    Forward-fill is deliberately not applied here. Gap handling is the
    caller's responsibility so that each fill decision carries its own
    justification (see module docstring on the no-bfill policy).
    """
    raise NotImplementedError


def load_oxford_man_rv(symbol: str) -> "pd.DataFrame":
    """Return the frozen Oxford-Man 5-minute RV series for validation.

    This source is read-only for the study: it anchors the daily-OHLC range
    proxy against an intraday benchmark over the overlapping dates only.
    """
    raise NotImplementedError


def assert_no_backfill(frame: "pd.DataFrame") -> None:
    """Guard that no backward-fill has been applied to a frame.

    Used in tests and pipelines to make the no-bfill policy auditable rather
    than merely documented.
    """
    raise NotImplementedError
