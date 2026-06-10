"""HAR-RV point forecaster (Corsi 2009).

This is the primary credible base on which the conformal layer sits. The HAR
cascade is a parsimonious long-memory proxy; keeping it simple matters because
the study is about calibration of intervals, not about squeezing point error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class HARModel:
    """Ordinary least squares HAR-RV regression on log realised vol."""

    def fit(self, features: pd.DataFrame, target: pd.Series[float]) -> HARModel:
        raise NotImplementedError

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
