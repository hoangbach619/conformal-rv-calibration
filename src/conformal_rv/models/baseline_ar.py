"""Autoregressive / linear sanity baseline.

Not a serious competitor. Its job is to catch pipeline mistakes: if the HAR
base cannot beat a plain AR on point error, something upstream is wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class ARBaseline:
    """Linear autoregressive forecaster on log realised vol."""

    def __init__(self, order: int) -> None:
        self.order = order

    def fit(self, target: pd.Series[float]) -> ARBaseline:
        raise NotImplementedError

    def predict(self, target: pd.Series[float]) -> np.ndarray:
        raise NotImplementedError
