"""Quantile-regression HAR.

Produces conditional lower and upper quantiles directly, which is what CQR
consumes. This is the second arm of the credible base: it gives a
heteroscedastic interval before any online correction is applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class QuantileHARModel:
    """Pinball-loss HAR fit at a low and a high quantile."""

    def __init__(self, lower_quantile: float, upper_quantile: float) -> None:
        # Stored rather than hard-coded so the nominal level is set by the
        # caller and recorded alongside results.
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(
        self, features: "pd.DataFrame", target: "pd.Series"
    ) -> "QuantileHARModel":
        raise NotImplementedError

    def predict_interval(self, features: "pd.DataFrame") -> "np.ndarray":
        """Return an (n, 2) array of lower and upper conditional quantiles."""
        raise NotImplementedError
