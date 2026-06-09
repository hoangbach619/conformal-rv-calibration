"""Single deep probabilistic comparator.

Deliberately one model only: DLinear or a quantile LSTM, not a TFT. It is
included as a calibration stress test, with the pre-registered prior that it
under-covers in high-volatility windows (see H4 in the pre-registration). It
is not entered as a candidate for best forecaster.

Torch is imported lazily inside methods so that the rest of the package, and
the fast test subset, do not depend on a heavy import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class DeepComparator:
    """Quantile deep model used only to stress-test calibration.

    Determinism is required: the caller must have set global seeds and
    ``torch.use_deterministic_algorithms(True)``. Residual GPU
    nondeterminism is documented in the pre-registration rather than hidden.
    """

    def __init__(self, lower_quantile: float, upper_quantile: float) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(
        self, features: "pd.DataFrame", target: "pd.Series"
    ) -> "DeepComparator":
        raise NotImplementedError

    def predict_interval(self, features: "pd.DataFrame") -> "np.ndarray":
        """Return an (n, 2) array of lower and upper conditional quantiles."""
        raise NotImplementedError
