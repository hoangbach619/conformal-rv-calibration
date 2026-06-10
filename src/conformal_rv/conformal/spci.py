"""Sequential Predictive Conformal Inference (Xu, Xie 2023).

Residual-modelling comparator. SPCI fits a quantile model (a random forest in
the original) to the recent residual sequence, so it can exploit serial
dependence that exchangeability-based methods discard. It is included because
realised-vol residuals are strongly autocorrelated, which is precisely the
regime SPCI was designed for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class SPCI:
    """Conditional residual-quantile intervals over a rolling window."""

    def __init__(self, alpha: float, window: int) -> None:
        self.alpha = alpha
        self.window = window

    def run(self, point_forecasts: np.ndarray, targets: np.ndarray) -> np.ndarray:
        raise NotImplementedError
