"""Conformal PID control (Angelopoulos, Candes, Tibshirani 2023).

Primary online correction under test. It treats coverage tracking as a
control problem: a proportional term on the current error, an integral term
on accumulated error, and a scorecaster as the derivative-like forecast term.
The hypothesis is that the integral and forecast terms are what restore
coverage through breaks where plain ACI lags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ConformalPID:
    """Proportional-integral-control conformal updater with a scorecaster."""

    def __init__(
        self,
        alpha: float,
        *,
        k_p: float,
        k_i: float,
    ) -> None:
        # Gains are explicit so they can be reported, not buried as defaults.
        self.alpha = alpha
        self.k_p = k_p
        self.k_i = k_i

    def run(self, intervals: np.ndarray, targets: np.ndarray) -> np.ndarray:
        raise NotImplementedError
