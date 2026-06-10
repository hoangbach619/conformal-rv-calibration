"""Adaptive Conformal Inference (Gibbs, Candes 2021).

A named baseline to beat. ACI adjusts the effective miscoverage level with a
single fixed learning rate using the online coverage error. Its known
weakness through regime breaks is exactly what the primary endpoint probes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ACI:
    """Single-rate online miscoverage adjustment."""

    def __init__(self, alpha: float, gamma: float) -> None:
        # gamma is the fixed step size; its sensitivity is the whole reason
        # AgACI and DtACI exist as successors.
        self.alpha = alpha
        self.gamma = gamma

    def update(self, covered: bool) -> float:
        """Ingest one realised coverage outcome, return the next alpha_t."""
        raise NotImplementedError

    def run(self, intervals: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Replay a test stream, returning the adjusted intervals."""
        raise NotImplementedError
