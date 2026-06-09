"""Aggregated Adaptive Conformal Inference (Zaffran et al. 2022).

A named baseline to beat. AgACI runs a set of ACI experts at different
learning rates and aggregates them online, removing the need to pick a single
gamma. It is the strongest of the pre-2023 baselines in this study.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class AgACI:
    """Online aggregation of ACI experts across a grid of learning rates."""

    def __init__(self, alpha: float, gammas: "tuple[float, ...]") -> None:
        self.alpha = alpha
        self.gammas = gammas

    def run(
        self, intervals: "np.ndarray", targets: "np.ndarray"
    ) -> "np.ndarray":
        raise NotImplementedError
