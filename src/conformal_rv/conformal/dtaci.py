"""Dynamically-tuned Adaptive Conformal Inference (Gibbs, Candes 2024).

Primary online correction under test. DtACI runs ACI experts and reweights
them with an adaptive meta-learning rate, giving the regret guarantees of
expert aggregation without a hand-tuned gamma. It is the natural successor to
AgACI and a direct competitor to Conformal PID on the break windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class DtACI:
    """Expert-aggregated ACI with an adaptive tuning of the meta rate."""

    def __init__(self, alpha: float, gammas: "tuple[float, ...]") -> None:
        self.alpha = alpha
        self.gammas = gammas

    def run(
        self, intervals: "np.ndarray", targets: "np.ndarray"
    ) -> "np.ndarray":
        raise NotImplementedError
