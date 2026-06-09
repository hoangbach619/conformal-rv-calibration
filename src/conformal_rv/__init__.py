"""Conformal calibration of realised-volatility prediction intervals.

This package is a research scaffold. Modules are intentionally stubbed so
that the experimental design (see docs/preregistration.md) is fixed before
any model is run. No results are produced by importing this package.
"""

from __future__ import annotations

# Project-wide reproducibility conventions. See docs/preregistration.md.
# These are referenced rather than enforced here so that importing the
# package has no global side effects on RNG state.
SEED: int = 42
N_JOBS: int = 1

__all__ = ["SEED", "N_JOBS"]
