"""Evaluation metrics, backtests and multiple-testing control.

Coverage (marginal and regime-conditional) and sharpness are the headline
trade-off. Scoring rules and PIT diagnostics support them. Backtests are the
formal coverage tests. Multiplicity control keeps the family-wise error in
check across the many method-by-regime comparisons.
"""

from __future__ import annotations
