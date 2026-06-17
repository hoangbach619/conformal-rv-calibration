"""Post-hoc analysis of the persisted sweep results.

This package reads the frozen result parquets and produces the result tables,
the seed aggregation, the false-discovery control and the hypothesis verdicts.
It never refits a model: every number comes from the persisted intervals.
"""

from __future__ import annotations
