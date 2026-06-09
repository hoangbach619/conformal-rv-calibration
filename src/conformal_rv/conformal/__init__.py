"""Conformal interval constructors and online corrections.

CQR is the base interval constructor. ACI and AgACI are the named baselines
to beat. Conformal PID and DtACI are the primary online corrections under
test. SPCI is the residual-modelling comparator. EnbPI is intentionally not a
headline method here.
"""

from __future__ import annotations
