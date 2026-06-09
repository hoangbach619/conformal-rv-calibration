"""Realised-volatility estimators from daily OHLC bars.

Yang-Zhang is the primary estimator because it is robust to both opening
jumps and drift. Garman-Klass, Parkinson and Rogers-Satchell are kept as
robustness checks; agreement across estimators is part of the validation,
not an afterthought. The modelling target throughout is log realised vol,
which stabilises variance and keeps the conformal residuals better behaved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def parkinson(ohlc: "pd.DataFrame") -> "pd.Series":
    """Parkinson high-low range estimator."""
    raise NotImplementedError


def garman_klass(ohlc: "pd.DataFrame") -> "pd.Series":
    """Garman-Klass OHLC estimator."""
    raise NotImplementedError


def rogers_satchell(ohlc: "pd.DataFrame") -> "pd.Series":
    """Rogers-Satchell estimator, drift-independent."""
    raise NotImplementedError


def yang_zhang(ohlc: "pd.DataFrame") -> "pd.Series":
    """Yang-Zhang estimator. Primary target generator for the study."""
    raise NotImplementedError


def log_rv(rv: "pd.Series") -> "pd.Series":
    """Map a realised-vol series to log realised vol, the modelling target."""
    raise NotImplementedError
