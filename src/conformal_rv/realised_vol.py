"""Realised-volatility estimators from daily OHLC bars.

Yang-Zhang is the primary estimator because it is robust to both opening
jumps and drift. Garman-Klass, Parkinson and Rogers-Satchell are kept as
robustness checks; agreement across estimators is part of the validation,
not an afterthought. The modelling target throughout is log realised vol,
which stabilises variance and keeps the conformal residuals better behaved.

Conventions. Every estimator below returns a daily realised *variance*
series, not a standard deviation and not an annualised figure: there is no
sqrt(252) scaling anywhere in the study, so a "vol" here is per-trading-day.
``to_log_rv`` maps such a variance to the modelling target, log realised vol.

Reproducibility. These functions are pure and deterministic: they draw no
random numbers (so ``conformal_rv.SEED`` is irrelevant here) and run no
parallel work (so ``conformal_rv.N_JOBS = 1`` holds trivially). The
reproducibility knobs are honoured by construction rather than by seeding.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd

# Pre-computed because it recurs in two estimators and a literal hides intent.
_FOUR_LOG_TWO = 4.0 * math.log(2.0)


def parkinson(ohlc: pd.DataFrame) -> pd.Series[float]:
    """Daily Parkinson high-low variance.

    Parkinson sees only the intraday range, never the overnight jump. That
    blind spot is the point: it is a deliberate foil for Yang-Zhang, which
    does carry the overnight term.
    """
    log_high: pd.Series[float] = np.log(ohlc["high"])
    log_low: pd.Series[float] = np.log(ohlc["low"])
    variance: pd.Series[float] = (log_high - log_low) ** 2 / _FOUR_LOG_TWO
    return variance


def garman_klass(ohlc: pd.DataFrame) -> pd.Series[float]:
    """Daily Garman-Klass variance.

    Uses the full OHLC of a single bar. It is more efficient than Parkinson
    under a zero-drift, no-gap assumption, which is exactly where it can
    disagree with the others when those assumptions fail.
    """
    log_open: pd.Series[float] = np.log(ohlc["open"])
    log_high: pd.Series[float] = np.log(ohlc["high"])
    log_low: pd.Series[float] = np.log(ohlc["low"])
    log_close: pd.Series[float] = np.log(ohlc["close"])
    high_low = log_high - log_low
    close_open = log_close - log_open
    variance: pd.Series[float] = (
        0.5 * high_low**2 - (2.0 * math.log(2.0) - 1.0) * close_open**2
    )
    return variance


def rogers_satchell(ohlc: pd.DataFrame) -> pd.Series[float]:
    """Daily Rogers-Satchell variance.

    Drift-independent: it stays unbiased when the bar has a trend, which is
    why Yang-Zhang uses it as its intraday component rather than Garman-Klass.
    """
    log_open: pd.Series[float] = np.log(ohlc["open"])
    log_high: pd.Series[float] = np.log(ohlc["high"])
    log_low: pd.Series[float] = np.log(ohlc["low"])
    log_close: pd.Series[float] = np.log(ohlc["close"])
    variance: pd.Series[float] = (log_high - log_close) * (log_high - log_open) + (
        log_low - log_close
    ) * (log_low - log_open)
    return variance


def yang_zhang(ohlc: pd.DataFrame, window: int = 21) -> pd.Series[float]:
    """Daily Yang-Zhang variance, the primary estimator.

    Yang-Zhang is windowed by construction: its overnight and open-to-close
    components are sample variances taken over ``window`` trading days, so a
    single-bar Yang-Zhang is undefined. The series is therefore NaN until the
    first full window has accumulated, and that NaN is left in place rather
    than filled (the no-backfill policy applies to targets too).

    variance = var(overnight) + k * var(open-to-close)
               + (1 - k) * mean(Rogers-Satchell)

    with ``k = 0.34 / (1.34 + (window + 1) / (window - 1))``. The overnight
    return needs the prior close, so the first overnight observation is NaN by
    construction and the overnight variance lags the other two terms by a day.
    """
    log_open: pd.Series[float] = np.log(ohlc["open"])
    log_close: pd.Series[float] = np.log(ohlc["close"])

    # Overnight return uses the prior close, hence the shift; this is the term
    # the single-bar estimators structurally cannot capture.
    overnight = log_open - log_close.shift(1)
    open_to_close = log_close - log_open

    var_overnight = overnight.rolling(window).var(ddof=1)
    var_open_close = open_to_close.rolling(window).var(ddof=1)
    rs_mean = rogers_satchell(ohlc).rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    variance: pd.Series[float] = (
        var_overnight + k * var_open_close + (1.0 - k) * rs_mean
    )
    return variance


def rolling_parkinson(ohlc: pd.DataFrame, window: int = 21) -> pd.Series[float]:
    """Window-averaged Parkinson variance.

    Provided so the single-bar estimators can be compared with Yang-Zhang on
    the same window, rather than comparing a one-day figure with a windowed
    one and mistaking the smoothing for an estimator disagreement.
    """
    return parkinson(ohlc).rolling(window).mean()


def rolling_garman_klass(ohlc: pd.DataFrame, window: int = 21) -> pd.Series[float]:
    """Window-averaged Garman-Klass variance (see ``rolling_parkinson``)."""
    return garman_klass(ohlc).rolling(window).mean()


def rolling_rogers_satchell(ohlc: pd.DataFrame, window: int = 21) -> pd.Series[float]:
    """Window-averaged Rogers-Satchell variance (see ``rolling_parkinson``)."""
    return rogers_satchell(ohlc).rolling(window).mean()


def to_log_rv(variance: pd.Series[float]) -> pd.Series[float]:
    """Map a daily realised variance to log realised vol, the model target.

    log rv = 0.5 * log(variance), because vol = sqrt(variance) and the study
    models the log of the daily, non-annualised vol. NaNs pass through; a
    non-positive variance yields -inf / NaN rather than being clipped, so that
    bad inputs surface instead of being silently repaired.
    """
    # np.log over a Series returns a Series at runtime, but mypy infers an
    # ndarray from the concrete float dtype; cast back to the true type.
    return cast("pd.Series[float]", 0.5 * np.log(variance))
