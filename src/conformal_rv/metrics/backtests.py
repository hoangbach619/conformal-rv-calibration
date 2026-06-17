"""Interval-coverage backtests.

Kupiec POF tests the unconditional violation rate. Christoffersen adds an
independence test (violations should not cluster) and a joint conditional-
coverage test. Both are reported because a method can pass the unconditional
test while failing badly on clustering through a break, which is the failure
mode this study cares about.

All functions take a boolean ``violations`` array (True where the outcome falls
outside the interval) and the nominal violation probability ``expected_rate``
(``alpha`` for the whole interval, ``alpha/2`` for a single tail), so the same
function backtests the interval or one tail by choice of indicator and rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


def _binomial_loglik(n_success: int, n_failure: int, rate: float) -> float:
    """Log-likelihood of a Bernoulli with the ``0 * log(0) = 0`` convention.

    A positive count always carries a positive matching rate (the rate is the
    MLE for those counts, or a nominal probability strictly inside (0, 1)), so
    the guarded terms never evaluate ``log(0)``.
    """
    loglik = 0.0
    if n_success > 0:
        loglik += n_success * math.log(rate)
    if n_failure > 0:
        loglik += n_failure * math.log(1.0 - rate)
    return loglik


def kupiec_pof(violations: np.ndarray, expected_rate: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR test of the unconditional rate.

    Returns the likelihood-ratio statistic and its chi-square(1) p-value. The
    zero-violation and all-violation boundaries are handled by the ``0 log 0``
    convention, so the statistic stays finite there (the unconstrained MLE
    log-likelihood is zero and the statistic reduces to ``-2 log L`` under the
    nominal rate).
    """
    indicator = np.asarray(violations, dtype=bool)
    n = int(indicator.shape[0])
    if n == 0:
        raise ValueError("need at least one observation")

    failures = int(indicator.sum())
    rate_hat = failures / n
    loglik_null = _binomial_loglik(failures, n - failures, expected_rate)
    loglik_mle = _binomial_loglik(failures, n - failures, rate_hat)

    statistic = max(-2.0 * (loglik_null - loglik_mle), 0.0)
    return statistic, float(chi2.sf(statistic, 1))


@dataclass(frozen=True)
class ChristoffersenResult:
    """Christoffersen independence and conditional-coverage LR tests.

    ``independence`` (chi-square(1)) detects first-order clustering of
    violations; ``conditional_coverage`` (chi-square(2)) is the Kupiec POF plus
    the independence statistic.
    """

    independence_statistic: float
    independence_pvalue: float
    conditional_coverage_statistic: float
    conditional_coverage_pvalue: float


def christoffersen(
    violations: np.ndarray, expected_rate: float
) -> ChristoffersenResult:
    """Christoffersen independence and conditional-coverage LR tests.

    The independence test compares a first-order Markov chain (the violation
    probability may depend on the previous state) with an independent chain,
    from the four transition counts. Degenerate cases -- no violations, no
    consecutive violations, an empty source state -- give a zero conditional
    probability whose ``0 log 0`` term vanishes, so the statistic is finite.
    """
    indicator = np.asarray(violations, dtype=bool).astype(int)
    if indicator.shape[0] == 0:
        raise ValueError("need at least one observation")

    previous, current = indicator[:-1], indicator[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    from_calm = n00 + n01
    from_violation = n10 + n11
    total = from_calm + from_violation
    # Conditional violation probabilities; a zero denominator gives a zero rate
    # whose paired count is also zero, so its log term drops out.
    pi_after_calm = n01 / from_calm if from_calm > 0 else 0.0
    pi_after_violation = n11 / from_violation if from_violation > 0 else 0.0
    pi_marginal = (n01 + n11) / total if total > 0 else 0.0

    loglik_markov = _binomial_loglik(n01, n00, pi_after_calm) + _binomial_loglik(
        n11, n10, pi_after_violation
    )
    loglik_independent = _binomial_loglik(n01 + n11, n00 + n10, pi_marginal)
    independence = max(-2.0 * (loglik_independent - loglik_markov), 0.0)

    pof, _ = kupiec_pof(violations, expected_rate)
    conditional_coverage = pof + independence
    return ChristoffersenResult(
        independence_statistic=independence,
        independence_pvalue=float(chi2.sf(independence, 1)),
        conditional_coverage_statistic=conditional_coverage,
        conditional_coverage_pvalue=float(chi2.sf(conditional_coverage, 2)),
    )
