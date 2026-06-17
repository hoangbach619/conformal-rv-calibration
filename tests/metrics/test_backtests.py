"""Unit tests for the coverage backtests.

Kupiec is checked at the nominal rate, with far too many violations, on a
hand-checked example, and at the zero-violation boundary. Christoffersen
independence is checked on independent versus clustered violations, including
the degenerate no-violation case, and the conditional-coverage statistic is
checked to equal Kupiec plus independence. None touch the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_rv.metrics import backtests


def test_kupiec_does_not_reject_at_the_nominal_rate() -> None:
    # 20 violations in 100 at a nominal 0.20: the MLE equals the nominal rate.
    violations = np.array([True] * 20 + [False] * 80)
    statistic, p_value = backtests.kupiec_pof(violations, 0.2)
    assert statistic == pytest.approx(0.0)
    assert p_value > 0.5


def test_kupiec_rejects_far_too_many_violations() -> None:
    violations = np.array([True] * 80 + [False] * 120)  # 40% against nominal 5%
    statistic, p_value = backtests.kupiec_pof(violations, 0.05)
    assert statistic > 50.0
    assert p_value < 0.01


def test_kupiec_statistic_is_hand_checked() -> None:
    # 10 violations in 100 against nominal 0.05; LR = -2(ll_null - ll_mle).
    violations = np.array([True] * 10 + [False] * 90)
    statistic, _ = backtests.kupiec_pof(violations, 0.05)
    assert statistic == pytest.approx(4.13084, abs=1e-4)


def test_kupiec_zero_violation_boundary_is_finite() -> None:
    violations = np.zeros(100, dtype=bool)
    statistic, p_value = backtests.kupiec_pof(violations, 0.05)
    assert np.isfinite(statistic)
    # With no violations the statistic reduces to -2 * n * log(1 - rate).
    assert statistic == pytest.approx(-2.0 * 100 * np.log(0.95), abs=1e-4)
    assert 0.0 < p_value < 1.0


def test_christoffersen_does_not_reject_independent_violations() -> None:
    rng = np.random.default_rng(2)
    violations = rng.random(800) < 0.2  # i.i.d., so no first-order dependence
    result = backtests.christoffersen(violations, 0.2)
    assert result.independence_pvalue > 0.1


def test_christoffersen_rejects_clustered_violations() -> None:
    # Violations arranged in runs: a violation is far more likely to follow a
    # violation than a calm point, which the independence test must catch.
    violations = np.array(([True] * 10 + [False] * 10) * 10)
    result = backtests.christoffersen(violations, 0.5)
    assert result.independence_pvalue < 0.01


def test_conditional_coverage_is_kupiec_plus_independence() -> None:
    violations = np.array(([True] * 10 + [False] * 10) * 10)
    pof, _ = backtests.kupiec_pof(violations, 0.5)
    result = backtests.christoffersen(violations, 0.5)
    assert result.conditional_coverage_statistic == pytest.approx(
        pof + result.independence_statistic
    )


def test_christoffersen_no_violation_case_is_finite() -> None:
    result = backtests.christoffersen(np.zeros(50, dtype=bool), 0.05)
    assert np.isfinite(result.independence_statistic)
    assert result.independence_statistic == pytest.approx(0.0)
