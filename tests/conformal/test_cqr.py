"""Unit tests for the CQR base interval constructor.

The marginal-coverage test uses exchangeable i.i.d. draws, where CQR's
finite-sample guarantee holds; the rest pin the conformity score, the
correction quantile, the over/under-cover behaviour, and determinism. None
touch the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_rv.conformal import cqr


def test_conformity_scores_are_signed_distance_outside_the_band() -> None:
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])
    y = np.array([0.5, 1.5, -0.3])
    # max(lower - y, y - upper): inside is negative, outside positive.
    scores = cqr.conformity_scores(lower, upper, y)
    assert np.allclose(scores, [-0.5, 0.5, 0.3])


def test_correction_is_the_finite_sample_quantile() -> None:
    scores = np.array([0.1, 0.5, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6, 0.8])  # n = 9
    # k = ceil((9 + 1) * (1 - 0.2)) = 8, so Q is the 8th smallest score = 0.8.
    assert cqr.conformal_correction(scores, alpha=0.2) == pytest.approx(0.8)
    # Too high a level for the sample size yields an infinite (covers-all) Q.
    assert cqr.conformal_correction(np.array([0.0, 1.0]), alpha=0.01) == float("inf")


def test_marginal_coverage_holds_on_exchangeable_data() -> None:
    rng = np.random.default_rng(0)
    n_cal, n_test = 3000, 40000
    cal_y = rng.normal(0.0, 1.0, n_cal)
    test_y = rng.normal(0.0, 1.0, n_test)
    # Deliberately biased constant "quantile forecasts": CQR must correct them.
    cal_lower = np.full(n_cal, -1.0)
    cal_upper = np.full(n_cal, 1.0)
    test_lower = np.full(n_test, -1.0)
    test_upper = np.full(n_test, 1.0)

    result = cqr.conformalise_cqr(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )

    # Exchangeable, so coverage sits at or just above the 0.80 nominal level.
    assert 0.79 <= result.coverage <= 0.82


def test_over_cover_tightens_under_cover_widens() -> None:
    rng = np.random.default_rng(1)
    n = 2000

    # Over-covering raw band: every outcome sits well inside, scores negative.
    wide_lower = np.full(n, -10.0)
    wide_upper = np.full(n, 10.0)
    inside_y = rng.uniform(-1.0, 1.0, n)
    wide_q = cqr.conformal_correction(
        cqr.conformity_scores(wide_lower, wide_upper, inside_y), alpha=0.2
    )
    assert wide_q < 0.0
    wide = cqr.conformalise_cqr(
        wide_lower, wide_upper, inside_y, wide_lower, wide_upper, inside_y, alpha=0.2
    )
    assert ((wide.upper - wide.lower) < (wide_upper - wide_lower)).all()

    # Under-covering raw band: most outcomes fall outside, scores positive.
    tight_lower = np.full(n, -0.1)
    tight_upper = np.full(n, 0.1)
    spread_y = rng.normal(0.0, 1.0, n)
    tight_q = cqr.conformal_correction(
        cqr.conformity_scores(tight_lower, tight_upper, spread_y), alpha=0.2
    )
    assert tight_q > 0.0
    tight = cqr.conformalise_cqr(
        tight_lower,
        tight_upper,
        spread_y,
        tight_lower,
        tight_upper,
        spread_y,
        alpha=0.2,
    )
    assert ((tight.upper - tight.lower) > (tight_upper - tight_lower)).all()


def test_conformalise_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    cal_lower, cal_upper = np.full(500, -1.0), np.full(500, 1.0)
    cal_y = rng.normal(0.0, 1.0, 500)
    test_lower, test_upper = np.full(300, -1.0), np.full(300, 1.0)
    test_y = rng.normal(0.0, 1.0, 300)

    first = cqr.conformalise_cqr(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )
    second = cqr.conformalise_cqr(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )

    assert np.array_equal(first.lower, second.lower)
    assert np.array_equal(first.upper, second.upper)
    assert np.array_equal(first.covered, second.covered)
