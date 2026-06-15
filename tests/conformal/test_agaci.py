"""Unit tests for Aggregated Adaptive Conformal Inference.

AgACI is checked for near-nominal coverage without being told the best gamma,
for tracking the best single-gamma expert's loss up to the aggregation regret,
and for determinism. None touch the network.
"""

from __future__ import annotations

import numpy as np

from conformal_rv.conformal import agaci
from conformal_rv.conformal.aci import conformalise_aci


def _shift_stream(seed: int, n_cal: int, n_test: int) -> tuple[np.ndarray, ...]:
    """Calibration N(0, 1), test N(0, 2): a band tuned on calibration drifts."""
    rng = np.random.default_rng(seed)
    cal_y = rng.normal(0.0, 1.0, n_cal)
    test_y = rng.normal(0.0, 2.0, n_test)
    cal_lower, cal_upper = np.full(n_cal, -1.0), np.full(n_cal, 1.0)
    test_lower, test_upper = np.full(n_test, -1.0), np.full(n_test, 1.0)
    return cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y


def _pinball(y: np.ndarray, bound: np.ndarray, tau: float) -> float:
    """Total pinball loss of a bound at level ``tau`` over the stream."""
    residual = y - bound
    per_point = np.where(residual >= 0.0, tau * residual, (tau - 1.0) * residual)
    return float(per_point.sum())


def _interval_loss(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> float:
    """Pinball loss of both bounds at their matching quantile levels."""
    return _pinball(y, lower, alpha / 2.0) + _pinball(y, upper, 1.0 - alpha / 2.0)


def test_agaci_reaches_near_nominal_coverage() -> None:
    cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y = _shift_stream(
        seed=0, n_cal=1000, n_test=800
    )
    result = agaci.conformalise_agaci(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )
    # Parameter-free in gamma, yet coverage sits near the 0.80 nominal level.
    assert 0.72 <= result.conformal.coverage <= 0.88


def test_agaci_tracks_the_best_single_gamma_expert() -> None:
    alpha = 0.2
    cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y = _shift_stream(
        seed=1, n_cal=1000, n_test=800
    )
    result = agaci.conformalise_agaci(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=alpha
    )
    aggregate_loss = _interval_loss(
        test_y, result.conformal.lower, result.conformal.upper, alpha
    )

    expert_losses = []
    for gamma in agaci.DEFAULT_GAMMAS:
        expert = conformalise_aci(
            cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha, gamma
        )
        expert_losses.append(
            _interval_loss(
                test_y, expert.conformal.lower, expert.conformal.upper, alpha
            )
        )

    best = min(expert_losses)
    worst = max(expert_losses)
    # No worse than the best single gamma up to the aggregation regret, and a
    # clear improvement on the worst choice of gamma.
    assert aggregate_loss <= best * 1.2
    assert aggregate_loss < worst


def test_final_weights_are_a_distribution() -> None:
    cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y = _shift_stream(
        seed=2, n_cal=600, n_test=400
    )
    result = agaci.conformalise_agaci(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )
    expert_count = len(agaci.DEFAULT_GAMMAS)
    for weights in (result.weights_lower, result.weights_upper):
        assert weights.shape == (expert_count,)
        assert np.all(weights >= 0.0)
        assert abs(float(weights.sum()) - 1.0) < 1e-9  # a distribution


def test_conformalise_agaci_is_deterministic() -> None:
    stream = _shift_stream(seed=7, n_cal=500, n_test=300)
    first = agaci.conformalise_agaci(*stream, alpha=0.2)
    second = agaci.conformalise_agaci(*stream, alpha=0.2)

    assert np.array_equal(first.conformal.lower, second.conformal.lower)
    assert np.array_equal(first.conformal.upper, second.conformal.upper)
    assert np.array_equal(first.weights_lower, second.weights_lower)
    assert np.array_equal(first.weights_upper, second.weights_upper)
