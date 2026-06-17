"""Sequential Predictive Conformal Inference (Xu and Xie 2023).

Residual-modelling comparator. The CQR conformity scores form a *dependent*
series -- realised-vol residuals cluster -- so instead of taking their fixed
empirical quantile (as CQR does), SPCI fits a quantile model on a window of
lagged scores and predicts the *conditional* quantile of the next score. During
a calm cluster the predicted quantile shrinks and the interval narrows; during
a volatile cluster it grows. That conditional radius gives a narrower interval
at the same coverage when the scores are autocorrelated.

The conformity score here is the one-sided CQR score ``max(lower - y, y -
upper)`` (the signed distance the outcome falls outside the band), so coverage
is ``s_t <= Q_t`` and a single conditional ``1 - alpha`` score quantile is the
right radius -- the two-tail ``alpha/2`` / ``1 - alpha/2`` split of the original
SPCI applies to a *signed* residual; with this one-sided score the upper
``1 - alpha`` quantile is the level that yields a ``1 - alpha`` interval. The
radius may be negative (tightening) when the band over-covers.

Quantile model. The paper uses a quantile random forest; this is a
gradient-boosted-quantile variant (``HistGradientBoostingRegressor`` with the
quantile loss) chosen to avoid a new dependency. It is refitted on a fixed
cadence (``refit_every`` steps) rather than every step: fitting a boosted model
at each step would dominate the cost, and between refits the lagged-score
features still update so the predicted quantile keeps adapting -- only the
learned mapping is held fixed. The number of fits is ``ceil(n_test /
refit_every)``.

Sequential and no-lookahead. The model and the prediction window at step t use
only scores dated before t. Reproducibility: the run seed is threaded into the
booster's ``random_state`` (the engine passes the configuration seed; 42 is the
standalone default), and early stopping is disabled (no random validation
split), so the fit is deterministic; no parallelism is introduced
(``conformal_rv.N_JOBS = 1``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from conformal_rv.conformal.cqr import ConformalResult, conformity_scores

_SEED = 42


def _lagged_windows(scores: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix of ``window`` lagged scores and the next-score target.

    Row ``i`` is ``scores[i:i+window]`` predicting ``scores[i+window]``; every
    feature is dated strictly before its target, so there is no lookahead.
    """
    length = scores.shape[0]
    if length <= window:
        return np.empty((0, window)), np.empty(0)
    starts = np.arange(length - window)
    indices = starts[:, None] + np.arange(window)[None, :]
    return scores[indices], scores[window:]


def conformalise_spci(
    cal_lower: np.ndarray,
    cal_upper: np.ndarray,
    cal_y: np.ndarray,
    test_lower: np.ndarray,
    test_upper: np.ndarray,
    test_y: np.ndarray,
    alpha: float,
    window: int = 20,
    refit_every: int = 25,
    max_iter: int = 100,
    random_state: int = _SEED,
) -> ConformalResult:
    """Run SPCI over one index and one horizon's calibration and test streams.

    ``max_iter`` is the number of boosting iterations of the quantile model;
    fewer trades accuracy for speed.
    """
    test_lower = np.asarray(test_lower, dtype=float)
    test_upper = np.asarray(test_upper, dtype=float)
    test_y = np.asarray(test_y, dtype=float)

    calibration_scores = conformity_scores(cal_lower, cal_upper, cal_y)
    test_scores = conformity_scores(test_lower, test_upper, test_y)
    quantile_level = 1.0 - alpha

    # The score history seen so far, growing as each test score is revealed.
    seen: list[float] = [float(s) for s in calibration_scores]

    steps = int(test_y.shape[0])
    lowers = np.empty(steps, dtype=float)
    uppers = np.empty(steps, dtype=float)
    covered = np.empty(steps, dtype=bool)

    model: Any = None
    steps_since_refit = 0
    for t in range(steps):
        history = np.asarray(seen, dtype=float)

        if model is None or steps_since_refit >= refit_every:
            features, targets = _lagged_windows(history, window)
            if targets.shape[0] >= window:
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=quantile_level,
                    max_iter=max_iter,
                    random_state=random_state,
                    early_stopping=False,
                )
                model.fit(features, targets)
                steps_since_refit = 0

        if model is not None and history.shape[0] >= window:
            radius = float(model.predict(history[-window:].reshape(1, -1))[0])
        else:
            # Warm-up before a model can be fitted: the unconditional quantile.
            radius = float(np.quantile(history, quantile_level))

        lowers[t] = float(test_lower[t]) - radius
        uppers[t] = float(test_upper[t]) + radius
        covered[t] = bool(float(test_scores[t]) <= radius)

        # Reveal this step's score only after the interval is formed.
        seen.append(float(test_scores[t]))
        steps_since_refit += 1

    return ConformalResult(lower=lowers, upper=uppers, y=test_y, covered=covered)
