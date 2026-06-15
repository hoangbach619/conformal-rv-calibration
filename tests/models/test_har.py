"""Unit tests for the HAR-RV point model.

Coefficient recovery uses a synthetic HAR data-generating process; the other
tests fix horizon alignment, the no-lookahead property, and determinism. None
touch the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_rv import features
from conformal_rv.models import har


def _simulate_har(
    n: int,
    intercept: float,
    beta_daily: float,
    beta_weekly: float,
    beta_monthly: float,
    sigma: float,
    seed: int,
) -> pd.Series[float]:
    """Simulate log-RV from a stationary one-step HAR process.

    Each next value is the HAR cascade at t plus Gaussian noise, exactly the
    relationship HARModel is meant to recover.
    """
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    values[:22] = rng.normal(0.0, 0.1, 22)  # warm-up before a full month exists
    for t in range(21, n - 1):
        daily = values[t]
        weekly = values[t - 4 : t + 1].mean()
        monthly = values[t - 21 : t + 1].mean()
        values[t + 1] = (
            intercept
            + beta_daily * daily
            + beta_weekly * weekly
            + beta_monthly * monthly
            + rng.normal(0.0, sigma)
        )
    index = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(values, index=index, name="log_rv")


def test_fit_recovers_true_har_coefficients() -> None:
    truth = (0.10, 0.30, 0.20, 0.10)  # intercept, daily, weekly, monthly
    log_rv = _simulate_har(8000, *truth, sigma=0.03, seed=42)

    cascade = features.har_features(log_rv)
    model = har.HARModel().fit(cascade, log_rv.shift(-1))

    assert np.allclose(model.coefficients, np.array(truth), atol=0.05)


def test_horizon_target_is_t_plus_h() -> None:
    # On a pure counter series x[t] = t, the cascade is an exact linear function
    # of t, so the direct-h fit reproduces the target x[t + h] = x[t] + h.
    n = 200
    counter = pd.Series(
        np.arange(n, dtype=float), index=pd.bdate_range("2000-01-03", periods=n)
    )
    models = har.fit_multi_horizon(counter, horizons=(1, 5, 10))
    cascade = features.har_features(counter).dropna()

    for horizon, model in models.items():
        forecast = model.predict(cascade)
        expected = counter.loc[cascade.index].to_numpy() + horizon
        assert np.allclose(forecast, expected, atol=1e-6)


def test_cascade_and_forecast_have_no_lookahead() -> None:
    log_rv = _simulate_har(120, 0.1, 0.3, 0.2, 0.1, sigma=0.05, seed=1)
    cascade = features.har_features(log_rv)

    cutoff = 60
    perturbed = log_rv.copy()
    perturbed.iloc[cutoff + 1 :] += 5.0  # change strictly future values only
    cascade_perturbed = features.har_features(perturbed)

    # The cascade at t uses only log-RV dated <= t, so rows up to the cutoff are
    # untouched by a purely future perturbation.
    pd.testing.assert_frame_equal(
        cascade.iloc[: cutoff + 1], cascade_perturbed.iloc[: cutoff + 1]
    )

    # A forecast for origin t depends only on the cascade row at t: scoring the
    # single row at the cutoff matches scoring the whole history up to it.
    model = har.HARModel().fit(cascade.iloc[:50], log_rv.shift(-1).iloc[:50])
    full = model.predict(cascade.iloc[: cutoff + 1])
    single = model.predict(cascade.iloc[[cutoff]])
    assert full[-1] == pytest.approx(single[0])


def test_fit_is_deterministic() -> None:
    log_rv = _simulate_har(500, 0.1, 0.3, 0.2, 0.1, sigma=0.05, seed=7)
    cascade = features.har_features(log_rv)
    target = log_rv.shift(-1)

    first = har.HARModel().fit(cascade, target)
    second = har.HARModel().fit(cascade, target)

    complete = cascade.dropna()
    assert np.array_equal(first.coefficients, second.coefficients)
    assert np.array_equal(first.predict(complete), second.predict(complete))
