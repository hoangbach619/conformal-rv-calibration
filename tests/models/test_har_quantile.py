"""Unit tests for the quantile-regression HAR.

Recovery uses a HAR data-generating process with location-shift Gaussian noise,
so the true conditional quantiles are known in closed form. The remaining tests
fix monotonicity, the no-lookahead property, and determinism. None touch the
network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from conformal_rv import features
from conformal_rv.models import har_quantile

# Standard-normal quantiles: with homoskedastic noise the conditional quantile
# at tau sits a fixed multiple of sigma above or below the conditional mean.
_Z = {0.1: float(norm.ppf(0.1)), 0.5: 0.0, 0.9: float(norm.ppf(0.9))}


def _simulate_har(
    n: int,
    intercept: float,
    beta_daily: float,
    beta_weekly: float,
    beta_monthly: float,
    sigma: float,
    seed: int,
) -> pd.Series[float]:
    """Simulate log-RV from a stationary HAR process with location-shift noise."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    values[:22] = rng.normal(0.0, 0.1, 22)
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
    return pd.Series(values, index=pd.bdate_range("2000-01-03", periods=n))


def test_predicted_quantiles_recover_the_truth() -> None:
    intercept, beta_d, beta_w, beta_m, sigma = 0.1, 0.3, 0.2, 0.1, 0.08
    log_rv = _simulate_har(12000, intercept, beta_d, beta_w, beta_m, sigma, seed=42)

    cascade = features.har_features(log_rv)
    model = har_quantile.QRHARModel().fit(cascade, log_rv.shift(-1))

    complete = cascade.dropna()
    location = (
        intercept
        + beta_d * complete["har_daily"]
        + beta_w * complete["har_weekly"]
        + beta_m * complete["har_monthly"]
    ).to_numpy()
    predictions = model.predict_quantiles(complete)

    # Each predicted quantile tracks the true conditional quantile
    # location + sigma * z_tau across the sample.
    for column, tau in enumerate(model.quantiles):
        true_quantile = location + sigma * _Z[tau]
        assert np.abs(predictions[:, column] - true_quantile).mean() < 0.06


def test_quantiles_are_monotone_after_rearrangement() -> None:
    # Regardless of whether the separate fits cross, the rearrangement must
    # deliver a non-decreasing row.
    log_rv = _simulate_har(2000, 0.1, 0.3, 0.2, 0.1, sigma=0.15, seed=3)
    cascade = features.har_features(log_rv)
    model = har_quantile.QRHARModel().fit(cascade, log_rv.shift(-1))

    predictions = model.predict_quantiles(cascade.dropna())
    assert (np.diff(predictions, axis=1) >= 0.0).all()


def test_predict_quantiles_have_no_lookahead() -> None:
    log_rv = _simulate_har(200, 0.1, 0.3, 0.2, 0.1, sigma=0.1, seed=1)
    cascade = features.har_features(log_rv)
    model = har_quantile.QRHARModel().fit(cascade.iloc[:80], log_rv.shift(-1).iloc[:80])

    # A forecast for one origin depends only on that origin's cascade row, which
    # itself uses only log-RV dated <= t: scoring a single row reproduces the
    # batch result exactly.
    complete = cascade.dropna()
    position = 100
    batch = model.predict_quantiles(complete)
    single = model.predict_quantiles(complete.iloc[[position]])
    assert np.allclose(batch[position], single[0])


def test_fit_is_deterministic() -> None:
    log_rv = _simulate_har(500, 0.1, 0.3, 0.2, 0.1, sigma=0.1, seed=7)
    cascade = features.har_features(log_rv)
    target = log_rv.shift(-1)

    first = har_quantile.QRHARModel().fit(cascade, target)
    second = har_quantile.QRHARModel().fit(cascade, target)

    complete = cascade.dropna()
    for tau in first.quantiles:
        assert np.array_equal(first.coefficients(tau), second.coefficients(tau))
    assert np.array_equal(
        first.predict_quantiles(complete), second.predict_quantiles(complete)
    )
