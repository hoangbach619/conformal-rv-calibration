"""Unit tests for Sequential Predictive Conformal Inference.

The headline test drives a stream with autocorrelated conformity scores and
checks that SPCI keeps coverage near nominal while being sharper than CQR --
the autocorrelation-exploitation claim. The rest pin the no-lookahead property
and determinism. None touch the network.
"""

from __future__ import annotations

import numpy as np

from conformal_rv.conformal import spci
from conformal_rv.conformal.cqr import conformalise_cqr


def _clustered(n: int, seed: int, phi: float = 0.9, sigma: float = 0.35) -> np.ndarray:
    """Outcomes with volatility clustering, so the CQR scores autocorrelate."""
    rng = np.random.default_rng(seed)
    log_vol = 0.0
    out = np.empty(n)
    for t in range(n):
        log_vol = phi * log_vol + rng.normal(0.0, sigma)
        out[t] = np.exp(log_vol) * rng.normal()
    return out


def test_spci_is_sharper_than_cqr_at_near_nominal_coverage() -> None:
    n_cal, n_test = 1000, 350
    # One stationary realisation split into calibration and test, so CQR is
    # well-calibrated and the width comparison is fair.
    full = _clustered(n_cal + n_test, seed=0)
    cal_y, test_y = full[:n_cal], full[n_cal:]
    cal_lower, cal_upper = np.full(n_cal, -1.0), np.full(n_cal, 1.0)
    test_lower, test_upper = np.full(n_test, -1.0), np.full(n_test, 1.0)

    cqr = conformalise_cqr(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, alpha=0.2
    )
    sp = spci.conformalise_spci(
        cal_lower,
        cal_upper,
        cal_y,
        test_lower,
        test_upper,
        test_y,
        alpha=0.2,
        window=20,
        refit_every=70,
        max_iter=60,
    )

    cqr_width = float(np.median(cqr.upper - cqr.lower))
    spci_width = float(np.median(sp.upper - sp.lower))

    # Both near the 0.80 nominal level, and at essentially the same coverage SPCI
    # is the sharper interval because it exploits the score autocorrelation.
    assert 0.72 <= cqr.coverage <= 0.90
    assert 0.68 <= sp.coverage <= 0.86
    assert abs(sp.coverage - cqr.coverage) < 0.12
    assert spci_width < cqr_width


def test_spci_has_no_lookahead() -> None:
    n_cal, n_test = 200, 80
    rng = np.random.default_rng(1)
    cal_y = rng.normal(0.0, 1.0, n_cal)
    test_y = rng.normal(0.0, 1.5, n_test)
    cal_lower, cal_upper = np.full(n_cal, -1.0), np.full(n_cal, 1.0)
    test_lower, test_upper = np.full(n_test, -1.0), np.full(n_test, 1.0)

    settings = {"alpha": 0.2, "window": 12, "refit_every": 40, "max_iter": 40}
    base = spci.conformalise_spci(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, **settings
    )

    # Perturb only strictly future outcomes; rows at or before the cutoff use
    # only scores dated before them, so they must be untouched.
    cutoff = 40
    perturbed_y = test_y.copy()
    perturbed_y[cutoff + 1 :] += 10.0
    perturbed = spci.conformalise_spci(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, perturbed_y, **settings
    )

    upto = cutoff + 1
    assert np.array_equal(base.lower[:upto], perturbed.lower[:upto])
    assert np.array_equal(base.upper[:upto], perturbed.upper[:upto])
    assert np.array_equal(base.covered[:upto], perturbed.covered[:upto])
    # Sanity: the future perturbation did change later rows.
    assert not np.array_equal(base.covered, perturbed.covered)


def test_conformalise_spci_is_deterministic() -> None:
    n_cal, n_test = 200, 100
    rng = np.random.default_rng(7)
    cal_y = rng.normal(0.0, 1.0, n_cal)
    test_y = rng.normal(0.0, 1.5, n_test)
    cal_lower, cal_upper = np.full(n_cal, -1.0), np.full(n_cal, 1.0)
    test_lower, test_upper = np.full(n_test, -1.0), np.full(n_test, 1.0)

    settings = {"alpha": 0.2, "window": 12, "refit_every": 40, "max_iter": 40}
    first = spci.conformalise_spci(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, **settings
    )
    second = spci.conformalise_spci(
        cal_lower, cal_upper, cal_y, test_lower, test_upper, test_y, **settings
    )

    assert np.array_equal(first.lower, second.lower)
    assert np.array_equal(first.upper, second.upper)
    assert np.array_equal(first.covered, second.covered)
