"""Unit tests for the single-configuration experiment engine.

Alignment is the fragile part, so the tests pin: the regime labels against known
break onsets on a constructed range; the pooled results, regime and days sharing
one length and one (sorted, unique) date index; the realised target being log-RV
at t + h; and the engine's folds respecting the embargo without overlapping.
Run on synthetic series with small split windows, so none touch the network.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from conformal_rv.experiment import engine
from conformal_rv.metrics.coverage import CALM
from conformal_rv.splits import walk_forward_splits

# Small split windows so the pipeline runs quickly while still spanning folds.
_SPLIT = {"train": 400, "embargo": 44, "calibration": 100, "test": 100, "step": 100}
_HORIZON = 5


def _synthetic(n: int, seed: int) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Synthetic log-RV, OHLC and VIX on a shared business-day calendar."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2005-01-03", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    ohlc = pd.DataFrame(
        {
            "open": close + rng.normal(0.0, 0.5, n),
            "high": close + rng.uniform(0.5, 2.0, n),
            "low": close - rng.uniform(0.5, 2.0, n),
            "close": close,
        },
        index=dates,
    )
    log_rv = pd.Series(rng.normal(-1.0, 0.5, n), index=dates, name="log_rv")
    vix = pd.Series(15.0 + rng.uniform(0.0, 5.0, n), index=dates, name="close")
    return log_rv, ohlc, vix


@pytest.fixture(scope="module")
def configuration() -> tuple[pd.Series, engine.ConfigurationRun]:
    """Run the engine once on synthetic data and reuse it across tests."""
    log_rv, ohlc, vix = _synthetic(900, seed=0)
    run = engine.run_on_series(
        "^GSPC", log_rv, ohlc, vix, horizon=_HORIZON, seed=42, **_SPLIT
    )
    return log_rv, run


def test_regime_labels_match_known_onsets() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    onsets = {"COVID": "2020-02-24"}
    regime, days = engine.regime_and_days(dates, onsets, window=60)

    onset = int(dates.searchsorted(pd.Timestamp("2020-02-24")))
    labels = regime.to_numpy()
    assert (labels[onset : onset + 60] == "COVID").all()
    assert (labels[:onset] == CALM).all()
    assert (labels[onset + 60 :] == CALM).all()

    elapsed = days.to_numpy()
    assert np.isinf(elapsed[:onset]).all()  # no onset yet
    assert elapsed[onset] == 0.0
    assert elapsed[onset + 10] == 10.0


def test_pooled_results_regime_and_days_share_one_index(
    configuration: tuple[pd.Series, engine.ConfigurationRun],
) -> None:
    _, run = configuration
    n = run.test_dates.shape[0]
    assert n > 0
    assert run.regime.shape[0] == n
    assert run.days_since_break.shape[0] == n
    # The six QR-HAR methods plus CQR on the AR-baseline band all share the index.
    assert set(run.results) == set(engine._RESULT_KEYS)
    for result in run.results.values():
        assert result.lower.shape[0] == n
        assert result.upper.shape[0] == n
        assert result.y.shape[0] == n
        assert result.covered.shape[0] == n
    # Folds pooled in date order, no overlap, so the index is sorted and unique.
    assert run.test_dates.is_monotonic_increasing
    assert run.test_dates.is_unique
    assert isinstance(run.qr_converged, bool)


def test_realised_target_is_log_rv_at_t_plus_h(
    configuration: tuple[pd.Series, engine.ConfigurationRun],
) -> None:
    log_rv, run = configuration
    expected = log_rv.shift(-_HORIZON).reindex(run.test_dates).to_numpy()
    # Every result shares the realised target, and it is log-RV at t + h: this is
    # the no-lookahead alignment the engine must get right.
    for result in run.results.values():
        assert np.allclose(result.y, expected)


def test_engine_folds_respect_embargo_and_do_not_overlap(
    configuration: tuple[pd.Series, engine.ConfigurationRun],
) -> None:
    log_rv, _ = configuration
    dates = pd.DatetimeIndex(log_rv.index)
    blocks = walk_forward_splits(dates, **_SPLIT)
    assert len(blocks) >= 2

    for block in blocks:
        max_train = int(dates.get_loc(block.train[-1]))
        min_calibration = int(dates.get_loc(block.calibration[0]))
        max_calibration = int(dates.get_loc(block.calibration[-1]))
        min_test = int(dates.get_loc(block.test[0]))
        assert min_calibration - max_train - 1 >= _SPLIT["embargo"]
        assert min_test - max_calibration - 1 >= _SPLIT["embargo"]

    for earlier, later in itertools.pairwise(blocks):
        assert int(dates.get_loc(earlier.test[-1])) < int(dates.get_loc(later.test[0]))
