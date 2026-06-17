"""Slow tests for the TFT comparator band (the H4 model).

The TFT pulls in the optional ``tft`` extra (torch/lightning/pytorch-forecasting)
and trains a network, so these are gated behind ``--slow`` and skipped entirely
when the extra is not installed. They pin the one property the conformal layer
depends on -- the TFT forecasts land on exactly the QR-HAR origin dates -- plus
ordered, finite quantiles. A deliberately tiny config keeps the train cheap.
None touch the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pytorch_forecasting")

from conformal_rv import features
from conformal_rv.models import har_quantile, tft
from conformal_rv.splits import walk_forward_splits

# Small split windows and a near-minimal network: enough to span a fold and emit
# the three quantiles, fast enough to run under --slow.
_SPLIT = {"train": 300, "embargo": 20, "calibration": 60, "test": 60, "step": 60}
_HORIZON = 5
_TINY = tft.TFTConfig(
    encoder_length=20,
    hidden_size=4,
    attention_heads=1,
    hidden_continuous_size=2,
    dropout=0.0,
    batch_size=32,
    max_epochs=1,
)


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
def forecasts() -> tuple[har_quantile.HorizonQuantileForecast, ...]:
    """QR-HAR and TFT forecasts for one fold of one synthetic index."""
    log_rv, ohlc, vix = _synthetic(600, seed=0)
    index = "^GSPC"
    panel = features.build_features({index: log_rv}, {index: ohlc}, vix)
    block = walk_forward_splits(pd.DatetimeIndex(log_rv.index), **_SPLIT)[0]

    qr = har_quantile.block_quantile_forecasts(log_rv, block, horizons=(_HORIZON,))[
        _HORIZON
    ]
    deep = tft.block_quantile_forecasts(
        panel, log_rv, block, _HORIZON, seed=42, config=_TINY
    )[_HORIZON]
    return qr, deep


@pytest.mark.slow
def test_tft_origins_match_qr_har(
    forecasts: tuple[har_quantile.HorizonQuantileForecast, ...],
) -> None:
    qr, deep = forecasts
    # The conformal layer pools the TFT band beside QR-HAR fold by fold, so the
    # two must agree on the calibration and test origin dates exactly.
    assert deep.calibration.index.equals(qr.calibration.index)
    assert deep.test.index.equals(qr.test.index)


@pytest.mark.slow
def test_tft_emits_ordered_finite_quantiles(
    forecasts: tuple[har_quantile.HorizonQuantileForecast, ...],
) -> None:
    _, deep = forecasts
    columns = [har_quantile._quantile_column(tau) for tau in tft.QUANTILES]
    for block in (deep.calibration, deep.test):
        values = block.loc[:, columns].to_numpy()
        assert np.isfinite(values).all()
        # The model rearranges the quantile heads, so the band is always
        # non-crossing (lower <= median <= upper) for the conformal layer.
        assert (np.diff(values, axis=1) >= 0.0).all()
