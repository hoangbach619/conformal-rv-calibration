"""Quantile AR(1) sanity baseline.

The complexity-justification floor for the HAR cascade. It is the same
unregularised quantile regression as QR-HAR, with the same fixed regressor
order and the same monotone rearrangement, but stripped to a single regressor:
the most recent realised log-RV at the forecast origin (the one-day, "lag-1"
HAR term). HAR adds the weekly and monthly cascade terms on top, so this is the
nested model the cascade must beat on pinball loss.

Reproducibility follows QR-HAR: deterministic, single-threaded, no RNG, so
``conformal_rv.SEED`` does not enter and ``conformal_rv.N_JOBS = 1`` holds by
construction.
"""

from __future__ import annotations

import pandas as pd

from conformal_rv.models.har_quantile import (
    DEFAULT_HORIZONS,
    HorizonQuantileForecast,
    QRHARModel,
    predict_quantile_window,
)
from conformal_rv.splits import WalkForwardBlock


class QRARBaseline(QRHARModel):
    """Quantile AR(1) baseline: QR-HAR restricted to the single lag-1 regressor.

    No new behaviour over :class:`QRHARModel`; the difference is entirely in the
    design it is fed (see :func:`ar1_features`). Kept a distinct type so the
    baseline is identifiable in results and ``isinstance`` checks.
    """


def ar1_features(log_rv: pd.Series[float]) -> pd.DataFrame:
    """Single regressor: the realised log-RV at the origin ``t``.

    This is the one-day ("lag-1") HAR component and the most recent value in the
    information set at ``t`` -- the same daily term the HAR cascade uses, here on
    its own. No fill is applied; back-fill is never used.
    """
    return pd.DataFrame({"lag1_log_rv": log_rv})


def fit_multi_horizon(
    log_rv: pd.Series[float], horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> dict[int, QRARBaseline]:
    """Fit one direct quantile AR(1) per horizon for a single index's log-RV."""
    ordered = log_rv.sort_index()
    design = ar1_features(ordered)
    return {
        horizon: QRARBaseline().fit(design, ordered.shift(-horizon))
        for horizon in horizons
    }


def block_quantile_forecasts(
    log_rv: pd.Series[float],
    block: WalkForwardBlock,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[int, HorizonQuantileForecast]:
    """Fit on a block's train window; score its calibration and test windows.

    The exact mirror of QR-HAR's block helper, over the one-regressor design.
    """
    ordered = log_rv.sort_index()
    design = ar1_features(ordered)

    forecasts: dict[int, HorizonQuantileForecast] = {}
    for horizon in horizons:
        target = ordered.shift(-horizon)
        model = QRARBaseline().fit(
            design.loc[design.index.isin(block.train)],
            target.loc[target.index.isin(block.train)],
        )
        forecasts[horizon] = HorizonQuantileForecast(
            horizon=horizon,
            quantiles=model.quantiles,
            calibration=predict_quantile_window(model, design, block.calibration),
            test=predict_quantile_window(model, design, block.test),
        )
    return forecasts
