"""HAR-RV point forecaster (Corsi 2009).

This is the primary credible base on which the conformal layer sits. The HAR
cascade is a parsimonious long-memory proxy; keeping it simple matters because
the study is about calibration of intervals, not about squeezing point error.

Multi-horizon forecasting is *direct*, not iterated: a separate HAR is fitted
per horizon, with the target for horizon ``h`` being log-RV at ``t + h`` and
the regressors the cascade at ``t``. Direct forecasting avoids compounding a
one-step model's error and keeps each horizon's calibration self-contained.

Reproducibility. Fitting is ordinary least squares: deterministic, drawing no
random numbers (so ``conformal_rv.SEED`` does not enter) and single-threaded
(so ``conformal_rv.N_JOBS = 1`` holds by construction). Identical inputs give
identical coefficients and forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from conformal_rv.features import har_features
from conformal_rv.splits import WalkForwardBlock

# Pre-registered forecast horizons in trading days (see docs/preregistration).
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 22)

# Internal target column name, kept out of the way of any cascade column.
_TARGET = "__target__"


class HARModel:
    """Ordinary least squares HAR-RV regression on log realised vol.

    Coefficients are stored as ``[intercept, beta_daily, beta_weekly,
    beta_monthly]`` in the order of the cascade columns, so the same model can
    score any cascade frame with those columns.
    """

    def __init__(self) -> None:
        self._columns: list[str] | None = None
        self._coefficients: np.ndarray | None = None

    def fit(self, cascade: pd.DataFrame, target: pd.Series[float]) -> HARModel:
        """Fit by OLS on the rows where both cascade and target are present.

        Rows with any missing cascade component (the leading months) or a
        missing target (the trailing horizon) are dropped, so the regression
        never sees a partial window or an unrealised target.
        """
        columns = list(cascade.columns)
        frame = cascade.loc[:, columns].copy()
        frame[_TARGET] = target
        frame = frame.dropna()

        design = frame.loc[:, columns].to_numpy(dtype=float)
        response = frame[_TARGET].to_numpy(dtype=float)
        if design.shape[0] <= design.shape[1]:
            raise ValueError("not enough complete observations to fit HAR")

        # Explicit intercept column rather than sm.add_constant, so the
        # coefficient order is fixed regardless of statsmodels' conventions.
        observations = design.shape[0]
        with_intercept = np.hstack([np.ones((observations, 1)), design])
        results = sm.OLS(response, with_intercept).fit()

        self._columns = columns
        self._coefficients = np.asarray(results.params, dtype=float)
        return self

    def predict(self, cascade: pd.DataFrame) -> np.ndarray:
        """Point forecast of log-RV for each row of ``cascade``."""
        if self._coefficients is None or self._columns is None:
            raise RuntimeError("HARModel.predict called before fit")
        design = cascade.loc[:, self._columns].to_numpy(dtype=float)
        intercept = float(self._coefficients[0])
        betas = self._coefficients[1:]
        forecast: np.ndarray = intercept + design @ betas
        return forecast

    @property
    def coefficients(self) -> np.ndarray:
        """The fitted ``[intercept, beta_daily, beta_weekly, beta_monthly]``."""
        if self._coefficients is None:
            raise RuntimeError("HARModel is not fitted")
        return self._coefficients


def fit_multi_horizon(
    log_rv: pd.Series[float], horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> dict[int, HARModel]:
    """Fit one direct HAR per horizon for a single index's log-RV series.

    The target for horizon ``h`` is log-RV at ``t + h`` (``log_rv.shift(-h)``
    aligned to origin ``t``); the regressors are the cascade at ``t``. No
    iterated forecasting: each horizon is its own regression.
    """
    ordered = log_rv.sort_index()
    cascade = har_features(ordered)
    return {
        horizon: HARModel().fit(cascade, ordered.shift(-horizon))
        for horizon in horizons
    }


@dataclass(frozen=True)
class HorizonForecast:
    """Out-of-sample point forecasts for one horizon, indexed by origin date."""

    horizon: int
    calibration: pd.Series[float]
    test: pd.Series[float]


def block_point_forecasts(
    log_rv: pd.Series[float],
    block: WalkForwardBlock,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[int, HorizonForecast]:
    """Fit on a block's train window; score its calibration and test windows.

    Operates on one index's log-RV series. For each horizon, a HAR is fitted on
    the train origins and then applied, unchanged, to the calibration and test
    origins, giving genuinely out-of-sample point forecasts. The embargo built
    into the block keeps the train targets clear of the calibration window.
    """
    ordered = log_rv.sort_index()
    cascade = har_features(ordered)

    forecasts: dict[int, HorizonForecast] = {}
    for horizon in horizons:
        target = ordered.shift(-horizon)
        model = HARModel().fit(
            cascade.loc[cascade.index.isin(block.train)],
            target.loc[target.index.isin(block.train)],
        )
        forecasts[horizon] = HorizonForecast(
            horizon=horizon,
            calibration=_predict_window(model, cascade, block.calibration),
            test=_predict_window(model, cascade, block.test),
        )
    return forecasts


def _predict_window(
    model: HARModel, cascade: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.Series[float]:
    """Score the cascade rows whose origin falls in ``dates`` (complete rows)."""
    window = cascade.loc[cascade.index.isin(dates)].dropna()
    return pd.Series(model.predict(window), index=window.index, name="forecast")
