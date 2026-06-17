"""Quantile-regression HAR.

Produces conditional quantiles of log realised vol directly, which is what CQR
consumes. This is the second arm of the credible base: it gives a
heteroscedastic interval before any online correction is applied.

The estimator is an unregularised linear quantile regression (statsmodels
``QuantReg``) on the same HAR cascade, at each quantile tau, with the same
fixed regressor order as the point HAR: intercept, daily, weekly, monthly.
Separate per-quantile fits can cross, so ``predict_quantiles`` rearranges the
predictions to be monotone (a standard fix); the downstream CQR layer then
calibrates the interval, so this only needs to be a valid ordering.

Multi-horizon forecasting is direct, mirroring har.py: one model per horizon
(each carrying its three quantile fits), target log-RV at ``t + h``.

Reproducibility. Quantile regression here is a deterministic convex programme,
drawing no random numbers (so ``conformal_rv.SEED`` does not enter) and run
single-threaded (so ``conformal_rv.N_JOBS = 1`` holds by construction).
Identical inputs give identical coefficients and forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
import statsmodels.api as sm

from conformal_rv.features import har_features
from conformal_rv.splits import WalkForwardBlock

# Pre-registered nominal quantiles and horizons (see docs/preregistration).
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 22)

# Internal target column name, kept out of the way of any cascade column.
_TARGET = "__target__"

# QuantReg iteration cap. Raised above the statsmodels default of 1000 because
# real log-RV cascades occasionally need more IRLS steps; a run that still hits
# this limit emits an IterationLimitWarning, which the engine records.
_QUANTREG_MAX_ITER = 2000


def _quantile_column(tau: float) -> str:
    """Stable column label for a quantile forecast, e.g. ``q0.1``."""
    return f"q{tau:g}"


class QRHARModel:
    """Linear quantile regression of log-RV on the HAR cascade.

    Coefficients are stored per quantile as ``[intercept, beta_daily,
    beta_weekly, beta_monthly]`` in cascade-column order, so the same model can
    score any frame carrying those columns.
    """

    def __init__(self, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> None:
        # Sorted ascending so the rearrangement in predict_quantiles lines up
        # the j-th smallest prediction with the j-th smallest quantile.
        self.quantiles: tuple[float, ...] = tuple(sorted(quantiles))
        self._columns: list[str] | None = None
        self._coefficients: dict[float, np.ndarray] | None = None

    def fit(self, cascade: pd.DataFrame, target: pd.Series[float]) -> Self:
        """Fit one quantile regression per quantile on the complete rows.

        Rows with any missing cascade component (the leading months) or a
        missing target (the trailing horizon) are dropped, so no fit sees a
        partial window or an unrealised target.
        """
        columns = list(cascade.columns)
        frame = cascade.loc[:, columns].copy()
        frame[_TARGET] = target
        frame = frame.dropna()

        design = frame.loc[:, columns].to_numpy(dtype=float)
        response = frame[_TARGET].to_numpy(dtype=float)
        if design.shape[0] <= design.shape[1] + 1:
            raise ValueError("not enough complete observations to fit QR-HAR")

        # Explicit intercept column so the coefficient order is fixed: QuantReg
        # does not add a constant of its own.
        observations = design.shape[0]
        with_intercept = np.hstack([np.ones((observations, 1)), design])

        coefficients: dict[float, np.ndarray] = {}
        for tau in self.quantiles:
            fitted = sm.QuantReg(response, with_intercept).fit(
                q=tau, max_iter=_QUANTREG_MAX_ITER
            )
            coefficients[tau] = np.asarray(fitted.params, dtype=float)

        self._columns = columns
        self._coefficients = coefficients
        return self

    def predict_quantiles(self, cascade: pd.DataFrame) -> np.ndarray:
        """Return an ``(n, len(quantiles))`` array of monotone quantile forecasts.

        The raw per-quantile predictions are sorted ascending within each row.
        Rearrangement (Chernozhukov, Fernandez-Val and Galichon, 2010) repairs
        quantile crossing without changing the marginal distribution of each
        predicted quantile; CQR downstream calibrates the resulting interval.
        """
        if self._coefficients is None or self._columns is None:
            raise RuntimeError("QRHARModel.predict_quantiles called before fit")
        design = cascade.loc[:, self._columns].to_numpy(dtype=float)
        per_quantile = [
            self._coefficients[tau][0] + design @ self._coefficients[tau][1:]
            for tau in self.quantiles
        ]
        rearranged: np.ndarray = np.sort(np.column_stack(per_quantile), axis=1)
        return rearranged

    def coefficients(self, tau: float) -> np.ndarray:
        """Fitted ``[intercept, beta_daily, beta_weekly, beta_monthly]`` at ``tau``."""
        if self._coefficients is None:
            raise RuntimeError("QRHARModel is not fitted")
        if tau not in self._coefficients:
            raise KeyError(f"quantile {tau} was not fitted")
        return self._coefficients[tau]


def fit_multi_horizon(
    log_rv: pd.Series[float], horizons: tuple[int, ...] = DEFAULT_HORIZONS
) -> dict[int, QRHARModel]:
    """Fit one direct QR-HAR per horizon for a single index's log-RV series.

    The target for horizon ``h`` is log-RV at ``t + h``; the regressors are the
    cascade at ``t``. Each returned model holds the three per-quantile fits.
    """
    ordered = log_rv.sort_index()
    cascade = har_features(ordered)
    return {
        horizon: QRHARModel().fit(cascade, ordered.shift(-horizon))
        for horizon in horizons
    }


@dataclass(frozen=True)
class HorizonQuantileForecast:
    """Out-of-sample quantile forecasts for one horizon.

    ``calibration`` and ``test`` are frames indexed by origin date with one
    column per quantile (``q0.1`` etc.), in ascending quantile order.
    """

    horizon: int
    quantiles: tuple[float, ...]
    calibration: pd.DataFrame
    test: pd.DataFrame


def block_quantile_forecasts(
    log_rv: pd.Series[float],
    block: WalkForwardBlock,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[int, HorizonQuantileForecast]:
    """Fit on a block's train window; score its calibration and test windows.

    Mirrors har.py's ``block_point_forecasts`` for quantiles. The embargo built
    into the block keeps the train targets clear of the calibration window.
    """
    ordered = log_rv.sort_index()
    cascade = har_features(ordered)

    forecasts: dict[int, HorizonQuantileForecast] = {}
    for horizon in horizons:
        target = ordered.shift(-horizon)
        model = QRHARModel().fit(
            cascade.loc[cascade.index.isin(block.train)],
            target.loc[target.index.isin(block.train)],
        )
        forecasts[horizon] = HorizonQuantileForecast(
            horizon=horizon,
            quantiles=model.quantiles,
            calibration=predict_quantile_window(model, cascade, block.calibration),
            test=predict_quantile_window(model, cascade, block.test),
        )
    return forecasts


def predict_quantile_window(
    model: QRHARModel, cascade: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Score the complete cascade rows whose origin falls in ``dates``.

    Shared by the QR-HAR and the AR baseline (which passes its own one-column
    design), so the quantile column labelling stays in one place.
    """
    window = cascade.loc[cascade.index.isin(dates)].dropna()
    predictions = model.predict_quantiles(window)
    columns = [_quantile_column(tau) for tau in model.quantiles]
    return pd.DataFrame(predictions, index=window.index, columns=columns)
