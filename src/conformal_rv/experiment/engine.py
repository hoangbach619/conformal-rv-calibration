"""Single-configuration experiment engine.

``run_configuration(index, horizon, seed)`` goes from raw OHLC to a pooled
out-of-sample ``ConformalResult`` for each of the six conformal methods, with a
regime-label array and a days-since-break array aligned to the pooled test
dates. This is the step that first touches real data, so it stays auditable and
deliberately does not loop over the panel yet.

Pipeline. Yang-Zhang log realised vol is the target; the feature panel is built
through features.py (exercising the leakage-disciplined pipeline on real data).
Walk-forward splits use the pre-registered windows. Per fold the QR-HAR base
produces the 0.1/0.5/0.9 quantile band that the conformal layer calibrates; the
point HAR and the AR baseline are fitted alongside and checked to share the same
test origins. The six methods (CQR and the five online corrections) calibrate on
the calibration block and predict the test block, and the test folds are pooled
into one result per method.

Alignment is the fragile part: every per-fold test origin keeps its date, the
realised target at ``t + h`` is joined by that date, the pooled results share a
single date index, and the regime labels and days-since-break are looked up on
that index.

Reproducibility. The base models are deterministic; ``seed`` sets SPCI's
``random_state``. ``conformal_rv.N_JOBS = 1`` holds (no parallelism here).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from conformal_rv import features
from conformal_rv.conformal.aci import conformalise_aci
from conformal_rv.conformal.agaci import conformalise_agaci
from conformal_rv.conformal.conformal_pid import conformalise_pid
from conformal_rv.conformal.cqr import ConformalResult, conformalise_cqr
from conformal_rv.conformal.dtaci import conformalise_dtaci
from conformal_rv.conformal.spci import conformalise_spci
from conformal_rv.data import load_index_ohlc, load_vix
from conformal_rv.metrics.coverage import CALM
from conformal_rv.models import baseline_ar, har, har_quantile
from conformal_rv.realised_vol import to_log_rv, yang_zhang
from conformal_rv.splits import walk_forward_splits

# 80% interval (0.1/0.9), the primary nominal level (see docs/preregistration).
ALPHA: float = 0.2
# Fixed learning rate for the single-gamma ACI baseline.
ACI_GAMMA: float = 0.05
# Post-break window length in trading days.
POST_BREAK_WINDOW: int = 60
# Band columns of the QR-HAR forecast consumed as the conformal interval.
_LOWER, _UPPER = "q0.1", "q0.9"

# Named break onsets (see docs/preregistration); the post-break window is the
# POST_BREAK_WINDOW trading days starting at the first trading day on or after.
BREAK_ONSETS: dict[str, str] = {
    "GFC2008": "2008-09-15",
    "EuroDebt2011": "2011-08-01",
    "China2015": "2015-08-24",
    "Volmageddon2018": "2018-02-05",
    "COVID2020": "2020-02-24",
    "RateShock2022": "2022-01-03",
}

METHOD_NAMES: tuple[str, ...] = (
    "CQR",
    "ACI",
    "AgACI",
    "ConformalPID",
    "DtACI",
    "SPCI",
)

# Result keys are (band, method). The six conformal methods calibrate the QR-HAR
# band; CQR additionally calibrates the AR-baseline band as the H3/H4 floor; and,
# when TFT is requested, the six methods also calibrate the TFT comparator band.
_QR_KEYS: tuple[tuple[str, str], ...] = tuple(
    ("qr_har", method) for method in METHOD_NAMES
)
_TFT_KEYS: tuple[tuple[str, str], ...] = tuple(
    ("tft", method) for method in METHOD_NAMES
)
_RESULT_KEYS: tuple[tuple[str, str], ...] = (*_QR_KEYS, ("ar_baseline", "CQR"))


def _result_keys(with_tft: bool) -> tuple[tuple[str, str], ...]:
    """Result keys for a run, optionally including the TFT comparator band."""
    return (*_RESULT_KEYS, *_TFT_KEYS) if with_tft else _RESULT_KEYS


@dataclass(frozen=True)
class ConfigurationRun:
    """A single (index, horizon, seed) run.

    ``results`` is the pooled out-of-sample result keyed by ``(band, method)``;
    ``regime`` and ``days_since_break`` are aligned to ``test_dates``;
    ``qr_converged`` records whether every quantile-regression fit converged.
    """

    index: str
    horizon: int
    seed: int
    test_dates: pd.DatetimeIndex
    regime: np.ndarray
    days_since_break: np.ndarray
    results: dict[tuple[str, str], ConformalResult]
    qr_converged: bool


@dataclass(frozen=True)
class _Band:
    """A conformal interval band joined to its realised target, by origin date."""

    lower: np.ndarray
    upper: np.ndarray
    y: np.ndarray
    index: pd.DatetimeIndex


def regime_and_days(
    dates: pd.DatetimeIndex,
    onsets: dict[str, str] = BREAK_ONSETS,
    window: int = POST_BREAK_WINDOW,
) -> tuple[pd.Series[str], pd.Series[float]]:
    """Regime label and days-since-most-recent-onset for every date.

    A date is labelled with a break if it falls in that break's ``window``
    trading days (the most recent break wins any overlap), otherwise ``CALM``.
    Days-since-break counts trading days from the most recent onset on or before
    the date, and is ``inf`` before the first onset.
    """
    axis = pd.DatetimeIndex(dates)
    n = axis.shape[0]
    labels = np.full(n, CALM, dtype="<U16")
    days = np.full(n, np.inf, dtype=float)

    positions = sorted(
        int(axis.searchsorted(pd.Timestamp(iso)))
        for iso in onsets.values()
        if int(axis.searchsorted(pd.Timestamp(iso))) < n
    )
    names_by_position = {
        int(axis.searchsorted(pd.Timestamp(iso))): name for name, iso in onsets.items()
    }
    # Ascending so a later (more recent) window overwrites an earlier overlap.
    for position in positions:
        labels[position : min(position + window, n)] = names_by_position[position]

    if positions:
        onset_positions = np.asarray(positions)
        all_positions = np.arange(n)
        most_recent = np.searchsorted(onset_positions, all_positions, side="right") - 1
        seen = most_recent >= 0
        days[seen] = all_positions[seen] - onset_positions[most_recent[seen]]

    regime: pd.Series[str] = pd.Series(labels, index=axis)
    days_since: pd.Series[float] = pd.Series(days, index=axis)
    return regime, days_since


def run_configuration(
    index: str, horizon: int, seed: int = 42, *, with_tft: bool = False
) -> ConfigurationRun:
    """Run one configuration, loading the index OHLC and VIX (cached)."""
    ohlc = load_index_ohlc(index)[index]
    log_rv = to_log_rv(yang_zhang(ohlc)).dropna()
    vix = load_vix()["close"]
    return run_on_series(index, log_rv, ohlc, vix, horizon, seed, with_tft=with_tft)


def run_on_series(
    index: str,
    log_rv: pd.Series[float],
    ohlc: pd.DataFrame,
    vix: pd.Series[float],
    horizon: int,
    seed: int = 42,
    *,
    with_tft: bool = False,
    train: int = 2000,
    embargo: int = 44,
    calibration: int = 250,
    test: int = 250,
    step: int = 250,
) -> ConfigurationRun:
    """Run one configuration on already-loaded series (no network).

    The split windows default to the pre-registered values; tests pass smaller
    ones. Folds whose test block has no realisable target are skipped. With
    ``with_tft`` the TFT comparator band is fitted and pooled alongside QR-HAR
    (its optional torch dependency is imported only then).
    """
    log_rv = log_rv.sort_index()

    # Build the feature panel through features.py: this exercises the
    # leakage-disciplined pipeline on real data even though the HAR base reads
    # its cascade from log-RV directly.
    panel = features.build_features({index: log_rv}, {index: ohlc}, vix)
    if panel.empty:
        raise RuntimeError(f"empty feature panel for {index}")

    dates = pd.DatetimeIndex(log_rv.index)
    blocks = walk_forward_splits(
        dates,
        train=train,
        embargo=embargo,
        calibration=calibration,
        test=test,
        step=step,
    )
    regime_full, days_full = regime_and_days(dates)
    target = log_rv.shift(-horizon)

    result_keys = _result_keys(with_tft)
    chunks: dict[tuple[str, str], list[ConformalResult]] = {
        key: [] for key in result_keys
    }
    pooled_dates: list[pd.DatetimeIndex] = []
    # Capture QuantReg non-convergence (IterationLimitWarning) over the whole run
    # so it is recorded in the result rather than printed and lost.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IterationLimitWarning)
        for block in blocks:
            quantile = har_quantile.block_quantile_forecasts(
                log_rv, block, horizons=(horizon,)
            )[horizon]
            # The point HAR and AR baseline are fitted alongside; all three base
            # models must forecast the same test origins (the alignment risk).
            point = har.block_point_forecasts(log_rv, block, horizons=(horizon,))[
                horizon
            ]
            ar = baseline_ar.block_quantile_forecasts(
                log_rv, block, horizons=(horizon,)
            )[horizon]
            if not (
                point.test.index.equals(quantile.test.index)
                and ar.test.index.equals(quantile.test.index)
            ):
                raise RuntimeError("base-model test origins are misaligned")

            qr_cal = _band(quantile.calibration, target)
            qr_test = _band(quantile.test, target)
            ar_cal = _band(ar.calibration, target)
            ar_test = _band(ar.test, target)
            if qr_cal.index.shape[0] == 0 or qr_test.index.shape[0] == 0:
                continue
            if not ar_test.index.equals(qr_test.index):
                raise RuntimeError("QR-HAR and AR-baseline test origins differ")

            # Six methods on the QR-HAR band; CQR on the AR-baseline band.
            for method, result in _apply_methods(qr_cal, qr_test, seed).items():
                chunks[("qr_har", method)].append(result)
            chunks[("ar_baseline", "CQR")].append(
                conformalise_cqr(
                    ar_cal.lower,
                    ar_cal.upper,
                    ar_cal.y,
                    ar_test.lower,
                    ar_test.upper,
                    ar_test.y,
                    ALPHA,
                )
            )

            # The TFT comparator band: same six methods, same origins.
            if with_tft:
                from conformal_rv.models import tft

                tft_forecast = tft.block_quantile_forecasts(
                    panel, log_rv, block, horizon, seed
                )[horizon]
                tft_cal = _band(tft_forecast.calibration, target)
                tft_test = _band(tft_forecast.test, target)
                if not tft_test.index.equals(qr_test.index):
                    raise RuntimeError("TFT and QR-HAR test origins differ")
                for method, result in _apply_methods(tft_cal, tft_test, seed).items():
                    chunks[("tft", method)].append(result)

            pooled_dates.append(qr_test.index)

        qr_converged = not any(
            issubclass(warning.category, IterationLimitWarning) for warning in caught
        )

    test_dates = (
        pd.DatetimeIndex(np.concatenate([idx.to_numpy() for idx in pooled_dates]))
        if pooled_dates
        else pd.DatetimeIndex([])
    )
    results = {key: _pool(chunks[key]) for key in result_keys}
    regime = regime_full.reindex(test_dates).to_numpy()
    days_since_break = days_full.reindex(test_dates).to_numpy()
    return ConfigurationRun(
        index=index,
        horizon=horizon,
        seed=seed,
        test_dates=test_dates,
        regime=regime,
        days_since_break=days_since_break,
        results=results,
        qr_converged=qr_converged,
    )


def _band(forecast: pd.DataFrame, target: pd.Series[float]) -> _Band:
    """Join the QR band columns to the realised target by origin date, drop NaN.

    A test origin t is kept only if both its forecast and its target at t + h
    are available, so the late origins whose target runs off the end are dropped.
    """
    frame = forecast.loc[:, [_LOWER, _UPPER]].copy()
    frame["_target"] = target.reindex(forecast.index)
    frame = frame.dropna()
    return _Band(
        lower=frame[_LOWER].to_numpy(),
        upper=frame[_UPPER].to_numpy(),
        y=frame["_target"].to_numpy(),
        index=pd.DatetimeIndex(frame.index),
    )


def _apply_methods(cal: _Band, tst: _Band, seed: int) -> dict[str, ConformalResult]:
    """Apply the six conformal methods to one fold's calibration and test band."""
    base = (cal.lower, cal.upper, cal.y, tst.lower, tst.upper, tst.y)
    return {
        "CQR": conformalise_cqr(*base, ALPHA),
        "ACI": conformalise_aci(*base, ALPHA, ACI_GAMMA).conformal,
        "AgACI": conformalise_agaci(*base, ALPHA).conformal,
        "ConformalPID": conformalise_pid(*base, ALPHA).conformal,
        "DtACI": conformalise_dtaci(*base, ALPHA).conformal,
        "SPCI": conformalise_spci(*base, ALPHA, random_state=seed),
    }


def _pool(parts: list[ConformalResult]) -> ConformalResult:
    """Concatenate per-fold results over the full out-of-sample test span."""
    if not parts:
        empty = np.empty(0, dtype=float)
        return ConformalResult(
            lower=empty, upper=empty, y=empty, covered=np.empty(0, dtype=bool)
        )
    return ConformalResult(
        lower=np.concatenate([part.lower for part in parts]),
        upper=np.concatenate([part.upper for part in parts]),
        y=np.concatenate([part.y for part in parts]),
        covered=np.concatenate([part.covered for part in parts]),
    )
