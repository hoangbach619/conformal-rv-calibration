"""Temporal Fusion Transformer comparator band (the H4 model).

A genuine TFT (pytorch-forecasting's ``TemporalFusionTransformer``) with native
quantile outputs at 0.1/0.5/0.9, trained with the pinball (quantile) loss. It is
the deep comparator the study expects to under-cover in the stressed bucket
relative to HAR (H4). The architecture is the reference implementation, not
hand-rolled.

Why pytorch-forecasting over neuralforecast. We need a forecast at *each*
calibration and test origin, aligned to exactly the QR-HAR origins. The
``TimeSeriesDataSet`` plus ``predict(mode="quantiles")`` gives a per-origin
prediction over an arbitrary span, and ``QuantileLoss`` emits exactly the three
quantiles the conformal layer consumes. neuralforecast's ``predict`` forecasts
only from the series end, so matching arbitrary deep origins would need awkward
rolling cutoffs.

Contract (matching the HAR models). It reads the feature panel and the log-RV
target, fits on the train block, and returns the 0.1/0.5/0.9 quantiles on the
calibration and test blocks indexed by the same origin dates as QR-HAR. It is
direct multi-horizon: one model per horizon, predicting ``h`` steps and reading
the h-ahead step, never recursive.

Determinism. ``_seed_everything`` seeds python, numpy and torch and turns on
deterministic algorithms (``warn_only`` so an op without a deterministic kernel
warns rather than aborts), data loading is single-threaded, and training runs on
CPU to avoid MPS/GPU nondeterminism. Residual nondeterminism: ``warn_only`` means
any op lacking a deterministic kernel is *not* forced, so bit-for-bit identity is
not guaranteed across hardware or library versions; runs on one machine with a
fixed seed are reproducible in practice.

Isolation. torch, lightning and pytorch-forecasting are an optional ``tft``
extra; this module is imported lazily by the engine only when TFT is requested,
and is excluded from mypy strict (the torch stack lacks complete stubs).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from conformal_rv.models.har_quantile import HorizonQuantileForecast, _quantile_column
from conformal_rv.splits import WalkForwardBlock

# Native TFT quantiles; match the conformal band consumed downstream.
QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)

# Calendar covariates known into the future; lagged covariates known only to t.
_KNOWN_REALS = [
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "is_month_end",
    "is_quarter_end",
]
_UNKNOWN_REALS = ["log_rv", "lag_log_rv", "lag_return", "lag_range", "vix_level"]


@dataclass(frozen=True)
class TFTConfig:
    """Modest training budget for the comparator (documented as smoke-scale)."""

    encoder_length: int = 63
    hidden_size: int = 16
    attention_heads: int = 2
    hidden_continuous_size: int = 8
    dropout: float = 0.1
    learning_rate: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 5


# Module-level default so callers can omit a config without a call in defaults.
_DEFAULT_CONFIG = TFTConfig()


def _seed_everything(seed: int) -> None:
    """Seed python, numpy and torch deterministically (see module docstring)."""
    import torch
    from lightning.pytorch import seed_everything

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed_everything(seed, workers=True)


def _build_frame(features: pd.DataFrame, log_rv: pd.Series) -> pd.DataFrame:
    """Long frame for the TimeSeriesDataSet: target, reals, contiguous time_idx.

    The cross-index average is dropped (it is NaN for a single index), the bool
    calendar flags are cast to float, and rows with any missing real are dropped
    before a contiguous ``time_idx`` is assigned so windows have no gaps.
    """
    frame = features.sort_values("date").reset_index(drop=True)
    frame["log_rv"] = log_rv.reindex(frame["date"]).to_numpy()
    frame = frame.drop(columns=["cross_index_avg_rv"], errors="ignore")
    for flag in ("is_month_end", "is_quarter_end"):
        frame[flag] = frame[flag].astype(float)

    frame = frame.dropna(subset=[*_KNOWN_REALS, *_UNKNOWN_REALS]).reset_index(drop=True)
    frame["time_idx"] = np.arange(len(frame), dtype=int)
    frame["series"] = frame["index"].astype(str)
    return frame


def _training_dataset(
    frame: pd.DataFrame, train_cutoff: int, horizon: int, config: TFTConfig
):
    """TimeSeriesDataSet over the train block (decoder windows within train)."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    return TimeSeriesDataSet(
        frame[frame["time_idx"] <= train_cutoff],
        time_idx="time_idx",
        target="log_rv",
        group_ids=["series"],
        max_encoder_length=config.encoder_length,
        min_encoder_length=config.encoder_length,
        max_prediction_length=horizon,
        min_prediction_length=horizon,
        time_varying_known_reals=_KNOWN_REALS,
        time_varying_unknown_reals=_UNKNOWN_REALS,
        static_categoricals=["series"],
        target_normalizer=GroupNormalizer(groups=["series"]),
        allow_missing_timesteps=False,
    )


def _predict_origins(
    model,
    training,
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    horizon: int,
    config: TFTConfig,
) -> pd.DataFrame:
    """Predict the h-ahead quantiles at each origin date, indexed by that date.

    A prediction sample's first decoder step is ``origin + 1``; the h-ahead value
    is the last decoder step. The span is sliced to cover the encoder of the
    earliest origin and the decoder of the latest.
    """
    from pytorch_forecasting import TimeSeriesDataSet

    date_to_idx = dict(zip(frame["date"], frame["time_idx"], strict=True))
    idx_to_date = dict(zip(frame["time_idx"], frame["date"], strict=True))
    wanted = sorted(int(date_to_idx[d]) for d in origins if d in date_to_idx)
    if not wanted:
        columns = [_quantile_column(tau) for tau in QUANTILES]
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    lo = min(wanted) - config.encoder_length
    hi = max(wanted) + horizon
    span = frame[(frame["time_idx"] >= lo) & (frame["time_idx"] <= hi)]
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training, span, predict=False, stop_randomization=True
    )

    prediction = model.predict(
        prediction_dataset,
        mode="quantiles",
        return_index=True,
        trainer_kwargs={
            "accelerator": "cpu",
            "logger": False,
            "enable_progress_bar": False,
        },
    )
    # output is (n_samples, horizon, n_quantiles); the h-ahead step is the last.
    h_ahead = np.asarray(prediction.output[:, -1, :])
    # Quantile rearrangement (Chernozhukov et al.): QuantileLoss does not constrain
    # the heads to be ordered, so sort each row into a non-crossing band, as the
    # QR-HAR base does, guaranteeing lower <= median <= upper for the conformal layer.
    h_ahead = np.sort(h_ahead, axis=1)
    first_decoder_idx = prediction.index["time_idx"].to_numpy()
    origin_dates = [idx_to_date[int(t) - 1] for t in first_decoder_idx]

    columns = [_quantile_column(tau) for tau in QUANTILES]
    forecast = pd.DataFrame(
        h_ahead, index=pd.DatetimeIndex(origin_dates), columns=columns
    )
    forecast = forecast.sort_index()
    wanted_dates = pd.DatetimeIndex([idx_to_date[i] for i in wanted])
    return forecast.reindex(wanted_dates)


def block_quantile_forecasts(
    features: pd.DataFrame,
    log_rv: pd.Series,
    block: WalkForwardBlock,
    horizon: int,
    seed: int = 42,
    config: TFTConfig = _DEFAULT_CONFIG,
) -> dict[int, HorizonQuantileForecast]:
    """Fit a TFT on the train block; predict the calibration and test origins.

    Returns ``{horizon: HorizonQuantileForecast}`` with calibration and test
    quantile frames indexed by origin date, ready for the conformal layer.
    """
    from lightning.pytorch import Trainer
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    _seed_everything(seed)
    frame = _build_frame(features, log_rv)
    date_to_idx = dict(zip(frame["date"], frame["time_idx"], strict=True))

    train_idx = [date_to_idx[d] for d in block.train if d in date_to_idx]
    if not train_idx:
        raise RuntimeError("no training rows for the TFT in this block")
    train_cutoff = max(train_idx)

    training = _training_dataset(frame, train_cutoff, horizon, config)
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_heads,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        loss=QuantileLoss(quantiles=list(QUANTILES)),
        log_interval=-1,
        optimizer="adam",
    )
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="cpu",
        devices=1,
        deterministic="warn",
        gradient_clip_val=0.1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(
        model,
        train_dataloaders=training.to_dataloader(
            train=True, batch_size=config.batch_size, num_workers=0
        ),
    )

    calibration = _predict_origins(
        model, training, frame, block.calibration, horizon, config
    )
    test = _predict_origins(model, training, frame, block.test, horizon, config)
    return {
        horizon: HorizonQuantileForecast(
            horizon=horizon,
            quantiles=QUANTILES,
            calibration=calibration,
            test=test,
        )
    }
