"""Unit tests for the sweep harness.

The harness is tested independently of the heavy pipeline: a fake ``run_fn``
returns hand-built (deterministic) runs, so the tests pin the file and manifest
writing, the resumable skip on a second invocation, and a persisted file
round-tripping to the same coverage numbers. None touch the network.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest

from conformal_rv.conformal.cqr import ConformalResult
from conformal_rv.experiment import engine, sweep

_INDICES = ["^GSPC", "^FTSE"]
_HORIZONS = [5, 10]
_SEEDS = [42]


def _fake_run(index: str, horizon: int, seed: int) -> engine.ConfigurationRun:
    """A deterministic ConfigurationRun without the pipeline, for harness tests."""
    rng = np.random.default_rng([seed, horizon, sum(map(ord, index))])
    n = 40
    dates = pd.bdate_range("2010-01-01", periods=n)
    results = {
        key: ConformalResult(
            lower=rng.normal(-1.0, 0.5, n),
            upper=rng.normal(1.0, 0.5, n),
            y=rng.normal(0.0, 1.0, n),
            covered=rng.random(n) < 0.8,
        )
        for key in engine._RESULT_KEYS
    }
    return engine.ConfigurationRun(
        index=index,
        horizon=horizon,
        seed=seed,
        test_dates=dates,
        regime=np.full(n, "calm"),
        days_since_break=np.arange(n, dtype=float),
        results=results,
        qr_converged=True,
    )


def test_sweep_writes_files_and_manifest(tmp_path) -> None:
    calls: list[tuple[str, int, int]] = []

    def counting(index: str, horizon: int, seed: int) -> engine.ConfigurationRun:
        calls.append((index, horizon, seed))
        return _fake_run(index, horizon, seed)

    summary = sweep.run_sweep(
        tmp_path,
        indices=_INDICES,
        horizons=_HORIZONS,
        seeds=_SEEDS,
        run_fn=counting,
    )

    assert len(summary.written) == 4
    assert summary.skipped == []
    assert len(calls) == 4
    for index, horizon, seed in product(_INDICES, _HORIZONS, _SEEDS):
        assert sweep.configuration_path(tmp_path, index, horizon, seed).exists()

    manifest = pd.read_csv(tmp_path / "manifest.csv")
    assert len(manifest) == 4
    assert set(manifest["index"]) == set(_INDICES)
    assert manifest["qr_converged"].all()


def test_sweep_is_resumable(tmp_path) -> None:
    calls: list[tuple[str, int, int]] = []

    def counting(index: str, horizon: int, seed: int) -> engine.ConfigurationRun:
        calls.append((index, horizon, seed))
        return _fake_run(index, horizon, seed)

    kwargs = {"indices": _INDICES, "horizons": _HORIZONS, "seeds": _SEEDS}
    sweep.run_sweep(tmp_path, run_fn=counting, **kwargs)
    assert len(calls) == 4

    # Second invocation: every configuration already exists, so all are skipped
    # and the run function is not called again.
    second = sweep.run_sweep(tmp_path, run_fn=counting, **kwargs)
    assert second.written == []
    assert len(second.skipped) == 4
    assert len(calls) == 4


def test_persisted_file_round_trips_to_the_same_coverage(tmp_path) -> None:
    sweep.run_sweep(
        tmp_path, indices=["^GSPC"], horizons=[5], seeds=[42], run_fn=_fake_run
    )
    frame = pd.read_parquet(sweep.configuration_path(tmp_path, "^GSPC", 5, 42))

    # The fake run is deterministic, so re-running gives the in-memory truth.
    run = _fake_run("^GSPC", 5, 42)
    expected_columns = {
        "date",
        "method",
        "band",
        "lower",
        "upper",
        "y",
        "covered",
        "regime",
        "days_since_break",
        "index",
        "horizon",
        "seed",
        "qr_converged",
    }
    assert set(frame.columns) == expected_columns

    for (band, method), result in run.results.items():
        rows = frame[(frame["band"] == band) & (frame["method"] == method)]
        assert float(rows["covered"].mean()) == pytest.approx(result.coverage)
