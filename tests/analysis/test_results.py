"""Unit tests for the results analysis.

The analysis is tested on hand-built parquets persisted through the sweep, so
nothing refits and nothing touches the network. The fake runs carry a real
post-break window so the regime split, the aggregation shape and the verdicts
all exercise non-trivial values. One coverage-gap aggregate is hand-checked
against a direct computation on a single parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_rv.analysis import results
from conformal_rv.conformal.cqr import ConformalResult
from conformal_rv.experiment import engine, sweep
from conformal_rv.metrics.coverage import CALM

_INDICES = ["^GSPC", "^FTSE"]
_HORIZONS = [5, 10]
_SEEDS = [42, 43]
_N = 80
# A 20-day post-break window sits in the middle of each fake test span.
_BREAK = slice(30, 50)


def _fake_run(index: str, horizon: int, seed: int) -> engine.ConfigurationRun:
    """A deterministic ConfigurationRun with a real post-break window."""
    rng = np.random.default_rng([seed, horizon, sum(map(ord, index))])
    dates = pd.bdate_range("2010-01-01", periods=_N)
    regime = np.full(_N, CALM, dtype="<U16")
    regime[_BREAK] = "GFC2008"
    days = np.full(_N, np.inf)
    days[_BREAK] = np.arange(_BREAK.stop - _BREAK.start, dtype=float)
    band = {
        key: ConformalResult(
            lower=rng.normal(-1.0, 0.5, _N),
            upper=rng.normal(1.0, 0.5, _N),
            y=rng.normal(0.0, 1.0, _N),
            covered=rng.random(_N) < 0.8,
        )
        for key in engine._RESULT_KEYS
    }
    return engine.ConfigurationRun(
        index=index,
        horizon=horizon,
        seed=seed,
        test_dates=dates,
        regime=regime,
        days_since_break=days,
        results=band,
        qr_converged=True,
    )


@pytest.fixture
def results_dir(tmp_path):
    for index in _INDICES:
        for horizon in _HORIZONS:
            for seed in _SEEDS:
                sweep.persist_run(_fake_run(index, horizon, seed), tmp_path)
    return tmp_path


def test_loader_assembles_the_expected_long_frame(results_dir) -> None:
    frame = results.load_results(results_dir)

    assert set(results.EXPECTED_COLUMNS) <= set(frame.columns)
    assert set(frame["index"]) == set(_INDICES)
    assert set(frame["horizon"]) == set(_HORIZONS)
    assert set(frame["seed"]) == set(_SEEDS)

    pairs = set(map(tuple, frame[["band", "method"]].drop_duplicates().to_numpy()))
    assert pairs == set(engine._RESULT_KEYS)

    n_configs = len(_INDICES) * len(_HORIZONS) * len(_SEEDS)
    assert len(frame) == n_configs * len(engine._RESULT_KEYS) * _N


def test_aggregation_has_one_row_per_cell(results_dir) -> None:
    frame = results.load_results(results_dir)
    per_config = results.per_configuration_metrics(frame)

    n_configs = len(_INDICES) * len(_HORIZONS) * len(_SEEDS)
    assert len(per_config) == n_configs * len(engine._RESULT_KEYS)

    aggregated = results.aggregate_over_seeds(per_config)
    cells = per_config[list(results.CELL_KEYS)].drop_duplicates()
    assert len(aggregated) == len(cells)
    assert not aggregated.duplicated(subset=list(results.CELL_KEYS)).any()
    # Each metric expands to mean/std/min/max.
    for metric in results.METRIC_COLUMNS:
        for stat in ("mean", "std", "min", "max"):
            assert f"{metric}_{stat}" in aggregated.columns


def test_fdr_control_flags_a_subset(results_dir) -> None:
    frame = results.load_results(results_dir)
    per_config = results.apply_fdr_control(results.per_configuration_metrics(frame))

    for column in (
        "kupiec_reject_fdr",
        "kupiec_qvalue",
        "christoffersen_indep_reject_fdr",
        "christoffersen_indep_qvalue",
    ):
        assert column in per_config.columns
    # Rejections are a (possibly empty) subset, and q-values never undershoot p.
    assert per_config["kupiec_reject_fdr"].sum() <= len(per_config)
    assert (per_config["kupiec_qvalue"] >= per_config["kupiec_pvalue"] - 1e-12).all()


def test_verdict_table_runs_end_to_end_and_writes(results_dir) -> None:
    tables = results.run_analysis(results_dir)

    verdicts = tables.verdicts
    assert set(verdicts["hypothesis"]) == {
        "H1_static_undercovers",
        "H2_online_restores",
        "H3_calm_width_bounded",
    }
    assert set(verdicts["horizon"]) == set(_HORIZONS)
    assert verdicts["passed"].dtype == bool

    written = results.write_tables(results_dir, tables)
    assert [path.name for path in written] == [
        "per_configuration_metrics.csv",
        "seed_aggregates.csv",
        "headline_coverage_gap.csv",
        "verdicts.csv",
    ]
    assert all(path.exists() for path in written)


def test_coverage_gap_matches_a_direct_computation(results_dir) -> None:
    frame = results.load_results(results_dir)
    per_config = results.per_configuration_metrics(frame)

    index, horizon, seed, band, method = "^GSPC", 5, 42, "qr_har", "CQR"
    rows = frame[
        (frame["index"] == index)
        & (frame["horizon"] == horizon)
        & (frame["seed"] == seed)
        & (frame["band"] == band)
        & (frame["method"] == method)
    ]
    covered = rows["covered"].to_numpy(dtype=bool)
    regime = rows["regime"].to_numpy()
    direct_gap = covered[regime == CALM].mean() - covered[regime != CALM].mean()

    cell = per_config[
        (per_config["index"] == index)
        & (per_config["horizon"] == horizon)
        & (per_config["seed"] == seed)
        & (per_config["band"] == band)
        & (per_config["method"] == method)
    ]
    assert cell["coverage_gap"].iloc[0] == pytest.approx(float(direct_gap))
