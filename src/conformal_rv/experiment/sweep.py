"""Sweep harness: loop the engine over the pre-registered grid and persist.

The grid is the eight indices by the four horizons (1, 5, 10, 22) by the five
seeds (42..46). Each configuration is run by the engine and persisted as one
tidy long parquet under a gitignored results directory, with a manifest listing
the completed configurations. The sweep is resumable: a configuration whose
parquet already exists is skipped, so the full grid can run in stages.

No analysis happens here; seed aggregation and the FDR control come later. The
run itself is network-free given a warm data cache and deterministic per
configuration (the engine threads the seed through every stochastic step).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

from conformal_rv.data import INDICES
from conformal_rv.experiment.engine import ConfigurationRun, run_configuration

# Pre-registered grid (see docs/preregistration).
HORIZONS: tuple[int, ...] = (1, 5, 10, 22)
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

_DEFAULT_RESULTS_DIR: Path = Path(__file__).resolve().parents[3] / "results"
_MANIFEST_NAME = "manifest.csv"

# One run produces one ConfigurationRun for (index, horizon, seed).
RunFn = Callable[[str, int, int], ConfigurationRun]


def _config_name(index: str, horizon: int, seed: int) -> str:
    """Unambiguous, filesystem-safe stem for a configuration."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", index).strip("_")
    return f"{safe}__h{horizon}__seed{seed}"


def configuration_path(
    results_dir: str | Path, index: str, horizon: int, seed: int
) -> Path:
    """Parquet path for a configuration's persisted result."""
    return Path(results_dir) / f"{_config_name(index, horizon, seed)}.parquet"


def run_to_frame(run: ConfigurationRun) -> pd.DataFrame:
    """Flatten a run to one tidy long frame, one row per (date, band, method)."""
    frames = [
        pd.DataFrame(
            {
                "date": run.test_dates,
                "method": method,
                "band": band,
                "lower": result.lower,
                "upper": result.upper,
                "y": result.y,
                "covered": result.covered,
                "regime": run.regime,
                "days_since_break": run.days_since_break,
            }
        )
        for (band, method), result in run.results.items()
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["index"] = run.index
    frame["horizon"] = run.horizon
    frame["seed"] = run.seed
    frame["qr_converged"] = run.qr_converged
    return frame


def persist_run(run: ConfigurationRun, results_dir: str | Path) -> Path:
    """Write a run's tidy frame to its configuration parquet."""
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = configuration_path(directory, run.index, run.horizon, run.seed)
    run_to_frame(run).to_parquet(path)
    return path


def rebuild_manifest(results_dir: str | Path) -> pd.DataFrame:
    """Rewrite the manifest by scanning the completed configuration parquets.

    Rebuilt from the files rather than appended to, so the manifest is correct
    even when the sweep is resumed across separate invocations.
    """
    directory = Path(results_dir)
    records = []
    for path in sorted(directory.glob("*.parquet")):
        meta = pd.read_parquet(
            path, columns=["index", "horizon", "seed", "qr_converged"]
        ).iloc[0]
        records.append(
            {
                "index": str(meta["index"]),
                "horizon": int(meta["horizon"]),
                "seed": int(meta["seed"]),
                "file": path.name,
                "qr_converged": bool(meta["qr_converged"]),
            }
        )
    manifest = pd.DataFrame(
        records, columns=["index", "horizon", "seed", "file", "qr_converged"]
    )
    directory.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(directory / _MANIFEST_NAME, index=False)
    return manifest


@dataclass(frozen=True)
class SweepSummary:
    """What a sweep invocation did: configurations newly written and skipped."""

    written: list[Path]
    skipped: list[Path]


def run_sweep(
    results_dir: str | Path = _DEFAULT_RESULTS_DIR,
    *,
    indices: Sequence[str] = INDICES,
    horizons: Sequence[int] = HORIZONS,
    seeds: Sequence[int] = SEEDS,
    run_fn: RunFn = run_configuration,
) -> SweepSummary:
    """Run (or resume) the sweep over ``indices x horizons x seeds``.

    A configuration whose parquet already exists is skipped, so the grid can be
    completed in stages. The manifest is rebuilt from the files at the end.
    """
    directory = Path(results_dir)
    written: list[Path] = []
    skipped: list[Path] = []
    for index, horizon, seed in product(indices, horizons, seeds):
        path = configuration_path(directory, index, horizon, seed)
        if path.exists():
            skipped.append(path)
            continue
        persist_run(run_fn(index, horizon, seed), directory)
        written.append(path)

    rebuild_manifest(directory)
    return SweepSummary(written=written, skipped=skipped)
