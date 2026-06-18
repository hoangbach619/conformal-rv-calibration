"""Headline result figures (Phase 4g): plotting only, no new computation.

Reads the persisted sweep parquets and renders the four headline PNGs to the
committed ``figures/`` directory. Every number comes from the frozen intervals
and the existing coverage metrics (``coverage_decay_curve``,
``regime_conditional_coverage``); nothing is refitted and no new metric is
defined here. All four are on the ``qr_har`` band, pooled across indices and
seeds, at horizon 22 unless a figure states otherwise, with the 0.80 nominal
coverage line drawn for reference.

    PYTHONPATH=src python -m conformal_rv.analysis.figures
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless PNG rendering, no display required

import matplotlib.pyplot as plt

from conformal_rv.analysis.results import BASE_BAND, BASE_METHOD, DEFAULT_RESULTS_DIR
from conformal_rv.conformal.cqr import ConformalResult
from conformal_rv.experiment.engine import ALPHA, BREAK_ONSETS, METHOD_NAMES
from conformal_rv.metrics.coverage import (
    coverage_decay_curve,
    regime_conditional_coverage,
)

# Committed output directory, anchored to the repo root.
FIGURES_DIR: Path = Path(__file__).resolve().parents[3] / "figures"

NOMINAL: float = 1.0 - ALPHA  # 0.80 interval coverage
HEADLINE_HORIZON: int = 22

# Days-since-break bins for the decay curve: the 60-day post-break window in
# 5-day steps.
_DECAY_BUCKETS: np.ndarray = np.arange(0, 61, 5)

# One stable colour per method so figures 1 and 3 read consistently; CQR (the
# band that fails) is the bold red.
_METHOD_COLOURS: dict[str, str] = {
    "CQR": "#C44E52",
    "ACI": "#4C72B0",
    "AgACI": "#55A868",
    "ConformalPID": "#8172B3",
    "DtACI": "#CCB974",
    "SPCI": "#937860",
}
_CALM_COLOUR = "#4C72B0"
_POST_COLOUR = "#C44E52"


def _load_band(results_dir: str | Path, horizon: int, band: str) -> pd.DataFrame:
    """Pool the rows for one band and horizon across every persisted parquet."""
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(results_dir).glob("*.parquet")):
        df = pd.read_parquet(path)
        sub = df[(df["horizon"] == horizon) & (df["band"] == band)]
        if not sub.empty:
            frames.append(sub)
    if not frames:
        raise FileNotFoundError(f"no {band} h{horizon} rows under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def _result(rows: pd.DataFrame) -> ConformalResult:
    """Reconstruct a ConformalResult from persisted interval rows."""
    return ConformalResult(
        lower=rows["lower"].to_numpy(dtype=float),
        upper=rows["upper"].to_numpy(dtype=float),
        y=rows["y"].to_numpy(dtype=float),
        covered=rows["covered"].to_numpy(dtype=bool),
    )


def _draw_nominal(ax: Any) -> None:
    """Draw the 0.80 nominal-coverage reference line on an axis."""
    ax.axhline(
        NOMINAL, color="black", linestyle="--", linewidth=1.0, label="0.80 nominal"
    )


def fig_coverage_decay(band_rows: pd.DataFrame, path: Path) -> Path:
    """Coverage versus days since the most recent break, one line per method."""
    midpoints = (_DECAY_BUCKETS[:-1] + _DECAY_BUCKETS[1:]) / 2.0
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for method in METHOD_NAMES:
        rows = band_rows[band_rows["method"] == method]
        curve = coverage_decay_curve(
            _result(rows),
            rows["days_since_break"].to_numpy(dtype=float),
            _DECAY_BUCKETS,
        )
        ax.plot(
            midpoints,
            curve.coverage,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=_METHOD_COLOURS[method],
            label=method,
        )
    _draw_nominal(ax)
    ax.set_xlabel("Trading days since most recent break onset")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage decay after a regime break (qr_har band, h=22, pooled)")
    ax.set_ylim(0.55, 0.95)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_coverage_by_index(band_rows: pd.DataFrame, path: Path) -> Path:
    """Static-CQR calm vs post-break coverage, grouped bars per index by gap."""
    cqr = band_rows[band_rows["method"] == BASE_METHOD]
    records: list[tuple[str, float, float, float]] = []
    for index, rows in cqr.groupby("index"):
        coverage = regime_conditional_coverage(_result(rows), rows["regime"].to_numpy())
        records.append((str(index), coverage.calm, coverage.post_break, coverage.gap))
    records.sort(key=lambda record: record[3], reverse=True)

    labels = [record[0] for record in records]
    calm = [record[1] for record in records]
    post = [record[2] for record in records]
    positions = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(positions - width / 2, calm, width, label="calm", color=_CALM_COLOUR)
    ax.bar(positions + width / 2, post, width, label="post-break", color=_POST_COLOUR)
    _draw_nominal(ax)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Index (ordered by calm - post-break gap)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Static CQR coverage by index (qr_har band, h=22)")
    ax.set_ylim(0.5, 0.95)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_coverage_by_method(band_rows: pd.DataFrame, path: Path) -> Path:
    """Calm vs post-break coverage, grouped bars per conformal method."""
    labels = list(METHOD_NAMES)
    calm: list[float] = []
    post: list[float] = []
    for method in labels:
        rows = band_rows[band_rows["method"] == method]
        coverage = regime_conditional_coverage(_result(rows), rows["regime"].to_numpy())
        calm.append(coverage.calm)
        post.append(coverage.post_break)
    positions = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(positions - width / 2, calm, width, label="calm", color=_CALM_COLOUR)
    ax.bar(positions + width / 2, post, width, label="post-break", color=_POST_COLOUR)
    _draw_nominal(ax)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Conformal method (qr_har band)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calm vs post-break coverage by method (h=22, pooled)")
    ax.set_ylim(0.5, 0.95)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_interval_band_stress(band_rows: pd.DataFrame, path: Path) -> Path:
    """One under-covering index's static CQR band through the COVID break."""
    cqr = band_rows[band_rows["method"] == BASE_METHOD].copy()
    cqr["date"] = pd.to_datetime(cqr["date"])
    onset = pd.Timestamp(BREAK_ONSETS["COVID2020"])

    # The index that covers worst inside the COVID post-break window.
    covid = cqr[cqr["regime"] == "COVID2020"]
    cov_by_index = covid.groupby("index")["covered"].mean()
    chosen = str(cov_by_index.idxmin())
    chosen_cov = float(cov_by_index.min())

    # CQR is seed-invariant, so a single seed gives the band without duplicate
    # origin dates; a few months either side of the onset frame the spike.
    window = cqr[(cqr["index"] == chosen) & (cqr["seed"] == 42)].sort_values("date")
    lo, hi = onset - pd.Timedelta(days=120), onset + pd.Timedelta(days=170)
    window = window[(window["date"] >= lo) & (window["date"] <= hi)]

    dates = window["date"].to_numpy()
    realised = window["y"].to_numpy(dtype=float)
    lower = window["lower"].to_numpy(dtype=float)
    upper = window["upper"].to_numpy(dtype=float)
    uncovered = ~window["covered"].to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(
        dates, lower, upper, color="#4C72B0", alpha=0.25, label="static CQR 80% band"
    )
    ax.plot(dates, lower, color="#4C72B0", linewidth=0.8)
    ax.plot(dates, upper, color="#4C72B0", linewidth=0.8)
    ax.plot(
        dates, realised, color="black", linewidth=1.3, label="realised log-RV (h=22)"
    )
    ax.scatter(
        dates[uncovered],
        realised[uncovered],
        color="#C44E52",
        s=16,
        zorder=5,
        label="uncovered",
    )
    ax.axvline(
        onset, color="darkorange", linestyle="--", linewidth=1.2, label="COVID onset"
    )
    ax.set_xlabel("Origin date")
    ax.set_ylabel("log realised volatility (22-day-ahead target)")
    ax.set_title(
        f"Static CQR band fails through COVID — {chosen} "
        f"(post-break coverage {chosen_cov:.2f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_all(results_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    """Render the four headline figures and return their paths."""
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    band_rows = _load_band(results_dir, HEADLINE_HORIZON, BASE_BAND)
    return [
        fig_coverage_decay(band_rows, out / "coverage_decay.png"),
        fig_coverage_by_index(band_rows, out / "coverage_by_index.png"),
        fig_coverage_by_method(band_rows, out / "coverage_by_method.png"),
        fig_interval_band_stress(band_rows, out / "interval_band_stress.png"),
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the headline result figures from the persisted sweep."
    )
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    for path in render_all(args.results_dir, args.figures_dir):
        print(path)


if __name__ == "__main__":
    main()
