"""Read the persisted sweep, score every interval, and adjudicate the thesis.

This is the analysis step: it operates only on the frozen result parquets and
never refits anything. From the per-point intervals it computes, per
``(index, horizon, seed, band, method)``, the coverage (marginal, calm,
post-break and the calm-minus-post-break gap), the median width calm versus
post-break, the pinball loss, the PIT KS p-value, and the Kupiec and
Christoffersen statistics. The Christoffersen *independence* p-value is kept
separate from the conditional-coverage one because clustering of post-break
misses is the formal statement of the thesis.

It then aggregates across the five seeds (mean, standard deviation, min-to-max)
per ``(index, horizon, band, method)``, pools across indices for the headline
``(horizon, method)`` coverage gap, and applies Benjamini-Hochberg at the
pre-registered 0.10 across the family of Christoffersen independence tests and,
separately, across the Kupiec tests. Finally it adjudicates the three frozen
hypotheses (H1/H2/H3) into a pass-or-fail verdict per horizon, each reported
with the number that decided it.

The aggregated tables are written to CSV under the results directory, and
``python -m conformal_rv.analysis.results`` prints the headline coverage table
and the verdict table so the numbers are visible without opening a file.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from conformal_rv.conformal.cqr import ConformalResult
from conformal_rv.experiment.engine import ALPHA, METHOD_NAMES
from conformal_rv.metrics.backtests import christoffersen, kupiec_pof
from conformal_rv.metrics.coverage import CALM, regime_conditional_coverage
from conformal_rv.metrics.multiplicity import FDR_LEVEL, benjamini_hochberg
from conformal_rv.metrics.scoring import ks_uniformity, pinball_loss, pit_values
from conformal_rv.metrics.sharpness import summarise_width

# Default results directory, anchored to the repo root (mirrors the sweep).
DEFAULT_RESULTS_DIR: Path = Path(__file__).resolve().parents[3] / "results"

# Columns every persisted configuration parquet carries (see sweep.run_to_frame).
EXPECTED_COLUMNS: tuple[str, ...] = (
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
)

# Grouping keys: one configuration, and the seed-aggregated cell.
CONFIG_KEYS: tuple[str, ...] = ("index", "horizon", "seed", "band", "method")
CELL_KEYS: tuple[str, ...] = ("index", "horizon", "band", "method")

# Metrics carried through the seed aggregation (counts are diagnostics, not
# aggregated). The order fixes the column order of the per-configuration table.
METRIC_COLUMNS: tuple[str, ...] = (
    "coverage",
    "coverage_calm",
    "coverage_post_break",
    "coverage_gap",
    "width_median_calm",
    "width_median_post_break",
    "pinball_loss",
    "pit_ks_stat",
    "pit_ks_pvalue",
    "kupiec_stat",
    "kupiec_pvalue",
    "christoffersen_indep_stat",
    "christoffersen_indep_pvalue",
    "christoffersen_cc_stat",
    "christoffersen_cc_pvalue",
)

# The static base band/method and the online corrections layered on it.
BASE_BAND = "qr_har"
BASE_METHOD = "CQR"
ONLINE_METHODS: tuple[str, ...] = tuple(m for m in METHOD_NAMES if m != BASE_METHOD)

# Frozen hypothesis thresholds (see docs/preregistration).
NOMINAL: float = 1.0 - ALPHA  # 0.80 interval coverage
H1_COVERAGE_DROP: float = 0.05  # post-break coverage at least 5pp under nominal
H1_KUPIEC_ALPHA: float = 0.05  # one-sided Kupiec rejection level
H2_RESTORED_BAND: tuple[float, float] = (0.76, 0.84)  # restored post-break band
H3_WIDTH_RATIO: float = 1.25  # max calm-width inflation over the CQR base

# Headline horizon for the per-index H1 coverage breakdown.
HEADLINE_HORIZON: int = 22

# The two interval quantile levels (alpha/2 and 1 - alpha/2).
_LEVELS: np.ndarray = np.array([ALPHA / 2.0, 1.0 - ALPHA / 2.0])


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Load every configuration parquet under ``results_dir`` into one frame.

    The result is the tidy long frame keyed by index, horizon, seed, band and
    method (one row per test date within each). Raises if no parquet is found or
    a file is missing the expected columns.
    """
    directory = Path(results_dir)
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no result parquets under {directory}")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    missing = set(EXPECTED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"result parquets missing columns: {sorted(missing)}")
    return frame


def _result_of(group: pd.DataFrame) -> ConformalResult:
    """Reconstruct a ConformalResult from one configuration's rows."""
    return ConformalResult(
        lower=group["lower"].to_numpy(dtype=float),
        upper=group["upper"].to_numpy(dtype=float),
        y=group["y"].to_numpy(dtype=float),
        covered=group["covered"].to_numpy(dtype=bool),
    )


def _metrics_for_group(group: pd.DataFrame) -> dict[str, float]:
    """All per-configuration metrics, reusing the metrics and backtest modules."""
    result = _result_of(group)
    regime = group["regime"].to_numpy()
    calm_mask = regime == CALM

    coverage = regime_conditional_coverage(result, regime)
    width = summarise_width(result, regime)
    # PIT through the two-quantile predictive CDF; sort the pair so the
    # interpolation grid is ascending even if an interval inverts.
    forecasts = np.sort(np.column_stack([result.lower, result.upper]), axis=1)
    pit = pit_values(forecasts, result.y, _LEVELS)
    ks_stat, ks_pvalue = ks_uniformity(pit)

    violations = ~result.covered
    kupiec_stat, kupiec_pvalue = kupiec_pof(violations, ALPHA)
    chris = christoffersen(violations, ALPHA)

    return {
        "n": float(result.covered.shape[0]),
        "n_calm": float(calm_mask.sum()),
        "n_post_break": float((~calm_mask).sum()),
        "coverage": result.coverage,
        "coverage_calm": coverage.calm,
        "coverage_post_break": coverage.post_break,
        "coverage_gap": coverage.gap,
        "width_median_calm": width.calm_median,
        "width_median_post_break": width.post_break_median,
        "pinball_loss": pinball_loss(result.lower, result.upper, result.y, ALPHA),
        "pit_ks_stat": ks_stat,
        "pit_ks_pvalue": ks_pvalue,
        "kupiec_stat": kupiec_stat,
        "kupiec_pvalue": kupiec_pvalue,
        "christoffersen_indep_stat": chris.independence_statistic,
        "christoffersen_indep_pvalue": chris.independence_pvalue,
        "christoffersen_cc_stat": chris.conditional_coverage_statistic,
        "christoffersen_cc_pvalue": chris.conditional_coverage_pvalue,
    }


def per_configuration_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Score every ``(index, horizon, seed, band, method)`` configuration.

    One row per configuration, in sorted key order, carrying the coverage,
    width, scoring and backtest metrics. Rows are sorted by date within each
    group so the Christoffersen transition counts are temporally ordered.
    """
    records: list[dict[str, object]] = []
    for keys, group in frame.groupby(list(CONFIG_KEYS), sort=True):
        ordered = group.sort_values("date")
        record: dict[str, object] = dict(zip(CONFIG_KEYS, keys, strict=True))
        record.update(_metrics_for_group(ordered))
        records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate_over_seeds(per_config: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation and min-to-max across seeds per cell.

    One row per ``(index, horizon, band, method)``; each metric becomes four
    columns ``{metric}_{mean,std,min,max}``. The std uses ``ddof=1`` so a single
    seed yields NaN, flagging that the dispersion is not yet estimable.
    """
    grouped = per_config.groupby(list(CELL_KEYS), sort=True)[list(METRIC_COLUMNS)]
    aggregated = grouped.agg(["mean", "std", "min", "max"])
    # Flatten the (metric, stat) MultiIndex columns to "{metric}_{stat}".
    aggregated.columns = aggregated.columns.map("_".join)
    return aggregated.reset_index()


def headline_coverage_gap(per_config: pd.DataFrame) -> pd.DataFrame:
    """Pool across indices for the headline ``(horizon, band, method)`` gap.

    The coverage gap is pooled over indices and seeds, reported as its mean and
    dispersion (std, min, max) with the post-break coverage mean for context.
    The ``qr_har`` rows are the headline; the comparator bands ride alongside.
    """
    grouped = per_config.groupby(["horizon", "band", "method"], sort=True)
    headline = grouped.agg(
        coverage_gap_mean=("coverage_gap", "mean"),
        coverage_gap_std=("coverage_gap", "std"),
        coverage_gap_min=("coverage_gap", "min"),
        coverage_gap_max=("coverage_gap", "max"),
        coverage_post_break_mean=("coverage_post_break", "mean"),
        n_configs=("coverage_gap", "count"),
    )
    return headline.reset_index()


def apply_fdr_control(per_config: pd.DataFrame) -> pd.DataFrame:
    """Benjamini-Hochberg at 0.10 across the Christoffersen and Kupiec families.

    The independence tests and the Kupiec tests are each their own family across
    every ``(index, horizon, seed, band, method)`` configuration. Adds the
    rejection flag and the BH-adjusted q-value for each, leaving the per-test
    p-values untouched.
    """
    out = per_config.copy()
    kupiec = benjamini_hochberg(out["kupiec_pvalue"].to_numpy(dtype=float), FDR_LEVEL)
    indep = benjamini_hochberg(
        out["christoffersen_indep_pvalue"].to_numpy(dtype=float), FDR_LEVEL
    )
    out["kupiec_reject_fdr"] = kupiec.rejected
    out["kupiec_qvalue"] = kupiec.adjusted_pvalues
    out["christoffersen_indep_reject_fdr"] = indep.rejected
    out["christoffersen_indep_qvalue"] = indep.adjusted_pvalues
    return out


def _pooled_post_break_coverage(rows: pd.DataFrame) -> tuple[float, int]:
    """Post-break coverage pooled over the rows, with the post-break count."""
    regime = rows["regime"].to_numpy()
    covered = rows["covered"].to_numpy(dtype=bool)
    mask = regime != CALM
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(covered[mask].mean()), n


def _pooled_calm_coverage(rows: pd.DataFrame) -> float:
    """Calm coverage pooled over the rows."""
    regime = rows["regime"].to_numpy()
    covered = rows["covered"].to_numpy(dtype=bool)
    mask = regime == CALM
    if not bool(mask.any()):
        return float("nan")
    return float(covered[mask].mean())


def _pooled_calm_width_median(rows: pd.DataFrame) -> float:
    """Median calm interval width pooled over the rows."""
    regime = rows["regime"].to_numpy()
    width = (rows["upper"] - rows["lower"]).to_numpy(dtype=float)
    calm = width[regime == CALM]
    return float(np.median(calm)) if calm.size else float("nan")


def _one_sided_kupiec_post_break(rows: pd.DataFrame) -> float:
    """One-sided (under-coverage) Kupiec p-value on the pooled post-break points.

    The Kupiec LR is two-sided on a chi-square(1); halve it on the
    under-coverage side (observed violation rate above nominal) and take the
    complement otherwise, so a small value means significant under-coverage.
    """
    regime = rows["regime"].to_numpy()
    covered = rows["covered"].to_numpy(dtype=bool)
    mask = regime != CALM
    if not bool(mask.any()):
        return float("nan")
    violations = ~covered[mask]
    _, pvalue = kupiec_pof(violations, ALPHA)
    rate = float(violations.mean())
    return pvalue / 2.0 if rate > ALPHA else 1.0 - pvalue / 2.0


def _verdict_row(
    hypothesis: str,
    horizon: int,
    passed: bool,
    value: float,
    threshold: str,
    detail: str,
) -> dict[str, object]:
    return {
        "hypothesis": hypothesis,
        "horizon": horizon,
        "passed": passed,
        "value": value,
        "threshold": threshold,
        "detail": detail,
    }


def _adjudicate_h1(base: pd.DataFrame, horizon: int) -> dict[str, object]:
    """H1: the static CQR band under-covers post-break (>=5pp, one-sided Kupiec)."""
    cqr = base[base["method"] == BASE_METHOD]
    cov, n = _pooled_post_break_coverage(cqr)
    pvalue = _one_sided_kupiec_post_break(cqr)
    passed = bool(
        not np.isnan(cov)
        and cov <= NOMINAL - H1_COVERAGE_DROP
        and pvalue < H1_KUPIEC_ALPHA
    )
    return _verdict_row(
        "H1_static_undercovers",
        horizon,
        passed,
        cov,
        f"<= {NOMINAL - H1_COVERAGE_DROP:.2f} and kupiec_p_1s < {H1_KUPIEC_ALPHA}",
        f"post_break_cov={cov:.3f}, kupiec_p_1s={pvalue:.3f}, n_post_break={n}",
    )


def _adjudicate_h2(base: pd.DataFrame, horizon: int) -> dict[str, object]:
    """H2: at least one online method restores post-break coverage into the band."""
    lo, hi = H2_RESTORED_BAND
    covers = {
        method: _pooled_post_break_coverage(base[base["method"] == method])[0]
        for method in ONLINE_METHODS
    }
    valid = {m: c for m, c in covers.items() if not np.isnan(c)}
    in_band = {m: c for m, c in valid.items() if lo <= c <= hi}
    pool = in_band or valid
    if pool:
        method = min(pool, key=lambda m: abs(pool[m] - NOMINAL))
        value = pool[method]
    else:
        method, value = "none", float("nan")
    return _verdict_row(
        "H2_online_restores",
        horizon,
        bool(in_band),
        value,
        f"in [{lo}, {hi}]",
        f"method={method}, post_break_cov={value:.3f}",
    )


def _adjudicate_h3(base: pd.DataFrame, horizon: int) -> dict[str, object]:
    """H3: an online method keeps calm width within 1.25x the CQR base."""
    cqr_base = _pooled_calm_width_median(base[base["method"] == BASE_METHOD])
    ratios = {
        method: _pooled_calm_width_median(base[base["method"] == method]) / cqr_base
        for method in ONLINE_METHODS
        if cqr_base and not np.isnan(cqr_base)
    }
    valid = {m: r for m, r in ratios.items() if not np.isnan(r)}
    if valid:
        method = min(valid, key=lambda m: valid[m])
        ratio = valid[method]
    else:
        method, ratio = "none", float("nan")
    return _verdict_row(
        "H3_calm_width_bounded",
        horizon,
        bool(valid and ratio <= H3_WIDTH_RATIO),
        ratio,
        f"<= {H3_WIDTH_RATIO}",
        f"method={method}, calm_width_ratio={ratio:.3f}, cqr_base={cqr_base:.4f}",
    )


def adjudicate_hypotheses(frame: pd.DataFrame) -> pd.DataFrame:
    """Pass-or-fail verdicts for H1/H2/H3 at every horizon, from the raw frame.

    The deciding quantities pool the raw post-break (or calm) test points over
    all indices and seeds for the base ``qr_har`` band at each horizon, so each
    verdict rests on the full out-of-sample sample rather than an average of
    per-configuration coverages.
    """
    records: list[dict[str, object]] = []
    for horizon in sorted(int(h) for h in frame["horizon"].unique()):
        base = frame[(frame["horizon"] == horizon) & (frame["band"] == BASE_BAND)]
        records.append(_adjudicate_h1(base, horizon))
        records.append(_adjudicate_h2(base, horizon))
        records.append(_adjudicate_h3(base, horizon))
    return pd.DataFrame.from_records(records)


def per_index_h1(frame: pd.DataFrame, horizon: int = HEADLINE_HORIZON) -> pd.DataFrame:
    """Static-CQR calm/post-break coverage and gap per index at one horizon.

    Pools across seeds and dates for the base ``qr_har`` CQR band; one row per
    index, ordered by the calm-minus-post-break gap (largest failure first).
    These are the numbers behind the per-index figure.
    """
    base = frame[
        (frame["horizon"] == horizon)
        & (frame["band"] == BASE_BAND)
        & (frame["method"] == BASE_METHOD)
    ]
    records: list[dict[str, object]] = []
    for index, rows in base.groupby("index"):
        calm = _pooled_calm_coverage(rows)
        post, n_post = _pooled_post_break_coverage(rows)
        records.append(
            {
                "index": str(index),
                "coverage_calm": calm,
                "coverage_post_break": post,
                "coverage_gap": calm - post,
                "n_post_break": n_post,
            }
        )
    # Explicit columns so an absent horizon yields an empty (not column-less)
    # frame the sort can still operate on.
    columns = [
        "index",
        "coverage_calm",
        "coverage_post_break",
        "coverage_gap",
        "n_post_break",
    ]
    table = pd.DataFrame(records, columns=columns)
    return table.sort_values("coverage_gap", ascending=False).reset_index(drop=True)


def robustness_verdicts(frame: pd.DataFrame) -> pd.DataFrame:
    """H1/H2/H3 verdicts on all configs vs dropping non-converged QuantReg fits.

    Reruns the adjudication on the full frame and on the subset with
    ``qr_converged`` True, tagging each block, so the verdict can be read with
    and without the non-converged configurations (the result should survive
    dropping them). Rows are ordered so each hypothesis/horizon shows the two
    config sets adjacent.
    """
    full = adjudicate_hypotheses(frame)
    full.insert(0, "configs", "all")
    converged = adjudicate_hypotheses(frame[frame["qr_converged"]])
    converged.insert(0, "configs", "qr_converged_only")
    combined = pd.concat([full, converged], ignore_index=True)
    ordered = combined.sort_values(["hypothesis", "horizon", "configs"])
    return ordered.reset_index(drop=True)


@dataclass(frozen=True)
class AnalysisTables:
    """The tables the analysis produces from the persisted results."""

    per_configuration: pd.DataFrame
    seed_aggregates: pd.DataFrame
    headline_coverage_gap: pd.DataFrame
    verdicts: pd.DataFrame
    per_index_h1: pd.DataFrame
    robustness_verdicts: pd.DataFrame


def run_analysis(results_dir: str | Path) -> AnalysisTables:
    """Run the full analysis on the persisted parquets (no refitting, no writes)."""
    frame = load_results(results_dir)
    per_config = apply_fdr_control(per_configuration_metrics(frame))
    return AnalysisTables(
        per_configuration=per_config,
        seed_aggregates=aggregate_over_seeds(per_config),
        headline_coverage_gap=headline_coverage_gap(per_config),
        verdicts=adjudicate_hypotheses(frame),
        per_index_h1=per_index_h1(frame),
        robustness_verdicts=robustness_verdicts(frame),
    )


def write_tables(results_dir: str | Path, tables: AnalysisTables) -> list[Path]:
    """Write the aggregated tables to CSV under the results directory."""
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_configuration_metrics.csv": tables.per_configuration,
        "seed_aggregates.csv": tables.seed_aggregates,
        "headline_coverage_gap.csv": tables.headline_coverage_gap,
        "verdicts.csv": tables.verdicts,
        "per_index_h1.csv": tables.per_index_h1,
        "verdicts_robustness.csv": tables.robustness_verdicts,
    }
    written: list[Path] = []
    for name, table in outputs.items():
        path = directory / name
        table.to_csv(path, index=False)
        written.append(path)
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse the persisted sweep and adjudicate the hypotheses."
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="directory of persisted result parquets (default: repo results/)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    tables = run_analysis(args.results_dir)
    written = write_tables(args.results_dir, tables)

    headline = tables.headline_coverage_gap.round(4)
    print("Headline coverage gap (calm - post-break), pooled across indices:\n")
    print(headline.to_string(index=False))
    print("\nHypothesis verdicts (per horizon):\n")
    print(tables.verdicts.round(4).to_string(index=False))
    print(
        f"\nPer-index H1: static CQR coverage at h={HEADLINE_HORIZON}, "
        "ordered by gap:\n"
    )
    print(tables.per_index_h1.round(4).to_string(index=False))
    print("\nVerdict robustness (all configs vs dropping non-converged QuantReg):\n")
    print(tables.robustness_verdicts.round(4).to_string(index=False))
    print(f"\nWrote {len(written)} tables to {Path(args.results_dir)}")


if __name__ == "__main__":
    main()
