#!/usr/bin/env python
"""Command-line entry point for launching (or resuming) the experiment sweep.

A full grid is a multi-hour job, so it belongs in a terminal rather than an
interactive session. This wraps ``run_sweep`` with argument parsing and a
one-line-per-configuration progress log. It defaults to the full HAR grid (all
indices, horizons 1/5/10/22, seeds 42..46, TFT off); the same script launches
the scoped TFT grid later via ``--with-tft`` with reduced ``--horizons``.

The sweep is resumable: a configuration whose parquet already exists is skipped,
so the run can be stopped with Ctrl-C and restarted, picking up where it left
off. Results land in the default results directory.

    python scripts/run_grid.py                          # full HAR grid
    python scripts/run_grid.py --with-tft --horizons 5 22 --indices ^GSPC
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import partial
from itertools import product

from conformal_rv.data import INDICES
from conformal_rv.experiment.engine import ConfigurationRun, run_configuration
from conformal_rv.experiment.sweep import HORIZONS, SEEDS, configuration_path, run_sweep


def _label(index: str, horizon: int, seed: int) -> str:
    """Uniform, aligned label for one configuration."""
    return f"{index:<8} h={horizon:<2} seed={seed}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch or resume the conformal-RV experiment sweep."
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        choices=list(INDICES),
        default=list(INDICES),
        metavar="TICKER",
        help="indices to sweep (default: all)",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        choices=list(HORIZONS),
        default=list(HORIZONS),
        metavar="H",
        help="forecast horizons (default: 1 5 10 22)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=list(SEEDS),
        default=list(SEEDS),
        metavar="SEED",
        help="seeds (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--with-tft",
        action="store_true",
        help="also fit the TFT comparator band (heavy; needs the tft extra)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    indices: list[str] = args.indices
    horizons: list[int] = args.horizons
    seeds: list[int] = args.seeds

    total = len(indices) * len(horizons) * len(seeds)
    tft = "on" if args.with_tft else "off"
    print(
        f"launching sweep: {len(indices)} indices x {len(horizons)} horizons "
        f"x {len(seeds)} seeds = {total} configs (TFT {tft})",
        flush=True,
    )

    base_run_fn = partial(run_configuration, with_tft=args.with_tft)

    def run_fn(index: str, horizon: int, seed: int) -> ConfigurationRun:
        run = base_run_fn(index, horizon, seed)
        print(f"wrote   {_label(index, horizon, seed)}", flush=True)
        return run

    summary = run_sweep(indices=indices, horizons=horizons, seeds=seeds, run_fn=run_fn)

    # Skipped configurations are resolved instantly inside run_sweep (their
    # parquet already exists), so they are reported after the live "wrote" stream
    # rather than interleaved. The full grid is replayed to keep them in order.
    results_dir = (summary.written or summary.skipped)[0].parent
    skipped = set(summary.skipped)
    for index, horizon, seed in product(indices, horizons, seeds):
        if configuration_path(results_dir, index, horizon, seed) in skipped:
            print(f"skipped {_label(index, horizon, seed)}", flush=True)

    print(
        f"done: {len(summary.written)} written, {len(summary.skipped)} skipped "
        f"-> {results_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
