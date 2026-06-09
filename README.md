# conformal-rv-calibration

## Research question

Does any post-2023 online conformal method restore nominal *conditional*
coverage of realised-volatility prediction intervals through named regime
breaks, on a credible HAR base, and at what interval-width cost?

The intervals are built on a credible base (HAR-RV and quantile-regression
HAR), corrected online, and judged primarily on the gap between calm-period
coverage and coverage in the 60-trading-day window after each named break. The
study asks whether the recent online methods (Conformal PID, DtACI) close that
gap where the earlier baselines (ACI, AgACI) do not, and what that costs in
interval width.

The full design is frozen in [docs/preregistration.md](docs/preregistration.md)
before any model is run.

## Status

Design stage. No models have been run and there are no results yet. The source
tree is a typed scaffold of stubs; the pre-registration fixes the hypotheses,
endpoints, data and tests in advance.

## Layout

- `src/conformal_rv/` library: data, realised-vol estimators, features,
  walk-forward splits, models, conformal methods, metrics.
- `tests/` mirror the modules; the end-to-end integration test is gated behind
  `--slow`.
- `docs/preregistration.md` the frozen design.
- `notebooks/` exploratory work.

## Conventions

- Reproducibility: `SEED=42` and `n_jobs=1` throughout (see
  `conformal_rv.SEED`, `conformal_rv.N_JOBS`).
- Python 3.12, src-layout, setuptools.
- British spelling in prose and docstrings.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
ruff check src tests
mypy src
pytest            # fast subset; add --slow for the integration test
```
