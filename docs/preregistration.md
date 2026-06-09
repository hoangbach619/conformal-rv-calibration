# Pre-registration: conditional coverage of realised-volatility intervals through regime breaks

This document fixes the design before any data is loaded or any model is run.
Every hypothesis and decision threshold here is fixed and frozen as of this
commit, with no remaining TODOs; the commit timestamp is the evidence of the
freeze. Where a number is already a structural part of the design (the
60-trading-day post-break window, the forecast horizons, the
false-discovery-rate level, the seed) it is stated directly; these are design
choices, not the thresholds the study is powered to detect.

## 1. Question

Does any post-2023 online conformal method restore nominal *conditional*
coverage of realised-volatility prediction intervals through named regime
breaks, on a credible HAR base, and at what interval-width cost?

The emphasis is on conditional coverage. A method can hold marginal coverage
across a long sample while badly under-covering in the weeks after a
volatility regime change, and it is precisely those windows that matter for
risk use. The secondary half of the question is the price of any restored
coverage, measured as interval width relative to the split-conformal baseline.

## 2. Target and data

The modelling target is log realised volatility. The log transform stabilises
variance and keeps conformal residuals better behaved across the wide dynamic
range of volatility.

Realised volatility is estimated from daily open-high-low-close bars. This is a
range proxy, chosen deliberately because daily OHLC is free and freely
redistributable. It is not intraday integrated variance. The study is explicit
that the daily-OHLC range estimator is a proxy and that any conclusion is a
conclusion about that proxy unless corroborated by the frozen intraday target
below.

The Yang-Zhang estimator is the primary realised-vol measure because it is
robust to both opening jumps and drift. Garman-Klass, Parkinson and
Rogers-Satchell are computed as robustness checks; material disagreement
between estimators is reported rather than smoothed away.

Data sources:

- An equity-index panel pulled via `yfinance` (daily OHLC). This is the working
  panel for the walk-forward study.
- The legacy Oxford-Man Institute 5-minute realised-volatility panel (Heber,
  Lunde, Shephard and Sheppard, 2009) is treated as a frozen, out-of-sample
  validation target where its dates overlap the working panel. It is read only.
  It is used to check whether the daily-OHLC range proxy tracks an intraday
  benchmark over the overlap, and is never used for fitting or for tuning.

Missing-data policy:

- Backward-fill is never permitted, anywhere. Back-filling a forecasting target
  or feature injects future information and would invalidate every coverage
  claim.
- Forward-fill is allowed per feature only, and only with an explicit
  justification recorded at the call site. The default is to leave gaps and
  drop the affected windows.

## 3. Models

The base forecasters are deliberately credible and simple, because the study is
about interval calibration, not about minimising point error.

- HAR-RV (Corsi, 2009) is the primary credible base. The cascade of daily,
  weekly and monthly realised-vol averages (windows 1, 5 and 22 trading days)
  is a parsimonious long-memory proxy.
- Quantile-regression HAR is the second arm of the base. It produces
  conditional lower and upper quantiles directly, which is what the base
  conformal constructor consumes, and it gives a heteroscedastic interval
  before any online correction.
- An autoregressive / linear model is a sanity baseline only. If HAR cannot
  beat a plain AR on point error, something upstream is wrong.
- Exactly one deep probabilistic comparator is included: either DLinear or a
  quantile LSTM. A Temporal Fusion Transformer is explicitly excluded as
  over-parameterised for this sample. The deep model is a calibration stress
  test, not a candidate for best forecaster. It carries the pre-registered
  prior that it under-covers in high-volatility windows (see H4).

## 4. Conformal methods

- Conformalised Quantile Regression (CQR; Romano, Patterson and Candès, 2019)
  is the base interval constructor. It calibrates the quantile-HAR band on a
  held-out calibration set so that marginal coverage is exact under
  exchangeability. Every online method is a correction layered on top of this.
- Adaptive Conformal Inference (ACI; Gibbs and Candès, 2021) is a named
  baseline to beat. Single fixed learning rate.
- Aggregated ACI (AgACI; Zaffran and co-authors, 2022) is a named baseline to
  beat. Online aggregation of ACI experts across a grid of learning rates.
- Conformal PID control (Angelopoulos, Candès and Tibshirani, 2023) is a
  primary online correction under test.
- Dynamically-tuned ACI (DtACI; Gibbs and Candès, 2024) is the second primary
  online correction under test.
- Sequential Predictive Conformal Inference (SPCI; Xu and Xie, 2023) is the
  residual-modelling comparator, included because realised-vol residuals are
  strongly autocorrelated, the regime SPCI was designed for.

EnbPI is not a headline method in this study. It may appear only as an
auxiliary reference if space allows, never as a primary comparison.

The target nominal coverage level is **0.80 (1 - alpha = 0.80, i.e. 0.10/0.90
per side)**, with a secondary level of **0.90 (0.05/0.95 per side)** (and
therefore the per-side miscoverage used by CQR and the online methods).

## 5. Endpoints

Primary endpoint: the regime-conditional coverage gap, defined as empirical
coverage in calm periods minus empirical coverage in the 60-trading-day window
following each named break. Smaller in magnitude is better; the ideal is zero,
with both arms at the nominal level.

Secondary endpoints:

- Marginal coverage over the full test stream.
- Pinball (quantile) loss at forecast horizons 1, 5, 10 and 22 trading days.
- Probability integral transform (PIT) calibration, summarised by a
  Kolmogorov-Smirnov statistic of the PIT against the uniform.
- Post-break median interval width, reported descriptively for the raw base,
  the CQR base and the online method, and broken out for calm versus post-break
  periods. This is reporting only and is kept separate from the H3 threshold,
  which is defined on median calm width.

## 6. Regime breaks

The named break onsets are:

- Global Financial Crisis, 2008.
- Euro-area / US debt-ceiling turmoil, 2011.
- China devaluation and August volatility spike, 2015.
- The February 2018 volatility event.
- COVID-19 crash, onset March 2020.
- The 2022 rate-shock drawdown.

The precise onset dates for each are **Global Financial Crisis 2008-09-15;
2011 turmoil 2011-08-01; China devaluation 2015-08-24; February 2018 event
2018-02-05; COVID-19 2020-02-24; 2022 rate shock 2022-01-03**, fixed before
any run.

The post-break window is the 60 trading days starting at the onset date
inclusive. Calm periods are all test observations that fall in no post-break
window. The minimum separation required to treat a stretch as calm is **120 trading
days from any onset and from the end of any post-break window**.

## 7. Hypotheses

Each hypothesis is one-sided and is tested per the analysis plan in section 9.
The effect sizes below are the thresholds the author sets before committing;
they are not yet filled in so that the design cannot be reverse-engineered from
a desired result.

- H1. Split-conformal / CQR under-covers in the post-break window relative to
  nominal. Predicted direction: post-break coverage is below nominal by at
  least **5** coverage points (post-break 80% coverage at or below 0.75),
  tested one-sided by Kupiec at alpha = 0.05.
- H2. At least one online method (Conformal PID, DtACI, AgACI, ACI or SPCI)
  restores marginal coverage to within tolerance of nominal. Tolerance band:
  **+/- 4** coverage points (coverage restored to the band [0.76, 0.84], with
  Kupiec not rejected at alpha = 0.05).
- H3. The width cost of any coverage restoration is bounded relative to
  split-conformal / CQR. Predicted bound: the median interval width in calm
  periods for the online method is no more than **1.25** x the median calm
  interval width of the CQR/split base.
- H4. The deep comparator under-covers more than HAR in high-volatility
  windows. Predicted gap: deep-comparator coverage in high-vol windows below
  HAR-based coverage by at least **5** coverage points (deep-comparator
  coverage in the stressed bucket at least 5pp below HAR). The high-volatility
  window is defined by realised vol above the **80th percentile (top quintile),
  with the cut defined on the training window**.

## 8. Backtests

Coverage is tested formally, not only summarised:

- Kupiec proportion-of-failures test for the unconditional hit rate.
- Christoffersen independence test, so that hits are not allowed to cluster.
- Christoffersen joint conditional-coverage test.

Both Christoffersen components are reported, because a method can pass the
unconditional test while failing the independence test through a break, which
is the failure mode of interest. The per-test significance level is **0.05**
(for both the Kupiec and Christoffersen tests).

## 9. Multiplicity

The study runs many tests: several methods, several break windows, several
estimators and several horizons. The false-discovery rate is controlled with
Benjamini-Hochberg at 0.10. The effective number of tests is reported
alongside the nominal count, because the tests are correlated (the same methods
across overlapping windows) and the nominal count overstates the family size.

## 10. Splits

Evaluation is walk-forward. Each fold is ordered as train, embargo,
calibration, embargo, test. Two embargoes appear because conformal calibration
needs a clean separation from both the training fit and the test evaluation.

Each embargo is at least the maximum forecast horizon (22 trading days) plus
the maximum feature look-back (22 trading days for the monthly HAR term). The
exact embargo, train, calibration and test window lengths are **train 2000 /
calibration 250 / test 250 trading days, rolled forward by 250, with an embargo
of 44 trading days each side (22 horizon + 22 monthly RV look-back)**.

## 11. Reproducibility

- A single seed, `SEED = 42`, is set across NumPy, PyTorch, the Python `random`
  module and any data-loader worker seeding.
- `n_jobs = 1` everywhere, to remove scheduling nondeterminism.
- `torch.use_deterministic_algorithms(True)` is set for the deep comparator.
- Results are reported over 5 seeds (the exact set is **42, 43, 44, 45, 46**),
  giving the mean coverage and a dispersion measure (the dispersion statistics
  are **the sample standard deviation and the min-max range across seeds**).
- Residual GPU nondeterminism is documented rather than hidden: some CUDA
  kernels remain nondeterministic even under deterministic algorithms, and any
  run that touches such a kernel is flagged in the results.

## 12. Economic overlay

Any economic or risk-management overlay (for example, the cost of a coverage
miss in a value-at-risk setting) is confined to a short secondary appendix. It
is not the headline and does not feed back into model selection. The headline
result is statistical: conditional coverage restoration and its width cost.
