# Code documentation

This document describes the active code-bearing structure of the repository after the empirical-pipeline cleanup. The maintained workflow is intentionally narrow: five canonical notebooks, shared helpers under `notebooks/_lib/`, scientific/runtime code under `src/`, and a compact deterministic regression suite under `tests/`.

## Global conventions

- `F0` and `K` are USD-per-coin futures/forward prices and strikes.
- Calibration targets coin-denominated option values (`mid_price_clean`).
- Snapshot timestamps are UTC; `time_to_maturity` is measured in years.
- The canonical empirical split is deterministic option-level 70% train / 30% test. There is no validation split in the active workflow.
- WLS uses `lambda_TS = 0`; PWLS uses `lambda_TS = 1`. No regularization sweep or lambda selection is performed.
- BTC and ETH raw coin errors are reported separately. Pooled pricing metrics use compatible USD-equivalent or spread-normalized units.
- SVCJ `rho_j` is an unbounded jump-size loading. Heston/SVCJ spot-variance `rho` remains a correlation parameter.
- The Feller condition is a soft penalty/diagnostic. SVCJ martingale and Carr-Madan contour moment conditions are explicit admissibility checks.
- Canonical hedging evaluates hedge errors on the test panel. It is not a trading-P&L simulation and does not add transaction fees, slippage, funding P&L, or deterministic discount-rate scenarios.

## Canonical notebook pipeline

The only maintained notebooks under `notebooks/` are:

1. `00_check_data.ipynb` — inventory snapshots and validate schema/timestamp/currency coverage.
2. `01_run_calibration.ipynb` — generate the fixed WLS and PWLS calibration workbooks.
3. `02_pricing_analysis.ipynb` — evaluate both fixed specifications on `test_data` only.
4. `03_run_hedging.ipynb` — run canonical test-only hedging, PWLS by default and WLS optionally.
5. `04_final_analysis.ipynb` — assemble existing pricing/hedging CSV and HTML artifacts without recalibrating, repricing, or rerunning hedges.

The notebooks are thin runners. Substantive orchestration/report logic belongs in `notebooks/_lib/`; pricing, calibration, data, and hedging logic belongs in `src/`.

## Runtime/scientific modules (`src/`)

### `inverse_fft_pricer.py`

Carr-Madan FFT pricing and characteristic functions for Black, Heston, and SVCJ, inverse-option conversion, caching, FFT-grid utilities, and analytical/semi-analytical reference pricers. The canonical numerical configuration is independently regression-tested at `N=4096`, `eta=0.10`, `alpha=1.5`, `use_simpson=True`.

### `calibration.py`

Liquid-quote filtering, residual scaling, transformed parameterizations, vectorized pricing plans, least-squares calibration, temporal regularization, Feller diagnostics, and SVCJ moment admissibility. Temporal anchoring is applied in packed/transformed optimizer coordinates. `rho_j` is not bounded to `[-1,1]`.

### `splitting.py`

Deterministic option-level cross-sectional train/test assignment. The canonical split is 70/30 and does not create a validation subset.

### `snapshot_job.py`

One-snapshot orchestration: load/filter, assign the deterministic split, calibrate all models on training rows only, reprice the full eligible snapshot, and return workbook-ready parameter/train/test frames.

### `results_store.py`

Owns the canonical calibration workbook schema, append/load helpers, timestamp normalization, resume helpers, and atomic Excel persistence. Canonical option-data sheets are `train_data` and `test_data`.

### `batch_runner.py`

Chronological multi-snapshot/multi-currency orchestration. WLS can execute independently; positive temporal regularization is sequential within currency so each successful snapshot parameter vector can anchor the next snapshot.

### `analysis_window.py`

Normalizes optional inclusive UTC start/end bounds. `None` bounds preserve the full available sample.

### `greeks.py` and `hedge_mapping.py`

Centralize inverse-option delta definitions and the term-futures-to-inverse-perpetual notional conversion. For the current short-one-option convention and local `dF_term/dH_perp = 1`, both regular and net delta use `N_USD = (H_perp^2 / F_term) * delta`. Net delta is `regular_delta - V_coin`.

### `hedging_analysis.py`

Canonical hedge engine: loads persisted calibration outputs, defaults to the test split, reconstructs prices/deltas, restricts forward snapshot pairing to the configured analysis window, attaches perpetual/funding histories, builds interval panels, and writes hedging workbooks.

### `hedging_analytics.py`

Reusable hedge-error summaries, equal-count bucket diagnostics, tail summaries, tables, figures, and exports.

### `collect_deribit_snapshot.py`

Live Deribit option/futures snapshot collection used by `.github/workflows/collect_deribit_snapshot.yml`.

### `backfill_futures_marks.py`

Historical term-futures and perpetual-price-proxy backfill. The perpetual series is a nearest-1-minute-candle open/close proxy rather than an exchange historical mark series; provenance columns are retained for downstream hedging analysis.

## Notebook support (`notebooks/_lib/`)

### `stage2_workflow.py`

Canonical Stage 2 orchestration: snapshot pre-flight, fixed WLS/PWLS configuration, calibration-workbook generation, and test-holdout pricing report assembly. `START_DT=None` and `END_DT=None` imply all available snapshots.

### `chapter3_final.py`

Pricing-report helpers: test-workbook loading, common-unit error construction, overall/snapshot/bucket metrics, SVCJ parameter summaries, and Plotly figures. The active loader is hard-wired to `test_data`; regularization-sweep and validation-selection APIs have been removed.

### `stage3_workflow.py`

Canonical Stage 3 orchestration: test-only hedging configuration, hedging reporting, and final artifact packaging. `04_final_analysis.ipynb` calls only packaging functions from this layer.

### `robustness.py`

Parameter-identification and admissibility diagnostics. This module does not implement deterministic discounting/rate-scenario robustness. In particular, `rho_j` diagnostics treat it as an unbounded loading rather than a correlation-boundary parameter.

### `common.py`

Shared project-path resolution, display settings, Plotly styling, and export helpers.

Other `_lib` modules are narrow reporting/data helpers used by the canonical pipeline; none reintroduce model-selection or discount-rate scenario logic.

## Generated outputs

Canonical calibration workbooks are under `excel files/canonical/`. Pricing reports are under `notebooks/pricing_outputs/canonical/`; hedging reports are under `notebooks/hedging_outputs/canonical/`; the final package is assembled under `notebooks/final_outputs/canonical/`.

Generated pricing and hedging outputs are not source-of-truth model inputs. `04_final_analysis.ipynb` only copies existing report artifacts into a stable final tree.

## Tests (`tests/`)

The permanent suite is compact and behavior-focused:

- `test_fft_numerical_validation.py` — independent Black-76 and converged Heston checks at the canonical FFT grid.
- `test_sanity_pricing.py` / `test_regression_reference_outputs.py` — pricing invariants, parity, SVCJ→Heston limit, and a small deterministic reference fixture.
- `test_snapshot_splitting.py` — deterministic 70/30 assignment, no dropped rows, training-only calibration, and train/test workbook schema.
- `test_calibration_objective.py` / `test_batch_regularization_sequence.py` — residual weighting, transformed-coordinate regularization, soft constraints, chronological anchoring, and resume equivalence.
- `test_svcj_admissibility.py` — unbounded `rho_j` and moment-domain checks.
- `test_chapter3_metrics.py` — explicit units and prohibition of pooled raw BTC+ETH coin RMSE.
- `test_measure_delta_identities.py`, `test_hedge_mapping.py`, `test_hedging_invariants.py` — regular/net delta identities and the common `H^2/F` notional map.
- `test_perp_price_provenance.py` / Stage 3 window tests — perpetual proxy labeling and forward-snapshot bounds.
- `test_canonical_notebooks.py` — clean JSON, unique cell IDs, no stored execution/output state, no superseded selection/discounting machinery, and artifact-only final notebook semantics.
- `test_mini_pipeline.py` — temporary synthetic calibration → persisted workbook → pricing → test-only hedge → report → final package integration smoke.

The suite is designed to finish quickly and does not run the full historical calibration/pricing/hedging job.

## Automation

Maintained GitHub workflows are:

- `.github/workflows/ci.yml` — clean Python 3.11 canonical verification.
- `.github/workflows/collect_deribit_snapshot.yml` — option snapshot collection.
- `.github/workflows/update-futures-history.yml` — futures/proxy-history update.

Task13/remediation workflows and scripts are not part of the active tree.

## Historical provenance

Git history contains the superseded regularization-sweep notebooks, selection code, Task13 automation, and remediation artifacts. The active tree retains only provenance that remains useful to reproduce or interpret the submitted thesis baseline:

- `docs/reproducibility/remediation_baseline.md`
- `docs/reproducibility/thesis_sample_manifest.csv`
- the submitted thesis PDF under `docs/`

Those historical files are not active methodological specifications. The current workflow is defined by the canonical notebooks, runtime code, tests, and README.
