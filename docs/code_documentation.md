# Code documentation

This document describes the current code-bearing structure of the repository: runtime modules in `src/`, notebook-support modules in `notebooks/_lib/`, notebook/report orchestration entry points, and the regression test harness in `tests/`.

## Global conventions

- **Underlying and strike units** — `F0` and `K` are treated as USD-per-coin futures/forward prices and strikes.
- **Market fitting target** — calibration is performed against **coin-denominated** option values (`mid_price_clean`).
- **Inverse option parity** — calls are priced directly from FFT call grids; puts are derived in coin terms via inverse put-call parity and floored at zero where appropriate.
- **Time variables** — `time_to_maturity` is in years; timestamps are normalized to UTC and serialized with trailing `Z`.
- **Workbook conventions** — the main calibration workbooks carry parameter sheets plus `train_data` / `test_data`; hedging workbooks carry priced option panels, interval panels, summaries, and notes.
- **Generated artifact conventions** — chapter-3 exports live under `notebooks/chapter3_outputs/`; chapter-4 hedging exports live under `hedging_output/hedging_chapter4/`.

## Technical architecture

The current end-to-end architecture is:

1. **Snapshot collection** — `src/collect_deribit_snapshot.py` fetches option/futures data from the Deribit public API into `data/deribit_options_snapshot_*.csv`.
2. **Calibration preparation and fitting** — `src/calibration.py` filters liquid quotes, builds per-expiry pricing plans, and calibrates Black, Heston, and SVCJ.
3. **Single-snapshot orchestration** — `src/snapshot_job.py` turns one snapshot into calibrated parameter rows and repriced train/test panels.
4. **Workbook persistence and batch execution** — `src/results_store.py` and `src/batch_runner.py` manage schema, resume logic, warm starts, and ordered writes to Excel workbooks.
5. **Notebook analysis** — root-level example notebooks demonstrate pricing/calibration in isolation; maintained notebooks under `notebooks/` generate the thesis/report outputs.
6. **Futures backfill and hedging** — `src/backfill_futures_marks.py` reconstructs futures histories; `src/hedging_analysis.py` builds hedge interval panels; `src/hedging_analytics.py` summarizes and visualizes those results.
7. **Regression harness** — `tests/` locks down pricing outputs, workbook I/O, notebook structure/output semantics, and hedging invariants.

## Production/runtime code (`src/`)

### `src/__init__.py`

- Package marker for the runtime modules.
- Contains no runtime logic of its own.

### `src/inverse_fft_pricer.py`

**Role**
- Numerical pricing core for inverse options.

**Main responsibilities**
- Defines the characteristic functions `cf_black`, `cf_heston`, and `cf_svcj`.
- Implements Carr-Madan FFT pricing via `carr_madan_call_fft(...)`.
- Exposes `price_inverse_option(...)` as the main public inverse-option pricing interface.
- Provides `center_fft_params_on_strikes(...)` to center FFT grids on observed strikes.
- Includes `clear_fft_cache()` plus cached grid helpers for repeated pricing during calibration.
- Provides reference pricers `black76_call_price`, `black76_put_price`, `heston_call_price`, and `heston_put_price` for validation.

**Important types**
- `FFTParams`: controls the FFT grid (`N`, `alpha`, `eta`, `b`, `use_simpson`).

### `src/calibration.py`

**Role**
- Snapshot-level data preparation, vectorized repricing, and least-squares calibration.

**Main responsibilities**
- `filter_liquid_options(...)` validates and filters raw option rows and creates cleaned columns such as `mid_price_clean`, spreads, moneyness, and per-expiry `F0`.
- `_build_pricing_plan(...)` and `_price_with_plan(...)` create per-expiry FFT pricing plans and fill model prices efficiently.
- `price_dataframe(...)` is the public vectorized repricing entry point for a panel of quotes.
- `calibrate_model(...)` fits Black, Heston, or SVCJ with transformed parameters, weighted residuals, optional warm-start anchoring, and soft penalties.
- `clear_fft_cache()` forwards to the FFT grid cache reset so calibration workflows can control cache state.

**Important types**
- `WeightConfig`: residual-weight construction from spreads, vegas, and open interest.
- `CalibrationResult`: normalized container for fit status, parameters, messages, and diagnostics.

### `src/snapshot_job.py`

**Role**
- One-file orchestration layer between raw snapshot CSVs and workbook-ready outputs.

**Main responsibilities**
- `timestamp_from_filename(...)` / `timestamp_to_iso_z(...)` parse snapshot timestamps from the Deribit naming convention.
- `restrict_for_runtime(...)` applies optional runtime throttles for smoke runs or thesis experiments.
- `compute_errors(...)` calculates train/test summary errors.
- `process_snapshot_to_payload(...)` performs the full snapshot workflow: load CSV, filter, split train/test, calibrate all models, reprice filtered rows, and return parameter rows plus train/test tables.

### `src/results_store.py`

**Role**
- Workbook schema owner and persistence layer for calibration outputs.

**Main responsibilities**
- Defines the workbook schema via `WorkbookSchema` and the canonical sheet names.
- `init_empty_workbook(...)` creates empty sheets with the expected columns.
- `load_existing_workbook(...)` reloads an existing workbook in a schema-tolerant way.
- `append_df(...)` appends new rows while preserving canonical ordering.
- `get_latest_processed_timestamp(...)` and `latest_successful_params_before(...)` support resume logic and warm starts.
- `flush_workbook_atomic(...)` writes a workbook through a temporary file and atomic replace.

### `src/batch_runner.py`

**Role**
- Multi-snapshot, multi-currency batch coordinator.

**Main responsibilities**
- `BatchConfig` collects runtime, filtering, optimization, FFT, regularization, and threading settings.
- `list_snapshot_files(...)` discovers available snapshot CSVs.
- `_make_chunks(...)` and `_chunk_warm_start_from_workbook(...)` organize per-currency chunk execution and warm-start state.
- `run_currency_to_excel(...)` executes one currency’s pending snapshots.
- `run_all_snapshots_to_excel(...)` is the main top-level entry point for full workbook runs.

### `src/collect_deribit_snapshot.py`

**Role**
- Live data collection script for Deribit public-market data.

**Main responsibilities**
- Wraps the relevant API endpoints with `http_get(...)`, `get_instruments(...)`, `get_book_summary_options(...)`, and `get_ticker(...)`.
- Collects option metadata, quotes, Greeks, OI/volume, implied vol, maturity fields, and supporting futures information.
- `main()` implements the CLI for writing fresh snapshot CSVs under `data/`.

### `src/backfill_futures_marks.py`

**Role**
- Historical futures backfill aligned to the local snapshot set.

**Main responsibilities**
- Scans the existing snapshot universe with `list_snapshot_files(...)`, `build_snapshot_currency_combos(...)`, and `build_term_snapshot_requests(...)`.
- Downloads or reconstructs futures candles/marks in aligned time windows.
- Uses incremental merge helpers such as `filter_new_rows(...)`, `concat_dedup_sort(...)`, `build_term_incremental(...)`, and `build_perp_incremental(...)`.
- `main()` writes `term_futures_marks.csv`, `perpetual_futures_prices.csv`, and supporting combo files.

### `src/hedging_analysis.py`

**Role**
- Hedging engine that reconstructs model outputs and creates interval-level hedge panels.

**Main responsibilities**
- `HedgingConfig` defines the calibration workbook, data paths, FFT settings, rebalance frequency, currency filters, and hedging toggles.
- `load_calibration_workbook(...)` and `combine_option_sheets(...)` reload the saved calibration outputs into one option panel.
- `build_param_lookup(...)`, `list_snapshot_files(...)`, and `build_forward_snapshot_lookup(...)` align calibration rows with future evaluation snapshots.
- `load_perp_history(...)` and `load_funding_history(...)` attach hedging-market data.
- `compute_prices_and_deltas(...)` reconstructs per-model prices, regular deltas, and net deltas.
- `enrich_options_with_model_outputs(...)`, `attach_evaluation_snapshot_data(...)`, and `attach_perp_and_funding(...)` build the hedging-ready option panel.
- `build_hedge_interval_panel(...)`, `add_analysis_buckets(...)`, `summarize_panel(...)`, and `make_summary_tables_from_panel(...)` produce interval diagnostics and summary tables.
- `prepare_output_option_sheets(...)`, `write_hedge_workbook(...)`, and `run_hedging_analysis(...)` write the final hedging workbook.

### `src/hedging_analytics.py`

**Role**
- Analysis and visualization layer for the hedging workbook.

**Main responsibilities**
- `AnalysisPaths` and `resolve_default_paths(...)` find calibration workbooks, data folders, and output locations.
- `normalize_perp_history_to_seconds(...)` and `run_engine_with_normalized_perp(...)` bridge notebook workflows to the hedging engine.
- `load_output_workbook(...)`, `ensure_analysis_columns(...)`, and `make_interval_long(...)` reshape the saved workbook into analysis-friendly tables.
- `pooled_summary(...)`, `representative_panel(...)`, and `unhedged_vs_hedged_panel(...)` create pooled and representative-sample diagnostics.
- `build_equal_count_bucket_summary(...)`, `build_sample_construction(...)`, and `build_sensitivity_table(...)` support the chapter-4 bucket and robustness analysis.
- `figure_*` functions build the Plotly visuals used in the maintained hedging notebook.
- `export_figures_html(...)` and `export_tables_csv(...)` write thesis-ready exports.

## Notebook/report orchestration

### Primary example notebooks

- `pricing_examples.ipynb`
  - Lightweight pricing walkthrough for Black, Heston, and SVCJ.
  - Useful for validating pricing intuition independently of the batch/workbook machinery.

- `calibration_example.ipynb`
  - Single-snapshot calibration walkthrough.
  - Shows filtering, random train/test splitting, model calibration, and repricing in a compact example setting.

### Maintained notebooks under `notebooks/`

- `notebooks/calibrate_all_to_excel.ipynb`
  - Thin orchestration notebook that builds a `BatchConfig` and runs the full calibration batch pipeline.

- `notebooks/calibration_analysis_complete_reg_0.ipynb`
- `notebooks/calibration_analysis_complete_reg_1.ipynb`
- `notebooks/calibration_analysis_complete_reg_3.ipynb`
- `notebooks/calibration_analysis_complete_reg_5.ipynb`
- `notebooks/calibration_analysis_complete_reg_10.ipynb`
- `notebooks/calibration_analysis_complete_reg_50.ipynb`
- `notebooks/calibration_analysis_complete_reg_100.ipynb`
  - Chapter-3 calibration-analysis notebooks parameterized by regularization strength.
  - After cleanup, they share one maintained code-cell structure and differ only in notebook-specific configuration and generated outputs.

- `notebooks/calibration_results_final.ipynb`
  - Final chapter-3 summary notebook.
  - Compares regularization choices, generates the main time-series and bucket figures, and exports to `notebooks/chapter3_outputs/`.

- `notebooks/futures_plotly_viewer.ipynb`
  - Interactive viewer for perpetual and term futures histories.

- `notebooks/hedging_analytics.ipynb`
  - Final chapter-4 notebook.
  - Loads or runs the hedging engine, produces pooled metrics and bucket diagnostics, and exports to `hedging_output/hedging_chapter4/`.

### Notebook support code (`notebooks/_lib/`)

- `notebooks/_lib/common.py`
  - Shared notebook path discovery, display configuration, Plotly styling, and export helpers.

- `notebooks/_lib/chapter3_analysis.py`
  - Shared logic for the regularization analysis notebooks: workbook loading, snapshot metrics, bootstrap summaries, bucket analysis, parameter diagnostics, and report assembly.

- `notebooks/_lib/chapter3_final.py`
  - Shared logic for the final chapter-3 summary notebook: regularization sweeps, baseline metrics, bucket panels, and parameter-path plots.

- `notebooks/_lib/futures_viewer.py`
  - Loaders, plotting helpers, and diagnostics tables for the futures viewer notebook.

- `notebooks/_lib/hedging_report.py`
  - Export glue used by the hedging analytics notebook to collect named figures/tables and write them to disk.

- `notebooks/_lib/__init__.py`
  - Package marker for the notebook support layer.

### Sync utility

- `notebooks/sync_regularization_notebooks.py`
  - Source-of-truth synchronizer for the `calibration_analysis_complete_reg_*.ipynb` family.
  - Rewrites code cells from one maintained template while preserving per-notebook configuration and allowing outputs to be regenerated afterward.

## Tests and fixtures (`tests/`)

The test suite is part of the documented structure because it now enforces behavior preservation for both runtime code and notebooks.

### Package/bootstrap helpers

- `tests/__init__.py` — test package marker.
- `tests/_path.py` — ensures the project root is on `sys.path` for the test suite.

### Pricing regression and sanity fixtures

- `tests/reference_inputs.py`
  - Defines the deterministic reference case used by the pricing regression suite.
  - Owns the canonical fixture paths in `tests/fixtures/`.

- `tests/generate_reference_outputs.py`
  - Regenerates the core pricing regression fixture `tests/fixtures/reference_outputs.npz`.

- `tests/test_regression_reference_outputs.py`
  - Exact-output regression checks for `price_inverse_option(...)`, `price_dataframe(...)`, and a raw FFT grid.

- `tests/test_sanity_pricing.py`
  - Invariant checks for characteristic functions, Black/Heston pricing consistency, put-call parity, and vectorized pricing.

### Workbook and hedging invariants

- `tests/test_results_store_io.py`
  - Smoke test for workbook flush/reload semantics in `src/results_store.py`.

- `tests/test_hedging_invariants.py`
  - Locks down the exact identity `net_delta = regular_delta - price_coin`.

### Notebook fixture generation and utilities

- `tests/generate_notebook_reference_outputs.py`
  - Builds the semantic notebook-output baseline `tests/fixtures/notebook_reference_outputs.json`.

- `tests/notebook_utils.py`
  - Shared helpers for notebook execution, snapshotting, temporary copies, and artifact inspection.

### Notebook structure and regression tests

- `tests/test_notebook_integrity.py`
  - Verifies notebook metadata, markdown headers, and cell-type ordering remain stable.

- `tests/test_notebook_structure.py`
  - Ensures the regularization notebook family matches the sync template and that maintained notebooks use shared path/bootstrap logic.

- `tests/test_notebook_regression_outputs.py`
  - Compares committed notebook outputs and exported artifact trees against the semantic reference fixtures.

- `tests/test_notebook_smoke.py`
  - Executes representative notebooks with `nbclient` when the environment permits kernel launch; skips cleanly in restricted environments.

### Fixture files

- `tests/fixtures/reference_outputs.npz` — frozen pricing regression outputs.
- `tests/fixtures/notebook_reference_outputs.json` — frozen notebook semantic-output baseline.

## What is production code vs support code

- **Production/runtime code** — everything under `src/`.
- **Notebook support code** — `notebooks/_lib/*.py` and `notebooks/sync_regularization_notebooks.py`.
- **Documentation/report orchestration** — the notebooks themselves plus generated chapter output folders.
- **Tests/fixtures** — everything under `tests/` and `tests/fixtures/`.
