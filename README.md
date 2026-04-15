# Deribit Inverse Options Pricer

This repository studies and implements pricing, calibration, and hedging workflows for **Deribit inverse options**: options quoted against crypto futures but settled in the underlying coin rather than in USD. That settlement convention matters economically because option value, put-call parity, and hedge interpretation all live in a **coin numeraire**, not in a cash numeraire. The project therefore combines familiar option models with inverse-option pricing conventions:

- **Black / Black-76** for a liquid benchmark surface,
- **Heston** for stochastic volatility,
- **SVCJ** for stochastic volatility with correlated jumps.

The repo is built around one end-to-end problem: collect market snapshots, filter liquid quotes, calibrate the three models on inverse-option prices, persist results to Excel, analyze the fitted surfaces in notebooks, and evaluate hedging performance using perpetual futures data.

## What the project does

At a high level, the workflow is:

1. Collect Deribit option snapshots and futures marks.
2. Filter snapshots down to liquid, model-usable option quotes.
3. Price inverse options with a Carr-Madan FFT engine.
4. Calibrate Black, Heston, and SVCJ by weighted least squares.
5. Save parameters and repriced train/test panels to Excel workbooks.
6. Produce chapter-style analysis figures/tables in notebooks.
7. Reconstruct deltas, hedge intervals, and hedging diagnostics from the saved calibration outputs.

## Project structure

```text
.
├── src/                         # Runtime pricing, calibration, data, and hedging code
├── notebooks/                   # Maintained thesis/reporting notebooks
│   ├── _lib/                    # Shared notebook helper modules
│   ├── chapter3_outputs/        # Generated chapter-3 tables and Plotly HTML
│   └── *.ipynb
├── docs/                        # Code documentation and thesis materials
├── data/                        # Snapshot CSVs and futures histories
├── excel files/                 # Calibration and hedging Excel workbooks
├── hedging_output/              # Generated chapter-4 hedging exports
├── tests/                       # Regression, invariants, notebook, and I/O checks
├── pricing_examples.ipynb       # Primary pricing walkthrough
├── calibration_example.ipynb    # Primary single-snapshot calibration walkthrough
├── README.md
└── requirements.txt
```

## How the project is organized

- **Core pricing and calibration engine** — `src/inverse_fft_pricer.py`, `src/calibration.py`, `src/snapshot_job.py`, `src/results_store.py`, and `src/batch_runner.py` implement the numerical core and workbook pipeline.
- **Data collection and market backfill** — `src/collect_deribit_snapshot.py` collects live Deribit snapshots; `src/backfill_futures_marks.py` reconstructs perpetual and term futures histories aligned to the snapshot set.
- **Hedging pipeline** — `src/hedging_analysis.py` rebuilds model prices/deltas and creates interval hedging panels; `src/hedging_analytics.py` turns those outputs into chapter-ready tables and figures.
- **Maintained notebook layer** — notebooks under `notebooks/` are the main reporting/orchestration layer for chapter 3 and chapter 4 outputs.
- **Primary example notebooks** — `pricing_examples.ipynb` and `calibration_example.ipynb` are standalone walkthroughs for pricing and single-snapshot calibration.
- **Documentation and generated artifacts** — `docs/` contains technical documentation and thesis materials, while `notebooks/chapter3_outputs/` and `hedging_output/hedging_chapter4/` contain generated reporting artifacts.

## Runtime module guide (`src/`)

- `src/__init__.py` — package marker for the runtime code.
- `src/inverse_fft_pricer.py` — characteristic functions, Carr-Madan FFT grids, inverse-option call/put pricing, cache helpers, and reference Black/Heston pricing routines.
- `src/calibration.py` — liquid-quote filtering, residual weighting, per-expiry pricing plans, vectorized repricing, parameter transforms, and model calibration.
- `src/snapshot_job.py` — one-snapshot orchestration: load CSV, filter data, split train/test, calibrate all models, price the filtered panel, and assemble Excel-ready payloads.
- `src/results_store.py` — workbook schema definitions, append/load helpers, timestamp normalization, latest-success lookup, and atomic workbook flushes.
- `src/batch_runner.py` — multi-snapshot, multi-currency orchestration with resume logic, warm starts, chunking, and ordered workbook commits.
- `src/collect_deribit_snapshot.py` — CLI collector for Deribit option snapshots and associated market data.
- `src/backfill_futures_marks.py` — backfill utility for perpetual and term futures marks aligned to the existing snapshot set.
- `src/hedging_analysis.py` — hedging engine that reloads calibration outputs, reconstructs prices/deltas, joins evaluation snapshots and perp data, and writes hedging workbooks.
- `src/hedging_analytics.py` — analysis/figure layer for hedging results, including pooled summaries, equal-count buckets, and Plotly export helpers.

## Notebook and script guide

### Primary example notebooks

- `pricing_examples.ipynb` — compact demonstrations of inverse-option pricing under Black, Heston, and SVCJ.
- `calibration_example.ipynb` — a single-snapshot calibration walkthrough showing filtering, train/test splitting, calibration, and repricing.

### Maintained notebooks under `notebooks/`

- `notebooks/calibrate_all_to_excel.ipynb` — short orchestration notebook for running the batch calibration pipeline to an Excel workbook.
- `notebooks/calibration_analysis_complete_reg_0.ipynb`
- `notebooks/calibration_analysis_complete_reg_1.ipynb`
- `notebooks/calibration_analysis_complete_reg_3.ipynb`
- `notebooks/calibration_analysis_complete_reg_5.ipynb`
- `notebooks/calibration_analysis_complete_reg_10.ipynb`
- `notebooks/calibration_analysis_complete_reg_50.ipynb`
- `notebooks/calibration_analysis_complete_reg_100.ipynb` — chapter-3 calibration analysis notebooks for different regularization strengths.
- `notebooks/calibration_results_final.ipynb` — compact final chapter-3 reporting notebook that compares regularization choices and exports the main calibration figures/tables.
- `notebooks/futures_plotly_viewer.ipynb` — interactive viewer for perpetual and term futures histories.
- `notebooks/hedging_analytics.ipynb` — chapter-4 reporting notebook for pooled hedging metrics, time-series diagnostics, bucket analyses, and exports.

### Notebook support code

- `notebooks/sync_regularization_notebooks.py` — keeps the regularization notebook family synchronized to one maintained code-cell template.
- `notebooks/_lib/common.py` — shared path resolution, Plotly styling, and export helpers.
- `notebooks/_lib/chapter3_analysis.py` — shared logic used by the regularization analysis notebooks.
- `notebooks/_lib/chapter3_final.py` — shared logic used by the final chapter-3 notebook.
- `notebooks/_lib/futures_viewer.py` — loader, plotting, and diagnostics helpers for the futures viewer notebook.
- `notebooks/_lib/hedging_report.py` — export glue for the hedging analytics notebook.

## Setup

Install the core pipeline dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Current notebook/reporting workflows also assume a Jupyter + Plotly environment. Optional static image export from Plotly figures additionally requires `kaleido`.

## Typical workflows

### 1) Collect or refresh market data

```bash
python -m src.collect_deribit_snapshot --currency BOTH --outdir data
```

For futures-history reconstruction aligned to the snapshot set:

```bash
python -m src.backfill_futures_marks --data-dir data
```

### 2) Run calibration

- For a single snapshot with full control, use `src.snapshot_job.process_snapshot_to_payload(...)`.
- For a full batch run to Excel, use `src.batch_runner.run_all_snapshots_to_excel(...)` or the notebook `notebooks/calibrate_all_to_excel.ipynb`.

The main calibration outputs are Excel workbooks such as:

- `excel files/calibration_results_reg_0.xlsx`
- `excel files/calibration_results_reg_1.xlsx`
- `excel files/calibration_results_reg_3.xlsx`
- `excel files/calibration_results_reg_5.xlsx`
- `excel files/calibration_results_reg_10.xlsx`
- `excel files/calibration_results_reg_25.xlsx`
- `excel files/calibration_results_reg_50.xlsx`
- `excel files/calibration_results_reg_100.xlsx`
- `excel files/calibration_results_reg_250.xlsx`
- `excel files/calibration_results_reg_500.xlsx`
- `excel files/calibration_results_reg_1000.xlsx`

### 3) Analyze calibration outputs

Use the chapter-3 notebooks in `notebooks/` to inspect fit quality, regularization trade-offs, bucket diagnostics, and parameter stability. Generated artifacts are written to `notebooks/chapter3_outputs/`.

### 4) Run and analyze hedging outputs

- Use `src.hedging_analysis.run_hedging_analysis(...)` for the hedging engine.
- Use `notebooks/hedging_analytics.ipynb` for the chapter-4 tables/figures and export flow.

Generated hedging artifacts are written to `hedging_output/hedging_chapter4/`.

## Documentation

- `docs/code_documentation.md` — technical documentation for runtime modules, notebook support code, orchestration scripts, and tests.
- `docs/main.tex` and `docs/MASTERS_DIPLOM.pdf` — thesis materials and longer theoretical context.
