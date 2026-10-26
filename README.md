# Deribit Inverse Options Pricer

This repository implements pricing, calibration, and hedging workflows for **Deribit inverse options**: options quoted against crypto futures and settled in the underlying coin. The empirical workflow compares three models:

- **Black / Black-76** as the benchmark,
- **Heston** for stochastic volatility,
- **SVCJ** for stochastic volatility with correlated price/variance jumps.

The current paper workflow is deliberately narrow: validate the snapshot dataset, run two pre-specified calibration regimes, evaluate pricing on a deterministic test holdout, and then pass the resulting workbooks to the hedging/robustness stages.

## Canonical paper workflow

Run the maintained notebooks in this order:

1. `notebooks/00_check_data.ipynb` — inventory all option snapshots and fail fast on schema, timestamp, or BTC/ETH coverage problems.
2. `notebooks/01_run_calibration.ipynb` — run the two fixed calibration specifications on the complete dataset.
3. `notebooks/02_pricing_analysis.ipynb` — analyze only the untouched 30% test holdout and export pricing tables/figures.

The two fixed specifications are:

| Specification | Temporal penalty | Output workbook |
| --- | ---: | --- |
| WLS | `lambda_TS = 0` | `excel files/canonical/calibration_wls.xlsx` |
| PWLS | `lambda_TS = 1` | `excel files/canonical/calibration_pwls.xlsx` |

Both use the deterministic **70/30 train/test split** implemented in the core pipeline. There is no validation split in the canonical workflow and no regularization sweep or data-driven lambda selection. Calibration uses the 70% training subset; final pricing reporting reads the 30% test subset only.

By default, `01_run_calibration.ipynb` sets:

```python
START_DT = None
END_DT = None
```

so every snapshot currently present in `data/` is included. Explicit start/end dates remain available for bounded replication or robustness exercises, but they are not the default paper run.

## Canonical outputs

Calibration workbooks are written to:

```text
excel files/canonical/
├── calibration_wls.xlsx
└── calibration_pwls.xlsx
```

Pricing analysis is written to:

```text
notebooks/pricing_outputs/canonical/
├── overall_metrics_by_specification.csv
├── snapshot_win_rates_by_specification.csv
├── wls/
└── pwls/
```

Within each specification, the pricing report includes overall metrics, snapshot-level RMSE/MAE and within-spread diagnostics, equal-count maturity and moneyness buckets, SVCJ parameter summaries, and figures. BTC and ETH coin-denominated errors are reported separately; pooled cross-asset reporting uses unit-coherent USD-equivalent or spread-normalized errors rather than pooling raw BTC and ETH coin RMSE.

## Project structure

```text
.
├── src/                         # Pricing, calibration, data and hedging runtime code
├── notebooks/
│   ├── 00_check_data.ipynb      # Canonical data pre-flight
│   ├── 01_run_calibration.ipynb # Canonical WLS/PWLS runner
│   ├── 02_pricing_analysis.ipynb# Canonical test-holdout pricing report
│   ├── _lib/                    # Shared notebook workflow/report helpers
│   └── ...                      # Legacy/provenance and later-stage notebooks
├── data/                        # Option snapshots and futures histories
├── excel files/                 # Generated/external workbook artifacts
├── docs/                        # Methodology and reproducibility documentation
├── tests/                       # Numerical, pipeline and notebook regressions
├── pricing_examples.ipynb       # Standalone pricing walkthrough
├── calibration_example.ipynb    # Standalone single-snapshot calibration walkthrough
└── requirements.txt
```

## Core runtime modules

- `src/inverse_fft_pricer.py` — characteristic functions, Carr-Madan FFT pricing, inverse-option conversion, and reference pricing routines.
- `src/calibration.py` — liquid-quote filtering, residual scaling, parameter transforms, admissibility checks, and model calibration.
- `src/snapshot_job.py` — one-snapshot orchestration, deterministic 70/30 split, training-only calibration, and repricing of train/test panels.
- `src/results_store.py` — workbook schemas, append/load helpers, timestamp normalization, and atomic persistence.
- `src/batch_runner.py` — chronological multi-snapshot orchestration, warm starts, resume behavior, and sequential temporal regularization.
- `src/analysis_window.py` — optional inclusive UTC sample-window normalization.
- `src/hedge_mapping.py` and `src/greeks.py` — centralized hedge-notional and delta identities.
- `src/hedging_analysis.py` / `src/hedging_analytics.py` — hedging engine and reporting layer.

## Notebook support code

- `notebooks/_lib/stage2_workflow.py` — canonical Stage 2 data checks, WLS/PWLS configuration, calibration orchestration, and pricing-report assembly.
- `notebooks/_lib/common.py` — shared path resolution, display configuration, Plotly styling, and export helpers.
- `notebooks/_lib/chapter3_final.py` — pricing metrics, buckets, figures, and parameter diagnostics reused by `02_pricing_analysis.ipynb`.
- `notebooks/_lib/hedging_report.py` — export helpers for the hedging reporting stage.

The older `calibrate_all_to_excel.ipynb`, `calibration_analysis_complete_reg_*.ipynb`, `regularization_selection.ipynb`, and `calibration_results_final.ipynb` are retained for thesis/remediation provenance. They are **not** the canonical paper rerun interface.

## Setup

Create an environment and install the runtime/reporting dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To execute notebooks, install Jupyter tooling if it is not already present:

```bash
pip install jupyter ipykernel
```

Plotly is included in `requirements.txt`. Static PNG export is optional and additionally requires `kaleido`; HTML figure export works without it.

## Data and calibration details

The canonical data check expects every `deribit_options_snapshot_*.csv` to contain the calibration-required fields and both BTC and ETH rows. The calibration filters then apply the model-useability rules in the runtime pipeline, including bid/ask availability, time-to-maturity, open-interest, relative-spread, and moneyness constraints.

The weighted least-squares residual is scaled by the quoted spread. With `spread_power=1`, residuals are divided by spread, so the squared objective is effectively inverse-spread-squared. PWLS adds the fixed transformed-parameter temporal penalty with `lambda_TS = 1`; each currency is therefore processed chronologically so snapshot `t+1` uses the successful parameters from snapshot `t`.

`run_calibration_suite()` intentionally discards the large in-memory workbook object after each specification has been persisted and returns only the two output paths. This keeps the full-sample WLS→PWLS rerun memory-bounded.

## Pricing-report conventions

`02_pricing_analysis.ipynb` loads `test_data` explicitly. It does not inspect a validation set or select between WLS and PWLS using test performance; both fixed specifications are reported.

For SVCJ, `rho_j` is treated as an **unbounded jump-size loading**, not as a Pearson correlation coefficient. Canonical parameter diagnostics therefore do not impose or plot artificial ±0.95 boundaries for `rho_j`; the usual correlation boundary diagnostic is retained only for the Heston/SVCJ spot-variance correlation parameter `rho`.

## Tests

The repository contains numerical invariants, calibration-objective tests, deterministic splitting tests, workbook I/O tests, provenance checks, and notebook smoke tests. The Stage 2 notebooks are smoke-tested against temporary synthetic snapshot/workbook fixtures so CI verifies the interfaces without launching the full historical calibration.

Run the full suite with:

```bash
python -m unittest discover -s tests -v
```

## Legacy and reproducibility material

- `docs/reproducibility/remediation_baseline.md` — pre-remediation result/sample provenance.
- `docs/reproducibility/thesis_sample_manifest.csv` — submitted-thesis sample manifest.
- `docs/Pricing and Hedging Deribit Inverse Options under Affine Stochastic Volatility Models with Jumps.pdf` — submitted thesis baseline.
- `docs/superpowers/` — remediation design and implementation notes.

Large historical Excel artifacts are stored externally; see `excel files/link.txt` where applicable.

## Perpetual hedge price provenance

The historical hedge input in `data/perpetual_futures_prices.csv` is a **price proxy**, not an exchange-provided historical Deribit mark series. New backfills use `perp_futures_price_proxy`: the closer-in-time open or close of the matched Deribit TradingView candle, subject to the configured maximum-gap filter. Provenance fields record source, price kind, candle choice, time gap, and resolution. Legacy `perp_futures_mark_price` files remain readable but are explicitly labeled as legacy by the hedging loader.
