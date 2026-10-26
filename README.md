# Deribit Inverse Options Pricer

This repository implements pricing, calibration, and hedging workflows for **Deribit inverse options**: options quoted against crypto futures and settled in the underlying coin. The empirical workflow compares three models:

- **Black / Black-76** as the benchmark,
- **Heston** for stochastic volatility,
- **SVCJ** for stochastic volatility with correlated price/variance jumps.

The current paper workflow is deliberately narrow: validate the snapshot dataset, run two pre-specified calibration regimes, evaluate pricing on a deterministic test holdout, run test-only hedging from the persisted calibration workbooks, and assemble the generated tables/figures into one final output tree.

## Canonical paper workflow

Run the maintained notebooks in this order:

1. `notebooks/00_check_data.ipynb` — inventory all option snapshots and fail fast on schema, timestamp, or BTC/ETH coverage problems.
2. `notebooks/01_run_calibration.ipynb` — run the two fixed calibration specifications on the complete dataset.
3. `notebooks/02_pricing_analysis.ipynb` — analyze only the untouched 30% test holdout and export pricing tables/figures.
4. `notebooks/03_run_hedging.ipynb` — run test-only hedging from the persisted calibration workbooks and export hedging diagnostics.
5. `notebooks/04_final_analysis.ipynb` — collect existing pricing/hedging CSV and HTML artifacts into the canonical final package; it performs no model recomputation.

The two fixed calibration specifications are:

| Specification | Temporal penalty | Output workbook |
| --- | ---: | --- |
| WLS | `lambda_TS = 0` | `excel files/canonical/calibration_wls.xlsx` |
| PWLS | `lambda_TS = 1` | `excel files/canonical/calibration_pwls.xlsx` |

Both use the deterministic **70/30 train/test split** implemented in the core pipeline. There is no validation split in the canonical workflow and no regularization sweep or data-driven lambda selection. Calibration uses the 70% training subset; final pricing and canonical hedging reporting use the 30% test subset only.

By default the long-running calibration and hedging notebooks set:

```python
START_DT = None
END_DT = None
```

so every snapshot currently present in `data/` is included. Explicit start/end dates remain available for bounded runs. Calibration and hedging use the same inclusive UTC analysis-window semantics; hedging filters the snapshot index before constructing forward pairs, so a bounded run cannot hedge into a snapshot outside its configured window.

## Canonical outputs

Calibration workbooks:

```text
excel files/canonical/
├── calibration_wls.xlsx
└── calibration_pwls.xlsx
```

Pricing analysis:

```text
notebooks/pricing_outputs/canonical/
├── overall_metrics_by_specification.csv
├── snapshot_win_rates_by_specification.csv
├── wls/
└── pwls/
```

Canonical hedging defaults to PWLS; WLS hedging can be enabled for direct benchmark comparison:

```text
excel files/canonical/
├── hedging_pwls.xlsx
└── hedging_wls.xlsx        # optional

notebooks/hedging_outputs/canonical/
├── pwls/
└── wls/                    # optional
```

The final packaging notebook creates:

```text
notebooks/final_outputs/canonical/
├── tables/
│   ├── pricing/
│   └── hedging/
├── figures/
│   ├── pricing/
│   └── hedging/
└── manifest.csv
```

`04_final_analysis.ipynb` only copies already-generated CSV/HTML artifacts into this stable tree. It does not recalibrate, reprice, or rerun hedges.

Within each pricing specification, the report includes overall metrics, snapshot-level RMSE/MAE and within-spread diagnostics, equal-count maturity and moneyness buckets, SVCJ parameter summaries, and figures. BTC and ETH coin-denominated errors are reported separately; pooled cross-asset reporting uses unit-coherent USD-equivalent or spread-normalized errors rather than pooling raw BTC and ETH coin RMSE.

The hedging report contains sample-construction diagnostics, pooled test-sample hedge metrics, timestamp summaries, equal-count maturity/moneyness/basis diagnostics, tail-risk summaries, and HTML figures. The final PWLS hedge workbook is `hedging_pwls.xlsx`; WLS hedging is optional rather than required for the canonical run.

## Project structure

```text
.
├── src/                         # Pricing, calibration, data and hedging runtime code
├── notebooks/
│   ├── 00_check_data.ipynb      # Canonical data pre-flight
│   ├── 01_run_calibration.ipynb # Canonical WLS/PWLS runner
│   ├── 02_pricing_analysis.ipynb# Canonical test-holdout pricing report
│   ├── 03_run_hedging.ipynb     # Canonical test-only hedging runner/report
│   ├── 04_final_analysis.ipynb  # Final artifact packager; no recomputation
│   ├── _lib/                    # Shared notebook workflow/report helpers
│   └── ...                      # Legacy/provenance notebooks
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
- `src/hedging_analysis.py` — test-panel hedging engine, optional analysis-window filtering, forward snapshot alignment, perpetual proxy joins, delta reconstruction, and workbook persistence.
- `src/hedging_analytics.py` — reusable hedging summaries, equal-count buckets, tail diagnostics, figures, and exports.

## Notebook support code

- `notebooks/_lib/stage2_workflow.py` — canonical data checks, WLS/PWLS calibration configuration, calibration orchestration, and pricing-report assembly.
- `notebooks/_lib/stage3_workflow.py` — canonical test-only hedging configuration/reporting and final artifact packaging.
- `notebooks/_lib/common.py` — shared path resolution, display configuration, Plotly styling, and export helpers.
- `notebooks/_lib/chapter3_final.py` — pricing metrics, buckets, figures, and parameter diagnostics reused by `02_pricing_analysis.ipynb`.
- `notebooks/_lib/hedging_report.py` — legacy export glue retained for the historical hedging notebook.

The older `calibrate_all_to_excel.ipynb`, `calibration_analysis_complete_reg_*.ipynb`, `regularization_selection.ipynb`, `calibration_results_final.ipynb`, and `hedging_analytics.ipynb` are retained for thesis/remediation provenance until final repository cleanup. They are **not** the canonical paper rerun interface.

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

## Calibration conventions

The canonical data check expects every `deribit_options_snapshot_*.csv` to contain the calibration-required fields and both BTC and ETH rows. The calibration filters then apply the model-useability rules in the runtime pipeline, including bid/ask availability, time-to-maturity, open-interest, relative-spread, and moneyness constraints.

The weighted least-squares residual is scaled by the quoted spread. With `spread_power=1`, residuals are divided by spread, so the squared objective is effectively inverse-spread-squared. PWLS adds the fixed transformed-parameter temporal penalty with `lambda_TS = 1`; each currency is therefore processed chronologically so snapshot `t+1` uses the successful parameters from snapshot `t`.

`run_calibration_suite()` intentionally discards the large in-memory workbook object after each specification has been persisted and returns only output paths. This keeps the full-sample WLS→PWLS rerun memory-bounded.

## Pricing-report conventions

`02_pricing_analysis.ipynb` loads `test_data` explicitly. It does not inspect a validation set or select between WLS and PWLS using test performance; both fixed specifications are reported.

For SVCJ, `rho_j` is treated as an **unbounded jump-size loading**, not as a Pearson correlation coefficient. Canonical parameter diagnostics therefore do not impose or plot artificial ±0.95 boundaries for `rho_j`; the usual correlation boundary diagnostic is retained only for the Heston/SVCJ spot-variance correlation parameter `rho`.

## Hedging conventions

`03_run_hedging.ipynb` defaults to the fixed PWLS specification and `evaluation_splits=("test",)`. WLS hedging is optional. The hedge engine reconstructs regular and net deltas and applies the same term/perpetual notional map to both hedge definitions. For a short option and local `dF/dH = 1`, the USD perpetual notional is proportional to

```text
(H_perp^2 / F_term) * delta
```

The historical perpetual input is a candle-derived price proxy, not a historical exchange mark series. Provenance fields are retained in the hedge panel/workbook.

When explicit analysis dates are supplied, the engine filters both calibration-option observations and available snapshot CSVs to that window **before** building the forward-snapshot lookup. Therefore the terminal observation of a bounded run cannot pair with a later snapshot outside the window.

## Tests

The repository contains numerical invariants, calibration-objective tests, deterministic splitting tests, workbook I/O tests, provenance checks, hedging-window tests, and notebook-interface tests. Tests use small synthetic fixtures; they do not launch a full historical calibration or hedging rerun.

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
