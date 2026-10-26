# Deribit Inverse Options Pricer

This repository implements the empirical pricing and hedging pipeline for Deribit inverse options used in the paper. It compares Black/Black-76, Heston, and SVCJ models, with calibration performed on coin-denominated option prices and final model evaluation performed on a deterministic holdout.

## Canonical workflow

Install the runtime dependencies first:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run the maintained notebooks in order:

1. `notebooks/00_check_data.ipynb` — inventory option snapshots and verify schema/timestamp/BTC-ETH coverage.
2. `notebooks/01_run_calibration.ipynb` — run the two fixed calibration specifications.
3. `notebooks/02_pricing_analysis.ipynb` — evaluate both specifications on the untouched test holdout.
4. `notebooks/03_run_hedging.ipynb` — run test-only hedging from the persisted calibration workbook(s).
5. `notebooks/04_final_analysis.ipynb` — assemble existing pricing and hedging artifacts; no model recomputation occurs here.

The two fixed calibration specifications are:

| Specification | Temporal penalty | Workbook |
| --- | ---: | --- |
| WLS | `lambda_TS = 0` | `outputs/canonical/calibration_wls.xlsx` |
| PWLS | `lambda_TS = 1` | `outputs/canonical/calibration_pwls.xlsx` |

There is no regularization sweep, validation-set model selection, or data-driven lambda choice in the canonical workflow. Both specifications use the same deterministic option-level **70% train / 30% test** split. Calibration uses only the 70% training subset; final pricing and canonical hedging reporting use only the 30% test subset.

`START_DT = None` and `END_DT = None` mean all available snapshots. Explicit bounds are optional and use inclusive UTC semantics.

## Canonical outputs

All generated artifacts are written under the root-level `outputs/` directory, which is gitignored. The canonical tree is:

```text
outputs/canonical/
├── calibration_wls.xlsx
├── calibration_pwls.xlsx
├── hedging_wls.xlsx
├── hedging_pwls.xlsx
├── .checkpoints/
│   ├── calibration_wls/
│   └── calibration_pwls/
├── pricing/
│   ├── overall_metrics_by_specification.csv
│   ├── snapshot_win_rates_by_specification.csv
│   ├── wls/
│   └── pwls/
├── hedging/
│   ├── wls/
│   └── pwls/
└── final/
    ├── tables/
    │   ├── pricing/
    │   └── hedging/
    ├── figures/
    │   ├── pricing/
    │   └── hedging/
    └── manifest.csv
```

Hedging runs both WLS and PWLS by default; PWLS remains the final specification. `04_final_analysis.ipynb` only collects already-generated CSV/HTML artifacts. It does not recalibrate, reprice, or rerun hedges.

## Methodological conventions

- BTC and ETH raw coin errors are never pooled into one cross-asset coin RMSE. Pooled pricing statistics use compatible USD-equivalent or spread-normalized units.
- SVCJ `rho_j` is an **unbounded jump-size loading**, not a Pearson correlation coefficient. It is not constrained to `[-1, 1]`.
- The Heston Feller condition is handled as a soft calibration penalty/diagnostic, not a hard admissibility bound.
- SVCJ martingale and Carr-Madan contour moment conditions remain explicit admissibility checks.
- Positive temporal regularization is sequential within currency: the previous successful parameter vector anchors the next snapshot in transformed optimizer coordinates.
- Regular delta and net delta use the same inverse-perpetual notional mapping. Under the short-one-option convention and local `dF/dH = 1`, `N_USD = (H_perp^2 / F_term) * delta`.
- Net delta is `regular_delta - V_coin`.
- Historical perpetual prices are a nearest-1-minute-candle open/close **proxy**, not an exchange historical mark-price series. The hedge workbook retains provenance fields.
- Hedging results are hedge-error comparisons, not trading P&L. The canonical analysis does not add transaction fees, slippage, funding P&L, or other trading frictions.
- No deterministic discounting/rate-scenario robustness layer is part of the final workflow.

## Repository structure

```text
.
├── src/                         # Pricing, calibration, collection and hedging runtime code
├── notebooks/
│   ├── 00_check_data.ipynb
│   ├── 01_run_calibration.ipynb
│   ├── 02_pricing_analysis.ipynb
│   ├── 03_run_hedging.ipynb
│   ├── 04_final_analysis.ipynb
│   └── _lib/                    # Shared notebook/report helpers
├── data/                        # Option snapshots and futures histories
├── outputs/                     # Generated canonical outputs (gitignored)
├── docs/                        # Methodology and reproducibility documentation
├── tests/                       # Numerical and pipeline regression suite
├── pricing_examples.ipynb       # Standalone pricing walkthrough
├── calibration_example.ipynb    # Standalone single-snapshot calibration walkthrough
└── requirements.txt
```

The maintained notebook surface under `notebooks/` is deliberately limited to `00`–`04` plus `_lib/`. Superseded lambda-sweep notebooks, remediation outputs, and Task13 orchestration are available through Git history rather than retained in the active tree.

## Core runtime modules

- `src/inverse_fft_pricer.py` — characteristic functions, Carr-Madan FFT pricing, inverse-option conversion, and analytical/semi-analytical references.
- `src/calibration.py` — quote filtering, residual weights, parameter transforms, admissibility diagnostics, and calibration.
- `src/snapshot_job.py` — deterministic train/test assignment, training-only calibration, and repricing for one snapshot.
- `src/results_store.py` — canonical workbook schema and atomic persistence.
- `src/batch_runner.py` — chronological multi-snapshot orchestration, warm starts, resume behavior, and sequential temporal regularization.
- `src/analysis_window.py` — optional inclusive UTC sample-window normalization.
- `src/greeks.py` and `src/hedge_mapping.py` — delta identities and inverse-perpetual hedge mapping.
- `src/hedging_analysis.py` — test-panel hedge engine and workbook persistence.
- `src/hedging_analytics.py` — hedging summaries, buckets, tail diagnostics, figures, and exports.
- `src/backfill_futures_marks.py` — term-futures histories and perpetual candle-price proxy backfill.

## Notebook support

- `notebooks/_lib/stage2_workflow.py` — data checks, fixed WLS/PWLS calibration configuration, calibration orchestration, and test-holdout pricing analysis.
- `notebooks/_lib/chapter3_final.py` — pricing metrics, common-unit errors, buckets, figures, and SVCJ parameter diagnostics. Its workbook loader is test-only.
- `notebooks/_lib/stage3_workflow.py` — canonical test-only hedging configuration/reporting and final artifact packaging.
- `notebooks/_lib/robustness.py` — parameter-identification and admissibility diagnostics; this is not a discount-rate robustness layer.
- `notebooks/_lib/common.py` — shared paths, Plotly styling, and export helpers.

## Data collection

`stage4-5-final-validation-cleanup` is the offline analysis branch and intentionally contains no GitHub Actions workflows. The authoritative data collectors and the main-to-Stage data synchronization workflow live on `main`.

Before an offline run, update or replace the local `data/` directory from `main` as needed. The Stage analysis code reads the local `data/` directory and does not download data automatically.

Historical perpetual hedge input in `data/perpetual_futures_prices.csv` is stored as `perp_futures_price_proxy`, with source, candle choice, time gap, resolution, and `price_kind=nearest_1m_candle_open_close` provenance where available.

## Verification

The regression suite uses deterministic synthetic/minimal fixtures; it does not launch a full historical calibration or hedge run.

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Important permanent checks include independent Black/Heston FFT benchmarks, deterministic splitting, calibration objective/temporal sequencing, SVCJ admissibility, common-unit pricing metrics, hedge identities/mapping, perpetual-price provenance, canonical notebook integrity, and a miniature calibration → pricing → hedging → final-package integration test.

The canonical FFT validation and its numerical evidence are documented in `docs/reproducibility/fft_numerical_validation.md`.

## Reproducibility provenance

- `docs/reproducibility/remediation_baseline.md` — historical pre-remediation baseline. It is provenance only, not an active workflow specification.
- `docs/reproducibility/thesis_sample_manifest.csv` — submitted-thesis sample manifest.
- `docs/Pricing and Hedging Deribit Inverse Options under Affine Stochastic Volatility Models with Jumps.pdf` — submitted thesis baseline.
