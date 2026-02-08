# Deribit Inverse Options — FFT Pricing & Calibration

This repository contains the core code used to price and calibrate **Deribit inverse options** (coin‑denominated calls and puts) under three models: **Black (Black–76)**, **Heston** and **SVCJ**.  These options are written on futures prices and settle in coins (BTC or ETH), which requires careful handling of the payoff and pricing convention.  The project implements a complete pipeline to ingest market snapshots, clean and filter the data, price options via a fast Fourier transform (FFT) engine, calibrate model parameters using a weighted least‑squares objective, and save results for multiple snapshots into a single workbook.

---

## Pipeline overview

For each Deribit snapshot (CSV) the pipeline performs the following steps:

1. **Ingest snapshot data** – Load a CSV containing bid/ask quotes, greeks, open interest, strikes and expiry times for all listed options, together with a snapshot of the perpetual futures price.
2. **Filter and clean** – Validate required fields, compute mid prices and relative spreads, derive a forward proxy `F₀` from the median futures price, and filter by time to maturity, moneyness band, open interest and vega.
3. **Construct FFT strike grids** – For each expiry, build a log‑strike grid and run a single **Carr–Madan FFT** to obtain a call price grid in USD.  Interpolate to the requested strikes, convert to coin units and obtain puts via inverse put–call parity.
4. **Split into training and test sets** – Shuffle the cleaned dataset deterministically and split it according to a configurable fraction.
5. **Calibrate model parameters** – Solve a weighted nonlinear least‑squares problem on the training set for each model.  Parameters are transformed to enforce positivity and correlation bounds, and soft penalties enforce the Feller condition and moment stability for models with stochastic volatility and jumps.
6. **Re‑price the entire snapshot** – Price both the training and test rows once per model using the calibrated parameters to compute consistent model prices for error metrics.
7. **Compute fit metrics and persist** – Compute RMSE and MAE on the train and test sets.  Append a row of parameters for each model and the priced option data to an Excel workbook, and periodically flush the workbook to disk.

When run over multiple snapshots and currencies, the pipeline supports **warm starts** (initialising parameters for each snapshot from the previous fit) and **resumes** from the last processed timestamp.

---

## Repository layout

```
.
├── src/
│   ├── inverse_fft_pricer.py       # Carr–Madan FFT pricer and characteristic functions
│   ├── calibration.py              # Filtering, weighting and single‑snapshot calibration
│   ├── collect_deribit_snapshot.py # Deribit API collector for options and perp snapshots
│   ├── results_store.py            # Excel workbook schema and persistence helpers
│   ├── snapshot_job.py             # Process one snapshot: filter → split → calibrate → price
│   ├── batch_runner.py             # Run calibration across snapshots with multithreading
│   └── __init__.py
├── data/
│   ├── deribit_options_snapshot_*.csv  # Option snapshots
│   └── perpetual_futures_prices.csv    # Perp snapshot
├── docs/
│   ├── code_documentation.md       # Detailed documentation of modules and functions
│   └── MASTERS_DIPLOM.pdf          # Thesis PDF (theory and derivations)
├── pricing_examples.ipynb          # Demo: price inverse calls/puts across strikes
├── calibration_example.ipynb       # Demo: filter, calibrate, price a single snapshot
├── calibrate_all_to_excel.ipynb    # Configures and runs batch calibration, writing results to Excel
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Module summaries

* **`src/inverse_fft_pricer.py`** – Implements the Carr–Madan FFT engine for European options on futures and provides characteristic functions for the Black, Heston and SVCJ models.  It exposes `price_inverse_option` to compute coin‑denominated calls and puts from model parameters.
* **`src/calibration.py`** – Contains the data‑to‑calibration pipeline for a single snapshot.  Includes functions to clean and filter option data, build per‑expiry pricing plans, compute weights, price large datasets efficiently, and solve the weighted least‑squares calibration problem with soft constraints.
* **`src/collect_deribit_snapshot.py`** – A script to collect a consistent snapshot from the Deribit public API, writing CSVs for options and perpetual futures.  The resulting files match the expected input schema of the calibration functions.
* **`src/results_store.py`** – Defines the Excel workbook schema (parameter sheets and train/test data sheets) and provides helpers to initialise, load, append to and atomically flush the workbook.  Results are appended in chronological order to enable safe resume.
* **`src/snapshot_job.py`** – Orchestrates the end‑to‑end processing of a single snapshot: loading the CSV, filtering, optional runtime throttles, splitting into train/test sets, calibrating each model sequentially (with warm starts), pricing the full dataset and assembling the payload for persistence.
* **`src/batch_runner.py`** – Runs calibration across multiple snapshots and currencies.  It enumerates snapshot files, resumes from the last processed timestamp, splits the workload across worker threads, warms up each worker with parameters from previous snapshots, commits results to the workbook in order, and flushes periodically to disk.

---

## Notebook summaries

* **`pricing_examples.ipynb`** – Demonstrates inverse call/put pricing for the Black, Heston and SVCJ models across a range of strikes and maturities.
* **`calibration_example.ipynb`** – Walks through a single snapshot: loading data, filtering and cleaning, calibrating each model, pricing train/test sets and plotting diagnostics.
* **`calibrate_all_to_excel.ipynb`** – Provides a thin front‑end for the batch pipeline.  It defines a `BatchConfig` with FFT parameters, filtering rules, weighting and worker settings, calls `run_all_snapshots_to_excel` from `src`, and displays progress and results.
