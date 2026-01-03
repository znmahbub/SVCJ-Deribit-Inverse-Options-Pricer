# Deribit Inverse Options — FFT Pricing & Calibration

This repository contains the core research code behind a thesis project on **pricing and calibrating Deribit-style inverse options** under **affine stochastic volatility models with jumps (SVCJ)**.

The code is organized around a simple pipeline:

1. **Ingest Deribit option snapshot data** (either collected via the included API script or loaded from CSV snapshots).
2. **Filter / clean** the cross-section down to “reasonably liquid” quotes.
3. **Price inverse options in coin units** using a **Carr–Madan FFT** engine (one FFT produces a full strike grid for a given expiry).
4. **Calibrate model parameters** (Black, Heston, SVCJ) by solving a **weighted nonlinear least-squares** problem on coin-denominated prices.

---

## Repository layout

- `src/`
  - `inverse_fft_pricer.py` — Carr–Madan FFT pricer + characteristic functions + (semi-)analytical Black and Heston prices for testing
  - `calibration.py` — filtering/cleaning + fast batch pricing + weighted least-squares calibration
  - `collect_deribit_snapshot.py` — Deribit public-API snapshot collector (options + perp futures snapshot)
- `docs/`
  - `MASTERS_DIPLOM.pdf` — the thesis PDF (context + methodology)
- `data/`
  - `deribit_options_snapshot_*.csv` — timestamped option cross-sections (bid/ask/mid/greeks/OI, etc.)
  - `perpetual_futures_prices.csv` — perp mark/bid/ask snapshots

Notebooks:
- `pricing_examples.ipynb` - minimal pricing example using the Black, Heston and SVCJ models
- `calibration_example.ipynb` - minimal calibration example
- `calibrate_all.ipynb` - calibration over each of the collected Deribit snapshots in the `data` folder

---

## Core module: `src/inverse_fft_pricer.py`

### What it implements
A **fast Fourier transform (FFT) engine** for **European option pricing on a futures price** under:

- **Black (Black–76, r = 0)** with constant volatility
- **Heston** stochastic volatility
- **SVCJ** stochastic volatility with **correlated jumps** in returns and variance

The implementation follows the **Carr–Madan (1999)** approach:

- Build a frequency grid `v_j = j * eta`
- Evaluate the model characteristic function at a shifted argument `v - i(α+1)` (exponential damping)
- Apply quadrature weights (trapezoid or Simpson)
- FFT to obtain a **grid of (strike, call-price)** pairs in USD terms

### Inverse-option convention (coin-denominated pricing)
Deribit inverse options are quoted in **coin units**. The pricer therefore:
- Computes a **USD call price grid** on strikes
- Interpolates to the requested strike(s)
- Converts to **coin prices** via a simple scaling by the current futures price `F0`
- Produces inverse puts via **inverse put–call parity** inside the main wrapper

### Numerical / implementation details
- **Quadrature choices**: trapezoidal or Simpson weights (Simpson requires even `N`)
- **Caching**:
  - Gauss–Legendre nodes/weights are cached for repeated quadratures
  - FFT call-price grids can be cached by `(model, T, F0, params, FFT grid)` — useful for repeated pricing at fixed parameters, but typically disabled during calibration (low hit rate)
- **Heston branch handling**: enforces a stable branch for the complex square root discriminant (mitigates discontinuities)
- **SVCJ jump component**: evaluates an integral term via **Gauss–Legendre quadrature**; includes basic domain checks to avoid singularities in the jump MGF

### Built-in “reference” pricers (for sanity checks)
To validate the FFT logic and characteristic functions:
- **Analytic Black–76 call/put** (`black76_call_price`, `black76_put_price`)
- **Semi-analytical Heston call/put** via Gil–Pelaez inversion (`heston_call_price`, `heston_put_price`)
  - The Heston *put* reference is computed **directly** (not via put–call parity), specifically for testing correctness.

---

## Core module: `src/calibration.py`

### What it implements
A complete **data → calibration** pipeline around the FFT pricer.

#### 1) Filtering and cleaning
`filter_liquid_options(df, ...)` enforces:
- Required fields present (quotes, expiry, forward proxy inputs, greeks/OI, etc.)
- Positive time-to-maturity, bid/ask sanity (`ask >= bid`), finite numerics
- Optional screens:
  - max relative spread
  - moneyness band (strike relative to a per-expiry forward proxy)
  - minimum open interest / vega thresholds
  - optional removal of synthetic underlyings

It also constructs:
- `mid_price_clean = (bid + ask)/2`
- `spread`, `rel_spread`
- A per-expiry forward proxy `F0` (median futures price within expiry bucket)
- `moneyness = K / F0` and `log_moneyness`

#### 2) Fast batch pricing across a dataset
To price many options efficiently, the module:
- Builds a **per-expiry pricing plan**
- For each expiry:
  - runs **one FFT call** to price calls across strikes
  - obtains puts via inverse put–call parity *within that expiry bucket*  
This reduces calibration cost significantly because one FFT covers a full strike slice.

It also supports **dynamic log-strike centering**:
- Adjusts the FFT grid location (`b`) around the median strike of an expiry bucket to keep the strike grid “centered” on the data (improves interpolation quality and reduces the need for huge grids).

#### 3) Weighted least-squares calibration
`calibrate_model(df, model, ...)` solves:

- Objective: minimize weighted residuals  
  `r_i = w_i * (P_model_coin(i) - P_market_coin(i))`

- Weight model (`WeightConfig`) can combine:
  - spread-based downweighting (tighter markets matter more)
  - vega-based scaling
  - open-interest scaling
  Each component has a power and stabilization epsilons; weights can be capped.

- Parameter constraints are enforced through **reparameterization**:
  - positive parameters via `log`
  - correlations via `arctanh/tanh` mapping into `(-1, 1)`
- Additional model constraints:
  - **Feller-type** constraint penalty for Heston/SVCJ
  - **moment/existence** constraint penalty for SVCJ jump term stability

- Uses `scipy.optimize.least_squares` with bounded variables and robust fallback penalties when pricing fails.

The module returns a compact `CalibrationResult` including parameter estimates and fit metrics (RMSE/MAE in coin units).

---

## Data collection: `src/collect_deribit_snapshot.py`

This script pulls a cross-sectional snapshot from the **Deribit public API**, assembling:

- Instrument metadata (strike, expiry, option type)
- Book summary by currency (bid/ask/mid, open interest, volume, mark IV)
- Per-instrument ticker calls (for Greeks like delta/vega)
- A separate small snapshot of **perpetual futures** (mark/bid/ask)

It writes:
- `data/deribit_options_snapshot_<timestamp>.csv`
- `data/perpetual_futures_prices.csv`

The output is designed to match the fields expected by `filter_liquid_options()` and downstream calibration.

---

## Notebooks (what they contain)

- `pricing_examples.ipynb`  
  Demonstrates inverse call/put pricing across strikes for Black, Heston, and SVCJ using the FFT pricer.

- `calibration_example.ipynb`  
  Walks through a single-snapshot workflow: load → filter → calibrate models → price train/test sets → produce diagnostic plots (true vs model).

- `calibrate_all.ipynb`  
  Batch-runs calibration across multiple stored snapshots for BTC and ETH, collecting:
  - time series of fitted parameters (by model)
  - time series of fit metrics  
  Includes basic parallelization patterns to speed up multi-snapshot processing.
