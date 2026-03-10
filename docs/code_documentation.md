# Code documentation

This document describes the implementation in `src/` and how modules interact in the end-to-end calibration workflow.

## Global conventions

- **Underlying and strike units**: `F0` and `K` are in USD per coin.
- **Market target prices**: calibration fits **coin-denominated** option prices (`mid_price_clean`).
- **Inverse put-call parity used by the pricer**:
  - Calls are priced directly in coin units from FFT call grids.
  - Puts are computed as `P_coin = C_coin - (1 - K/F0)` and floored at 0.
- **Time to maturity**: `time_to_maturity` is in years.

---

## Module: `src/inverse_fft_pricer.py`

### Responsibilities

- Defines model characteristic functions:
  - `cf_black`
  - `cf_heston`
  - `cf_svcj`
- Implements Carr-Madan FFT call pricing:
  - `carr_madan_call_fft(...)` returns `(K_grid, C_grid_usd)`.
- Provides a user-facing inverse option pricer:
  - `price_inverse_option(...)`.
- Includes reference closed/semi-closed pricers used for validation:
  - `black76_call_price`, `black76_put_price`
  - `heston_call_price`, `heston_put_price`

### FFT parameterization

`FFTParams` controls the grid:

- `N`: number of frequency/strike grid points.
- `alpha`: Carr-Madan damping coefficient.
- `eta`: frequency spacing.
- `b`: log-strike shift.
- `use_simpson`: Simpson vs trapezoidal quadrature weights.

### Caching

The module memoizes computed pricing grids using `_cached_pricing_grid(...)` so repeated evaluations for identical `(model, T, F0, params, fft_params)` avoid recomputing FFTs.

---

## Module: `src/calibration.py`

### Responsibilities

This module handles single-snapshot preparation and model fitting:

1. Filter and clean option rows (`filter_liquid_options`).
2. Build per-expiry pricing plans (`_build_pricing_plan`).
3. Vectorized pricing over a DataFrame (`price_dataframe`).
4. Weighted least-squares calibration with constraints (`calibrate_model`).

### Key data classes

- `WeightConfig`: controls residual weighting from spread/vega/open-interest.
- `CalibrationResult`: normalized fit result container.

### Filtering (`filter_liquid_options`)

Required columns are validated up front:

- `currency`, `option_type`, `strike`, `time_to_maturity`,
- `bid_price`, `ask_price`, `futures_price`, `vega`,
- `open_interest`, `expiry_datetime`.

Main transformations/screens:

- optional currency filter
- numeric coercion for key quote fields
- maturity bounds
- bid/ask validity checks (`ask >= bid`, both positive when required)
- vega/open-interest thresholds
- optional synthetic-underlying exclusion
- creation of:
  - `mid_price_clean`
  - `spread`
  - `rel_spread`
  - per-expiry median `F0`
  - `moneyness = strike / F0`
  - `log_moneyness`
- option type normalization to `call|put`

### Pricing plan and vectorized pricing

`_build_pricing_plan(...)` groups rows by expiry and creates tuples containing:

- positional row indices,
- strike arrays,
- put masks,
- per-group `T`, `F0`, and `FFTParams`.

`_price_with_plan(...)` then:

- prices calls once per expiry,
- fills output arrays in-place,
- derives puts via inverse parity where needed.

`price_dataframe(...)` is the public vectorized wrapper and returns a NumPy array of coin prices.

### Calibration (`calibrate_model`)

The solver uses `scipy.optimize.least_squares` with transformed parameters:

- positive parameters in log-space,
- correlations in `arctanh` space.

Residual vector is stacked from:

1. weighted pricing residuals, `w_i * (P_model - P_market)`,
2. optional soft-constraint penalties:
   - Heston/SVCJ Feller-type condition,
   - SVCJ moment-stability condition,
3. optional warm-start anchor term
   - active when `l2_prev_strength > 0` and `previous_params` has required keys.

Useful options:

- `dynamic_b` or explicit `fft_params_by_expiry` for stable grids,
- `use_cache_in_optimization` toggle,
- `clear_cache_before` helper,
- `max_nfev`, bounds, penalties, verbosity.

`clear_fft_cache()` clears the FFT grid LRU cache.

---

## Module: `src/snapshot_job.py`

### Responsibilities

`process_snapshot_to_payload(...)` processes one snapshot file and returns an Excel-ready payload consumed by `batch_runner`/`results_store`.

Workflow:

1. Parse snapshot timestamp from filename (`deribit_options_snapshot_YYYYMMDDTHHMMSSZ.csv`).
2. Load CSV and validate key columns.
3. Filter options via `filter_liquid_options(...)`.
4. Optionally throttle runtime via `restrict_for_runtime(...)`.
5. Build per-expiry FFT plans (stable between train/test for a snapshot).
6. Deterministically split to train/test.
7. Calibrate Black → Heston → SVCJ (with warm starts when available).
8. Reprice the full filtered dataset with all calibrated models.
9. Compute train/test RMSE/MAE and assemble:
   - `param_rows` per model,
   - `train_df` and `test_df` with `price_black`, `price_heston`, `price_svcj`,
   - `warm_next` parameters for the next snapshot.

If reading/filtering/calibration fails, the function still returns parameter rows marked `success=False` so batch resume logic can skip already-attempted files.

---

## Module: `src/results_store.py`

### Responsibilities

Owns Excel workbook schema and persistence helpers.

Sheet names:

- `black_params`
- `heston_params`
- `svcj_params`
- `train_data`
- `test_data`

### Notable helpers

- `init_empty_workbook()`
- `load_existing_workbook(path)`
- `append_df(...)`
- `flush_workbook_atomic(...)`
- `get_latest_processed_timestamp(...)`
- `latest_successful_params_before(...)`

Design guarantees:

- schema-tolerant parameter sheet loading,
- normalized timestamp format (`...Z` UTC),
- atomic flush via temp file + replace.

---

## Module: `src/batch_runner.py`

### Responsibilities

Orchestrates multi-file, multi-currency runs with optional multithreading.

Core components:

- `BatchConfig`: central runtime/config object.
- `list_snapshot_files(...)`: discovers `deribit_options_snapshot_*.csv`.
- `run_currency_to_excel(...)`: per-currency orchestrator.
- `run_all_snapshots_to_excel(...)`: high-level multi-currency entry point.

### Execution model

- Reads/initializes workbook.
- Applies resume logic from latest timestamp in `black_params` per currency.
- Splits pending files into contiguous chunks.
- Worker threads process chunks with `process_snapshot_to_payload(...)`.
- Main thread commits results in strict timestamp order.
- Workbook flushed every `save_every_n_files` snapshots.

Warm starts are bootstrapped from the latest successful parameter row before each chunk start.

---

## Module: `src/collect_deribit_snapshot.py`

### Responsibilities

CLI script to pull market data from the Deribit public API and write:

1. `deribit_options_snapshot_<timestamp>.csv`
2. `perpetual_futures_prices.csv`

### API usage

- `/public/get_instruments`
- `/public/get_book_summary_by_currency`
- `/public/ticker`

For each option row, the script captures metadata, quotes, Greeks, OI/volume, implied vol, and maturity fields used later by filtering/calibration.

CLI flags:

- `--currency BTC|ETH|BOTH`
- `--outdir`
- `--max-instruments`
- `--sleep`

---

## Typical usage path

1. Collect snapshots with `collect_deribit_snapshot.py`.
2. Configure batch settings with `BatchConfig` (often in notebook).
3. Run `run_all_snapshots_to_excel(cfg)`.
4. Inspect `calibration_results.xlsx` parameter and train/test sheets.

This workflow is also reflected in `calibrate_all_to_excel.ipynb`.
