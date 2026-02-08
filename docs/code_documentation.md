# Code documentation

This document explains **what the main Python modules in this project do** and how their key functions fit together.  It is meant as a technical companion to the high‑level overview in the README, and it follows the same pricing conventions used throughout the thesis.  For mathematical derivations and theoretical background, see `docs/MASTERS_DIPLOM.pdf`.

---

## Conventions used throughout the code

### Underlying and strike

* Options are written on a **futures or forward price** `F` quoted in USD per coin (e.g., USD/BTC).
* Strikes `K` are also expressed in USD per coin.

### Coin‑denominated (inverse) option prices

Inverse options are quoted and settled in coins rather than in dollars.  The pricing workflow therefore:

1. Computes a **regular call price grid in USD** on a strike grid `K_grid`.
2. Converts to **coin call prices** via `C_coin(K) = C_usd(K) / F_0` where `F_0` is a proxy for the forward price at expiry.
3. Obtains **coin puts** by **inverse put–call parity**:

   $$P_{coin}(K) = C_{coin}(K) - \left(1 - \frac{K}{F_0}\right).$$

Calls are clipped to `[0,1]` and puts are floored at zero to reflect the maximum one‑coin payoff.

### Time to maturity

Time to maturity `T` is measured in years.  Snapshot CSVs carry a column `time_to_maturity` in years and this value is passed directly into the pricing functions.

---

## Module: `src/inverse_fft_pricer.py`

### What this module is responsible for

`inverse_fft_pricer.py` implements the **Carr–Madan FFT engine** for European options on a futures price and provides characteristic functions for the supported models.

* **Characteristic functions** – Functions `cf_black`, `cf_heston` and `cf_svcj` compute the log characteristic functions for the Black, Heston and SVCJ models, respectively.  Heston uses a stable complex square‑root branch, and SVCJ evaluates the jump term via Gauss–Legendre quadrature.
* **FFT pricing** – The function `carr_madan_call_fft(cf, T, F0, params, fft_params)` builds a frequency grid `v_j = j * eta`, damps the integrand with the Carr–Madan exponential factor, applies quadrature weights (trapezoid or Simpson) and uses an FFT to transform to log‑strikes.  It returns `(K_grid, C_grid_usd)` in **USD**.
* **High‑level pricing** – `price_inverse_option(model, K, T, F0, params, option_type='call', ...)` selects the appropriate characteristic function, calls the FFT engine to get a call price grid, interpolates to requested strikes, scales to coin units and converts puts via inverse parity.  It accepts scalar or vector strikes and can return the pricing grid for diagnostics.

### FFT parameters

The grid is controlled by a `FFTParams` dataclass with fields:

* `N` – number of grid points (power of two for FFT efficiency).
* `alpha` – exponential damping parameter (set so the call transform is integrable).
* `eta` – frequency step; determines spacing of log‑strike grid.
* `b` – shift of the log‑strike grid.  In practice, `calibration.py` or `snapshot_job.py` may compute `b` dynamically per expiry to centre the grid around the data.
* `use_simpson` – if `True`, Simpson quadrature weights are used; otherwise trapezoid weights.

Reference pricers (`black76_call_price`, `black76_put_price`, `heston_call_price`, `heston_put_price`) are provided for sanity checks and validation.

---

## Module: `src/calibration.py`

### What this module is responsible for

`calibration.py` implements the **data → calibration** pipeline for a single snapshot.  It wraps the FFT pricer with data filtering, weighting and parameter estimation logic.

* **Filtering and cleaning** – The function `filter_liquid_options(df, ...)` validates required columns (`bid_price`, `ask_price`, `strike`, `expiry_datetime`, `time_to_maturity`, `futures_price`, `vega`, `open_interest`), computes mid prices and spreads, derives a forward proxy `F0` per expiry, computes moneyness `K/F0` and log‑moneyness, and applies screens on time to maturity, relative spread, moneyness band, open interest and vega.  Synthetic instruments can be dropped via `drop_synthetic_underlyings`.
* **Pricing a dataset** – `price_dataframe(df, model, params, fft_params_base, dynamic_b, ...)` builds a per‑expiry pricing plan and runs the FFT once per expiry to price all calls.  Puts are computed via inverse parity.  A dynamic `b` option centres the log‑strike grid around the median strike of each expiry bucket to improve interpolation.
* **Weighting** – A `WeightConfig` class defines how residuals are weighted in the calibration objective.  Spread, vega and open interest can contribute to the weight, with configurable exponents and caps.  Weights down‑weight illiquid or wide‑spread options.
* **Calibration** – `calibrate_model(df, model, ..., initial_params, bounds, penalty_coin, constraint_penalty, feller_eps, svcj_moment_eps, max_nfev, verbose, ...)` solves a **weighted nonlinear least‑squares** problem for the specified model.  The parameter vector is transformed (log for positive parameters, arctanh for correlations) and additional residuals impose soft penalties for the Heston Feller condition (`sigma_v^2 ≤ 2 κ θ`) and SVCJ moment stability (`1 - ell_v * rho_j ≥ ε`).  `scipy.optimize.least_squares` is used under the hood and returns a `CalibrationResult` with parameter estimates, fit metrics and success flags.
* **Cache management** – `clear_fft_cache()` clears the internal LRU cache used by the pricer to store previously computed FFT grids.  This can reduce memory usage when calibrating many snapshots sequentially.

`calibration.py` does not perform any I/O; it expects a DataFrame and model configuration and returns calibrated parameters and prices.  Higher‑level modules orchestrate persistence and parallelism.

---

## Module: `src/results_store.py`

### What this module is responsible for

`results_store.py` manages persistence of calibration results in an **Excel workbook**.  It defines the workbook schema and provides helpers for initialisation, loading, appending and flushing.

* **Workbook structure** – The workbook contains five sheets: `black_params`, `heston_params`, `svcj_params`, `train_data` and `test_data`.  The parameter sheets hold one row per processed snapshot per currency (timestamp, currency, fitted parameters, success flag, message, number of function evaluations and error metrics).  The data sheets hold the rows used in the train and test sets along with the computed model prices.
* **Initialisation** – `init_empty_workbook()` returns a dictionary of empty DataFrames with the correct columns for each sheet.  New workbooks use this structure when none exists.
* **Loading** – `load_existing_workbook(path)` reads an existing workbook into DataFrames and adds any missing columns to accommodate schema evolution.  It returns an in‑memory dictionary of DataFrames.
* **Resuming** – `get_latest_processed_timestamp(workbook, currency)` reads the parameter sheets to determine the most recent timestamp for a given currency.  Batch runners use this to skip already processed snapshots.
* **Appending and flushing** – `append_df(workbook, sheet_name, df)` appends new rows to the given sheet in the in‑memory workbook.  `flush_workbook_atomic(workbook, output_path)` writes the workbook to a temporary file and replaces the existing file atomically.  Periodic flushing ensures that partial results are never left on disk and that the pipeline can resume safely.

`results_store.py` is used only by the batch runner; it is not imported into the pricing or calibration modules.

---

## Module: `src/snapshot_job.py`

### What this module is responsible for

`snapshot_job.py` orchestrates all tasks needed to process **one snapshot CSV** and prepare results for persistence.  It is the glue between the calibration pipeline and the batch runner.

* **Timestamp parsing** – Helpers `timestamp_from_filename` and `timestamp_to_iso_z` extract consistent ISO timestamps from snapshot file names and convert them to strings for workbook keys.
* **Data loading and filtering** – Reads the CSV into a DataFrame and calls `calibration.filter_liquid_options` with the filtering rules supplied by the batch configuration.
* **Runtime throttling** – An optional `restrict_for_runtime(df, top_expiries_by_oi, max_options)` can limit the number of expiries or options for smoke tests or time‑limited runs.
* **Train/test split** – `train_test_split_df(df, train_frac, seed)` shuffles the rows deterministically using a seeded random generator and splits the filtered dataset into a train and test set.
* **Calibration with warm starts** – Calls `calibration.calibrate_model` sequentially for the Black, Heston and SVCJ models.  Each model can be initialised from either the previous snapshot’s parameters or a simple function of the Black volatility (for Heston/SVCJ).  Maximum function evaluations are configurable per model.
* **Repricing full dataset** – After calibrating, `price_dataframe` is called once per model on the full filtered DataFrame to obtain consistent model prices for both train and test rows.
* **Error metrics and payload** – Computes unweighted RMSE and MAE for the train and test sets.  Returns a payload containing the parameter rows, the priced train and test DataFrames, and the warm‑start parameters for the next snapshot.

The batch runner uses this payload to append rows to the workbook and to initialise warm starts for subsequent snapshots.

---

## Module: `src/batch_runner.py`

### What this module is responsible for

`batch_runner.py` runs calibration across **multiple snapshots and currencies**.  It coordinates file enumeration, warm starts, multithreading and persistence.

* **Batch configuration** – Defines a `BatchConfig` dataclass that holds global configuration: project root, output workbook path, filtering rules, weighting parameters, FFT parameters, train fraction, maximum function evaluations per model, runtime throttles, number of workers, verbosity, etc.  This configuration is consumed by the runner and by `snapshot_job`.
* **Snapshot enumeration and resume** – `list_snapshot_files()` and related helpers locate snapshot CSV files in the `data/` directory and sort them by timestamp.  The function `get_latest_processed_timestamp` from `results_store` is used to skip snapshots that have already been processed when resuming.
* **Chunking and multithreading** – The list of pending snapshot files is divided into contiguous chunks and assigned to worker threads.  Each worker calls `process_snapshot_to_payload` sequentially on its chunk, using a warm‑start dictionary initialised from the last processed snapshot for that currency.
* **Ordered committing** – Worker threads place their results into a queue.  The main thread pops payloads and commits them to the workbook **in timestamp order**, ensuring that partial results always form a contiguous prefix of the chronology.  Skipped or failed snapshots still write a parameter row with `success=False` so they will not be retried.
* **Periodic flushing and progress reporting** – After a configurable number of committed snapshots the workbook is flushed atomically to disk.  The verbosity level controls whether the runner prints per‑snapshot status messages and calibration diagnostics.

High‑level functions `run_currency_to_excel(cfg, currency)` and `run_all_snapshots_to_excel(cfg)` hide these details and can be called from notebooks or scripts.

---

## Module: `src/collect_deribit_snapshot.py`

### What this module is responsible for

`collect_deribit_snapshot.py` fetches a market snapshot from the **Deribit public API** and writes two CSV files:

* `deribit_options_snapshot_<timestamp>.csv` – Contains one row per option instrument with metadata (expiry, strike, option type), bid/ask prices, mark price, implied volatility, greeks, open interest and other fields.
* `perpetual_futures_prices.csv` – Contains the best bid, best ask and mark price for the perpetual futures contract at the same timestamp.

The snapshot is built by calling Deribit API endpoints to list instruments, fetch book summaries, fetch per‑instrument tickers and fetch the perp ticker.  The output CSVs are ready for ingestion by `calibration.filter_liquid_options` and the batch pipeline.

---

## Extending the code

### Adding a new model

1. Implement a characteristic function `cf_newmodel(u, T, F0, ...)` in `inverse_fft_pricer.py` and ensure it remains stable for complex arguments `u - i(α+1)`.
2. Modify `price_inverse_option` to select your new characteristic function and define how to unpack parameters from a dictionary.
3. Provide initial guesses, bounds and parameter transformations in `calibration.calibrate_model`.
4. Add new parameter columns to the workbook schema in `results_store.py` and update the payload assembly in `snapshot_job.py`.
5. Update this documentation and the README to describe your new model and any additional notebooks.

### Changing filtering or weights

Filtering rules are implemented in `calibration.filter_liquid_options`.  They can be configured via the `BatchConfig.filter_rules` dictionary in `batch_runner.py`.  Weighting is controlled by `WeightConfig`, which allows weighting by spread, vega and open interest with configurable exponents.

### Performance considerations

* **FFT resolution** – `FFTParams.N` and `eta` determine the resolution and range of the strike grid.  Larger `N` increases accuracy but increases computation time approximately as `O(N log N)`.
* **Dynamic centering** – Setting `dynamic_b=True` in pricing functions centres the log‑strike grid around the median strike of each expiry bucket, improving interpolation quality and reducing the required grid size.
* **Parallelism** – The number of worker threads `n_workers` should be chosen to match your hardware.  Each worker processes a contiguous chunk of snapshots and should not oversubscribe CPU cores, especially when BLAS libraries spawn internal threads.  Limiting internal threads via environment variables (e.g. `OMP_NUM_THREADS=1`) can improve overall speed.
* **Flush frequency** – The `save_every_n_files` parameter controls how often the workbook is flushed to disk.  Smaller values reduce the amount of work lost on interruption but increase I/O overhead.
