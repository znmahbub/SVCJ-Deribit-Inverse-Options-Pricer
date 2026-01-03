# Code documentation

This document explains **what the main Python modules do** and how the key functions fit together.

---

## Conventions used throughout the code

### Underlying and strike
- The option is written on a **futures/forward price** `F` quoted in **USD per coin** (e.g., USD/BTC).
- Strikes `K` are also **USD per coin**.

### Coin-denominated (inverse) option prices
Deribit inverse options are quoted in **coin** (e.g., BTC). In this project:

- The FFT produces **regular call prices in USD** on a strike grid:  
  `C_usd(K)`.

- The module converts to **inverse (coin) call prices** by:
  ```math
  C_{coin}(K) = \frac{C_{usd}(K)}{F_0}.
  ```

- Inverse put prices are obtained using **inverse put–call parity**:
  ```math
  C_{coin}(K) - P_{coin}(K) = 1 - \frac{K}{F_0},
  \quad\Rightarrow\quad
  P_{coin}(K) = C_{coin}(K) - \left(1-\frac{K}{F_0}\right).
  ```
  
  The code floors puts at 0.

### Time to maturity
- `T` is measured in **years** (ACT/365 style implied by how snapshots are built).
- Input datasets carry `time_to_maturity` in years.

---

## Module: `src/inverse_fft_pricer.py`

### What this module is responsible for
- Computing a **strike grid** and the associated **USD call prices** via the **Carr–Madan FFT**.
- Converting those prices to **coin-denominated inverse calls/puts**.
- Providing “sanity-check” pricers for Black–76 and (semi-analytical) Heston.

### Key data structure: `FFTParams`

```python
class FFTParams:
    N: int = 2 ** 12
    alpha: float = 1.5
    eta: float = 0.1
    b: float = -5.0
    use_simpson: bool = True
```

Interpretation:
- `N`: number of FFT grid points (power of two recommended).
- `alpha`: Carr–Madan exponential **damping** parameter for calls.
- `eta`: frequency step; sets max frequency and (via the FFT) the log-strike spacing.
- `b`: log-strike grid shift. In practice, **`calibration.py` often chooses this dynamically** so the grid
  is centered around the strikes being fit.
- `use_simpson`: if `True`, Simpson weights are used; otherwise trapezoid weights.

### Primary entry point: `price_inverse_option`

```python
def price_inverse_option(
    model: str,
    K: float | np.ndarray,
    T: float,
    F0: float,
    params: dict,
    *,
    option_type: str = "call",
    fft_params: FFTParams | None = None,
    use_cache: bool = True,
    return_grid: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
```

What it does:
1. Selects a characteristic function `cf_*` based on `model`.
2. Calls the Carr–Madan FFT once to compute a **USD call price grid** `(K_grid, C_grid)`.
3. Interpolates `C_usd(K)` onto the requested strike(s) `K`.
4. Converts to **coin**: `C_coin = C_usd / F0`.
5. If `option_type="put"`, applies inverse put–call parity.

Important implementation notes:
- `np.interp` is used (linear interpolation) on the monotone `K_grid`.
- Call prices in coin are clipped to `[0, 1]` (consistent with the “max 1 coin” intuition for inverse payoffs).
- Puts are floored at 0.
- If `return_grid=True`, returns `(price_coin, K_grid, C_grid)` for debugging/plotting.

Parameter dictionaries expected by `price_inverse_option`:
- **Black**: `{"sigma": ...}`
- **Heston**: `{"kappa": ..., "theta": ..., "sigma_v": ..., "rho": ..., "v0": ...}`
- **SVCJ**: `{"kappa": ..., "theta": ..., "sigma_v": ..., "rho": ..., "v0": ..., "lam": ..., "ell_y": ..., "sigma_y": ..., "ell_v": ..., "rho_j": ...}`

### FFT engine: `carr_madan_call_fft`

```python
def carr_madan_call_fft(
    cf: callable,
    T: float,
    F0: float,
    params: tuple,
    fft_params: FFTParams,
) -> tuple[np.ndarray, np.ndarray]:
```

What it returns:
- `K_grid`: strikes (ascending)
- `C_grid`: **regular call prices in USD** (not coin)

Under the hood:
- Builds a frequency grid `v_j = j * eta`.
- Evaluates the CF at a shifted argument `v - i(α+1)` (call damping).
- Applies quadrature weights (`Simpson` or trapezoid).
- FFT transforms the damped integrand into a log-strike grid.
- Maps log-strikes `k` to strikes `K = exp(k)`.

### Characteristic functions

- **Black**: `cf_black(u, T, F0, sigma)`
  - log-futures is Gaussian with variance `sigma^2 T`.

- **Heston**: `cf_heston(u, T, F0, kappa, theta, sigma_v, rho, v0)`
  - Includes numerical guards for overflow/non-finite exponentials.
  - Uses a stable complex square-root branch handling to reduce discontinuities.

- **SVCJ**: `cf_svcj(u, T, F0, kappa, theta, sigma_v, rho, v0, lam, ell_y, sigma_y, ell_v, rho_j, quad_nodes=32)`
  - Adds a jump component with Gauss–Legendre quadrature over time.
  - Guards against numerical blow-ups similarly to Heston.

### Caching (performance)
The module uses `functools.lru_cache` for:
- Gauss–Legendre nodes/weights (`_leggauss_cached`).
- Quadrature weights for FFT grids (`_quadrature_weights_cached`).
- Entire pricing grids `(K_grid, C_grid)` keyed by `(model, T, F0, params, FFTParams)` via `_cached_pricing_grid`.

Calibration often sets `use_cache=False` because parameter vectors change at every residual evaluation, which yields
low cache hit rates and can increase memory churn.

### Reference pricers (sanity checks)
- `black76_call_price`, `black76_put_price`: analytic Black–76 (undiscounted, consistent with futures conventions).
- `heston_call_price`, `heston_put_price`: semi-analytical Heston via Gil–Pelaez inversion.
  - The **put** implementation is computed **directly** (no put–call parity shortcut), so it’s useful as a correctness check.

---

## Module: `src/calibration.py`

### What this module is responsible for
- Cleaning/filtering Deribit snapshot data into a “fit-ready” cross-section.
- Computing per-option weights and fit targets.
- Pricing a dataset efficiently by reusing **one FFT per expiry bucket**.
- Running **weighted nonlinear least squares** calibration for `black`, `heston`, and `svcj`.

### Expected input schema (minimum required columns)

`filter_liquid_options` expects at least:

```text
currency, option_type, strike, time_to_maturity,
bid_price, ask_price, futures_price, vega, open_interest, expiry_datetime
```

Additional columns (if present) may be used for initial guesses or extra filtering (e.g., `implied_volatility`).

### Weighting

`WeightConfig` controls per-row weights \(w_i\) used in the residual:
```math
r_i = w_i \cdot (P^{model}_{coin,i} - P^{mkt}_{coin,i})
```

Key behavior:
- `use_spread=True` downweights wide markets via \((spread+\varepsilon)^{-p}\).
- `use_vega=True` and `use_open_interest=True` scale weights by powers of `vega` and `open_interest`.
- `cap` limits extreme weights.

### Filtering: `filter_liquid_options`

```python
def filter_liquid_options(
    df: pd.DataFrame,
    *,
    currency: Optional[str] = None,
    require_bid_ask: bool = True,
    min_time_to_maturity: float = 1.0 / 365.0,
    max_time_to_maturity: Optional[float] = None,
    min_open_interest: float = 1.0,
    min_vega: float = 0.0,
    max_rel_spread: Optional[float] = 0.5,
    moneyness_range: Optional[Tuple[float, float]] = (0.5, 2.0),
    drop_synthetic_underlyings: bool = False,
) -> pd.DataFrame:
```

What it does (high level):
- Validates required columns exist.
- Optional currency subset.
- Drops rows with missing bid/ask (if enabled), invalid/missing `T`, `K`, `F0` proxies, etc.
- Computes:
  - `mid_price_clean = 0.5*(bid+ask)`
  - `spread = ask-bid`
  - `rel_spread = spread / mid`
  - `F0` as **per-expiry median** of `futures_price`
  - `moneyness = K/F0` and `log_moneyness`
- Applies screens: `T` bounds, min `open_interest`, min `vega`, relative spread cap, moneyness band.
- Optionally removes “synthetic” underlyings (prefix `SYN.`).

The output DataFrame is “calibration-ready”: it contains the columns used by the pricing plan and weighting.

### Fast dataset pricing: `price_dataframe`

```python
def price_dataframe(
    df: pd.DataFrame,
    model: str,
    params: Dict[str, float],
    *,
    fft_params_base: Optional[FFTParams] = None,
    dynamic_b: bool = True,
    fft_params_by_expiry: Optional[dict] = None,
    use_cache: bool = True,
) -> np.ndarray:
```

Purpose:
- Prices every row in `df` **in coin units**, efficiently, by:
  1. Building a per-expiry “pricing plan”
  2. Running one FFT per expiry to price calls
  3. Computing puts by inverse parity (within the expiry bucket)

This is the same mechanism used inside calibration residuals.

### Per-expiry pricing plan (internal helpers)
These functions are internal but are central to performance:

- `_choose_fft_params_for_group(base, strikes)`
  - Computes a `b` that centers the FFT strike grid around the median strike of the group:
    ```math
    b \approx \log(\mathrm{median}(K)) - \tfrac{1}{2}N\lambda
    ```
  - This is used when `dynamic_b=True`.

- `_build_pricing_plan(df, ...)`
  - Groups rows by `expiry_datetime` into buckets.
  - Stores `(row_indices, strikes, put_mask, T_bucket, F0_bucket, FFTParams_bucket)`.

- `_price_with_plan(groups, model, params, out_prices, use_cache)`
  - For each expiry bucket:
    - calls `price_inverse_option(..., option_type="call")` once on all strikes
    - fills call rows directly
    - fills put rows via `P_coin = C_coin - (1 - K/F0)` floored at 0

### Cache management

```python
def clear_fft_cache() -> None:
```

Clears the global LRU cache in the pricer (the cached FFT grids). Helpful when:
- calibrating many independent snapshots sequentially,
- changing FFT grid settings frequently,
- or when memory use grows due to low cache hit rates.

### Calibration: `calibrate_model`

```python
def calibrate_model(
    df: pd.DataFrame,
    model: str,
    *,
    weight_config: WeightConfig = WeightConfig(),
    fft_params_base: Optional[FFTParams] = None,
    dynamic_b: bool = True,
    fft_params_by_expiry: Optional[dict] = None,
    use_cache_in_optimization: bool = False,
    initial_params: Optional[Dict[str, float]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    penalty_coin: float = 10.0,
    constraint_penalty: float = 100.0,
    feller_eps: float = 0.0,
    svcj_moment_eps: float = 1e-6,
    max_nfev: int = 50,
    verbose: int = 1,
    clear_cache_before: bool = False,
) -> CalibrationResult:
```

What it does:
1. Computes weights `w_i` from the cleaned DataFrame (spread/vega/OI, configurable).
2. Builds the per-expiry pricing plan once (so residuals are fast).
3. Sets up a SciPy `least_squares` problem over a **transformed parameter vector** `x`:
   - positive parameters use `log` transforms,
   - correlations use `tanh` (packed with `arctanh`).

4. On every residual evaluation:
   - unpack `x → params`
   - price the dataset with the per-expiry plan
   - compute residuals `r_i = w_i * (P_model_coin - P_mkt_coin)`
   - replace any non-finite model prices with a large penalty residual.

5. Adds constraint penalties (as extra residual components):
   - For Heston and SVCJ: a soft **Feller-type** penalty:
     ```math
     \sigma_v^2 \le 2\kappa\theta - \varepsilon
     ```
   - For SVCJ: a soft “moment stability” penalty:
     ```math
     1 - \ell_v \rho_j \ge \varepsilon
     ```
   These are implemented as nonnegative violations multiplied by `constraint_penalty`.

6. Returns `CalibrationResult(model, params_hat, success, message, nfev, rmse, mae)` computed on coin prices.

Parameterization / transforms used internally:
- Black: `x = [log(sigma)]`
- Heston: `x = [log(kappa), log(theta), log(sigma_v), arctanh(rho), log(v0)]`
- SVCJ adds `[log(lam), ell_y, log(sigma_y), log(ell_v), arctanh(rho_j)]`

Default initial guesses are inferred from the snapshot’s median implied vol (if available) and are otherwise
set to reasonable constants.

---

## Module: `src/collect_deribit_snapshot.py`

### What this module is responsible for
- Collecting a consistent “snapshot” of Deribit markets via the public HTTP API:
  - option instruments
  - option book summaries (bid/ask, open interest, etc.)
  - per-instrument ticker (greeks)
  - perpetual futures ticker

The key functions are thin endpoint wrappers:
- `get_instruments(...)`
- `get_book_summary_options(...)`
- `get_ticker(instrument_name)`
- `get_perp_ticker(currency)`
- `main(...)` orchestrates the fetch and writes CSVs.

The output CSVs are designed to match the schema expected by `filter_liquid_options`.

---

## Extending the code (common research tasks)

### Adding a new model (high level)
1. Implement a new characteristic function `cf_newmodel(u, T, F0, ...)`.
2. Add a new branch in `price_inverse_option` selecting the CF and packing parameters.
3. Optionally add:
   - analytic/semi-analytic check pricer (for validation)
   - new default bounds / initial params / packing transforms in `calibration.py`
4. Ensure the model returns stable values for `u - i(α+1)` given your choice of `alpha`
   (moment existence is the usual failure point for jump models).

### Changing filtering or weights
- Filtering lives in `filter_liquid_options`.
- Weighting lives in `_weights` and is configured by `WeightConfig`.
- If you change either, it will impact calibration results directly; keep notes consistent with the PDF spec.

### Performance knobs to be aware of
- `FFTParams.N` and `eta` (speed vs resolution).
- `dynamic_b` (reduces interpolation error; usually helps stability).
- `use_cache_in_optimization` (often **off** during least-squares).
- SVCJ `quad_nodes` inside `cf_svcj` (accuracy vs speed).

---

## Cross-reference: notebooks → modules
- `pricing_examples.ipynb` → `inverse_fft_pricer.price_inverse_option` (+ reference pricers)
- `calibration_example.ipynb` → `calibration.filter_liquid_options`, `calibration.calibrate_model`
- `calibrate_all.ipynb` → repeated calls to `calibrate_model` across snapshots (optionally clearing caches)
