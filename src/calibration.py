"""
Utilities to (1) clean Deribit option snapshot data and (2) calibrate
inverse-option pricing models (Black, Heston, SVCJ) by fitting model prices
(in coin units) to mid quotes using a weighted least-squares objective.

This module is designed to work with the user's Carr–Madan FFT pricer
`inverse_fft_pricer.py`, which must be importable.

Public API
----------
- filter_liquid_options(df, ...)
- calibrate_model(df, model, ...)

Notes
-----
- Requires bid and ask quotes (rows missing either are removed).
- Calibration is performed on *coin* prices (Deribit quote convention).
- The pricer's internal FFT grid cache is reused automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .inverse_fft_pricer import FFTParams, price_inverse_option, _cached_pricing_grid


# -----------------------------
# Weighting configuration
# -----------------------------

@dataclass(frozen=True)
class WeightConfig:
    """Controls how per-option residual weights are formed.

    Residual passed to least-squares:
        r_i = w_i * (P_model_i - P_mkt_i)
    where P are coin-denominated prices.
    """
    use_spread: bool = True
    use_vega: bool = True
    use_open_interest: bool = True

    spread_power: float = 1.0
    vega_power: float = 0.5
    oi_power: float = 0.5

    eps_spread: float = 1e-6
    eps_other: float = 1e-12

    cap: Optional[float] = 1e6


@dataclass(frozen=True)
class CalibrationResult:
    model: str
    params: Dict[str, float]
    success: bool
    message: str
    nfev: int
    rmse: float
    mae: float


# -----------------------------
# Cleaning / filtering
# -----------------------------

_REQUIRED_COLUMNS = {
    "currency",
    "option_type",
    "strike",
    "time_to_maturity",
    "bid_price",
    "ask_price",
    "futures_price",
    "vega",
    "open_interest",
    "expiry_datetime",
}


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
    """Filter a Deribit snapshot down to reasonably liquid option quotes."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    if currency is not None:
        out = out[out["currency"].astype(str).str.upper() == currency.upper()]

    for c in [
        "strike",
        "time_to_maturity",
        "bid_price",
        "ask_price",
        "futures_price",
        "vega",
        "open_interest",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out[np.isfinite(out["time_to_maturity"]) & (out["time_to_maturity"] > 0.0)]
    out = out[out["time_to_maturity"] >= float(min_time_to_maturity)]
    if max_time_to_maturity is not None:
        out = out[out["time_to_maturity"] <= float(max_time_to_maturity)]

    if require_bid_ask:
        out = out[
            np.isfinite(out["bid_price"])
            & np.isfinite(out["ask_price"])
            & (out["bid_price"] > 0.0)
            & (out["ask_price"] > 0.0)
        ]

    out = out[out["ask_price"] >= out["bid_price"]]

    out = out[np.isfinite(out["vega"]) & (out["vega"] >= float(min_vega))]
    out = out[np.isfinite(out["open_interest"]) & (out["open_interest"] >= float(min_open_interest))]

    if drop_synthetic_underlyings:
        if "underlying_name" in out.columns:
            out = out[~out["underlying_name"].astype(str).str.startswith("SYN.")]
        if "futures_instrument_name" in out.columns:
            out = out[~out["futures_instrument_name"].astype(str).str.startswith("SYN.")]

    out["mid_price_clean"] = 0.5 * (out["bid_price"] + out["ask_price"])
    out["spread"] = out["ask_price"] - out["bid_price"]
    out["rel_spread"] = out["spread"] / np.maximum(out["mid_price_clean"], 1e-18)

    if max_rel_spread is not None:
        out = out[np.isfinite(out["rel_spread"]) & (out["rel_spread"] <= float(max_rel_spread))]

    out["F0"] = out.groupby("expiry_datetime")["futures_price"].transform("median")
    out = out[np.isfinite(out["F0"]) & (out["F0"] > 0.0)]

    out["moneyness"] = out["strike"] / out["F0"]
    out["log_moneyness"] = np.log(out["moneyness"].astype(float))

    if moneyness_range is not None:
        lo, hi = moneyness_range
        out = out[(out["moneyness"] >= float(lo)) & (out["moneyness"] <= float(hi))]

    out["option_type"] = out["option_type"].astype(str).str.lower()
    out = out[out["option_type"].isin(["call", "put"])]

    out = out.sort_values(["expiry_datetime", "strike", "option_type"]).reset_index(drop=True)
    return out


# -----------------------------
# Pricing helper (fast path for calibration)
# -----------------------------

def _choose_fft_params_for_group(base: FFTParams, strikes: np.ndarray) -> FFTParams:
    """Create FFTParams with `b` centered around the group's strikes."""
    N = base.N
    eta = base.eta
    lam = 2.0 * np.pi / (N * eta)
    logK_center = float(np.log(np.median(strikes)))
    b = logK_center - 0.5 * N * lam
    return FFTParams(N=base.N, alpha=base.alpha, eta=base.eta, b=b, use_simpson=base.use_simpson)



def _build_pricing_plan(
    df: pd.DataFrame,
    *,
    fft_params_base: FFTParams,
    dynamic_b: bool,
    fft_params_by_expiry: Optional[dict] = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float, FFTParams]]:
    """Precompute a per-expiry pricing plan.

    Each plan entry corresponds to a single expiry and contains:
        (idx_positions, K_array, put_mask, T, F0, fft_params)

    This enables pricing all calls and puts for an expiry using *one* FFT call:
    we compute call prices for the expiry once and then obtain put prices via
    inverse put–call parity.

    Notes
    -----
    - ``idx_positions`` are positional indices (0..n-1) to support fast filling
      of preallocated numpy buffers.
    - If ``fft_params_by_expiry`` is provided, it overrides ``dynamic_b`` and
      ensures FFT grids (especially `b`) are stable across train/test splits.
    """
    n = len(df)
    if n == 0:
        return []

    expiry = df["expiry_datetime"].to_numpy()
    opt_type = df["option_type"].astype(str).str.lower().to_numpy()
    strike = df["strike"].to_numpy(dtype=float)
    ttm = df["time_to_maturity"].to_numpy(dtype=float)
    f0 = df["F0"].to_numpy(dtype=float)

    buckets: dict[object, list[int]] = {}
    order: list[object] = []

    for i in range(n):
        key = expiry[i]
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(i)

    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float, FFTParams]] = []
    for exp in order:
        idx = np.asarray(buckets[exp], dtype=int)
        K = strike[idx]
        put_mask = (opt_type[idx] == "put")

        T = float(np.median(ttm[idx]))
        F0 = float(np.median(f0[idx]))

        if fft_params_by_expiry is not None and exp in fft_params_by_expiry:
            fftp = fft_params_by_expiry[exp]
        else:
            fftp = _choose_fft_params_for_group(fft_params_base, K) if dynamic_b else fft_params_base

        groups.append((idx, K, put_mask, T, F0, fftp))

    return groups



def _price_with_plan(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float, FFTParams]],
    *,
    model: str,
    params: Dict[str, float],
    out_prices: np.ndarray,
    use_cache: bool = True,
) -> None:
    """Fill ``out_prices`` (coin prices) using the precomputed per-expiry plan.

    Implementation detail:
    - For each expiry, we price *calls* once via FFT, then compute puts via
      inverse put–call parity:
          P_coin = C_coin - (1 - K/F0)
      with a floor at 0 (as in `price_inverse_option`).
    """
    for idx, K, put_mask, T, F0, fftp in groups:
        call_coin = price_inverse_option(
            model=model,
            K=K,
            T=T,
            F0=F0,
            params=params,
            option_type="call",
            fft_params=fftp,
            use_cache=use_cache,
            return_grid=False,
        )

        out_prices[idx] = call_coin

        if np.any(put_mask):
            # Compute puts only where needed (avoid extra arrays)
            K_put = K[put_mask]
            p_put = call_coin[put_mask] - (1.0 - K_put / F0)
            p_put = np.maximum(p_put, 0.0)
            out_prices[idx[put_mask]] = p_put


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
    """Vectorized pricing via the FFT pricer (fast: no pandas groupby in hot path)."""
    if fft_params_base is None:
        fft_params_base = FFTParams()
    prices = np.empty(len(df), dtype=float)
    groups = _build_pricing_plan(
        df,
        fft_params_base=fft_params_base,
        dynamic_b=dynamic_b,
        fft_params_by_expiry=fft_params_by_expiry,
    )
    _price_with_plan(groups, model=model.lower(), params=params, out_prices=prices, use_cache=use_cache)
    return prices


def clear_fft_cache() -> None:
    """Clear the FFT grid cache in the pricer (useful between calibration runs)."""
    _cached_pricing_grid.cache_clear()


# -----------------------------
# Calibration
# -----------------------------

def _weights(df: pd.DataFrame, cfg: WeightConfig) -> np.ndarray:
    w = np.ones(len(df), dtype=float)

    if cfg.use_spread:
        spread = df["spread"].to_numpy(dtype=float)
        w *= (spread + cfg.eps_spread) ** (-cfg.spread_power)

    if cfg.use_vega:
        vega = df["vega"].to_numpy(dtype=float)
        w *= (vega + cfg.eps_other) ** (cfg.vega_power)

    if cfg.use_open_interest:
        oi = df["open_interest"].to_numpy(dtype=float)
        w *= (oi + cfg.eps_other) ** (cfg.oi_power)

    if cfg.cap is not None:
        w = np.minimum(w, float(cfg.cap))

    w = np.where(np.isfinite(w), w, 0.0)
    return w


# ---- parameter transforms (unconstrained -> constrained) ----

def _pack_black(params: Dict[str, float]) -> np.ndarray:
    return np.array([np.log(float(params["sigma"]))], dtype=float)


def _unpack_black(x: np.ndarray) -> Dict[str, float]:
    return {"sigma": float(np.exp(x[0]))}


def _pack_heston(params: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            np.log(float(params["kappa"])),
            np.log(float(params["theta"])),
            np.log(float(params["sigma_v"])),
            np.arctanh(np.clip(float(params["rho"]), -0.999, 0.999)),
            np.log(float(params["v0"])),
        ],
        dtype=float,
    )


def _unpack_heston(x: np.ndarray) -> Dict[str, float]:
    return {
        "kappa": float(np.exp(x[0])),
        "theta": float(np.exp(x[1])),
        "sigma_v": float(np.exp(x[2])),
        "rho": float(np.tanh(x[3])),
        "v0": float(np.exp(x[4])),
    }


def _pack_svcj(params: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            np.log(float(params["kappa"])),
            np.log(float(params["theta"])),
            np.log(float(params["sigma_v"])),
            np.arctanh(np.clip(float(params["rho"]), -0.999, 0.999)),
            np.log(float(params["v0"])),
            np.log(float(params["lam"])),
            float(params["ell_y"]),
            np.log(float(params["sigma_y"])),
            np.log(float(params["ell_v"])),
            np.arctanh(np.clip(float(params["rho_j"]), -0.999, 0.999)),
        ],
        dtype=float,
    )


def _unpack_svcj(x: np.ndarray) -> Dict[str, float]:
    return {
        "kappa": float(np.exp(x[0])),
        "theta": float(np.exp(x[1])),
        "sigma_v": float(np.exp(x[2])),
        "rho": float(np.tanh(x[3])),
        "v0": float(np.exp(x[4])),
        "lam": float(np.exp(x[5])),
        "ell_y": float(x[6]),
        "sigma_y": float(np.exp(x[7])),
        "ell_v": float(np.exp(x[8])),
        "rho_j": float(np.tanh(x[9])),
    }


def _default_initial_params(df: pd.DataFrame, model: str) -> Dict[str, float]:
    model = model.lower()
    if "implied_volatility" in df.columns and np.isfinite(df["implied_volatility"]).any():
        atm_iv = float(np.nanmedian(pd.to_numeric(df["implied_volatility"], errors="coerce")))
        atm_iv = max(atm_iv, 1e-3)
    else:
        atm_iv = 0.6

    if model == "black":
        return {"sigma": atm_iv}

    theta = atm_iv * atm_iv
    if model == "heston":
        return {"kappa": 2.0, "theta": theta, "sigma_v": 0.8, "rho": -0.5, "v0": theta}

    if model == "svcj":
        return {
            "kappa": 2.0,
            "theta": theta,
            "sigma_v": 0.8,
            "rho": -0.5,
            "v0": theta,
            "lam": 0.2,
            "ell_y": -0.05,
            "sigma_y": 0.25,
            "ell_v": 0.1,
            "rho_j": 0.0,
        }

    raise ValueError("model must be one of {'black','heston','svcj'}")


def _default_bounds(model: str) -> Tuple[np.ndarray, np.ndarray]:
    m = model.lower()
    if m == "black":
        return np.array([np.log(1e-4)]), np.array([np.log(5.0)])

    if m == "heston":
        lb = np.array([np.log(1e-4), np.log(1e-6), np.log(1e-4), -5.0, np.log(1e-6)], dtype=float)
        ub = np.array([np.log(50.0), np.log(5.0), np.log(10.0), 5.0, np.log(5.0)], dtype=float)
        return lb, ub

    if m == "svcj":
        lb = np.array(
            [
                np.log(1e-4),
                np.log(1e-6),
                np.log(1e-4),
                -5.0,
                np.log(1e-6),
                np.log(1e-6),
                -5.0,
                np.log(1e-4),
                np.log(1e-6),
                -5.0,
            ],
            dtype=float,
        )
        ub = np.array(
            [
                np.log(50.0),
                np.log(5.0),
                np.log(10.0),
                5.0,
                np.log(5.0),
                np.log(10.0),
                5.0,
                np.log(5.0),
                np.log(10.0),
                5.0,
            ],
            dtype=float,
        )
        return lb, ub

    raise ValueError("model must be one of {'black','heston','svcj'}")


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
    if fft_params_base is None:
        fft_params_base = FFTParams()

    model_l = model.lower()
    if initial_params is None:
        initial_params = _default_initial_params(df, model_l)

    if clear_cache_before:
        clear_fft_cache()

    if model_l == "black":
        pack, unpack = _pack_black, _unpack_black
    elif model_l == "heston":
        pack, unpack = _pack_heston, _unpack_heston
    elif model_l == "svcj":
        pack, unpack = _pack_svcj, _unpack_svcj
    else:
        raise ValueError("model must be one of {'black','heston','svcj'}")

    x0 = pack(initial_params)

    y_mkt_full = df["mid_price_clean"].to_numpy(dtype=float)
    w_full = _weights(df, weight_config)

    keep = np.isfinite(y_mkt_full) & (w_full > 0.0)
    df_fit = df.loc[keep].copy()
    y_mkt = y_mkt_full[keep]
    w = w_full[keep]

    pricing_plan = _build_pricing_plan(
        df_fit,
        fft_params_base=fft_params_base,
        dynamic_b=dynamic_b,
        fft_params_by_expiry=fft_params_by_expiry,
    )

    # preallocate buffers (reused in every residual eval)
    y_model_buf = np.empty(len(df_fit), dtype=float)
    r_buf = np.empty(len(df_fit), dtype=float)

    if bounds is None:
        bounds = _default_bounds(model_l)
    lb, ub = bounds
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    x0 = np.minimum(np.maximum(x0, lb), ub)

    def _constraint_residuals(p: Dict[str, float]) -> np.ndarray:
        pen = []

        if model_l in ("heston", "svcj"):
            feller_rhs = 2.0 * float(p["kappa"]) * float(p["theta"]) - float(feller_eps)
            feller_violation = max(0.0, float(p["sigma_v"]) ** 2 - feller_rhs)
            pen.append(float(constraint_penalty) * feller_violation)

        if model_l == "svcj":
            moment = 1.0 - float(p["ell_v"]) * float(p["rho_j"])
            moment_violation = max(0.0, float(svcj_moment_eps) - moment)
            pen.append(float(constraint_penalty) * moment_violation)

        if not pen:
            return np.empty(0, dtype=float)

        pen_arr = np.asarray(pen, dtype=float)
        return np.where(np.isfinite(pen_arr), pen_arr, float(constraint_penalty))

    def residuals(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        pen = _constraint_residuals(p)

        try:
            _price_with_plan(
                pricing_plan,
                model=model_l,
                params=p,
                out_prices=y_model_buf,
                use_cache=use_cache_in_optimization,
            )
        except Exception:
            r = penalty_coin * w
            r = np.where(np.isfinite(r), r, penalty_coin)
            return np.concatenate([r, pen]) if pen.size else r

        # r_buf = w * (y_model - y_mkt), computed in-place WITHOUT augmented assignment on r_buf
        np.subtract(y_model_buf, y_mkt, out=r_buf)
        np.multiply(r_buf, w, out=r_buf)  # <-- FIX: avoids UnboundLocalError in nested scope

        bad = ~np.isfinite(y_model_buf)
        if np.any(bad):
            r_buf[bad] = penalty_coin * w[bad]

        r = np.where(np.isfinite(r_buf), r_buf, penalty_coin)
        return np.concatenate([r, pen]) if pen.size else r

    res = least_squares(
        residuals,
        x0,
        method="trf",
        bounds=(lb, ub),
        max_nfev=int(max_nfev),
        verbose=int(verbose),
    )

    params_hat = unpack(res.x)

    _price_with_plan(pricing_plan, model=model_l, params=params_hat, out_prices=y_model_buf, use_cache=True)
    y_hat = y_model_buf

    if not np.all(np.isfinite(y_hat)):
        rmse = float("nan")
        mae = float("nan")
        success = False
        message = str(res.message) + " (non-finite prices at optimum)"
    else:
        err = y_hat - y_mkt
        rmse = float(np.sqrt(np.mean(err * err)))
        mae = float(np.mean(np.abs(err)))
        success = bool(res.success)
        message = str(res.message)

    return CalibrationResult(
        model=model_l,
        params=params_hat,
        success=bool(success),
        message=str(message),
        nfev=int(res.nfev),
        rmse=rmse,
        mae=mae,
    )
