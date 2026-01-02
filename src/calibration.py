"""deribit_calibration.py

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
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

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

    The residual vector passed to least-squares is:

        r_i = w_i * (P_model_i - P_mkt_i)

    where P are coin-denominated prices.

    Components
    ----------
    spread: uses bid-ask spread (ask - bid) in the denominator (downweights wide markets)
    vega:   upweights more vega (informational content)
    oi:     upweights higher open interest (liquidity proxy)

    Weight formula
    --------------
        w_i = (spread_i + eps_spread)^(-spread_power) *
              (vega_i   + eps_other)^(vega_power) *
              (oi_i     + eps_other)^(oi_power)

    You can disable any component by setting use_* = False.

    Practical defaults (reasonable, not sacred):
    - spread_power = 1.0 (divide by spread)
    - vega_power   = 0.5 (sqrt-weights)
    - oi_power     = 0.5
    """

    use_spread: bool = True
    use_vega: bool = True
    use_open_interest: bool = True

    spread_power: float = 1.0
    vega_power: float = 0.5
    oi_power: float = 0.5

    eps_spread: float = 1e-6
    eps_other: float = 1e-12

    # cap to prevent a single observation from dominating
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
    """Filter a Deribit snapshot down to reasonably liquid option quotes.

    Key extra rule (per your request): rows missing either bid or ask are removed.

    Parameters
    ----------
    df:
        Raw snapshot DataFrame.
    currency:
        'BTC' or 'ETH' to select one underlying. If None, keep all.
    require_bid_ask:
        If True, require bid_price>0 and ask_price>0 (and non-NaN).
    min_time_to_maturity:
        Minimum TTM in years.
    max_time_to_maturity:
        Optional maximum TTM.
    min_open_interest:
        Minimum open interest.
    min_vega:
        Minimum vega.
    max_rel_spread:
        Optional max (ask-bid)/mid.
    moneyness_range:
        Optional (min,max) for K/F per expiry, using per-expiry median F.
    drop_synthetic_underlyings:
        If True, drops rows whose underlying_name starts with 'SYN.' or
        whose futures_instrument_name starts with 'SYN.' (if present).

    Returns
    -------
    Cleaned DataFrame with extra columns:
        - mid_price_clean
        - spread
        - rel_spread
        - F0 (per-expiry median futures price)
        - moneyness (K/F0)
        - log_moneyness
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    if currency is not None:
        out = out[out["currency"].astype(str).str.upper() == currency.upper()]

    # numeric conversions
    for c in ["strike", "time_to_maturity", "bid_price", "ask_price", "futures_price", "vega", "open_interest"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # basic domain filters
    out = out[np.isfinite(out["time_to_maturity"]) & (out["time_to_maturity"] > 0.0)]
    out = out[out["time_to_maturity"] >= float(min_time_to_maturity)]
    if max_time_to_maturity is not None:
        out = out[out["time_to_maturity"] <= float(max_time_to_maturity)]

    # require bid/ask (requested)
    if require_bid_ask:
        out = out[np.isfinite(out["bid_price"]) & np.isfinite(out["ask_price"]) & (out["bid_price"] > 0.0) & (out["ask_price"] > 0.0)]

    # sanity: ask>=bid
    out = out[out["ask_price"] >= out["bid_price"]]

    # vega / OI filters
    out = out[np.isfinite(out["vega"]) & (out["vega"] >= float(min_vega))]
    out = out[np.isfinite(out["open_interest"]) & (out["open_interest"] >= float(min_open_interest))]

    # optional: remove synthetic underlyings
    if drop_synthetic_underlyings:
        if "underlying_name" in out.columns:
            out = out[~out["underlying_name"].astype(str).str.startswith("SYN.")]
        if "futures_instrument_name" in out.columns:
            out = out[~out["futures_instrument_name"].astype(str).str.startswith("SYN.")]

    # clean prices
    out["mid_price_clean"] = 0.5 * (out["bid_price"] + out["ask_price"])
    out["spread"] = out["ask_price"] - out["bid_price"]
    out["rel_spread"] = out["spread"] / np.maximum(out["mid_price_clean"], 1e-18)

    if max_rel_spread is not None:
        out = out[np.isfinite(out["rel_spread"]) & (out["rel_spread"] <= float(max_rel_spread))]

    # per-expiry forward proxy F0 (median across strikes)
    # expiry_datetime should be a stable grouping key
    out["F0"] = out.groupby("expiry_datetime")["futures_price"].transform("median")
    out = out[np.isfinite(out["F0"]) & (out["F0"] > 0.0)]

    out["moneyness"] = out["strike"] / out["F0"]
    out["log_moneyness"] = np.log(out["moneyness"].astype(float))

    if moneyness_range is not None:
        lo, hi = moneyness_range
        out = out[(out["moneyness"] >= float(lo)) & (out["moneyness"] <= float(hi))]

    # normalize option_type
    out["option_type"] = out["option_type"].astype(str).str.lower()
    out = out[out["option_type"].isin(["call", "put"])]

    # sort to make displays deterministic
    out = out.sort_values(["expiry_datetime", "strike", "option_type"]).reset_index(drop=True)
    return out


# -----------------------------
# Pricing helper
# -----------------------------

def _choose_fft_params_for_group(
    base: FFTParams,
    strikes: np.ndarray,
) -> FFTParams:
    """Create FFTParams with `b` centered around the group's strikes."""
    N = base.N
    eta = base.eta
    lam = 2.0 * np.pi / (N * eta)
    logK_center = float(np.log(np.median(strikes)))
    b = logK_center - 0.5 * N * lam
    return FFTParams(N=base.N, alpha=base.alpha, eta=base.eta, b=b, use_simpson=base.use_simpson)


def price_dataframe(
    df: pd.DataFrame,
    model: str,
    params: Dict[str, float],
    *,
    fft_params_base: Optional[FFTParams] = None,
    dynamic_b: bool = True,
) -> np.ndarray:
    """Vectorized pricing of options in df via the FFT pricer.

    Returns model prices in coin units in the same row order as df.
    """
    if fft_params_base is None:
        fft_params_base = FFTParams()

    # Work with positional indices to avoid surprises when df has a non-range index
    df_local = df.reset_index(drop=True)
    prices = np.empty(len(df_local), dtype=float)

    # group by expiry (T and F0 constant) AND option_type for fewer pricer calls
    for (_, opt_type), g in df_local.groupby(["expiry_datetime", "option_type"], sort=False):
        idx_pos = g.index.to_numpy()
        T = float(np.median(g["time_to_maturity"].to_numpy()))
        F0 = float(np.median(g["F0"].to_numpy()))
        K = g["strike"].to_numpy(dtype=float)

        fft_params = _choose_fft_params_for_group(fft_params_base, K) if dynamic_b else fft_params_base
        prices[idx_pos] = price_inverse_option(
            model=model,
            K=K,
            T=T,
            F0=F0,
            params=params,
            option_type=opt_type,
            fft_params=fft_params,
            return_grid=False,
        )

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

    # replace non-finite by 0 weight (effectively drop)
    w = np.where(np.isfinite(w), w, 0.0)
    return w


# ---- parameter transforms (unconstrained -> constrained) ----

def _pack_black(params: Dict[str, float]) -> np.ndarray:
    return np.array([np.log(float(params["sigma"]))], dtype=float)


def _unpack_black(x: np.ndarray) -> Dict[str, float]:
    sigma = float(np.exp(x[0]))
    return {"sigma": sigma}


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
    kappa = float(np.exp(x[0]))
    theta = float(np.exp(x[1]))
    sigma_v = float(np.exp(x[2]))
    rho = float(np.tanh(x[3]))
    v0 = float(np.exp(x[4]))
    return {"kappa": kappa, "theta": theta, "sigma_v": sigma_v, "rho": rho, "v0": v0}


def _pack_svcj(params: Dict[str, float]) -> np.ndarray:
    # use psi = ell_v * rho_j with |psi|<0.99 to guarantee 1 - ell_v*rho_j > 0
    ell_v = float(params["ell_v"])
    rho_j = float(params["rho_j"])
    psi = np.clip(ell_v * rho_j, -0.99, 0.99)
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
            np.arctanh(psi / 0.99),  # inverse of tanh scaling
        ],
        dtype=float,
    )


def _unpack_svcj(x: np.ndarray) -> Dict[str, float]:
    kappa = float(np.exp(x[0]))
    theta = float(np.exp(x[1]))
    sigma_v = float(np.exp(x[2]))
    rho = float(np.tanh(x[3]))
    v0 = float(np.exp(x[4]))
    lam = float(np.exp(x[5]))
    ell_y = float(x[6])
    sigma_y = float(np.exp(x[7]))
    ell_v = float(np.exp(x[8]))
    psi = 0.99 * float(np.tanh(x[9]))  # psi in (-0.99,0.99)
    rho_j = psi / ell_v
    return {
        "kappa": kappa,
        "theta": theta,
        "sigma_v": sigma_v,
        "rho": rho,
        "v0": v0,
        "lam": lam,
        "ell_y": ell_y,
        "sigma_y": sigma_y,
        "ell_v": ell_v,
        "rho_j": rho_j,
    }


def _default_initial_params(df: pd.DataFrame, model: str) -> Dict[str, float]:
    model = model.lower()
    if "implied_volatility" in df.columns and np.isfinite(df["implied_volatility"]).any():
        atm_iv = float(np.nanmedian(pd.to_numeric(df["implied_volatility"], errors="coerce")))
        atm_iv = max(atm_iv, 1e-3)
    else:
        atm_iv = 0.6  # fallback

    if model == "black":
        return {"sigma": atm_iv}

    if model == "heston":
        theta = atm_iv * atm_iv
        return {
            "kappa": 2.0,
            "theta": theta,
            "sigma_v": 0.8,
            "rho": -0.5,
            "v0": theta,
        }

    if model == "svcj":
        theta = atm_iv * atm_iv
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


def calibrate_model(
    df: pd.DataFrame,
    model: str,
    *,
    weight_config: WeightConfig = WeightConfig(),
    fft_params_base: Optional[FFTParams] = None,
    dynamic_b: bool = True,
    initial_params: Optional[Dict[str, float]] = None,
    max_nfev: int = 50,
    verbose: int = 1,
    clear_cache_before: bool = True,
) -> CalibrationResult:
    """Calibrate one model to a cleaned option dataset.

    Assumes df has been prepared by `filter_liquid_options` (so that mid_price_clean,
    spread, F0 exist, and bid/ask quotes are present).

    Returns fitted parameter dict and basic fit diagnostics.
    """
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

    y_mkt = df["mid_price_clean"].to_numpy(dtype=float)
    w = _weights(df, weight_config)

    # drop zero-weight rows defensively
    keep = np.isfinite(y_mkt) & (w > 0.0)
    df_fit = df.loc[keep].copy()
    y_mkt = y_mkt[keep]
    w = w[keep]

    def residuals(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        y_model = price_dataframe(df_fit, model_l, p, fft_params_base=fft_params_base, dynamic_b=dynamic_b)
        r = w * (y_model - y_mkt)
        # ensure finite residual vector
        r = np.where(np.isfinite(r), r, 0.0)
        return r

    res = least_squares(
        residuals,
        x0,
        method="trf",
        max_nfev=int(max_nfev),
        verbose=int(verbose),
    )

    params_hat = unpack(res.x)

    # diagnostics on the fitting set
    y_hat = price_dataframe(df_fit, model_l, params_hat, fft_params_base=fft_params_base, dynamic_b=dynamic_b)
    err = y_hat - y_mkt
    rmse = float(np.sqrt(np.mean(err * err)))
    mae = float(np.mean(np.abs(err)))

    return CalibrationResult(
        model=model_l,
        params=params_hat,
        success=bool(res.success),
        message=str(res.message),
        nfev=int(res.nfev),
        rmse=rmse,
        mae=mae,
    )
