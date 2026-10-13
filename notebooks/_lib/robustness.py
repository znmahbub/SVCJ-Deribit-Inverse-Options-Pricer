"""Robustness helpers for Chapter 3/4 remediation.

These helpers are deliberately diagnostic. They do not alter the baseline
calibration objective or pricing engine. Discounting scenarios apply a
deterministic discount factor to the regular USD futures-option value and,
therefore, equivalently to the inverse coin premium after division by F0.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_SCENARIO_RATES: tuple[float, ...] = (0.0, 0.025, 0.05, 0.075)
MODEL_KEYS: tuple[str, ...] = ("black", "heston", "svcj")
SVCJ_PARAMETER_COLUMNS: tuple[str, ...] = (
    "kappa",
    "theta",
    "sigma_v",
    "rho",
    "v0",
    "lam",
    "ell_y",
    "sigma_y",
    "ell_v",
    "rho_j",
)


def discount_factor(time_to_maturity, rate: float) -> np.ndarray:
    """Deterministic discount factor ``exp(-rT)`` with T in years."""
    T = np.asarray(time_to_maturity, dtype=float)
    r = float(rate)
    return np.exp(-r * T)


def apply_discounting_to_inverse_price(price_coin_zero_rate, time_to_maturity, rate: float) -> np.ndarray:
    """Apply deterministic discounting to a zero-rate inverse option premium.

    With the dated futures price distribution held fixed, the regular USD
    futures-option value is multiplied by D(0,T). Since ``V_coin = V_usd/F0``
    at the observation time, the inverse premium is multiplied by the same
    deterministic factor. At r=0 this function returns the original numerical
    values exactly (up to NumPy's no-op multiplication semantics).
    """
    price, T = np.broadcast_arrays(
        np.asarray(price_coin_zero_rate, dtype=float),
        np.asarray(time_to_maturity, dtype=float),
    )
    if float(rate) == 0.0:
        return price.copy()
    return price * discount_factor(T, float(rate))


def _finite_rmse(values: pd.Series | np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if len(x) else float("nan")


def _finite_mae(values: pd.Series | np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if len(x) else float("nan")


def build_discounting_sensitivity_table(
    quotes: pd.DataFrame,
    *,
    rates: Sequence[float] = DEFAULT_SCENARIO_RATES,
    models: Sequence[str] = MODEL_KEYS,
) -> pd.DataFrame:
    """Re-score fixed model prices under deterministic discounting scenarios.

    The market quote is kept fixed. This is intentionally a *pricing-convention
    sensitivity* diagnostic, not a recalibration. Rows are reported separately
    by currency and for a mechanical ``All`` scope. The pooled raw-coin scope is
    retained only for inspection and is never used for the promotion decision;
    BTC and ETH coin units are not economically commensurate.
    """
    required = {"time_to_maturity", "mid_price_clean", "bid_price", "ask_price"}
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"discounting sensitivity missing columns: {sorted(missing)}")

    work = quotes.copy()
    work["time_to_maturity"] = pd.to_numeric(work["time_to_maturity"], errors="coerce")
    work["mid_price_clean"] = pd.to_numeric(work["mid_price_clean"], errors="coerce")
    work["bid_price"] = pd.to_numeric(work["bid_price"], errors="coerce")
    work["ask_price"] = pd.to_numeric(work["ask_price"], errors="coerce")
    if "currency" not in work.columns:
        work["currency"] = "All"
    work["currency"] = work["currency"].astype(str).str.upper()

    rows: list[dict[str, float | int | str]] = []
    scopes = [(str(c), g.copy()) for c, g in work.groupby("currency", sort=True)]
    scopes.append(("All", work))

    for scope, frame in scopes:
        T = frame["time_to_maturity"].to_numpy(dtype=float)
        market = frame["mid_price_clean"].to_numpy(dtype=float)
        bid = frame["bid_price"].to_numpy(dtype=float)
        ask = frame["ask_price"].to_numpy(dtype=float)
        valid_market = np.isfinite(market) & np.isfinite(bid) & np.isfinite(ask) & (ask >= bid)
        for model in models:
            col = f"price_{model}"
            if col not in frame.columns:
                continue
            base = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            for rate in rates:
                shifted = apply_discounting_to_inverse_price(base, T, float(rate))
                valid = valid_market & np.isfinite(shifted)
                err = shifted[valid] - market[valid]
                base_valid = base[valid]
                within = (shifted[valid] >= bid[valid]) & (shifted[valid] <= ask[valid])
                rows.append(
                    {
                        "scope": scope,
                        "model": str(model).lower(),
                        "rate": float(rate),
                        "n": int(valid.sum()),
                        "rmse_coin": _finite_rmse(err),
                        "mae_coin": _finite_mae(err),
                        "within_spread_share": float(np.mean(within)) if len(within) else float("nan"),
                        "mean_abs_price_shift_coin": _finite_mae(shifted[valid] - base_valid),
                        "max_abs_price_shift_coin": float(np.max(np.abs(shifted[valid] - base_valid))) if len(base_valid) else float("nan"),
                    }
                )
    return pd.DataFrame(rows).sort_values(["scope", "model", "rate"]).reset_index(drop=True)


def assess_discounting_promotion(
    sensitivity: pd.DataFrame,
    *,
    scope: str,
    rmse_relative_threshold: float = 0.05,
    within_spread_absolute_threshold: float = 0.02,
) -> dict[str, object]:
    """Apply the pre-Task-13 discounting rule within one currency.

    BTC and ETH coin RMSE values are different units, so a cross-asset raw-coin
    RMSE must never decide whether discounting enters the baseline. Call this
    function separately for BTC and ETH. Promotion is triggered if a plausible
    positive-rate scenario reverses the within-currency zero-rate RMSE ranking,
    changes any model's RMSE by at least 5% relative, or changes within-spread
    share by at least 2 percentage points.
    """
    scope_name = str(scope).upper()
    if scope_name in {"ALL", "POOLED"}:
        raise ValueError(
            "cross-asset raw coin RMSE is not a valid discounting promotion criterion; "
            "assess BTC and ETH separately"
        )
    frame = sensitivity.loc[sensitivity["scope"].astype(str).str.upper() == scope_name].copy()
    if frame.empty:
        raise ValueError(f"no discounting rows for scope={scope!r}")
    base = frame.loc[frame["rate"] == 0.0].set_index("model")
    if base.empty:
        raise ValueError("discounting table has no zero-rate baseline")
    base_rank = tuple(base.sort_values("rmse_coin").index.astype(str))

    ranking_reversal = False
    max_rmse_relative_change = 0.0
    max_within_abs_change = 0.0
    for _, group in frame.loc[frame["rate"] > 0.0].groupby("rate"):
        current = group.set_index("model")
        common = base.index.intersection(current.index)
        if len(common) < 2:
            continue
        current_rank = tuple(current.loc[common].sort_values("rmse_coin").index.astype(str))
        base_common_rank = tuple(base.loc[common].sort_values("rmse_coin").index.astype(str))
        ranking_reversal |= current_rank != base_common_rank
        denom = base.loc[common, "rmse_coin"].abs().replace(0.0, np.nan)
        rel = ((current.loc[common, "rmse_coin"] - base.loc[common, "rmse_coin"]).abs() / denom).replace([np.inf, -np.inf], np.nan)
        if rel.notna().any():
            max_rmse_relative_change = max(max_rmse_relative_change, float(rel.max()))
        within_delta = (current.loc[common, "within_spread_share"] - base.loc[common, "within_spread_share"]).abs()
        if within_delta.notna().any():
            max_within_abs_change = max(max_within_abs_change, float(within_delta.max()))

    promote = bool(
        ranking_reversal
        or max_rmse_relative_change >= float(rmse_relative_threshold)
        or max_within_abs_change >= float(within_spread_absolute_threshold)
    )
    return {
        "scope": scope_name,
        "zero_rate_rmse_ranking": base_rank,
        "ranking_reversal": bool(ranking_reversal),
        "max_rmse_relative_change": float(max_rmse_relative_change),
        "max_within_spread_absolute_change": float(max_within_abs_change),
        "rmse_relative_threshold": float(rmse_relative_threshold),
        "within_spread_absolute_threshold": float(within_spread_absolute_threshold),
        "promote_discounting_to_baseline": promote,
    }


def parameter_identification_summary(
    svcj_params: pd.DataFrame,
    *,
    rho_boundary: float = 0.95,
    sigma_y_near_lower: float = 5e-4,
    lambda_near_lower: float = 5e-5,
    lambda_near_upper: float = 9.5,
) -> pd.DataFrame:
    """Summarise SVCJ identification/admissibility diagnostics by currency.

    ``rho_j`` is intentionally *not* assigned a boundary rate: it is an
    unbounded jump-size loading. ``rho_j_abs_gt_one_share`` is descriptive only
    and must not be interpreted as a constraint violation.
    """
    work = svcj_params.copy()
    if "success" in work.columns:
        work = work.loc[work["success"] == True].copy()
    if "currency" not in work.columns:
        work["currency"] = "All"
    work["currency"] = work["currency"].astype(str).str.upper()

    def num(frame: pd.DataFrame, col: str) -> pd.Series:
        if col not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[col], errors="coerce")

    rows: list[dict[str, object]] = []
    for currency, frame in work.groupby("currency", sort=True):
        rho = num(frame, "rho")
        rho_j = num(frame, "rho_j")
        sigma_y = num(frame, "sigma_y")
        lam = num(frame, "lam")
        feller_gap = num(frame, "feller_gap")
        if "feller_violated" in frame.columns:
            feller_violation = pd.to_numeric(frame["feller_violated"], errors="coerce")
        else:
            feller_violation = (feller_gap < 0.0).astype(float)
        mart = num(frame, "svcj_martingale_margin")
        contour = num(frame, "svcj_contour_margin")
        rows.append(
            {
                "currency": currency,
                "n_success": int(len(frame)),
                "feller_violation_share": float(feller_violation.mean()) if len(frame) else float("nan"),
                "feller_gap_median": float(feller_gap.median()),
                "feller_gap_min": float(feller_gap.min()),
                "rho_boundary_share": float((rho.abs() > float(rho_boundary)).mean()),
                "rho_j_abs_gt_one_share": float((rho_j.abs() > 1.0).mean()),
                "rho_j_median": float(rho_j.median()),
                "rho_j_min": float(rho_j.min()),
                "rho_j_max": float(rho_j.max()),
                "sigma_y_near_lower_bound_share": float((sigma_y <= float(sigma_y_near_lower)).mean()),
                "lambda_near_lower_bound_share": float((lam <= float(lambda_near_lower)).mean()),
                "lambda_near_upper_bound_share": float((lam >= float(lambda_near_upper)).mean()),
                "martingale_margin_median": float(mart.median()),
                "martingale_margin_min": float(mart.min()),
                "contour_margin_median": float(contour.median()),
                "contour_margin_min": float(contour.min()),
            }
        )
    return pd.DataFrame(rows)


def parameter_correlation_matrix(
    svcj_params: pd.DataFrame,
    *,
    currency: str,
    columns: Iterable[str] = SVCJ_PARAMETER_COLUMNS,
) -> pd.DataFrame:
    """Pearson parameter correlation matrix for successful SVCJ snapshots."""
    work = svcj_params.copy()
    if "success" in work.columns:
        work = work.loc[work["success"] == True].copy()
    if "currency" in work.columns:
        work = work.loc[work["currency"].astype(str).str.upper() == str(currency).upper()].copy()
    cols = [c for c in columns if c in work.columns]
    if not cols:
        return pd.DataFrame()
    numeric = work[cols].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="pearson")


def compare_identification_summaries(
    corrected_params: pd.DataFrame,
    legacy_params: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Side-by-side legacy/corrected identification summary when both exist."""
    corrected = parameter_identification_summary(corrected_params).add_prefix("corrected_")
    corrected = corrected.rename(columns={"corrected_currency": "currency"})
    if legacy_params is None:
        return corrected
    legacy = parameter_identification_summary(legacy_params).add_prefix("legacy_")
    legacy = legacy.rename(columns={"legacy_currency": "currency"})
    return legacy.merge(corrected, on="currency", how="outer")


__all__ = [
    "DEFAULT_SCENARIO_RATES",
    "MODEL_KEYS",
    "discount_factor",
    "apply_discounting_to_inverse_price",
    "build_discounting_sensitivity_table",
    "assess_discounting_promotion",
    "parameter_identification_summary",
    "parameter_correlation_matrix",
    "compare_identification_summaries",
]
