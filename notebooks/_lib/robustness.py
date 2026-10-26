"""SVCJ identification and admissibility diagnostics for the remediation analysis.

The helpers in this module inspect fitted parameter paths only. They do not
alter calibration, pricing, or hedging conventions.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

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
    "SVCJ_PARAMETER_COLUMNS",
    "parameter_identification_summary",
    "parameter_correlation_matrix",
    "compare_identification_summaries",
]
