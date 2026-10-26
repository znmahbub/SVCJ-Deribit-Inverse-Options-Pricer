"""Persistence helpers for calibration outputs.

New calibration workbooks contain model parameter sheets plus deterministic
``train_data`` and ``test_data`` option sheets. Legacy three-way workbooks are
read tolerantly; their validation sheet and validation-only parameter columns
are ignored by the active schema.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

import numpy as np
import pandas as pd


PARAM_SHEET_BLACK = "black_params"
PARAM_SHEET_HESTON = "heston_params"
PARAM_SHEET_SVCJ = "svcj_params"
TRAIN_SHEET = "train_data"
TEST_SHEET = "test_data"

# Legacy name retained only so old analysis modules can import it while the
# repository is migrated. New workbooks never create or write this sheet.
VALIDATION_SHEET = "validation_data"


PARAM_COLS_COMMON = [
    "timestamp",
    "currency",
    "success",
    "message",
    "nfev",
    "rmse_fit",
    "mae_fit",
    "rmse_train",
    "mae_train",
    "rmse_test",
    "mae_test",
    "n_options_total",
    "n_train",
    "n_test",
    "random_seed",
    "regularization_active",
    "l2_prev_strength",
    "feller_penalty_active",
    "feller_gap",
    "feller_violated",
    "svcj_moment_penalty_active",
    "svcj_martingale_margin",
    "svcj_martingale_violated",
    "svcj_contour_margin",
    "svcj_contour_violated",
]

PARAM_COLS_BLACK = PARAM_COLS_COMMON + ["sigma"]
PARAM_COLS_HESTON = PARAM_COLS_COMMON + ["kappa", "theta", "sigma_v", "rho", "v0"]
PARAM_COLS_SVCJ = PARAM_COLS_COMMON + [
    "kappa", "theta", "sigma_v", "rho", "v0",
    "lam", "ell_y", "sigma_y", "ell_v", "rho_j",
]

TRAIN_TEST_FRONT_COLS = [
    "snapshot_ts",
    "currency",
    "instrument_name",
    "option_type",
    "strike",
    "expiry_datetime",
    "time_to_maturity",
    "futures_price",
    "bid_price",
    "ask_price",
    "mid_price_clean",
    "rel_spread",
    "open_interest",
    "vega",
    "random_seed",
    "split",
    "price_black",
    "price_heston",
    "price_svcj",
]


@dataclass(frozen=True)
class WorkbookSchema:
    param_sheet_black: str = PARAM_SHEET_BLACK
    param_sheet_heston: str = PARAM_SHEET_HESTON
    param_sheet_svcj: str = PARAM_SHEET_SVCJ
    train_sheet: str = TRAIN_SHEET
    test_sheet: str = TEST_SHEET


def _empty_df(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def init_empty_workbook() -> Dict[str, pd.DataFrame]:
    """Return an empty workbook using the active train/test schema."""
    return {
        PARAM_SHEET_BLACK: _empty_df(PARAM_COLS_BLACK),
        PARAM_SHEET_HESTON: _empty_df(PARAM_COLS_HESTON),
        PARAM_SHEET_SVCJ: _empty_df(PARAM_COLS_SVCJ),
        TRAIN_SHEET: pd.DataFrame(),
        TEST_SHEET: pd.DataFrame(),
    }


def normalize_timestamp_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    out = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out.where(out.notna(), s.astype(str))


def ensure_param_columns(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Normalize a parameter sheet to the active schema, dropping legacy extras."""
    out = df.copy() if df is not None else pd.DataFrame()
    for c in expected_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[expected_cols]
    if "timestamp" in out.columns:
        out["timestamp"] = normalize_timestamp_series(out["timestamp"])
    return out


def order_train_test_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()
    front = [c for c in TRAIN_TEST_FRONT_COLS if c in df.columns]
    rest = sorted(c for c in df.columns if c not in front)
    return df[front + rest]


def load_existing_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    """Load an active or legacy workbook into the train/test schema."""
    if not path.exists():
        return init_empty_workbook()

    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    wb = init_empty_workbook()
    for sheet_name, expected_cols in [
        (PARAM_SHEET_BLACK, PARAM_COLS_BLACK),
        (PARAM_SHEET_HESTON, PARAM_COLS_HESTON),
        (PARAM_SHEET_SVCJ, PARAM_COLS_SVCJ),
    ]:
        if sheet_name in sheets:
            wb[sheet_name] = ensure_param_columns(sheets[sheet_name], expected_cols)

    # Legacy validation_data, if present, is deliberately not carried forward.
    for sheet_name in (TRAIN_SHEET, TEST_SHEET):
        if sheet_name in sheets:
            wb[sheet_name] = sheets[sheet_name].copy()
            if "snapshot_ts" in wb[sheet_name].columns:
                wb[sheet_name]["snapshot_ts"] = normalize_timestamp_series(
                    wb[sheet_name]["snapshot_ts"]
                )
    return wb


def get_latest_processed_timestamp(wb: Dict[str, pd.DataFrame], currency: str) -> Optional[pd.Timestamp]:
    df = wb.get(PARAM_SHEET_BLACK, pd.DataFrame())
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    sub = df[df["currency"].astype(str) == str(currency)].copy()
    if sub.empty:
        return None
    dt = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce").dropna()
    return None if dt.empty else dt.max()


def append_df(wb: Dict[str, pd.DataFrame], sheet: str, new: pd.DataFrame) -> None:
    if new is None or len(new) == 0:
        return
    if sheet not in wb or wb[sheet] is None or wb[sheet].empty:
        wb[sheet] = new.copy()
        return
    wb[sheet] = pd.concat([wb[sheet], new], ignore_index=True, sort=False)


def latest_successful_params_before(
    params_df: pd.DataFrame,
    *,
    currency: str,
    ts0: pd.Timestamp,
    required_cols: list[str],
) -> Optional[dict[str, float]]:
    if params_df is None or params_df.empty:
        return None
    sub = params_df.copy()
    sub = sub[sub["currency"].astype(str) == str(currency)]
    if sub.empty:
        return None
    sub = sub.assign(_ts=pd.to_datetime(sub["timestamp"], utc=True, errors="coerce"))
    sub = sub[sub["_ts"].notna() & (sub["_ts"] < ts0)]
    if "success" in sub.columns:
        sub = sub[sub["success"] == True]  # noqa: E712
    if sub.empty:
        return None
    row = sub.sort_values("_ts").iloc[-1]
    out: dict[str, float] = {}
    for c in required_cols:
        if c in row.index and pd.notna(row[c]):
            try:
                out[c] = float(row[c])
            except Exception:
                pass
    return out if out else None


def flush_workbook_atomic(wb: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Write the active train/test workbook schema atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Explicit sheet set prevents legacy validation sheets from being rewritten.
    wb_to_write: Dict[str, pd.DataFrame] = {
        PARAM_SHEET_BLACK: ensure_param_columns(wb.get(PARAM_SHEET_BLACK, pd.DataFrame()), PARAM_COLS_BLACK),
        PARAM_SHEET_HESTON: ensure_param_columns(wb.get(PARAM_SHEET_HESTON, pd.DataFrame()), PARAM_COLS_HESTON),
        PARAM_SHEET_SVCJ: ensure_param_columns(wb.get(PARAM_SHEET_SVCJ, pd.DataFrame()), PARAM_COLS_SVCJ),
        TRAIN_SHEET: order_train_test_columns(wb.get(TRAIN_SHEET, pd.DataFrame())),
        TEST_SHEET: order_train_test_columns(wb.get(TEST_SHEET, pd.DataFrame())),
    }

    for sh in (PARAM_SHEET_BLACK, PARAM_SHEET_HESTON, PARAM_SHEET_SVCJ):
        df = wb_to_write[sh].copy()
        if len(df):
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values(["currency", "timestamp_dt"]).drop(columns=["timestamp_dt"])
        wb_to_write[sh] = df

    for sh in (TRAIN_SHEET, TEST_SHEET):
        df = wb_to_write[sh].copy()
        if len(df):
            df["_snapshot_dt"] = pd.to_datetime(
                df.get("snapshot_ts", pd.Series([pd.NaT] * len(df))),
                utc=True,
                errors="coerce",
            )
            df["_expiry_dt"] = pd.to_datetime(
                df.get("expiry_datetime", pd.Series([pd.NaT] * len(df))),
                utc=True,
                errors="coerce",
            )
            sort_cols = ["currency", "_snapshot_dt", "_expiry_dt"]
            if "strike" in df.columns:
                sort_cols.append("strike")
            df = df.sort_values(sort_cols).drop(columns=["_snapshot_dt", "_expiry_dt"])
        wb_to_write[sh] = df

    with NamedTemporaryFile("wb", suffix=".xlsx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            for sheet_name, df in wb_to_write.items():
                (df if df is not None else pd.DataFrame()).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


__all__ = [
    "PARAM_SHEET_BLACK", "PARAM_SHEET_HESTON", "PARAM_SHEET_SVCJ",
    "TRAIN_SHEET", "TEST_SHEET", "VALIDATION_SHEET",
    "PARAM_COLS_BLACK", "PARAM_COLS_HESTON", "PARAM_COLS_SVCJ",
    "init_empty_workbook", "load_existing_workbook",
    "get_latest_processed_timestamp", "latest_successful_params_before",
    "append_df", "flush_workbook_atomic", "order_train_test_columns",
    "ensure_param_columns",
]
