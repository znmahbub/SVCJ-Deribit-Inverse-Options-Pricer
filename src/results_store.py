"""src.results_store

Persistence helpers for calibration outputs.

This module owns the schema for ``calibration_results.xlsx`` and provides:

- creation of an empty workbook (as a dict of DataFrames)
- loading an existing workbook (schema-tolerant)
- appending new rows (params + train/test option rows)
- crash-safe (atomic) flushing to disk

Design goals
------------
- **Resume-safe**: the batch runner uses the latest timestamp present in
  ``black_params`` as the resume key.
- **Robust to schema evolution**: if columns are missing in an existing
  workbook, we add them.
- **Crash-safe writes**: flushing writes to a temp file and atomically replaces
  the target file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Workbook schema
# ---------------------------------------------------------------------------

PARAM_SHEET_BLACK = "black_params"
PARAM_SHEET_HESTON = "heston_params"
PARAM_SHEET_SVCJ = "svcj_params"

TRAIN_SHEET = "train_data"
TEST_SHEET = "test_data"


PARAM_COLS_COMMON = [
    "timestamp",  # snapshot timestamp from filename (ISO string, UTC with trailing 'Z')
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
]

PARAM_COLS_BLACK = PARAM_COLS_COMMON + ["sigma"]
PARAM_COLS_HESTON = PARAM_COLS_COMMON + ["kappa", "theta", "sigma_v", "rho", "v0"]
PARAM_COLS_SVCJ = PARAM_COLS_COMMON + [
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
]


# Preferred (front) columns for train/test sheets; any other columns will be
# appended after these.
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
    "price_black",
    "price_heston",
    "price_svcj",
]


@dataclass(frozen=True)
class WorkbookSchema:
    """Lightweight container for sheet/column definitions."""

    param_sheet_black: str = PARAM_SHEET_BLACK
    param_sheet_heston: str = PARAM_SHEET_HESTON
    param_sheet_svcj: str = PARAM_SHEET_SVCJ
    train_sheet: str = TRAIN_SHEET
    test_sheet: str = TEST_SHEET


def _empty_df(cols: list[str]) -> pd.DataFrame:
    # Keep object dtype to be friendly to Excel roundtrips.
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def init_empty_workbook() -> Dict[str, pd.DataFrame]:
    """Return an empty workbook dict with the expected sheets present."""

    return {
        PARAM_SHEET_BLACK: _empty_df(PARAM_COLS_BLACK),
        PARAM_SHEET_HESTON: _empty_df(PARAM_COLS_HESTON),
        PARAM_SHEET_SVCJ: _empty_df(PARAM_COLS_SVCJ),
        TRAIN_SHEET: pd.DataFrame(),
        TEST_SHEET: pd.DataFrame(),
    }


def normalize_timestamp_series(s: pd.Series) -> pd.Series:
    """Normalize a timestamp series to ISO strings (UTC with trailing 'Z')."""

    dt = pd.to_datetime(s, utc=True, errors="coerce")
    out = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # preserve existing values when parsing fails
    out = out.where(out.notna(), s.astype(str))
    return out


def ensure_param_columns(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Ensure the param sheet has the expected columns (adds missing)."""

    out = df.copy() if df is not None else pd.DataFrame()
    for c in expected_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[expected_cols]
    if "timestamp" in out.columns:
        out["timestamp"] = normalize_timestamp_series(out["timestamp"])
    return out


def order_train_test_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Put important columns first; keep the rest in stable (sorted) order."""

    if df is None or len(df) == 0:
        return df if df is not None else pd.DataFrame()
    front = [c for c in TRAIN_TEST_FRONT_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    rest_sorted = sorted(rest)
    return df[front + rest_sorted]


def load_existing_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    """Load an existing workbook if present, otherwise return an empty one."""

    if not path.exists():
        return init_empty_workbook()

    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    wb = init_empty_workbook()

    # params sheets: enforce schema
    if PARAM_SHEET_BLACK in sheets:
        wb[PARAM_SHEET_BLACK] = ensure_param_columns(sheets[PARAM_SHEET_BLACK], PARAM_COLS_BLACK)
    if PARAM_SHEET_HESTON in sheets:
        wb[PARAM_SHEET_HESTON] = ensure_param_columns(sheets[PARAM_SHEET_HESTON], PARAM_COLS_HESTON)
    if PARAM_SHEET_SVCJ in sheets:
        wb[PARAM_SHEET_SVCJ] = ensure_param_columns(sheets[PARAM_SHEET_SVCJ], PARAM_COLS_SVCJ)

    # train/test: keep as-is; we'll align columns on append
    if TRAIN_SHEET in sheets:
        wb[TRAIN_SHEET] = sheets[TRAIN_SHEET].copy()
        if "snapshot_ts" in wb[TRAIN_SHEET].columns:
            wb[TRAIN_SHEET]["snapshot_ts"] = normalize_timestamp_series(wb[TRAIN_SHEET]["snapshot_ts"])
    if TEST_SHEET in sheets:
        wb[TEST_SHEET] = sheets[TEST_SHEET].copy()
        if "snapshot_ts" in wb[TEST_SHEET].columns:
            wb[TEST_SHEET]["snapshot_ts"] = normalize_timestamp_series(wb[TEST_SHEET]["snapshot_ts"])

    return wb


def get_latest_processed_timestamp(wb: Dict[str, pd.DataFrame], currency: str) -> Optional[pd.Timestamp]:
    """Resume key: latest timestamp present in the black params sheet."""

    df = wb.get(PARAM_SHEET_BLACK, pd.DataFrame())
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    sub = df[df["currency"].astype(str) == str(currency)].copy()
    if sub.empty:
        return None
    dt = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce").dropna()
    if dt.empty:
        return None
    return dt.max()


def append_df(wb: Dict[str, pd.DataFrame], sheet: str, new: pd.DataFrame) -> None:
    """Append a DataFrame to a sheet inside the workbook dict (in-place)."""

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
    """Return latest successful params row before ``ts0`` (as a float dict)."""

    if params_df is None or params_df.empty:
        return None

    sub = params_df.copy()
    sub = sub[sub["currency"].astype(str) == str(currency)]
    if sub.empty:
        return None

    sub_dt = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
    sub = sub.assign(_ts=sub_dt)
    sub = sub[(sub["_ts"].notna()) & (sub["_ts"] < ts0)]
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
    """Write workbook to disk (atomic replace)."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure stable ordering
    wb_to_write = dict(wb)

    wb_to_write[PARAM_SHEET_BLACK] = ensure_param_columns(wb_to_write.get(PARAM_SHEET_BLACK, pd.DataFrame()), PARAM_COLS_BLACK)
    wb_to_write[PARAM_SHEET_HESTON] = ensure_param_columns(wb_to_write.get(PARAM_SHEET_HESTON, pd.DataFrame()), PARAM_COLS_HESTON)
    wb_to_write[PARAM_SHEET_SVCJ] = ensure_param_columns(wb_to_write.get(PARAM_SHEET_SVCJ, pd.DataFrame()), PARAM_COLS_SVCJ)

    # sort param sheets
    for sh in [PARAM_SHEET_BLACK, PARAM_SHEET_HESTON, PARAM_SHEET_SVCJ]:
        df = wb_to_write[sh].copy()
        if len(df):
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values(["currency", "timestamp_dt"]).drop(columns=["timestamp_dt"])
        wb_to_write[sh] = df

    # order train/test columns
    wb_to_write[TRAIN_SHEET] = order_train_test_columns(wb_to_write.get(TRAIN_SHEET, pd.DataFrame()))
    wb_to_write[TEST_SHEET] = order_train_test_columns(wb_to_write.get(TEST_SHEET, pd.DataFrame()))

    # sort train/test rows
    for sh in [TRAIN_SHEET, TEST_SHEET]:
        df = wb_to_write.get(sh, pd.DataFrame()).copy()
        if len(df):
            df["_snapshot_dt"] = pd.to_datetime(df.get("snapshot_ts", pd.Series([pd.NaT] * len(df))), utc=True, errors="coerce")
            df["_expiry_dt"] = pd.to_datetime(df.get("expiry_datetime", pd.Series([pd.NaT] * len(df))), utc=True, errors="coerce")
            sort_cols = ["currency", "_snapshot_dt", "_expiry_dt"]
            if "strike" in df.columns:
                sort_cols.append("strike")
            df = df.sort_values(sort_cols)
            df = df.drop(columns=[c for c in ["_snapshot_dt", "_expiry_dt"] if c in df.columns])
        wb_to_write[sh] = df

    with NamedTemporaryFile("wb", suffix=".xlsx", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            for sheet_name, df in wb_to_write.items():
                if df is None:
                    df = pd.DataFrame()
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


__all__ = [
    "PARAM_SHEET_BLACK",
    "PARAM_SHEET_HESTON",
    "PARAM_SHEET_SVCJ",
    "TRAIN_SHEET",
    "TEST_SHEET",
    "PARAM_COLS_BLACK",
    "PARAM_COLS_HESTON",
    "PARAM_COLS_SVCJ",
    "init_empty_workbook",
    "load_existing_workbook",
    "get_latest_processed_timestamp",
    "latest_successful_params_before",
    "append_df",
    "flush_workbook_atomic",
    "order_train_test_columns",
    "ensure_param_columns",
]
