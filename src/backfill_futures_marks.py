#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

DERIBIT_BASE = "https://www.deribit.com/api/v2"
SNAPSHOT_GLOB = "deribit_options_snapshot_*.csv"
TERM_OUTPUT_NAME = "term_futures_marks.csv"
PERP_OUTPUT_NAME = "perpetual_futures_prices.csv"
COMBOS_OUTPUT_NAME = "snapshot_currency_combos.csv"

DEFAULT_RESOLUTION = "1"
DEFAULT_MAX_BARS_PER_CALL = 5000
DEFAULT_PERP_MAX_GAP_SECONDS = 1800.0
DEFAULT_TERM_MAX_GAP_SECONDS = 1800.0


# -----------------------------
# Basic helpers
# -----------------------------
def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def parse_iso_to_utc(ts: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(str(ts))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def iso_to_ms(ts: str) -> int:
    return int(parse_iso_to_utc(ts).timestamp() * 1000)


def resolution_to_ms(resolution: str) -> int:
    res = str(resolution).upper()
    if res == "1D":
        return 86_400_000
    return int(res) * 60_000


def http_get(
    session: requests.Session,
    path: str,
    params: Dict[str, Any] | None = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    url = f"{DERIBIT_BASE}{path}"
    response = session.get(url, params=params or {}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"Deribit API error for {path}: {payload['error']}")
    return payload


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _normalize_key_value(x: Any) -> str:
    if pd.isna(x):
        return "<NA>"
    return str(x)


def dataframe_key_tuples(df: pd.DataFrame, key_cols: list[str]) -> set[tuple[str, ...]]:
    if df.empty:
        return set()
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        return set()
    return {
        tuple(_normalize_key_value(v) for v in row)
        for row in df[key_cols].itertuples(index=False, name=None)
    }


def filter_new_rows(df: pd.DataFrame, existing_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    existing_keys = dataframe_key_tuples(existing_df, key_cols)
    if not existing_keys:
        return df.copy()
    mask = [
        tuple(_normalize_key_value(v) for v in row) not in existing_keys
        for row in df[key_cols].itertuples(index=False, name=None)
    ]
    return df.loc[mask].copy()


def select_rows_by_keys(df: pd.DataFrame, key_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if df.empty or key_df.empty:
        return df.iloc[0:0].copy()
    key_set = dataframe_key_tuples(key_df, key_cols)
    mask = [
        tuple(_normalize_key_value(v) for v in row) in key_set
        for row in df[key_cols].itertuples(index=False, name=None)
    ]
    return df.loc[mask].copy()


def exclude_rows_by_keys(df: pd.DataFrame, key_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if df.empty or key_df.empty:
        return df.copy()
    key_set = dataframe_key_tuples(key_df, key_cols)
    mask = [
        tuple(_normalize_key_value(v) for v in row) not in key_set
        for row in df[key_cols].itertuples(index=False, name=None)
    ]
    return df.loc[mask].copy()


def concat_dedup_sort(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    key_cols: list[str],
    sort_cols: list[str],
) -> pd.DataFrame:
    parts = [df for df in [existing_df, new_df] if not df.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    if all(c in out.columns for c in key_cols):
        out = out.drop_duplicates(subset=key_cols, keep="first")
    sort_cols_present = [c for c in sort_cols if c in out.columns]
    if sort_cols_present:
        out = out.sort_values(sort_cols_present).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


# -----------------------------
# Snapshot parsing
# -----------------------------
def list_snapshot_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.glob(SNAPSHOT_GLOB) if p.is_file())


def read_snapshot_minimal(path: Path) -> pd.DataFrame:
    wanted = [
        "timestamp",
        "currency",
        "futures_price",
        "futures_instrument_name",
        "underlying",
        "underlying_name",
    ]
    df = pd.read_csv(path)
    present = [c for c in wanted if c in df.columns]
    if "timestamp" not in df.columns or "currency" not in df.columns:
        raise ValueError(f"{path.name} is missing required columns timestamp/currency")
    return df[present].copy()


def build_snapshot_currency_combos(snapshot_files: Iterable[Path]) -> pd.DataFrame:
    snapshot_files = list(snapshot_files)
    chunks: list[pd.DataFrame] = []

    for path in tqdm(snapshot_files, desc="Scanning snapshots", unit="file"):
        df = read_snapshot_minimal(path)
        cur = df[["timestamp", "currency"]].dropna().drop_duplicates().copy()
        cur["snapshot_file"] = path.name
        cur["snapshot_timestamp_ms"] = cur["timestamp"].map(iso_to_ms)
        chunks.append(cur)

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "currency", "snapshot_file", "snapshot_timestamp_ms"])

    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values(["timestamp", "currency", "snapshot_file"]).reset_index(drop=True)


def build_term_snapshot_requests(
    snapshot_files: Iterable[Path],
    include_synthetics: bool = False,
) -> pd.DataFrame:
    snapshot_files = list(snapshot_files)
    rows: list[pd.DataFrame] = []

    for path in tqdm(snapshot_files, desc="Extracting term futures", unit="file"):
        df = read_snapshot_minimal(path)

        if "futures_instrument_name" not in df.columns:
            continue

        work = df[["timestamp", "currency", "futures_instrument_name"]].copy()
        work = work.dropna(subset=["timestamp", "currency", "futures_instrument_name"])

        if "futures_price" in df.columns:
            work["snapshot_term_mark_price"] = pd.to_numeric(df["futures_price"], errors="coerce")
        else:
            work["snapshot_term_mark_price"] = np.nan

        if not include_synthetics:
            work = work[~work["futures_instrument_name"].astype(str).str.startswith("SYN.")]

        if work.empty:
            continue

        grouped = (
            work.groupby(["timestamp", "currency", "futures_instrument_name"], as_index=False)
            .agg(
                snapshot_term_mark_price=("snapshot_term_mark_price", "median"),
                snapshot_term_mark_min=("snapshot_term_mark_price", "min"),
                snapshot_term_mark_max=("snapshot_term_mark_price", "max"),
                contributing_option_rows=("futures_instrument_name", "size"),
                distinct_prices_seen=("snapshot_term_mark_price", "nunique"),
            )
            .copy()
        )
        grouped["snapshot_file"] = path.name
        grouped["snapshot_timestamp_ms"] = grouped["timestamp"].map(iso_to_ms)
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "currency",
                "futures_instrument_name",
                "snapshot_term_mark_price",
                "snapshot_term_mark_min",
                "snapshot_term_mark_max",
                "contributing_option_rows",
                "distinct_prices_seen",
                "snapshot_file",
                "snapshot_timestamp_ms",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["timestamp", "currency", "futures_instrument_name"]).reset_index(drop=True)


# -----------------------------
# Candle fetching and matching
# -----------------------------
def fetch_tradingview_chart_chunk(
    session: requests.Session,
    instrument_name: str,
    start_ms: int,
    end_ms: int,
    resolution: str,
) -> pd.DataFrame:
    payload = http_get(
        session,
        "/public/get_tradingview_chart_data",
        {
            "instrument_name": instrument_name,
            "start_timestamp": int(start_ms),
            "end_timestamp": int(end_ms),
            "resolution": str(resolution),
        },
    )
    result = payload.get("result", {})

    if result.get("status") == "no_data":
        return pd.DataFrame(columns=["tick", "open", "high", "low", "close", "volume", "cost"])

    ticks = result.get("ticks", []) or []
    if not ticks:
        return pd.DataFrame(columns=["tick", "open", "high", "low", "close", "volume", "cost"])

    n = len(ticks)

    def _arr(name: str) -> list[Any]:
        arr = result.get(name, []) or []
        if len(arr) < n:
            arr = list(arr) + [None] * (n - len(arr))
        return arr[:n]

    return pd.DataFrame(
        {
            "tick": ticks,
            "open": _arr("open"),
            "high": _arr("high"),
            "low": _arr("low"),
            "close": _arr("close"),
            "volume": _arr("volume"),
            "cost": _arr("cost"),
        }
    )


def fetch_tradingview_candles_for_requests(
    request_rows: pd.DataFrame,
    instrument_col: str,
    timestamp_ms_col: str,
    resolution: str,
    sleep_seconds: float,
    max_gap_seconds: Optional[float],
    max_bars_per_call: int,
    progress_desc: str,
) -> pd.DataFrame:
    if request_rows.empty:
        return pd.DataFrame()

    resolution_ms = resolution_to_ms(resolution)
    chunk_ms = max(int(max_bars_per_call), 1) * resolution_ms

    session = requests.Session()
    session.headers.update({"User-Agent": "SVCJ-Deribit-Pricer/1.0 (candle backfill)"})

    req = request_rows.copy()
    req["__row_id"] = np.arange(len(req))
    all_rows: list[pd.DataFrame] = []
    instruments = sorted(req[instrument_col].astype(str).dropna().unique().tolist())

    for instrument_name in tqdm(instruments, desc=progress_desc, unit="inst"):
        sub = req[req[instrument_col].astype(str) == instrument_name].copy()
        sub[timestamp_ms_col] = pd.to_numeric(sub[timestamp_ms_col], errors="coerce")
        sub = sub.dropna(subset=[timestamp_ms_col]).sort_values(timestamp_ms_col)

        if sub.empty:
            continue

        min_ts = int(sub[timestamp_ms_col].min())
        max_ts = int(sub[timestamp_ms_col].max())
        start_ms = max(0, min_ts - resolution_ms)
        end_ms = max_ts + resolution_ms

        candle_parts: list[pd.DataFrame] = []
        chunk_start = start_ms

        while chunk_start <= end_ms:
            chunk_end = min(chunk_start + chunk_ms - resolution_ms, end_ms)
            candle_chunk = fetch_tradingview_chart_chunk(
                session=session,
                instrument_name=instrument_name,
                start_ms=chunk_start,
                end_ms=chunk_end,
                resolution=resolution,
            )
            if not candle_chunk.empty:
                candle_parts.append(candle_chunk)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if chunk_end >= end_ms:
                break
            chunk_start = chunk_end + resolution_ms

        candles = (
            pd.concat(candle_parts, ignore_index=True).drop_duplicates(subset=["tick"]).sort_values("tick")
            if candle_parts
            else pd.DataFrame(columns=["tick", "open", "high", "low", "close", "volume", "cost"])
        )

        if candles.empty:
            tmp = sub.copy()
            tmp["candle_found"] = False
            tmp["candle_tick_ms"] = pd.NA
            tmp["candle_open_time_ms"] = pd.NA
            tmp["candle_close_time_ms"] = pd.NA
            tmp["candle_gap_open_seconds"] = pd.NA
            tmp["candle_gap_close_seconds"] = pd.NA
            tmp["candle_price_choice"] = pd.NA
            tmp["candle_price_timestamp_ms"] = pd.NA
            tmp["candle_price_gap_seconds"] = pd.NA
            tmp["candle_open"] = pd.NA
            tmp["candle_high"] = pd.NA
            tmp["candle_low"] = pd.NA
            tmp["candle_close"] = pd.NA
            tmp["candle_volume"] = pd.NA
            tmp["candle_cost"] = pd.NA
            tmp["accepted_under_max_gap"] = False if max_gap_seconds is not None else pd.NA
            all_rows.append(tmp)
            continue

        candles = candles.copy()
        candles["candle_tick_ms"] = candles["tick"]
        candles["candle_open_time_ms"] = candles["tick"]
        candles["candle_close_time_ms"] = candles["tick"] + resolution_ms

        merged = pd.merge_asof(
            sub[["__row_id", instrument_col, timestamp_ms_col]].sort_values(timestamp_ms_col),
            candles[
                [
                    "candle_tick_ms",
                    "candle_open_time_ms",
                    "candle_close_time_ms",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "cost",
                ]
            ],
            left_on=timestamp_ms_col,
            right_on="candle_tick_ms",
            direction="backward",
            allow_exact_matches=True,
        )

        merged["candle_found"] = merged["candle_tick_ms"].notna()
        merged["candle_gap_open_seconds"] = np.where(
            merged["candle_found"],
            np.abs(merged[timestamp_ms_col] - merged["candle_open_time_ms"]) / 1000.0,
            np.nan,
        )
        merged["candle_gap_close_seconds"] = np.where(
            merged["candle_found"],
            np.abs(merged[timestamp_ms_col] - merged["candle_close_time_ms"]) / 1000.0,
            np.nan,
        )

        choose_open = merged["candle_gap_open_seconds"] <= merged["candle_gap_close_seconds"]
        merged["candle_price_choice"] = np.where(choose_open, "open", "close")
        merged["candle_price_timestamp_ms"] = np.where(
            choose_open,
            merged["candle_open_time_ms"],
            merged["candle_close_time_ms"],
        )
        merged["candle_price_gap_seconds"] = np.where(
            choose_open,
            merged["candle_gap_open_seconds"],
            merged["candle_gap_close_seconds"],
        )

        if max_gap_seconds is None:
            merged["accepted_under_max_gap"] = merged["candle_found"]
        else:
            merged["accepted_under_max_gap"] = merged["candle_found"] & (
                merged["candle_price_gap_seconds"] <= float(max_gap_seconds)
            )

        merged = merged.rename(
            columns={
                "open": "candle_open",
                "high": "candle_high",
                "low": "candle_low",
                "close": "candle_close",
                "volume": "candle_volume",
                "cost": "candle_cost",
            }
        )

        out = sub.merge(
            merged[
                [
                    "__row_id",
                    "candle_found",
                    "candle_tick_ms",
                    "candle_open_time_ms",
                    "candle_close_time_ms",
                    "candle_gap_open_seconds",
                    "candle_gap_close_seconds",
                    "candle_price_choice",
                    "candle_price_timestamp_ms",
                    "candle_price_gap_seconds",
                    "candle_open",
                    "candle_high",
                    "candle_low",
                    "candle_close",
                    "candle_volume",
                    "candle_cost",
                    "accepted_under_max_gap",
                ]
            ],
            on="__row_id",
            how="left",
        )
        all_rows.append(out)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True, sort=False)
    out = out.drop(columns=["__row_id"], errors="ignore")
    out["candle_resolution"] = str(resolution)
    out["candle_price_used"] = np.where(
        out["candle_price_choice"].astype(str).eq("open"),
        pd.to_numeric(out["candle_open"], errors="coerce"),
        pd.to_numeric(out["candle_close"], errors="coerce"),
    )
    return out


# -----------------------------
# Incremental builders
# -----------------------------
def build_term_incremental(
    term_requests: pd.DataFrame,
    existing_term: pd.DataFrame,
    skip_term_backfill: bool,
    sleep_seconds: float,
    max_gap_seconds: Optional[float],
    term_resolution: str,
    term_max_bars_per_call: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["timestamp", "currency", "futures_instrument_name"]
    new_term = filter_new_rows(term_requests, existing_term, key_cols)
    if new_term.empty:
        return existing_term.copy(), new_term

    term_new = new_term.copy()

    if skip_term_backfill:
        for col in [
            "term_candle_found",
            "term_candle_tick_ms",
            "term_candle_open_time_ms",
            "term_candle_close_time_ms",
            "term_candle_gap_open_seconds",
            "term_candle_gap_close_seconds",
            "term_candle_price_choice",
            "term_candle_price_timestamp_ms",
            "term_candle_gap_seconds",
            "term_candle_open",
            "term_candle_high",
            "term_candle_low",
            "term_candle_close",
            "term_candle_volume",
            "term_candle_cost",
            "term_candle_resolution",
            "term_candle_price_used",
            "term_candle_accepted_under_max_gap",
        ]:
            term_new[col] = pd.NA
    else:
        candle_requests = term_new[
            ["timestamp", "currency", "futures_instrument_name", "snapshot_timestamp_ms", "snapshot_file"]
        ].copy()
        term_candles = fetch_tradingview_candles_for_requests(
            request_rows=candle_requests.rename(columns={"futures_instrument_name": "instrument_name"}),
            instrument_col="instrument_name",
            timestamp_ms_col="snapshot_timestamp_ms",
            resolution=term_resolution,
            sleep_seconds=sleep_seconds,
            max_gap_seconds=max_gap_seconds,
            max_bars_per_call=term_max_bars_per_call,
            progress_desc="Fetching term candles",
        )
        if not term_candles.empty:
            term_candles = term_candles.rename(
                columns={
                    "instrument_name": "futures_instrument_name",
                    "candle_found": "term_candle_found",
                    "candle_tick_ms": "term_candle_tick_ms",
                    "candle_open_time_ms": "term_candle_open_time_ms",
                    "candle_close_time_ms": "term_candle_close_time_ms",
                    "candle_gap_open_seconds": "term_candle_gap_open_seconds",
                    "candle_gap_close_seconds": "term_candle_gap_close_seconds",
                    "candle_price_choice": "term_candle_price_choice",
                    "candle_price_timestamp_ms": "term_candle_price_timestamp_ms",
                    "candle_price_gap_seconds": "term_candle_gap_seconds",
                    "candle_open": "term_candle_open",
                    "candle_high": "term_candle_high",
                    "candle_low": "term_candle_low",
                    "candle_close": "term_candle_close",
                    "candle_volume": "term_candle_volume",
                    "candle_cost": "term_candle_cost",
                    "candle_resolution": "term_candle_resolution",
                    "candle_price_used": "term_candle_price_used",
                    "accepted_under_max_gap": "term_candle_accepted_under_max_gap",
                }
            )

            keep_cols = [
                "timestamp",
                "currency",
                "snapshot_file",
                "futures_instrument_name",
                "term_candle_found",
                "term_candle_tick_ms",
                "term_candle_open_time_ms",
                "term_candle_close_time_ms",
                "term_candle_gap_open_seconds",
                "term_candle_gap_close_seconds",
                "term_candle_price_choice",
                "term_candle_price_timestamp_ms",
                "term_candle_gap_seconds",
                "term_candle_open",
                "term_candle_high",
                "term_candle_low",
                "term_candle_close",
                "term_candle_volume",
                "term_candle_cost",
                "term_candle_resolution",
                "term_candle_price_used",
                "term_candle_accepted_under_max_gap",
            ]

            term_new = term_new.merge(
                term_candles[keep_cols],
                on=["timestamp", "currency", "snapshot_file", "futures_instrument_name"],
                how="left",
            )

    for col in [
        "term_candle_found",
        "term_candle_tick_ms",
        "term_candle_open_time_ms",
        "term_candle_close_time_ms",
        "term_candle_gap_open_seconds",
        "term_candle_gap_close_seconds",
        "term_candle_price_choice",
        "term_candle_price_timestamp_ms",
        "term_candle_gap_seconds",
        "term_candle_open",
        "term_candle_high",
        "term_candle_low",
        "term_candle_close",
        "term_candle_volume",
        "term_candle_cost",
        "term_candle_resolution",
        "term_candle_price_used",
        "term_candle_accepted_under_max_gap",
    ]:
        if col not in term_new.columns:
            term_new[col] = pd.NA

    term_new["term_mark_price_final"] = np.where(
        term_new["term_candle_accepted_under_max_gap"].fillna(False),
        pd.to_numeric(term_new["term_candle_price_used"], errors="coerce"),
        np.nan,
    )
    term_new["term_mark_price_final_source"] = np.where(
        term_new["term_candle_accepted_under_max_gap"].fillna(False),
        "tradingview_chart_open_close_nearest",
        pd.NA,
    )

    updated_term = concat_dedup_sort(
        existing_df=existing_term,
        new_df=term_new,
        key_cols=key_cols,
        sort_cols=["timestamp", "currency", "futures_instrument_name"],
    )
    return updated_term, term_new


def build_perp_incremental(
    combos: pd.DataFrame,
    existing_perp: pd.DataFrame,
    skip_perp_backfill: bool,
    sleep_seconds: float,
    max_gap_seconds: Optional[float],
    perp_resolution: str,
    perp_max_bars_per_call: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if combos.empty:
        return existing_perp.copy(), pd.DataFrame()

    perp_requests = combos.copy()
    perp_requests["obs_datetime"] = perp_requests["timestamp"]
    perp_requests["coin"] = perp_requests["currency"]
    perp_requests["instrument_name"] = perp_requests["coin"].astype(str) + "-PERPETUAL"
    key_cols = ["obs_datetime", "coin"]

    existing_norm = existing_perp.copy()
    if not existing_norm.empty:
        if "timestamp" in existing_norm.columns and "obs_datetime" not in existing_norm.columns:
            existing_norm["obs_datetime"] = existing_norm["timestamp"]
        if "currency" in existing_norm.columns and "coin" not in existing_norm.columns:
            existing_norm["coin"] = existing_norm["currency"]

    needs_refresh = perp_requests.copy()
    if not existing_norm.empty and all(c in existing_norm.columns for c in key_cols):
        existing_subset = select_rows_by_keys(existing_norm, perp_requests, key_cols)
        if not existing_subset.empty:
            merged = perp_requests.merge(existing_subset, on=key_cols, how="left", suffixes=("", "__existing"))
            existing_price = pd.to_numeric(merged.get("perp_futures_mark_price"), errors="coerce")
            existing_source = merged.get("data_source")
            existing_ok = existing_price.notna()
            if existing_source is not None:
                existing_ok &= existing_source.astype(str).eq("tradingview_chart_open_close_nearest")
            needs_refresh = merged.loc[~existing_ok, perp_requests.columns].copy()

    if needs_refresh.empty:
        return existing_perp.copy(), pd.DataFrame()

    if skip_perp_backfill:
        perp_new = needs_refresh.copy()
        perp_new["perp_futures_mark_price"] = pd.NA
        perp_new["perp_futures_best_bid"] = pd.NA
        perp_new["perp_futures_best_ask"] = pd.NA
        perp_new["data_source"] = "skipped"
        for col in [
            "candle_found",
            "candle_tick_ms",
            "candle_open_time_ms",
            "candle_close_time_ms",
            "candle_gap_open_seconds",
            "candle_gap_close_seconds",
            "candle_price_choice",
            "candle_price_timestamp_ms",
            "candle_price_gap_seconds",
            "candle_open",
            "candle_high",
            "candle_low",
            "candle_close",
            "candle_volume",
            "candle_cost",
            "accepted_under_max_gap",
            "candle_resolution",
            "candle_price_used",
        ]:
            perp_new[col] = pd.NA
    else:
        perp_new = fetch_tradingview_candles_for_requests(
            request_rows=needs_refresh,
            instrument_col="instrument_name",
            timestamp_ms_col="snapshot_timestamp_ms",
            resolution=perp_resolution,
            sleep_seconds=sleep_seconds,
            max_gap_seconds=max_gap_seconds,
            max_bars_per_call=perp_max_bars_per_call,
            progress_desc="Fetching perp candles",
        )
        perp_new["perp_futures_mark_price"] = np.where(
            perp_new["accepted_under_max_gap"].fillna(False),
            pd.to_numeric(perp_new["candle_price_used"], errors="coerce"),
            np.nan,
        )
        perp_new["perp_futures_best_bid"] = pd.NA
        perp_new["perp_futures_best_ask"] = pd.NA
        perp_new["data_source"] = "tradingview_chart_open_close_nearest"

    if "timestamp" in perp_new.columns:
        perp_new = perp_new.drop(columns=["timestamp"], errors="ignore")
    if "currency" in perp_new.columns:
        perp_new = perp_new.drop(columns=["currency"], errors="ignore")

    existing_keep = exclude_rows_by_keys(existing_perp, needs_refresh, key_cols)
    updated_perp = concat_dedup_sort(
        existing_df=existing_keep,
        new_df=perp_new,
        key_cols=key_cols,
        sort_cols=["obs_datetime", "coin"],
    )
    return updated_perp, perp_new


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Incrementally maintain futures data for the thesis repo. "
            "Perp and term futures are backfilled from TradingView chart candles using "
            "the closer of the candle open and candle close in time. "
            "Rows outside the max-gap threshold are left blank."
        )
    )
    ap.add_argument("--data-dir", default="data", help="Directory containing deribit_options_snapshot_*.csv")
    ap.add_argument("--outdir", default="data", help="Directory where output CSVs live")
    ap.add_argument(
        "--include-synthetics",
        action="store_true",
        help="Include SYN.* underlyings from option snapshots. Default is to keep only real term futures.",
    )
    ap.add_argument(
        "--skip-perp-backfill",
        action="store_true",
        help="Do not call Deribit for missing perpetual rows. Existing rows are preserved.",
    )
    ap.add_argument(
        "--skip-term-backfill",
        action="store_true",
        help="Do not call Deribit for missing term-future rows. Existing rows are preserved.",
    )
    ap.add_argument(
        "--perp-max-gap-seconds",
        type=float,
        default=DEFAULT_PERP_MAX_GAP_SECONDS,
        help="Maximum allowed distance between option snapshot time and chosen perp candle open/close time. Default: 1800.",
    )
    ap.add_argument(
        "--term-max-gap-seconds",
        type=float,
        default=DEFAULT_TERM_MAX_GAP_SECONDS,
        help="Maximum allowed distance between option snapshot time and chosen term candle open/close time. Default: 1800.",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Sleep between Deribit requests, in seconds.",
    )
    ap.add_argument(
        "--perp-resolution",
        default=DEFAULT_RESOLUTION,
        help="TradingView chart resolution for perp backfill. Default: 1 (1 minute).",
    )
    ap.add_argument(
        "--term-resolution",
        default=DEFAULT_RESOLUTION,
        help="TradingView chart resolution for term-futures backfill. Default: 1 (1 minute).",
    )
    ap.add_argument(
        "--perp-max-bars-per-call",
        type=int,
        default=DEFAULT_MAX_BARS_PER_CALL,
        help="Maximum perp candle bars to request per TradingView chart call. Default: 5000.",
    )
    ap.add_argument(
        "--term-max-bars-per-call",
        type=int,
        default=DEFAULT_MAX_BARS_PER_CALL,
        help="Maximum term candle bars to request per TradingView chart call. Default: 5000.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    snapshot_files = list_snapshot_files(data_dir)
    if not snapshot_files:
        raise FileNotFoundError(f"No files matching {SNAPSHOT_GLOB} were found in {data_dir}")

    combos = build_snapshot_currency_combos(snapshot_files)
    combos_out = outdir / COMBOS_OUTPUT_NAME
    combos.to_csv(combos_out, index=False)

    term_requests = build_term_snapshot_requests(
        snapshot_files=snapshot_files,
        include_synthetics=args.include_synthetics,
    )

    term_out_path = outdir / TERM_OUTPUT_NAME
    perp_out_path = outdir / PERP_OUTPUT_NAME
    existing_term = read_csv_if_exists(term_out_path)
    existing_perp = read_csv_if_exists(perp_out_path)

    updated_term, term_appended = build_term_incremental(
        term_requests=term_requests,
        existing_term=existing_term,
        skip_term_backfill=args.skip_term_backfill,
        sleep_seconds=args.sleep,
        max_gap_seconds=args.term_max_gap_seconds,
        term_resolution=args.term_resolution,
        term_max_bars_per_call=args.term_max_bars_per_call,
    )
    updated_term.to_csv(term_out_path, index=False)

    updated_perp, perp_appended = build_perp_incremental(
        combos=combos,
        existing_perp=existing_perp,
        skip_perp_backfill=args.skip_perp_backfill,
        sleep_seconds=args.sleep,
        max_gap_seconds=args.perp_max_gap_seconds,
        perp_resolution=args.perp_resolution,
        perp_max_bars_per_call=args.perp_max_bars_per_call,
    )
    updated_perp.to_csv(perp_out_path, index=False)

    print(f"Wrote snapshot combos: {combos_out}")
    print(f"Wrote term futures marks: {term_out_path}")
    print(f"Wrote perpetual futures history: {perp_out_path}")
    print(
        "Backfill source: TradingView chart open/close nearest "
        f"(perp resolution={args.perp_resolution}, term resolution={args.term_resolution})"
    )
    print(f"Appended term rows: {len(term_appended):,}")
    print(f"Appended perp rows: {len(perp_appended):,}")
    print(f"Total term rows saved: {len(updated_term):,}")
    print(f"Total perp rows saved: {len(updated_perp):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
