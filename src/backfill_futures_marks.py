#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

DERIBIT_BASE = "https://www.deribit.com/api/v2"
SNAPSHOT_GLOB = "deribit_options_snapshot_*.csv"
FUTURES_SNAPSHOT_GLOB = "deribit_futures_snapshot_*.csv"
TERM_OUTPUT_NAME = "term_futures_marks.csv"
PERP_OUTPUT_NAME = "perpetual_futures_prices.csv"
COMBOS_OUTPUT_NAME = "snapshot_currency_combos.csv"

DEFAULT_RESOLUTION = "1"
DEFAULT_MAX_BARS_PER_CALL = 5000
DEFAULT_PERP_MAX_GAP_SECONDS = 1800.0
DEFAULT_TERM_MAX_GAP_SECONDS = 1800.0
SNAPSHOT_ID_RE = re.compile(r"deribit_(?:options|futures)_snapshot_(\d{8}T\d{6}Z)\.csv$")


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


def http_get(session: requests.Session, path: str, params: Dict[str, Any] | None = None, timeout: int = 30) -> Dict[str, Any]:
    response = session.get(f"{DERIBIT_BASE}{path}", params=params or {}, timeout=timeout)
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
    if df.empty or any(c not in df.columns for c in key_cols):
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
    if any(c not in df.columns for c in key_cols):
        return df.copy()
    key_set = dataframe_key_tuples(key_df, key_cols)
    mask = [
        tuple(_normalize_key_value(v) for v in row) not in key_set
        for row in df[key_cols].itertuples(index=False, name=None)
    ]
    return df.loc[mask].copy()


def concat_dedup_sort(existing_df: pd.DataFrame, new_df: pd.DataFrame, key_cols: list[str], sort_cols: list[str]) -> pd.DataFrame:
    parts = [df for df in (existing_df, new_df) if not df.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    if all(c in out.columns for c in key_cols):
        out = out.drop_duplicates(subset=key_cols, keep="first")
    present = [c for c in sort_cols if c in out.columns]
    if present:
        out = out.sort_values(present).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def _snapshot_id_from_name(name: str) -> Optional[str]:
    match = SNAPSHOT_ID_RE.search(str(name))
    return match.group(1) if match else None


def list_snapshot_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.glob(SNAPSHOT_GLOB) if p.is_file())


def list_futures_snapshot_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.glob(FUTURES_SNAPSHOT_GLOB) if p.is_file())


def read_snapshot_minimal(path: Path) -> pd.DataFrame:
    wanted = ["timestamp", "currency", "futures_price", "futures_instrument_name", "underlying", "underlying_name"]
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "currency" not in df.columns:
        raise ValueError(f"{path.name} is missing required columns timestamp/currency")
    return df[[c for c in wanted if c in df.columns]].copy()


def build_snapshot_currency_combos(snapshot_files: Iterable[Path]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for path in list(snapshot_files):
        df = read_snapshot_minimal(path)
        cur = df[["timestamp", "currency"]].dropna().drop_duplicates().copy()
        cur["snapshot_file"] = path.name
        cur["snapshot_timestamp_ms"] = cur["timestamp"].map(iso_to_ms)
        cur["snapshot_id"] = _snapshot_id_from_name(path.name)
        chunks.append(cur)
    if not chunks:
        return pd.DataFrame(columns=["timestamp", "currency", "snapshot_file", "snapshot_timestamp_ms", "snapshot_id"])
    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values(["timestamp", "currency", "snapshot_file"]).reset_index(drop=True)


def build_term_snapshot_requests(snapshot_files: Iterable[Path], include_synthetics: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in list(snapshot_files):
        df = read_snapshot_minimal(path)
        if "futures_instrument_name" not in df.columns:
            continue
        work = df[["timestamp", "currency", "futures_instrument_name"]].copy()
        work = work.dropna(subset=["timestamp", "currency", "futures_instrument_name"])
        work["snapshot_term_mark_price"] = pd.to_numeric(df.get("futures_price"), errors="coerce") if "futures_price" in df.columns else np.nan
        if not include_synthetics:
            work = work[~work["futures_instrument_name"].astype(str).str.startswith("SYN.")]
        if work.empty:
            continue
        grouped = work.groupby(["timestamp", "currency", "futures_instrument_name"], as_index=False).agg(
            snapshot_term_mark_price=("snapshot_term_mark_price", "median"),
            snapshot_term_mark_min=("snapshot_term_mark_price", "min"),
            snapshot_term_mark_max=("snapshot_term_mark_price", "max"),
            contributing_option_rows=("futures_instrument_name", "size"),
            distinct_prices_seen=("snapshot_term_mark_price", "nunique"),
        )
        grouped["snapshot_file"] = path.name
        grouped["snapshot_timestamp_ms"] = grouped["timestamp"].map(iso_to_ms)
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "currency", "futures_instrument_name", "snapshot_term_mark_price", "snapshot_term_mark_min", "snapshot_term_mark_max", "contributing_option_rows", "distinct_prices_seen", "snapshot_file", "snapshot_timestamp_ms"])
    return pd.concat(rows, ignore_index=True).sort_values(["timestamp", "currency", "futures_instrument_name"]).reset_index(drop=True)


def build_direct_perp_rows(futures_snapshot_files: Iterable[Path], combos: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "snapshot_file", "snapshot_timestamp_ms", "obs_datetime", "coin", "instrument_name",
        "perp_futures_mark_price", "perp_futures_best_bid", "perp_futures_best_ask", "data_source",
        "futures_snapshot_file", "perp_exchange_timestamp", "perp_requested_at_utc",
        "perp_received_at_utc", "perp_collection_latency_ms",
    ]
    if combos.empty:
        return pd.DataFrame(columns=columns)
    combo = combos.copy()
    if "snapshot_id" not in combo.columns:
        combo["snapshot_id"] = combo["snapshot_file"].map(_snapshot_id_from_name)
    combo["currency"] = combo["currency"].astype(str).str.upper()
    rows: list[pd.DataFrame] = []
    for path in list(futures_snapshot_files):
        df = pd.read_csv(path)
        required = {"currency", "instrument_name", "mark_price", "bid_price", "ask_price"}
        if not required.issubset(df.columns):
            continue
        work = df.copy()
        work["currency"] = work["currency"].astype(str).str.upper()
        work = work[work["instrument_name"].astype(str).str.endswith("-PERPETUAL")].copy()
        if work.empty:
            continue
        snapshot_id = _snapshot_id_from_name(path.name)
        work["snapshot_id"] = work.get("snapshot_id", snapshot_id)
        work["snapshot_id"] = work["snapshot_id"].fillna(snapshot_id)
        merged = work.merge(
            combo[["snapshot_id", "timestamp", "currency", "snapshot_file", "snapshot_timestamp_ms"]],
            on=["snapshot_id", "currency"],
            how="inner",
            suffixes=("", "__option"),
        )
        if merged.empty:
            continue
        out = pd.DataFrame()
        out["snapshot_file"] = merged["snapshot_file"]
        out["snapshot_timestamp_ms"] = merged["snapshot_timestamp_ms"]
        out["obs_datetime"] = merged["timestamp__option"] if "timestamp__option" in merged.columns else merged["timestamp"]
        out["coin"] = merged["currency"]
        out["instrument_name"] = merged["instrument_name"]
        out["perp_futures_mark_price"] = pd.to_numeric(merged["mark_price"], errors="coerce")
        out["perp_futures_best_bid"] = pd.to_numeric(merged["bid_price"], errors="coerce")
        out["perp_futures_best_ask"] = pd.to_numeric(merged["ask_price"], errors="coerce")
        out["data_source"] = "deribit_snapshot"
        out["futures_snapshot_file"] = path.name
        out["perp_exchange_timestamp"] = merged.get("exchange_timestamp")
        out["perp_requested_at_utc"] = merged.get("requested_at_utc")
        out["perp_received_at_utc"] = merged.get("received_at_utc")
        out["perp_collection_latency_ms"] = pd.to_numeric(merged.get("collection_latency_ms"), errors="coerce")
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=columns)
    direct = pd.concat(rows, ignore_index=True, sort=False)
    direct = direct.drop_duplicates(subset=["obs_datetime", "coin"], keep="last")
    return direct[columns].sort_values(["obs_datetime", "coin"]).reset_index(drop=True)


def fetch_tradingview_chart_chunk(session: requests.Session, instrument_name: str, start_ms: int, end_ms: int, resolution: str) -> pd.DataFrame:
    payload = http_get(session, "/public/get_tradingview_chart_data", {"instrument_name": instrument_name, "start_timestamp": int(start_ms), "end_timestamp": int(end_ms), "resolution": str(resolution)})
    result = payload.get("result", {})
    if result.get("status") == "no_data":
        return pd.DataFrame(columns=["tick", "open", "high", "low", "close", "volume", "cost"])
    ticks = result.get("ticks", []) or []
    if not ticks:
        return pd.DataFrame(columns=["tick", "open", "high", "low", "close", "volume", "cost"])
    n = len(ticks)
    def arr(name: str) -> list[Any]:
        values = list(result.get(name, []) or [])
        return (values + [None] * n)[:n]
    return pd.DataFrame({"tick": ticks, "open": arr("open"), "high": arr("high"), "low": arr("low"), "close": arr("close"), "volume": arr("volume"), "cost": arr("cost")})


def fetch_tradingview_candles_for_requests(request_rows: pd.DataFrame, instrument_col: str, timestamp_ms_col: str, resolution: str, sleep_seconds: float, max_gap_seconds: Optional[float], max_bars_per_call: int, progress_desc: str) -> pd.DataFrame:
    if request_rows.empty:
        return pd.DataFrame()
    resolution_ms = resolution_to_ms(resolution)
    chunk_ms = max(int(max_bars_per_call), 1) * resolution_ms
    session = requests.Session()
    session.headers.update({"User-Agent": "SVCJ-Deribit-Pricer/1.0 (candle backfill)"})
    req = request_rows.copy()
    req["__row_id"] = np.arange(len(req))
    all_rows: list[pd.DataFrame] = []
    instruments = sorted(req[instrument_col].dropna().astype(str).unique().tolist())
    for instrument_name in instruments:
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
            chunk = fetch_tradingview_chart_chunk(session, instrument_name, chunk_start, chunk_end, resolution)
            if not chunk.empty:
                candle_parts.append(chunk)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if chunk_end >= end_ms:
                break
            chunk_start = chunk_end + resolution_ms
        candles = pd.concat(candle_parts, ignore_index=True).drop_duplicates(subset=["tick"]).sort_values("tick") if candle_parts else pd.DataFrame()
        if candles.empty:
            tmp = sub.copy()
            for col in ["candle_tick_ms", "candle_open_time_ms", "candle_close_time_ms", "candle_gap_open_seconds", "candle_gap_close_seconds", "candle_price_choice", "candle_price_timestamp_ms", "candle_price_gap_seconds", "candle_open", "candle_high", "candle_low", "candle_close", "candle_volume", "candle_cost"]:
                tmp[col] = pd.NA
            tmp["candle_found"] = False
            tmp["accepted_under_max_gap"] = False if max_gap_seconds is not None else pd.NA
            all_rows.append(tmp)
            continue
        candles = candles.copy()
        candles["candle_tick_ms"] = candles["tick"]
        candles["candle_open_time_ms"] = candles["tick"]
        candles["candle_close_time_ms"] = candles["tick"] + resolution_ms
        merged = pd.merge_asof(
            sub[["__row_id", instrument_col, timestamp_ms_col]].sort_values(timestamp_ms_col),
            candles[["candle_tick_ms", "candle_open_time_ms", "candle_close_time_ms", "open", "high", "low", "close", "volume", "cost"]],
            left_on=timestamp_ms_col, right_on="candle_tick_ms", direction="backward", allow_exact_matches=True,
        )
        merged["candle_found"] = merged["candle_tick_ms"].notna()
        merged["candle_gap_open_seconds"] = np.where(merged["candle_found"], np.abs(merged[timestamp_ms_col] - merged["candle_open_time_ms"]) / 1000.0, np.nan)
        merged["candle_gap_close_seconds"] = np.where(merged["candle_found"], np.abs(merged[timestamp_ms_col] - merged["candle_close_time_ms"]) / 1000.0, np.nan)
        choose_open = merged["candle_gap_open_seconds"] <= merged["candle_gap_close_seconds"]
        merged["candle_price_choice"] = np.where(choose_open, "open", "close")
        merged["candle_price_timestamp_ms"] = np.where(choose_open, merged["candle_open_time_ms"], merged["candle_close_time_ms"])
        merged["candle_price_gap_seconds"] = np.where(choose_open, merged["candle_gap_open_seconds"], merged["candle_gap_close_seconds"])
        merged["accepted_under_max_gap"] = merged["candle_found"] if max_gap_seconds is None else merged["candle_found"] & (merged["candle_price_gap_seconds"] <= float(max_gap_seconds))
        merged = merged.rename(columns={"open": "candle_open", "high": "candle_high", "low": "candle_low", "close": "candle_close", "volume": "candle_volume", "cost": "candle_cost"})
        keep = ["__row_id", "candle_found", "candle_tick_ms", "candle_open_time_ms", "candle_close_time_ms", "candle_gap_open_seconds", "candle_gap_close_seconds", "candle_price_choice", "candle_price_timestamp_ms", "candle_price_gap_seconds", "candle_open", "candle_high", "candle_low", "candle_close", "candle_volume", "candle_cost", "accepted_under_max_gap"]
        all_rows.append(sub.merge(merged[keep], on="__row_id", how="left"))
    if not all_rows:
        return pd.DataFrame()
    out = pd.concat(all_rows, ignore_index=True, sort=False).drop(columns=["__row_id"], errors="ignore")
    out["candle_resolution"] = str(resolution)
    out["candle_price_used"] = np.where(out["candle_price_choice"].astype(str).eq("open"), pd.to_numeric(out["candle_open"], errors="coerce"), pd.to_numeric(out["candle_close"], errors="coerce"))
    return out


def build_term_incremental(term_requests: pd.DataFrame, existing_term: pd.DataFrame, skip_term_backfill: bool, sleep_seconds: float, max_gap_seconds: Optional[float], term_resolution: str, term_max_bars_per_call: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["timestamp", "currency", "futures_instrument_name"]
    new_term = filter_new_rows(term_requests, existing_term, key_cols)
    if new_term.empty:
        return existing_term.copy(), new_term
    term_new = new_term.copy()
    candle_cols = ["term_candle_found", "term_candle_tick_ms", "term_candle_open_time_ms", "term_candle_close_time_ms", "term_candle_gap_open_seconds", "term_candle_gap_close_seconds", "term_candle_price_choice", "term_candle_price_timestamp_ms", "term_candle_gap_seconds", "term_candle_open", "term_candle_high", "term_candle_low", "term_candle_close", "term_candle_volume", "term_candle_cost", "term_candle_resolution", "term_candle_price_used", "term_candle_accepted_under_max_gap"]
    if skip_term_backfill:
        for col in candle_cols:
            term_new[col] = pd.NA
    else:
        requests_df = term_new[["timestamp", "currency", "futures_instrument_name", "snapshot_timestamp_ms", "snapshot_file"]].rename(columns={"futures_instrument_name": "instrument_name"})
        candles = fetch_tradingview_candles_for_requests(requests_df, "instrument_name", "snapshot_timestamp_ms", term_resolution, sleep_seconds, max_gap_seconds, term_max_bars_per_call, "Fetching term candles")
        if not candles.empty:
            candles = candles.rename(columns={
                "instrument_name": "futures_instrument_name", "candle_found": "term_candle_found", "candle_tick_ms": "term_candle_tick_ms",
                "candle_open_time_ms": "term_candle_open_time_ms", "candle_close_time_ms": "term_candle_close_time_ms", "candle_gap_open_seconds": "term_candle_gap_open_seconds",
                "candle_gap_close_seconds": "term_candle_gap_close_seconds", "candle_price_choice": "term_candle_price_choice", "candle_price_timestamp_ms": "term_candle_price_timestamp_ms",
                "candle_price_gap_seconds": "term_candle_gap_seconds", "candle_open": "term_candle_open", "candle_high": "term_candle_high", "candle_low": "term_candle_low",
                "candle_close": "term_candle_close", "candle_volume": "term_candle_volume", "candle_cost": "term_candle_cost", "candle_resolution": "term_candle_resolution",
                "candle_price_used": "term_candle_price_used", "accepted_under_max_gap": "term_candle_accepted_under_max_gap",
            })
            merge_cols = ["timestamp", "currency", "snapshot_file", "futures_instrument_name"] + candle_cols
            term_new = term_new.merge(candles[merge_cols], on=["timestamp", "currency", "snapshot_file", "futures_instrument_name"], how="left")
    for col in candle_cols:
        if col not in term_new.columns:
            term_new[col] = pd.NA
    accepted = term_new["term_candle_accepted_under_max_gap"].fillna(False).astype(bool)
    term_new["term_mark_price_final"] = np.where(accepted, pd.to_numeric(term_new["term_candle_price_used"], errors="coerce"), np.nan)
    term_new["term_mark_price_final_source"] = np.where(accepted, "tradingview_chart_open_close_nearest", pd.NA)
    updated = concat_dedup_sort(existing_term, term_new, key_cols, ["timestamp", "currency", "futures_instrument_name"])
    return updated, term_new


def _normalize_existing_perp(existing_perp: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy and current perpetual rows to the canonical proxy schema."""
    out = existing_perp.copy()
    if "timestamp" in out.columns and "obs_datetime" not in out.columns:
        out["obs_datetime"] = out["timestamp"]
    if "currency" in out.columns and "coin" not in out.columns:
        out["coin"] = out["currency"]

    legacy_price = (
        pd.to_numeric(out["perp_futures_mark_price"], errors="coerce")
        if "perp_futures_mark_price" in out.columns
        else pd.Series(np.nan, index=out.index, dtype=float)
    )
    if "perp_futures_price_proxy" in out.columns:
        proxy = pd.to_numeric(out["perp_futures_price_proxy"], errors="coerce")
        out["perp_futures_price_proxy"] = proxy.where(proxy.notna(), legacy_price)
    else:
        out["perp_futures_price_proxy"] = legacy_price

    data_source = (
        out["data_source"].astype("string")
        if "data_source" in out.columns
        else pd.Series(pd.NA, index=out.index, dtype="string")
    )
    inferred_source = data_source.replace(
        {
            "tradingview_chart_open_close_nearest": "deribit_tradingview_chart",
            "deribit_snapshot": "deribit_snapshot",
            "skipped": "skipped",
        }
    )
    if "price_source" in out.columns:
        current = out["price_source"].astype("string")
        out["price_source"] = current.where(current.notna(), inferred_source)
    else:
        out["price_source"] = inferred_source

    inferred_kind = pd.Series("nearest_1m_candle_open_close", index=out.index, dtype="string")
    inferred_kind = inferred_kind.where(~data_source.eq("deribit_snapshot"), "direct_perpetual_mark")
    if "price_kind" in out.columns:
        current_kind = out["price_kind"].astype("string")
        out["price_kind"] = current_kind.where(current_kind.notna(), inferred_kind)
    else:
        out["price_kind"] = inferred_kind
    return out


def build_perp_incremental(combos: pd.DataFrame, existing_perp: pd.DataFrame, skip_perp_backfill: bool, sleep_seconds: float, max_gap_seconds: Optional[float], perp_resolution: str, perp_max_bars_per_call: int, direct_perp_rows: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if combos.empty:
        return existing_perp.copy(), pd.DataFrame()
    requests_df = combos.copy()
    requests_df["obs_datetime"] = requests_df["timestamp"]
    requests_df["coin"] = requests_df["currency"].astype(str).str.upper()
    requests_df["instrument_name"] = requests_df["coin"] + "-PERPETUAL"
    key_cols = ["obs_datetime", "coin"]
    existing_norm = _normalize_existing_perp(existing_perp)
    direct = direct_perp_rows.copy() if direct_perp_rows is not None else pd.DataFrame()
    if not direct.empty:
        direct["coin"] = direct["coin"].astype(str).str.upper()
        direct = direct.drop_duplicates(subset=key_cols, keep="last")
    direct_keys = direct[key_cols] if not direct.empty else pd.DataFrame(columns=key_cols)
    unresolved = exclude_rows_by_keys(requests_df, direct_keys, key_cols)
    if not existing_norm.empty and not unresolved.empty:
        existing_subset = select_rows_by_keys(existing_norm, unresolved, key_cols)
        if not existing_subset.empty:
            merged = unresolved.merge(existing_subset[key_cols + [c for c in ["perp_futures_mark_price", "perp_futures_best_bid", "perp_futures_best_ask"] if c in existing_subset.columns]], on=key_cols, how="left")
            mark = pd.to_numeric(merged.get("perp_futures_mark_price"), errors="coerce")
            bid = pd.to_numeric(merged.get("perp_futures_best_bid"), errors="coerce")
            ask = pd.to_numeric(merged.get("perp_futures_best_ask"), errors="coerce")
            existing_ok = mark.notna() | (bid.notna() & ask.notna())
            unresolved = merged.loc[~existing_ok, requests_df.columns].copy()
    fallback_new = pd.DataFrame()
    if not unresolved.empty:
        if skip_perp_backfill:
            fallback_new = unresolved.copy()
            fallback_new["perp_futures_mark_price"] = pd.NA
            fallback_new["perp_futures_best_bid"] = pd.NA
            fallback_new["perp_futures_best_ask"] = pd.NA
            fallback_new["data_source"] = "skipped"
            for col in ["candle_found", "candle_tick_ms", "candle_open_time_ms", "candle_close_time_ms", "candle_gap_open_seconds", "candle_gap_close_seconds", "candle_price_choice", "candle_price_timestamp_ms", "candle_price_gap_seconds", "candle_open", "candle_high", "candle_low", "candle_close", "candle_volume", "candle_cost", "accepted_under_max_gap", "candle_resolution", "candle_price_used"]:
                fallback_new[col] = pd.NA
        else:
            fallback_new = fetch_tradingview_candles_for_requests(unresolved, "instrument_name", "snapshot_timestamp_ms", perp_resolution, sleep_seconds, max_gap_seconds, perp_max_bars_per_call, "Fetching perp candles")
            if not fallback_new.empty:
                accepted = fallback_new["accepted_under_max_gap"].fillna(False).astype(bool)
                fallback_new["perp_futures_mark_price"] = np.where(accepted, pd.to_numeric(fallback_new["candle_price_used"], errors="coerce"), np.nan)
                fallback_new["perp_futures_best_bid"] = pd.NA
                fallback_new["perp_futures_best_ask"] = pd.NA
                fallback_new["data_source"] = "tradingview_chart_open_close_nearest"
    direct = _normalize_existing_perp(direct)
    fallback_new = _normalize_existing_perp(fallback_new)
    new_parts = [df for df in (direct, fallback_new) if not df.empty]
    appended = pd.concat(new_parts, ignore_index=True, sort=False) if new_parts else pd.DataFrame()
    if not appended.empty:
        appended = appended.drop(columns=["timestamp", "currency"], errors="ignore")
    replace_keys = appended[key_cols] if not appended.empty else pd.DataFrame(columns=key_cols)
    existing_keep = exclude_rows_by_keys(existing_norm, replace_keys, key_cols)
    updated = concat_dedup_sort(existing_keep, appended, key_cols, ["obs_datetime", "coin"])
    return updated, appended


def main() -> int:
    ap = argparse.ArgumentParser(description="Incrementally maintain term and perpetual futures history. Direct futures snapshots are preferred; TradingView candles remain the fallback for older snapshots.")
    ap.add_argument("--data-dir", default="data", help="Directory containing Deribit option/futures snapshots")
    ap.add_argument("--outdir", default="data", help="Directory where output CSVs live")
    ap.add_argument("--include-synthetics", action="store_true")
    ap.add_argument("--skip-perp-backfill", action="store_true", help="Do not call Deribit TradingView candles for unresolved perpetual rows; direct futures snapshots are still ingested.")
    ap.add_argument("--skip-term-backfill", action="store_true")
    ap.add_argument("--perp-max-gap-seconds", type=float, default=DEFAULT_PERP_MAX_GAP_SECONDS)
    ap.add_argument("--term-max-gap-seconds", type=float, default=DEFAULT_TERM_MAX_GAP_SECONDS)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--perp-resolution", default=DEFAULT_RESOLUTION)
    ap.add_argument("--term-resolution", default=DEFAULT_RESOLUTION)
    ap.add_argument("--perp-max-bars-per-call", type=int, default=DEFAULT_MAX_BARS_PER_CALL)
    ap.add_argument("--term-max-bars-per-call", type=int, default=DEFAULT_MAX_BARS_PER_CALL)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    snapshot_files = list_snapshot_files(data_dir)
    if not snapshot_files:
        raise FileNotFoundError(f"No files matching {SNAPSHOT_GLOB} were found in {data_dir}")
    futures_snapshot_files = list_futures_snapshot_files(data_dir)
    combos = build_snapshot_currency_combos(snapshot_files)
    combos_out = outdir / COMBOS_OUTPUT_NAME
    combos.to_csv(combos_out, index=False)
    term_requests = build_term_snapshot_requests(snapshot_files, include_synthetics=args.include_synthetics)
    direct_perp = build_direct_perp_rows(futures_snapshot_files, combos)

    term_out = outdir / TERM_OUTPUT_NAME
    perp_out = outdir / PERP_OUTPUT_NAME
    existing_term = read_csv_if_exists(term_out)
    existing_perp = read_csv_if_exists(perp_out)
    updated_term, term_appended = build_term_incremental(term_requests, existing_term, args.skip_term_backfill, args.sleep, args.term_max_gap_seconds, args.term_resolution, args.term_max_bars_per_call)
    updated_term.to_csv(term_out, index=False)
    updated_perp, perp_appended = build_perp_incremental(combos, existing_perp, args.skip_perp_backfill, args.sleep, args.perp_max_gap_seconds, args.perp_resolution, args.perp_max_bars_per_call, direct_perp_rows=direct_perp)
    updated_perp.to_csv(perp_out, index=False)

    print(f"Option snapshots scanned: {len(snapshot_files):,}")
    print(f"Futures snapshots scanned: {len(futures_snapshot_files):,}")
    print(f"Direct perpetual rows available: {len(direct_perp):,}")
    print(f"Wrote snapshot combos: {combos_out}")
    print(f"Wrote term futures marks: {term_out}")
    print(f"Wrote perpetual futures history: {perp_out}")
    print(f"Appended/replaced term rows: {len(term_appended):,}")
    print(f"Appended/replaced perp rows: {len(perp_appended):,}")
    print(f"Total term rows saved: {len(updated_term):,}")
    print(f"Total perp rows saved: {len(updated_perp):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
