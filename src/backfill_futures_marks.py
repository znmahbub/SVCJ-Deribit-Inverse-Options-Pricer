#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from tqdm import tqdm

import pandas as pd
import requests

DERIBIT_BASE = "https://www.deribit.com/api/v2"
SNAPSHOT_GLOB = "deribit_options_snapshot_*.csv"
TERM_OUTPUT_NAME = "term_futures_marks.csv"
PERP_OUTPUT_NAME = "perpetual_futures_prices.csv"
COMBOS_OUTPUT_NAME = "snapshot_currency_combos.csv"


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
        return pd.DataFrame(
            columns=["timestamp", "currency", "snapshot_file", "snapshot_timestamp_ms"]
        )

    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values(["timestamp", "currency", "snapshot_file"]).reset_index(drop=True)


def build_term_futures_from_snapshots(
    snapshot_files: Iterable[Path],
    include_synthetics: bool = False,
) -> pd.DataFrame:
    snapshot_files = list(snapshot_files)
    rows: list[pd.DataFrame] = []

    for path in tqdm(snapshot_files, desc="Extracting term futures", unit="file"):
        df = read_snapshot_minimal(path)

        if "futures_instrument_name" not in df.columns or "futures_price" not in df.columns:
            continue

        work = df[["timestamp", "currency", "futures_instrument_name", "futures_price"]].copy()
        work = work.dropna(subset=["timestamp", "currency", "futures_instrument_name", "futures_price"])
        work["futures_price"] = pd.to_numeric(work["futures_price"], errors="coerce")
        work = work.dropna(subset=["futures_price"])

        if not include_synthetics:
            work = work[~work["futures_instrument_name"].astype(str).str.startswith("SYN.")]

        if work.empty:
            continue

        grouped = (
            work.groupby(["timestamp", "currency", "futures_instrument_name"], as_index=False)
            .agg(
                snapshot_term_mark_price=("futures_price", "median"),
                snapshot_term_mark_min=("futures_price", "min"),
                snapshot_term_mark_max=("futures_price", "max"),
                contributing_option_rows=("futures_price", "size"),
                distinct_prices_seen=("futures_price", "nunique"),
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
# Historical trade mark backfill
# -----------------------------
def get_trade_mark_before(
    session: requests.Session,
    instrument_name: str,
    ts_ms: int,
) -> Optional[Dict[str, Any]]:
    payload = http_get(
        session,
        "/public/get_last_trades_by_instrument_and_time",
        {
            "instrument_name": instrument_name,
            "end_timestamp": ts_ms,
            "count": 1,
            "sorting": "desc",
        },
    )
    trades = payload.get("result", {}).get("trades", [])
    return trades[0] if trades else None


def get_trade_mark_after(
    session: requests.Session,
    instrument_name: str,
    ts_ms: int,
) -> Optional[Dict[str, Any]]:
    payload = http_get(
        session,
        "/public/get_last_trades_by_instrument_and_time",
        {
            "instrument_name": instrument_name,
            "start_timestamp": ts_ms,
            "count": 1,
            "sorting": "asc",
        },
    )
    trades = payload.get("result", {}).get("trades", [])
    return trades[0] if trades else None


def choose_nearest_trade(
    before: Optional[Dict[str, Any]],
    after: Optional[Dict[str, Any]],
    ts_ms: int,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    candidates: list[tuple[int, str, Dict[str, Any]]] = []
    for label, trade in (("before", before), ("after", after)):
        if trade and trade.get("timestamp") is not None:
            gap_ms = abs(int(trade["timestamp"]) - ts_ms)
            candidates.append((gap_ms, label, trade))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (x[0], 0 if x[1] == "before" else 1))
    _, label, trade = candidates[0]
    return trade, label


def fetch_nearest_trade_mark(
    session: requests.Session,
    instrument_name: str,
    ts_ms: int,
    max_gap_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    before = get_trade_mark_before(session, instrument_name, ts_ms)
    after = get_trade_mark_after(session, instrument_name, ts_ms)
    trade, side = choose_nearest_trade(before, after, ts_ms)

    base: Dict[str, Any] = {
        "instrument_name": instrument_name,
        "query_timestamp_ms": ts_ms,
        "trade_found": False,
        "chosen_side": None,
        "trade_timestamp_ms": None,
        "gap_seconds": None,
        "trade_price": None,
        "trade_mark_price": None,
        "trade_index_price": None,
        "trade_amount": None,
        "trade_direction": None,
        "trade_id": None,
        "accepted_under_max_gap": None,
    }

    if trade is None:
        base["accepted_under_max_gap"] = False if max_gap_seconds is not None else None
        return base

    gap_seconds = abs(int(trade["timestamp"]) - ts_ms) / 1000.0
    accepted = True if max_gap_seconds is None else gap_seconds <= max_gap_seconds

    base.update(
        {
            "trade_found": True,
            "chosen_side": side,
            "trade_timestamp_ms": int(trade.get("timestamp")),
            "gap_seconds": gap_seconds,
            "trade_price": safe_float(trade.get("price")),
            "trade_mark_price": safe_float(trade.get("mark_price")),
            "trade_index_price": safe_float(trade.get("index_price")),
            "trade_amount": safe_float(trade.get("amount")),
            "trade_direction": trade.get("direction"),
            "trade_id": trade.get("trade_id"),
            "accepted_under_max_gap": accepted,
        }
    )
    return base


def backfill_trade_marks(
    request_rows: pd.DataFrame,
    instrument_col: str,
    timestamp_ms_col: str,
    sleep_seconds: float,
    max_gap_seconds: Optional[float],
) -> pd.DataFrame:
    if request_rows.empty:
        return pd.DataFrame()

    session = requests.Session()
    session.headers.update({"User-Agent": "SVCJ-Deribit-Pricer/1.0 (trade-mark backfill)"})

    out_rows: list[Dict[str, Any]] = []
    records = request_rows.to_dict("records")

    for rec in tqdm(records, desc="Fetching Deribit trade marks", unit="req"):
        instrument_name = str(rec[instrument_col])
        ts_ms = int(rec[timestamp_ms_col])
        result = fetch_nearest_trade_mark(
            session=session,
            instrument_name=instrument_name,
            ts_ms=ts_ms,
            max_gap_seconds=max_gap_seconds,
        )
        result.update(rec)
        out_rows.append(result)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return pd.DataFrame(out_rows)


# -----------------------------
# Incremental builders
# -----------------------------
def build_term_incremental(
    term_snapshot: pd.DataFrame,
    existing_term: pd.DataFrame,
    skip_term_trade_backfill: bool,
    sleep_seconds: float,
    max_gap_seconds: Optional[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["timestamp", "currency", "futures_instrument_name"]
    new_term = filter_new_rows(term_snapshot, existing_term, key_cols)
    if new_term.empty:
        return existing_term.copy(), new_term

    term_new = new_term.copy()

    if not skip_term_trade_backfill:
        term_requests = term_new[
            ["timestamp", "currency", "futures_instrument_name", "snapshot_timestamp_ms", "snapshot_file"]
        ].copy()
        term_trade = backfill_trade_marks(
            request_rows=term_requests.rename(columns={"futures_instrument_name": "instrument_name"}),
            instrument_col="instrument_name",
            timestamp_ms_col="snapshot_timestamp_ms",
            sleep_seconds=sleep_seconds,
            max_gap_seconds=max_gap_seconds,
        )
        if not term_trade.empty:
            term_trade = term_trade.rename(
                columns={
                    "trade_mark_price": "term_trade_mark_price",
                    "trade_price": "term_trade_price",
                    "trade_index_price": "term_trade_index_price",
                    "trade_amount": "term_trade_amount",
                    "trade_direction": "term_trade_direction",
                    "trade_timestamp_ms": "term_trade_timestamp_ms",
                    "gap_seconds": "term_trade_gap_seconds",
                    "trade_found": "term_trade_found",
                    "chosen_side": "term_trade_side",
                    "accepted_under_max_gap": "term_trade_accepted_under_max_gap",
                    "trade_id": "term_trade_id",
                    "instrument_name": "futures_instrument_name",
                }
            )
            term_new = term_new.merge(
                term_trade[
                    [
                        "timestamp",
                        "currency",
                        "snapshot_file",
                        "futures_instrument_name",
                        "term_trade_found",
                        "term_trade_side",
                        "term_trade_timestamp_ms",
                        "term_trade_gap_seconds",
                        "term_trade_price",
                        "term_trade_mark_price",
                        "term_trade_index_price",
                        "term_trade_amount",
                        "term_trade_direction",
                        "term_trade_id",
                        "term_trade_accepted_under_max_gap",
                    ]
                ],
                on=["timestamp", "currency", "snapshot_file", "futures_instrument_name"],
                how="left",
            )
    else:
        for col in [
            "term_trade_found",
            "term_trade_side",
            "term_trade_timestamp_ms",
            "term_trade_gap_seconds",
            "term_trade_price",
            "term_trade_mark_price",
            "term_trade_index_price",
            "term_trade_amount",
            "term_trade_direction",
            "term_trade_id",
            "term_trade_accepted_under_max_gap",
        ]:
            term_new[col] = pd.NA

    for col in [
        "term_trade_found",
        "term_trade_side",
        "term_trade_timestamp_ms",
        "term_trade_gap_seconds",
        "term_trade_price",
        "term_trade_mark_price",
        "term_trade_index_price",
        "term_trade_amount",
        "term_trade_direction",
        "term_trade_id",
        "term_trade_accepted_under_max_gap",
    ]:
        if col not in term_new.columns:
            term_new[col] = pd.NA

    term_new["term_mark_price_final"] = term_new["snapshot_term_mark_price"]
    term_new["term_mark_price_final_source"] = "snapshot_underlying_price"

    missing_snapshot = term_new["term_mark_price_final"].isna() & term_new["term_trade_mark_price"].notna()
    if missing_snapshot.any():
        term_new.loc[missing_snapshot, "term_mark_price_final"] = term_new.loc[
            missing_snapshot, "term_trade_mark_price"
        ].astype(float)
        term_new.loc[missing_snapshot, "term_mark_price_final_source"] = "nearest_trade_mark_price"

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if combos.empty:
        return existing_perp.copy(), pd.DataFrame()

    perp_requests = combos.copy()
    perp_requests["obs_datetime"] = perp_requests["timestamp"]
    perp_requests["coin"] = perp_requests["currency"]
    perp_requests["instrument_name"] = perp_requests["coin"].astype(str) + "-PERPETUAL"

    key_cols = ["obs_datetime", "coin"]
    new_requests = filter_new_rows(perp_requests, existing_perp, key_cols)
    if new_requests.empty:
        return existing_perp.copy(), pd.DataFrame()

    if skip_perp_backfill:
        perp_new = new_requests.copy()
        perp_new["perp_futures_mark_price"] = pd.NA
        perp_new["perp_futures_best_bid"] = pd.NA
        perp_new["perp_futures_best_ask"] = pd.NA
        perp_new["data_source"] = "skipped"
        for col in [
            "trade_found",
            "chosen_side",
            "trade_timestamp_ms",
            "gap_seconds",
            "trade_price",
            "trade_index_price",
            "trade_amount",
            "trade_direction",
            "trade_id",
            "accepted_under_max_gap",
        ]:
            perp_new[col] = pd.NA
    else:
        perp_backfill = backfill_trade_marks(
            request_rows=new_requests,
            instrument_col="instrument_name",
            timestamp_ms_col="snapshot_timestamp_ms",
            sleep_seconds=sleep_seconds,
            max_gap_seconds=max_gap_seconds,
        )
        perp_new = perp_backfill.rename(columns={"trade_mark_price": "perp_futures_mark_price"})
        perp_new["perp_futures_best_bid"] = pd.NA
        perp_new["perp_futures_best_ask"] = pd.NA
        perp_new["data_source"] = "nearest_trade_mark_price"

    if "timestamp" in perp_new.columns:
        perp_new = perp_new.drop(columns=["timestamp"], errors="ignore")
    if "currency" in perp_new.columns:
        perp_new = perp_new.drop(columns=["currency"], errors="ignore")

    updated_perp = concat_dedup_sort(
        existing_df=existing_perp,
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
            "The script reads existing CSVs if they exist, appends only missing rows, "
            "and rewrites the saved CSVs in place."
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
        "--skip-term-trade-backfill",
        action="store_true",
        help="Do not call Deribit for missing term-future nearest-trade validation rows.",
    )
    ap.add_argument(
        "--max-gap-seconds",
        type=float,
        default=900.0,
        help="Maximum allowed distance between snapshot time and chosen trade. Default: 900 seconds.",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Sleep between Deribit requests, in seconds.",
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

    term_snapshot = build_term_futures_from_snapshots(
        snapshot_files=snapshot_files,
        include_synthetics=args.include_synthetics,
    )

    term_out_path = outdir / TERM_OUTPUT_NAME
    perp_out_path = outdir / PERP_OUTPUT_NAME
    existing_term = read_csv_if_exists(term_out_path)
    existing_perp = read_csv_if_exists(perp_out_path)

    updated_term, term_appended = build_term_incremental(
        term_snapshot=term_snapshot,
        existing_term=existing_term,
        skip_term_trade_backfill=args.skip_term_trade_backfill,
        sleep_seconds=args.sleep,
        max_gap_seconds=args.max_gap_seconds,
    )
    updated_term.to_csv(term_out_path, index=False)

    updated_perp, perp_appended = build_perp_incremental(
        combos=combos,
        existing_perp=existing_perp,
        skip_perp_backfill=args.skip_perp_backfill,
        sleep_seconds=args.sleep,
        max_gap_seconds=args.max_gap_seconds,
    )
    updated_perp.to_csv(perp_out_path, index=False)

    print(f"Wrote snapshot combos: {combos_out}")
    print(f"Wrote term futures marks: {term_out_path}")
    print(f"Wrote perpetual futures history: {perp_out_path}")
    print(f"Appended term rows: {len(term_appended):,}")
    print(f"Appended perp rows: {len(perp_appended):,}")
    print(f"Total term rows saved: {len(updated_term):,}")
    print(f"Total perp rows saved: {len(updated_perp):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
