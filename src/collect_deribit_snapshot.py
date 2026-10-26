#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

DERIBIT_BASE = "https://www.deribit.com/api/v2"
ORDER_BOOK_DEPTH = 20
DERIBIT_ORDER_BOOK_DEPTHS = (1, 5, 10, 20, 50, 100, 1000, 10000)

LEGACY_OPTION_COLUMNS = [
    "timestamp",
    "instrument_name",
    "underlying",
    "underlying_name",
    "option_type",
    "strike",
    "expiry_datetime",
    "time_to_maturity",
    "bid_price",
    "ask_price",
    "mark_price",
    "mid_price",
    "futures_price",
    "futures_instrument_name",
    "implied_volatility",
    "delta",
    "vega",
    "open_interest",
    "volume",
    "bid_ask_spread",
]

OPTION_EXTRA_COLUMNS = [
    "snapshot_id",
    "requested_at_utc",
    "received_at_utc",
    "exchange_timestamp",
    "collection_latency_ms",
    "bid_amount",
    "ask_amount",
    "last_price",
    "currency",
    "index_price",
    "volume_usd",
]

FUTURE_COLUMNS = [
    "timestamp",
    "snapshot_id",
    "currency",
    "instrument_name",
    "is_perpetual",
    "expiry_datetime",
    "time_to_maturity",
    "index_price",
    "underlying_price",
    "underlying_name",
    "bid_price",
    "ask_price",
    "bid_amount",
    "ask_amount",
    "mid_price",
    "mark_price",
    "last_price",
    "open_interest",
    "volume",
    "volume_usd",
    "current_funding",
    "funding_8h",
    "estimated_delivery_price",
    "interest_value",
    "requested_at_utc",
    "received_at_utc",
    "exchange_timestamp",
    "collection_latency_ms",
]

ORDER_BOOK_COLUMNS = [
    "snapshot_id",
    "currency",
    "instrument_name",
    "instrument_kind",
    "requested_at_utc",
    "received_at_utc",
    "exchange_timestamp",
    "side",
    "level",
    "price",
    "amount",
]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def ms_to_dt_utc(ms: Any) -> Optional[dt.datetime]:
    try:
        ms_int = int(ms)
        return dt.datetime.fromtimestamp(ms_int / 1000.0, tz=dt.timezone.utc)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def latency_ms(requested_at: dt.datetime, received_at: dt.datetime) -> float:
    return (received_at - requested_at).total_seconds() * 1000.0


def api_order_book_depth(depth: int) -> int:
    if depth <= 0:
        raise ValueError("order-book depth must be positive")
    for supported in DERIBIT_ORDER_BOOK_DEPTHS:
        if depth <= supported:
            return supported
    raise ValueError(f"order-book depth cannot exceed {DERIBIT_ORDER_BOOK_DEPTHS[-1]}")


def _top_amount(order_book: Dict[str, Any], side: str) -> Optional[float]:
    direct_key = "best_bid_amount" if side == "bid" else "best_ask_amount"
    direct = safe_float(order_book.get(direct_key))
    if direct is not None:
        return direct
    levels = order_book.get("bids" if side == "bid" else "asks", [])
    if not levels:
        return None
    level = levels[0]
    if not isinstance(level, (list, tuple)) or len(level) < 2:
        return None
    return safe_float(level[-1])


def _stats_value(payload: Dict[str, Any], key: str) -> Any:
    stats = payload.get("stats")
    if isinstance(stats, dict):
        return stats.get(key)
    return None


def _ordered_dataframe(rows: List[Dict[str, Any]], columns: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = None
    extras = [column for column in df.columns if column not in columns]
    return df[columns + extras]


class RawResponseArchive:
    """Gzip-compressed JSONL archive of the exact Deribit API responses."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: Any = None

    def __enter__(self) -> "RawResponseArchive":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = gzip.open(self.path, "wt", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(
        self,
        *,
        path: str,
        params: Dict[str, Any],
        requested_at: dt.datetime,
        received_at: dt.datetime,
        payload: Any,
        status_code: Optional[int] = None,
    ) -> None:
        if self._fh is None:
            raise RuntimeError("RawResponseArchive must be opened with a context manager")
        record = {
            "path": path,
            "params": params,
            "requested_at_utc": requested_at.isoformat(),
            "received_at_utc": received_at.isoformat(),
            "latency_ms": latency_ms(requested_at, received_at),
            "status_code": status_code,
            "payload": payload,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self._fh.write("\n")
        self._fh.flush()


def http_get(
    session: requests.Session,
    path: str,
    params: Dict[str, Any] | None = None,
    timeout: int = 30,
    raw_archive: Optional[RawResponseArchive] = None,
) -> Tuple[Dict[str, Any], dt.datetime, dt.datetime]:
    request_params = dict(params or {})
    url = f"{DERIBIT_BASE}{path}"
    requested_at = utc_now()
    r = session.get(url, params=request_params, timeout=timeout)
    received_at = utc_now()
    try:
        payload: Any = r.json()
    except Exception:
        payload = {"_non_json_body": r.text}
    if raw_archive is not None:
        raw_archive.write(path=path, params=request_params, requested_at=requested_at, received_at=received_at, payload=payload, status_code=r.status_code)
    r.raise_for_status()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Deribit payload type from {path}: {type(payload).__name__}")
    if payload.get("error"):
        raise RuntimeError(f"Deribit API error: {payload['error']}")
    return payload, requested_at, received_at


def get_instruments(session: requests.Session, currency: str, kind: str = "option", raw_archive: Optional[RawResponseArchive] = None) -> Tuple[List[Dict[str, Any]], dt.datetime, dt.datetime]:
    payload, requested_at, received_at = http_get(session, "/public/get_instruments", {"currency": currency, "kind": kind, "expired": "false"}, raw_archive=raw_archive)
    result = payload.get("result", [])
    return (result if isinstance(result, list) else []), requested_at, received_at


def get_book_summary(session: requests.Session, currency: str, kind: str, raw_archive: Optional[RawResponseArchive] = None) -> Tuple[List[Dict[str, Any]], dt.datetime, dt.datetime]:
    payload, requested_at, received_at = http_get(session, "/public/get_book_summary_by_currency", {"currency": currency, "kind": kind}, raw_archive=raw_archive)
    result = payload.get("result", [])
    return (result if isinstance(result, list) else []), requested_at, received_at


def get_book_summary_options(session: requests.Session, currency: str, raw_archive: Optional[RawResponseArchive] = None) -> List[Dict[str, Any]]:
    result, _, _ = get_book_summary(session, currency, "option", raw_archive=raw_archive)
    return result


def get_order_book(session: requests.Session, instrument_name: str, depth: int, raw_archive: Optional[RawResponseArchive] = None) -> Tuple[Dict[str, Any], dt.datetime, dt.datetime]:
    payload, requested_at, received_at = http_get(session, "/public/get_order_book", {"instrument_name": instrument_name, "depth": api_order_book_depth(depth)}, raw_archive=raw_archive)
    result = payload.get("result", {})
    return (result if isinstance(result, dict) else {}), requested_at, received_at


def get_ticker(session: requests.Session, instrument_name: str) -> Dict[str, Any]:
    payload, _, _ = http_get(session, "/public/ticker", {"instrument_name": instrument_name})
    result = payload.get("result", {})
    return result if isinstance(result, dict) else {}


def normalize_order_book(order_book: Dict[str, Any], *, snapshot_id: str, currency: str, instrument_name: str, instrument_kind: str, requested_at: dt.datetime, received_at: dt.datetime, depth: int = ORDER_BOOK_DEPTH) -> List[Dict[str, Any]]:
    exchange_dt = ms_to_dt_utc(order_book.get("timestamp"))
    exchange_iso = exchange_dt.isoformat() if exchange_dt else None
    rows: List[Dict[str, Any]] = []
    for side, key in (("bid", "bids"), ("ask", "asks")):
        levels = order_book.get(key, [])
        if not isinstance(levels, list):
            continue
        for level_number, level in enumerate(levels[:depth], start=1):
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                continue
            rows.append({"snapshot_id": snapshot_id, "currency": currency, "instrument_name": instrument_name, "instrument_kind": instrument_kind, "requested_at_utc": requested_at.isoformat(), "received_at_utc": received_at.isoformat(), "exchange_timestamp": exchange_iso, "side": side, "level": level_number, "price": safe_float(level[-2]), "amount": safe_float(level[-1])})
    return rows


def build_option_row(*, inst: Dict[str, Any], summary: Dict[str, Any], order_book: Dict[str, Any], currency: str, run_ts: dt.datetime, snapshot_id: str, requested_at: dt.datetime, received_at: dt.datetime) -> Dict[str, Any]:
    greeks = order_book.get("greeks") if isinstance(order_book.get("greeks"), dict) else {}
    expiry_dt = ms_to_dt_utc(inst.get("expiration_timestamp"))
    expiry_iso = expiry_dt.isoformat() if expiry_dt else None
    ttm_years = (expiry_dt - run_ts).total_seconds() / (365.25 * 24 * 3600) if expiry_dt else None
    bid = safe_float(first_not_none(order_book.get("best_bid_price"), summary.get("bid_price")))
    ask = safe_float(first_not_none(order_book.get("best_ask_price"), summary.get("ask_price")))
    mid = 0.5 * (bid + ask) if bid is not None and ask is not None else safe_float(summary.get("mid_price"))
    mark = safe_float(first_not_none(order_book.get("mark_price"), summary.get("mark_price")))
    underlying_price = safe_float(first_not_none(order_book.get("underlying_price"), summary.get("underlying_price")))
    underlying_name = first_not_none(order_book.get("underlying_index"), summary.get("underlying_index"))
    index_price = safe_float(order_book.get("index_price"))
    mark_iv_pct = safe_float(first_not_none(order_book.get("mark_iv"), summary.get("mark_iv")))
    implied_vol = (mark_iv_pct / 100.0) if mark_iv_pct is not None else None
    futures_instrument_name = None
    futures_price = None
    if isinstance(underlying_name, str) and underlying_name != "index_price":
        futures_instrument_name = underlying_name
        futures_price = underlying_price
    exchange_dt = ms_to_dt_utc(order_book.get("timestamp"))
    spread = ask - bid if bid is not None and ask is not None else None
    return {
        "timestamp": run_ts.isoformat(), "instrument_name": inst.get("instrument_name"), "underlying": underlying_price, "underlying_name": underlying_name,
        "option_type": inst.get("option_type"), "strike": safe_float(inst.get("strike")), "expiry_datetime": expiry_iso, "time_to_maturity": ttm_years,
        "bid_price": bid, "ask_price": ask, "mark_price": mark, "mid_price": mid, "futures_price": futures_price, "futures_instrument_name": futures_instrument_name,
        "implied_volatility": implied_vol, "delta": safe_float(greeks.get("delta")), "vega": safe_float(greeks.get("vega")),
        "open_interest": safe_float(first_not_none(order_book.get("open_interest"), summary.get("open_interest"))),
        "volume": safe_float(first_not_none(_stats_value(order_book, "volume"), order_book.get("volume"), summary.get("volume"))), "bid_ask_spread": spread,
        "snapshot_id": snapshot_id, "requested_at_utc": requested_at.isoformat(), "received_at_utc": received_at.isoformat(),
        "exchange_timestamp": exchange_dt.isoformat() if exchange_dt else None, "collection_latency_ms": latency_ms(requested_at, received_at),
        "bid_amount": _top_amount(order_book, "bid"), "ask_amount": _top_amount(order_book, "ask"), "last_price": safe_float(order_book.get("last_price")),
        "currency": currency, "index_price": index_price,
        "volume_usd": safe_float(first_not_none(_stats_value(order_book, "volume_usd"), order_book.get("volume_usd"), summary.get("volume_usd"))),
    }


def build_future_row(*, inst: Dict[str, Any], summary: Dict[str, Any], order_book: Dict[str, Any], currency: str, run_ts: dt.datetime, snapshot_id: str, requested_at: dt.datetime, received_at: dt.datetime) -> Dict[str, Any]:
    instrument_name = str(inst.get("instrument_name") or order_book.get("instrument_name") or "")
    is_perpetual = instrument_name.endswith("-PERPETUAL")
    expiry_dt = None if is_perpetual else ms_to_dt_utc(inst.get("expiration_timestamp"))
    ttm_years = (expiry_dt - run_ts).total_seconds() / (365.25 * 24 * 3600) if expiry_dt else None
    bid = safe_float(first_not_none(order_book.get("best_bid_price"), summary.get("bid_price")))
    ask = safe_float(first_not_none(order_book.get("best_ask_price"), summary.get("ask_price")))
    mid = 0.5 * (bid + ask) if bid is not None and ask is not None else safe_float(summary.get("mid_price"))
    exchange_dt = ms_to_dt_utc(order_book.get("timestamp"))
    return {
        "timestamp": run_ts.isoformat(), "snapshot_id": snapshot_id, "currency": currency, "instrument_name": instrument_name, "is_perpetual": is_perpetual,
        "expiry_datetime": expiry_dt.isoformat() if expiry_dt else None, "time_to_maturity": ttm_years, "index_price": safe_float(order_book.get("index_price")),
        "underlying_price": safe_float(first_not_none(order_book.get("underlying_price"), summary.get("underlying_price"))),
        "underlying_name": first_not_none(order_book.get("underlying_index"), summary.get("underlying_index")), "bid_price": bid, "ask_price": ask,
        "bid_amount": _top_amount(order_book, "bid"), "ask_amount": _top_amount(order_book, "ask"), "mid_price": mid,
        "mark_price": safe_float(first_not_none(order_book.get("mark_price"), summary.get("mark_price"))), "last_price": safe_float(order_book.get("last_price")),
        "open_interest": safe_float(first_not_none(order_book.get("open_interest"), summary.get("open_interest"))),
        "volume": safe_float(first_not_none(_stats_value(order_book, "volume"), order_book.get("volume"), summary.get("volume"))),
        "volume_usd": safe_float(first_not_none(_stats_value(order_book, "volume_usd"), order_book.get("volume_usd"), summary.get("volume_usd"))),
        "current_funding": safe_float(order_book.get("current_funding")), "funding_8h": safe_float(order_book.get("funding_8h")),
        "estimated_delivery_price": safe_float(order_book.get("estimated_delivery_price")), "interest_value": safe_float(order_book.get("interest_value")),
        "requested_at_utc": requested_at.isoformat(), "received_at_utc": received_at.isoformat(),
        "exchange_timestamp": exchange_dt.isoformat() if exchange_dt else None, "collection_latency_ms": latency_ms(requested_at, received_at),
    }


def build_manifest(*, snapshot_id: str, started_at: dt.datetime, completed_at: dt.datetime, currencies: List[str], order_book_depth: int, max_instruments: int, option_stats: Dict[str, Dict[str, int]], future_stats: Dict[str, Dict[str, int]], failures: List[Dict[str, Any]], output_files: Dict[str, str], raw_archive_path: str) -> Dict[str, Any]:
    stats_complete = all(stats.get("collected", 0) == stats.get("selected", 0) and stats.get("failed", 0) == 0 for stats in list(option_stats.values()) + list(future_stats.values()))
    complete = stats_complete and not failures
    return {"snapshot_id": snapshot_id, "started_at_utc": started_at.isoformat(), "completed_at_utc": completed_at.isoformat(), "duration_seconds": (completed_at - started_at).total_seconds(), "currencies": currencies, "order_book_depth": order_book_depth, "deribit_order_book_request_depth": api_order_book_depth(order_book_depth), "max_option_instruments": max_instruments, "full_option_universe_requested": max_instruments <= 0, "status": "complete" if complete else "partial", "complete": complete, "options": option_stats, "futures": future_stats, "failures": failures, "output_files": output_files, "raw_archive": raw_archive_path}


def _failure(*, currency: str, kind: str, stage: str, error: Exception, instrument_name: Optional[str] = None) -> Dict[str, Any]:
    return {"currency": currency, "kind": kind, "stage": stage, "instrument_name": instrument_name, "error": f"{type(error).__name__}: {error}"}


def collect_kind(*, session: requests.Session, currency: str, kind: str, run_ts: dt.datetime, snapshot_id: str, raw_archive: RawResponseArchive, order_book_depth: int, sleep_seconds: float, failures: List[Dict[str, Any]], max_instruments: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    try:
        inst_list, _, _ = get_instruments(session, currency, kind=kind, raw_archive=raw_archive)
    except Exception as exc:
        failures.append(_failure(currency=currency, kind=kind, stage="get_instruments", error=exc))
        return [], [], {"discovered": 0, "selected": 0, "collected": 0, "failed": 0}
    inst_map = {inst["instrument_name"]: inst for inst in inst_list if isinstance(inst, dict) and inst.get("instrument_name")}
    instrument_names = list(inst_map)
    if kind == "option" and max_instruments > 0:
        instrument_names = instrument_names[:max_instruments]
    summary_map: Dict[str, Dict[str, Any]] = {}
    try:
        summary_list, _, _ = get_book_summary(session, currency, kind, raw_archive=raw_archive)
        summary_map = {item["instrument_name"]: item for item in summary_list if isinstance(item, dict) and item.get("instrument_name")}
    except Exception as exc:
        failures.append(_failure(currency=currency, kind=kind, stage="get_book_summary", error=exc))
    normalized_rows: List[Dict[str, Any]] = []
    order_book_rows: List[Dict[str, Any]] = []
    failed_count = 0
    for instrument_name in instrument_names:
        try:
            order_book, requested_at, received_at = get_order_book(session, instrument_name, depth=order_book_depth, raw_archive=raw_archive)
            inst = inst_map[instrument_name]
            summary = summary_map.get(instrument_name, {})
            if kind == "option":
                normalized_rows.append(build_option_row(inst=inst, summary=summary, order_book=order_book, currency=currency, run_ts=run_ts, snapshot_id=snapshot_id, requested_at=requested_at, received_at=received_at))
            else:
                normalized_rows.append(build_future_row(inst=inst, summary=summary, order_book=order_book, currency=currency, run_ts=run_ts, snapshot_id=snapshot_id, requested_at=requested_at, received_at=received_at))
            order_book_rows.extend(normalize_order_book(order_book, snapshot_id=snapshot_id, currency=currency, instrument_name=instrument_name, instrument_kind=kind, requested_at=requested_at, received_at=received_at, depth=order_book_depth))
        except Exception as exc:
            failed_count += 1
            failures.append(_failure(currency=currency, kind=kind, stage="get_order_book", instrument_name=instrument_name, error=exc))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    stats = {"discovered": len(inst_map), "selected": len(instrument_names), "collected": len(normalized_rows), "failed": failed_count}
    return normalized_rows, order_book_rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", choices=["BTC", "ETH", "BOTH"], default="BOTH")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--max-instruments", type=int, default=0, help="0 = no cap; cap applies to options only.")
    ap.add_argument("--sleep", type=float, default=0.02, help="Sleep between per-instrument order-book calls.")
    ap.add_argument("--order-book-depth", type=int, default=ORDER_BOOK_DEPTH, help="Number of order-book levels to persist per side (default: 20).")
    args = ap.parse_args()
    if args.order_book_depth <= 0:
        ap.error("--order-book-depth must be positive")
    api_order_book_depth(args.order_book_depth)
    currencies = ["BTC", "ETH"] if args.currency == "BOTH" else [args.currency]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_ts = utc_now()
    snapshot_id = run_ts.strftime("%Y%m%dT%H%M%SZ")
    raw_archive_path = outdir / "raw" / "deribit" / snapshot_id / "api_responses.jsonl.gz"
    session = requests.Session()
    session.headers.update({"User-Agent": "SVCJ-Deribit-Pricer/2.0 (snapshot collector)"})
    option_rows: List[Dict[str, Any]] = []
    future_rows: List[Dict[str, Any]] = []
    order_book_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    option_stats: Dict[str, Dict[str, int]] = {}
    future_stats: Dict[str, Dict[str, int]] = {}
    with RawResponseArchive(raw_archive_path) as raw_archive:
        for currency in currencies:
            options, option_books, stats = collect_kind(session=session, currency=currency, kind="option", run_ts=run_ts, snapshot_id=snapshot_id, raw_archive=raw_archive, order_book_depth=args.order_book_depth, sleep_seconds=args.sleep, failures=failures, max_instruments=args.max_instruments)
            option_rows.extend(options)
            order_book_rows.extend(option_books)
            option_stats[currency] = stats
            futures, future_books, stats = collect_kind(session=session, currency=currency, kind="future", run_ts=run_ts, snapshot_id=snapshot_id, raw_archive=raw_archive, order_book_depth=args.order_book_depth, sleep_seconds=args.sleep, failures=failures, max_instruments=0)
            future_rows.extend(futures)
            order_book_rows.extend(future_books)
            future_stats[currency] = stats
    options_path = outdir / f"deribit_options_snapshot_{snapshot_id}.csv"
    futures_path = outdir / f"deribit_futures_snapshot_{snapshot_id}.csv"
    order_book_path = outdir / f"deribit_order_book_snapshot_{snapshot_id}.csv.gz"
    manifest_path = outdir / f"deribit_snapshot_manifest_{snapshot_id}.json"
    option_df = _ordered_dataframe(option_rows, LEGACY_OPTION_COLUMNS + OPTION_EXTRA_COLUMNS)
    future_df = _ordered_dataframe(future_rows, FUTURE_COLUMNS)
    order_book_df = _ordered_dataframe(order_book_rows, ORDER_BOOK_COLUMNS)
    option_df.to_csv(options_path, index=False)
    future_df.to_csv(futures_path, index=False)
    order_book_df.to_csv(order_book_path, index=False, compression="gzip")
    completed_at = utc_now()
    output_files = {"options": str(options_path), "futures": str(futures_path), "order_books": str(order_book_path), "manifest": str(manifest_path)}
    manifest = build_manifest(snapshot_id=snapshot_id, started_at=run_ts, completed_at=completed_at, currencies=currencies, order_book_depth=args.order_book_depth, max_instruments=args.max_instruments, option_stats=option_stats, future_stats=future_stats, failures=failures, output_files=output_files, raw_archive_path=str(raw_archive_path))
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"Wrote {len(option_df):,} option rows to: {options_path}")
    print(f"Wrote {len(future_df):,} future rows to: {futures_path}")
    print(f"Wrote {len(order_book_df):,} order-book levels to: {order_book_path}")
    print(f"Wrote raw API archive to: {raw_archive_path}")
    print(f"Wrote {manifest['status']} manifest to: {manifest_path}")
    if failures:
        print(f"WARNING: snapshot completed with {len(failures)} recorded failure(s); see manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
