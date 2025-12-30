#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

DERIBIT_BASE = "https://www.deribit.com/api/v2"


# -----------------------------
# Helpers
# -----------------------------
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


def http_get(session: requests.Session, path: str, params: Dict[str, Any] | None = None, timeout: int = 30) -> Dict[str, Any]:
    url = f"{DERIBIT_BASE}{path}"
    r = session.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload and payload["error"]:
        raise RuntimeError(f"Deribit API error: {payload['error']}")
    return payload


# -----------------------------
# Deribit API wrappers
# -----------------------------
def get_instruments(session: requests.Session, currency: str) -> List[Dict[str, Any]]:
    # Needed for strike / expiry / option_type
    payload = http_get(session, "/public/get_instruments", {"currency": currency, "kind": "option", "expired": "false"})
    return payload.get("result", [])


def get_book_summary_options(session: requests.Session, currency: str) -> List[Dict[str, Any]]:
    # Fast: prices + OI + volume + mid + mark_iv for all options at once
    payload = http_get(session, "/public/get_book_summary_by_currency", {"currency": currency, "kind": "option"})
    return payload.get("result", [])


def get_ticker(session: requests.Session, instrument_name: str) -> Dict[str, Any]:
    payload = http_get(session, "/public/ticker", {"instrument_name": instrument_name})
    return payload.get("result", {})


# -----------------------------
# Main collection logic
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", choices=["BTC", "ETH", "BOTH"], default="BOTH")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--max-instruments", type=int, default=0, help="0 = no cap (useful for quick tests).")
    ap.add_argument("--sleep", type=float, default=0.02, help="Sleep between ticker calls (rate-limit friendly).")
    args = ap.parse_args()

    currencies = ["BTC", "ETH"] if args.currency == "BOTH" else [args.currency]

    os.makedirs(args.outdir, exist_ok=True)
    run_ts = utc_now()
    run_ts_str = run_ts.strftime("%Y%m%dT%H%M%SZ")

    session = requests.Session()
    session.headers.update({"User-Agent": "SVCJ-Deribit-Pricer/1.0 (snapshot collector)"})

    rows: List[Dict[str, Any]] = []

    for ccy in currencies:
        # 1) Instrument metadata (strike/expiry/type)
        inst_list = get_instruments(session, ccy)
        inst_map = {i["instrument_name"]: i for i in inst_list if "instrument_name" in i}

        # 2) Fast surface snapshot for prices/OI/volume
        summary_list = get_book_summary_options(session, ccy)

        # Filter to just options that appear in inst_map
        option_names = [s.get("instrument_name") for s in summary_list if s.get("instrument_name") in inst_map]

        # Optional cap for quick testing
        if args.max_instruments and args.max_instruments > 0:
            option_names = option_names[: args.max_instruments]

        # Build a quick summary dict by instrument_name
        summary_map = {s["instrument_name"]: s for s in summary_list if s.get("instrument_name") in option_names}

        # 3) For Greeks + mark IV consistent with Deribit ticker: call ticker per instrument
        for name in option_names:
            inst = inst_map.get(name, {})
            summ = summary_map.get(name, {})

            t = get_ticker(session, name)
            greeks = t.get("greeks", {}) if isinstance(t.get("greeks"), dict) else {}

            exp_ts = inst.get("expiration_timestamp")
            expiry_dt = ms_to_dt_utc(exp_ts)
            expiry_iso = expiry_dt.isoformat() if expiry_dt else None
            ttm_years = None
            if expiry_dt:
                ttm_years = (expiry_dt - run_ts).total_seconds() / (365.25 * 24 * 3600)

            # Prices from summary (fast and includes volume/open_interest)
            bid = safe_float(summ.get("bid_price"))
            ask = safe_float(summ.get("ask_price"))
            mid = safe_float(summ.get("mid_price"))
            mark = safe_float(summ.get("mark_price"))
            underlying_price = safe_float(summ.get("underlying_price"))  # often forward/synthetic underlying
            underlying_name = summ.get("underlying_index") or t.get("underlying_index")

            # Deribit ticker also provides index_price (spot index)
            index_price = safe_float(t.get("index_price"))

            # Spread (simple)
            spread = None
            if bid is not None and ask is not None:
                spread = ask - bid

            # If mid missing, compute it
            if mid is None and bid is not None and ask is not None:
                mid = 0.5 * (bid + ask)

            # Implied vol: use summary mark_iv (%). Convert to decimal.
            mark_iv_pct = safe_float(summ.get("mark_iv"))
            implied_vol = (mark_iv_pct / 100.0) if mark_iv_pct is not None else None

            # Futures identification (best available from underlying_index/underlying_price)
            # If underlying_name is "index_price", then it's spot index; otherwise it's usually SYN.<CCY>-<EXPIRY> or <CCY>-<EXPIRY>
            futures_instrument_name = None
            futures_price = None
            if isinstance(underlying_name, str) and underlying_name != "index_price":
                futures_instrument_name = underlying_name
                futures_price = underlying_price

            row = {
                # requested
                "timestamp": run_ts.isoformat(),
                "instrument_name": name,
                "underlying": underlying_price,
                "underlying_name": underlying_name,  # NEW column you requested
                "option_type": inst.get("option_type"),
                "strike": safe_float(inst.get("strike")),
                "expiry_datetime": expiry_iso,
                "time_to_maturity": ttm_years,

                "bid_price": bid,
                "ask_price": ask,
                "mark_price": mark,
                "mid_price": mid,

                "futures_price": futures_price,
                "futures_instrument_name": futures_instrument_name,

                "implied_volatility": implied_vol,
                "delta": safe_float(greeks.get("delta")),
                "vega": safe_float(greeks.get("vega")),

                "open_interest": safe_float(summ.get("open_interest")),
                "volume": safe_float(summ.get("volume")),
                "bid_ask_spread": spread,

                # extra useful fields (kept, but you can remove if you want)
                "currency": ccy,
                "index_price": index_price,
                "volume_usd": safe_float(summ.get("volume_usd")),
            }
            rows.append(row)

            time.sleep(args.sleep)

    df = pd.DataFrame(rows)

    # Reorder columns to match your requested list first, then extras
    requested_order = [
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
    extras = [c for c in df.columns if c not in requested_order]
    df = df[requested_order + extras]

    outpath = os.path.join(args.outdir, f"deribit_options_snapshot_{run_ts_str}.csv")
    df.to_csv(outpath, index=False)
    print(f"Wrote {len(df):,} rows to: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
