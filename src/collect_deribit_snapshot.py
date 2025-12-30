#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Any, Dict, List

import pandas as pd
import requests

DERIBIT_BASE = "https://www.deribit.com/api/v2"


def http_get(path: str, params: Dict[str, Any] | None = None, timeout: int = 30) -> Dict[str, Any]:
    url = f"{DERIBIT_BASE}{path}"
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload and payload["error"]:
        raise RuntimeError(f"Deribit API error: {payload['error']}")
    return payload


def get_instruments(currency: str) -> List[Dict[str, Any]]:
    payload = http_get("/public/get_instruments", {"currency": currency, "kind": "option", "expired": "false"})
    return payload.get("result", [])


def get_ticker(instrument_name: str) -> Dict[str, Any]:
    payload = http_get("/public/ticker", {"instrument_name": instrument_name})
    return payload.get("result", {})


def safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", choices=["BTC", "ETH", "BOTH"], default="BOTH")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--max-instruments", type=int, default=0, help="0 = no cap (can be slow).")
    args = ap.parse_args()

    currencies = ["BTC", "ETH"] if args.currency == "BOTH" else [args.currency]

    os.makedirs(args.outdir, exist_ok=True)
    ts = dt.datetime.utcnow().replace(microsecond=0).isoformat().replace(":", "").replace("-", "")
    rows: List[Dict[str, Any]] = []

    for ccy in currencies:
        instruments = get_instruments(ccy)

        if args.max_instruments and args.max_instruments > 0:
            instruments = instruments[: args.max_instruments]

        for inst in instruments:
            name = inst.get("instrument_name")
            if not name:
                continue

            t = get_ticker(name)

            rows.append(
                {
                    "ts_utc": dt.datetime.utcnow().isoformat(),
                    "currency": ccy,
                    "instrument_name": name,
                    "option_type": inst.get("option_type"),
                    "strike": safe_float(inst.get("strike")),
                    "expiration_timestamp": inst.get("expiration_timestamp"),
                    "underlying_price": safe_float(t.get("underlying_price")),
                    "mark_price": safe_float(t.get("mark_price")),
                    "best_bid_price": safe_float(t.get("best_bid_price")),
                    "best_ask_price": safe_float(t.get("best_ask_price")),
                    "index_price": safe_float(t.get("index_price")),
                    "mark_iv": safe_float(t.get("mark_iv")),
                    "delta": safe_float(t.get("greeks", {}).get("delta")) if isinstance(t.get("greeks"), dict) else None,
                    "vega": safe_float(t.get("greeks", {}).get("vega")) if isinstance(t.get("greeks"), dict) else None,
                    "open_interest": safe_float(t.get("open_interest")),
                    "volume": safe_float(t.get("volume")),
                    "volume_usd": safe_float(t.get("volume_usd")),
                }
            )

    df = pd.DataFrame(rows)
    outpath = os.path.join(args.outdir, f"deribit_options_snapshot_{ts}.csv")
    df.to_csv(outpath, index=False)
    print(f"Wrote {len(df):,} rows to: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
