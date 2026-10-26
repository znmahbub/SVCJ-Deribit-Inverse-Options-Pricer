from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import collect_deribit_snapshot as collector


UTC = dt.timezone.utc


class CollectorUnitTests(unittest.TestCase):
    def test_normalize_order_book_keeps_top_15_levels_per_side(self) -> None:
        requested_at = dt.datetime(2026, 10, 26, 8, 0, 0, tzinfo=UTC)
        received_at = requested_at + dt.timedelta(milliseconds=125)
        payload = {
            "timestamp": 1793001600123,
            "bids": [[100.0 - i, 1.0 + i] for i in range(20)],
            "asks": [[101.0 + i, 2.0 + i] for i in range(20)],
        }

        rows = collector.normalize_order_book(
            payload,
            snapshot_id="20261026T080000Z",
            currency="BTC",
            instrument_name="BTC-27NOV26-100000-C",
            instrument_kind="option",
            requested_at=requested_at,
            received_at=received_at,
            depth=15,
        )

        self.assertEqual(len(rows), 30)
        bids = [row for row in rows if row["side"] == "bid"]
        asks = [row for row in rows if row["side"] == "ask"]
        self.assertEqual([row["level"] for row in bids], list(range(1, 16)))
        self.assertEqual([row["level"] for row in asks], list(range(1, 16)))
        self.assertEqual(bids[0]["price"], 100.0)
        self.assertEqual(bids[0]["amount"], 1.0)
        self.assertEqual(asks[0]["price"], 101.0)
        self.assertEqual(asks[0]["amount"], 2.0)
        self.assertEqual(rows[0]["requested_at_utc"], requested_at.isoformat())
        self.assertEqual(rows[0]["received_at_utc"], received_at.isoformat())
        self.assertEqual(rows[0]["exchange_timestamp"], "2026-10-26T08:00:00.123000+00:00")

    def test_build_option_row_adds_per_call_timestamps_and_quote_amounts(self) -> None:
        run_ts = dt.datetime(2026, 10, 26, 8, 0, 0, tzinfo=UTC)
        requested_at = run_ts + dt.timedelta(seconds=1)
        received_at = requested_at + dt.timedelta(milliseconds=80)
        inst = {
            "instrument_name": "BTC-27NOV26-100000-C",
            "expiration_timestamp": 1795766400000,
            "option_type": "call",
            "strike": 100000.0,
        }
        summary = {
            "bid_price": 0.020,
            "ask_price": 0.021,
            "mid_price": 0.0205,
            "mark_price": 0.0204,
            "underlying_price": 101500.0,
            "underlying_index": "SYN.BTC-27NOV26",
            "mark_iv": 51.25,
            "open_interest": 123.0,
            "volume": 45.0,
            "volume_usd": 90000.0,
        }
        book = {
            "timestamp": 1793001601080,
            "index_price": 100900.0,
            "best_bid_amount": 4.5,
            "best_ask_amount": 7.0,
            "greeks": {"delta": 0.41, "vega": 12.5},
            "bids": [[0.020, 4.5]],
            "asks": [[0.021, 7.0]],
        }

        row = collector.build_option_row(
            inst=inst,
            summary=summary,
            order_book=book,
            currency="BTC",
            run_ts=run_ts,
            snapshot_id="20261026T080000Z",
            requested_at=requested_at,
            received_at=received_at,
        )

        self.assertEqual(row["timestamp"], run_ts.isoformat())
        self.assertEqual(row["snapshot_id"], "20261026T080000Z")
        self.assertEqual(row["requested_at_utc"], requested_at.isoformat())
        self.assertEqual(row["received_at_utc"], received_at.isoformat())
        self.assertEqual(row["exchange_timestamp"], "2026-10-26T08:00:01.080000+00:00")
        self.assertEqual(row["bid_amount"], 4.5)
        self.assertEqual(row["ask_amount"], 7.0)
        self.assertEqual(row["bid_price"], 0.020)
        self.assertEqual(row["ask_price"], 0.021)
        self.assertAlmostEqual(row["implied_volatility"], 0.5125)
        self.assertEqual(row["delta"], 0.41)
        self.assertEqual(row["vega"], 12.5)

    def test_build_future_row_classifies_perpetual_and_dated_futures(self) -> None:
        run_ts = dt.datetime(2026, 10, 26, 8, 0, 0, tzinfo=UTC)
        requested_at = run_ts + dt.timedelta(seconds=1)
        received_at = requested_at + dt.timedelta(milliseconds=50)
        perpetual = collector.build_future_row(
            inst={"instrument_name": "BTC-PERPETUAL", "expiration_timestamp": None},
            summary={"mark_price": 101000.0, "open_interest": 1000.0, "volume": 250.0},
            order_book={
                "timestamp": 1793001601050,
                "index_price": 100950.0,
                "best_bid_price": 100990.0,
                "best_ask_price": 101010.0,
                "best_bid_amount": 100.0,
                "best_ask_amount": 120.0,
                "current_funding": 0.0001,
                "funding_8h": 0.0008,
            },
            currency="BTC",
            run_ts=run_ts,
            snapshot_id="20261026T080000Z",
            requested_at=requested_at,
            received_at=received_at,
        )
        dated = collector.build_future_row(
            inst={"instrument_name": "BTC-27NOV26", "expiration_timestamp": 1795766400000},
            summary={"mark_price": 102000.0},
            order_book={"timestamp": 1793001601050, "index_price": 100950.0},
            currency="BTC",
            run_ts=run_ts,
            snapshot_id="20261026T080000Z",
            requested_at=requested_at,
            received_at=received_at,
        )

        self.assertTrue(perpetual["is_perpetual"])
        self.assertIsNone(perpetual["expiry_datetime"])
        self.assertEqual(perpetual["current_funding"], 0.0001)
        self.assertEqual(perpetual["funding_8h"], 0.0008)
        self.assertFalse(dated["is_perpetual"])
        self.assertEqual(dated["expiry_datetime"], "2026-11-27T08:00:00+00:00")

    def test_manifest_reports_partial_snapshot_when_any_instrument_fails(self) -> None:
        start = dt.datetime(2026, 10, 26, 8, 0, 0, tzinfo=UTC)
        end = start + dt.timedelta(seconds=10)
        manifest = collector.build_manifest(
            snapshot_id="20261026T080000Z",
            started_at=start,
            completed_at=end,
            currencies=["BTC", "ETH"],
            order_book_depth=15,
            max_instruments=0,
            option_stats={
                "BTC": {"discovered": 100, "selected": 100, "collected": 99, "failed": 1},
                "ETH": {"discovered": 80, "selected": 80, "collected": 80, "failed": 0},
            },
            future_stats={
                "BTC": {"discovered": 8, "selected": 8, "collected": 8, "failed": 0},
                "ETH": {"discovered": 7, "selected": 7, "collected": 7, "failed": 0},
            },
            failures=[{"currency": "BTC", "kind": "option", "instrument_name": "X", "error": "boom"}],
            output_files={"options": "data/options.csv"},
            raw_archive_path="data/raw/api_responses.jsonl.gz",
        )

        self.assertEqual(manifest["status"], "partial")
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["order_book_depth"], 15)
        self.assertEqual(manifest["failures"][0]["instrument_name"], "X")
        self.assertAlmostEqual(manifest["duration_seconds"], 10.0)

    def test_raw_response_archive_preserves_full_payload_and_call_metadata(self) -> None:
        requested_at = dt.datetime(2026, 10, 26, 8, 0, 0, tzinfo=UTC)
        received_at = requested_at + dt.timedelta(milliseconds=50)
        payload = {"jsonrpc": "2.0", "result": {"bids": [[1.0, 2.0]], "asks": [[1.1, 3.0]]}}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api.jsonl.gz"
            with collector.RawResponseArchive(path) as archive:
                archive.write(
                    path="/public/get_order_book",
                    params={"instrument_name": "BTC-PERPETUAL", "depth": 15},
                    requested_at=requested_at,
                    received_at=received_at,
                    payload=payload,
                )

            with gzip.open(path, "rt", encoding="utf-8") as fh:
                record = json.loads(fh.readline())

        self.assertEqual(record["path"], "/public/get_order_book")
        self.assertEqual(record["params"]["depth"], 15)
        self.assertEqual(record["requested_at_utc"], requested_at.isoformat())
        self.assertEqual(record["received_at_utc"], received_at.isoformat())
        self.assertEqual(record["payload"], payload)


@unittest.skipUnless(os.environ.get("DERIBIT_LIVE_TEST") == "1", "live Deribit test disabled")
class CollectorLiveTests(unittest.TestCase):
    def test_live_small_snapshot_writes_all_new_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "src" / "collect_deribit_snapshot.py"
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--currency",
                    "BTC",
                    "--max-instruments",
                    "2",
                    "--sleep",
                    "0",
                    "--outdir",
                    tmp,
                ],
                cwd=repo_root,
                check=True,
                timeout=240,
            )
            outdir = Path(tmp)
            option_files = list(outdir.glob("deribit_options_snapshot_*.csv"))
            future_files = list(outdir.glob("deribit_futures_snapshot_*.csv"))
            book_files = list(outdir.glob("deribit_order_book_snapshot_*.csv.gz"))
            manifest_files = list(outdir.glob("deribit_snapshot_manifest_*.json"))
            raw_files = list(outdir.glob("raw/deribit/*/api_responses.jsonl.gz"))

            self.assertEqual(len(option_files), 1)
            self.assertEqual(len(future_files), 1)
            self.assertEqual(len(book_files), 1)
            self.assertEqual(len(manifest_files), 1)
            self.assertEqual(len(raw_files), 1)

            options = pd.read_csv(option_files[0])
            books = pd.read_csv(book_files[0])
            with manifest_files[0].open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)

            for column in ["snapshot_id", "requested_at_utc", "received_at_utc", "exchange_timestamp", "bid_amount", "ask_amount"]:
                self.assertIn(column, options.columns)
            self.assertLessEqual(int(books["level"].max()), 15)
            self.assertEqual(manifest["order_book_depth"], 15)
            self.assertGreater(manifest["futures"]["BTC"]["discovered"], 0)
            self.assertGreater(len(future_files), 0)


if __name__ == "__main__":
    unittest.main()
