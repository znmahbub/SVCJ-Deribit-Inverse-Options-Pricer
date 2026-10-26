from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import backfill_futures_marks as backfill


class BackfillDirectPerpTests(unittest.TestCase):
    def test_build_direct_perp_rows_maps_futures_snapshot_to_legacy_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            futures_path = data_dir / "deribit_futures_snapshot_20261026T080000Z.csv"
            pd.DataFrame(
                [
                    {
                        "timestamp": "2026-10-26T08:00:00+00:00",
                        "snapshot_id": "20261026T080000Z",
                        "currency": "BTC",
                        "instrument_name": "BTC-PERPETUAL",
                        "is_perpetual": True,
                        "mark_price": 101000.0,
                        "bid_price": 100990.0,
                        "ask_price": 101010.0,
                        "exchange_timestamp": "2026-10-26T08:00:01+00:00",
                        "requested_at_utc": "2026-10-26T08:00:00.900000+00:00",
                        "received_at_utc": "2026-10-26T08:00:01.050000+00:00",
                        "collection_latency_ms": 150.0,
                    },
                    {
                        "timestamp": "2026-10-26T08:00:00+00:00",
                        "snapshot_id": "20261026T080000Z",
                        "currency": "BTC",
                        "instrument_name": "BTC-27NOV26",
                        "is_perpetual": False,
                        "mark_price": 102000.0,
                    },
                ]
            ).to_csv(futures_path, index=False)

            combos = pd.DataFrame(
                [
                    {
                        "timestamp": "2026-10-26T08:00:00+00:00",
                        "currency": "BTC",
                        "snapshot_file": "deribit_options_snapshot_20261026T080000Z.csv",
                        "snapshot_timestamp_ms": 1793001600000,
                    }
                ]
            )

            rows = backfill.build_direct_perp_rows([futures_path], combos)

        self.assertEqual(len(rows), 1)
        row = rows.iloc[0]
        self.assertEqual(row["obs_datetime"], "2026-10-26T08:00:00+00:00")
        self.assertEqual(row["coin"], "BTC")
        self.assertEqual(row["instrument_name"], "BTC-PERPETUAL")
        self.assertEqual(row["perp_futures_mark_price"], 101000.0)
        self.assertEqual(row["perp_futures_best_bid"], 100990.0)
        self.assertEqual(row["perp_futures_best_ask"], 101010.0)
        self.assertEqual(row["data_source"], "deribit_snapshot")
        self.assertEqual(row["snapshot_file"], "deribit_options_snapshot_20261026T080000Z.csv")
        self.assertEqual(row["futures_snapshot_file"], futures_path.name)
        self.assertEqual(row["perp_exchange_timestamp"], "2026-10-26T08:00:01+00:00")

    def test_direct_snapshot_replaces_existing_tradingview_row_without_api_call(self) -> None:
        combos = pd.DataFrame(
            [
                {
                    "timestamp": "2026-10-26T08:00:00+00:00",
                    "currency": "BTC",
                    "snapshot_file": "deribit_options_snapshot_20261026T080000Z.csv",
                    "snapshot_timestamp_ms": 1793001600000,
                }
            ]
        )
        existing = pd.DataFrame(
            [
                {
                    "obs_datetime": "2026-10-26T08:00:00+00:00",
                    "coin": "BTC",
                    "instrument_name": "BTC-PERPETUAL",
                    "perp_futures_mark_price": 100900.0,
                    "perp_futures_best_bid": None,
                    "perp_futures_best_ask": None,
                    "data_source": "tradingview_chart_open_close_nearest",
                }
            ]
        )
        direct = pd.DataFrame(
            [
                {
                    "obs_datetime": "2026-10-26T08:00:00+00:00",
                    "coin": "BTC",
                    "instrument_name": "BTC-PERPETUAL",
                    "perp_futures_mark_price": 101000.0,
                    "perp_futures_best_bid": 100990.0,
                    "perp_futures_best_ask": 101010.0,
                    "data_source": "deribit_snapshot",
                    "snapshot_file": "deribit_options_snapshot_20261026T080000Z.csv",
                    "snapshot_timestamp_ms": 1793001600000,
                }
            ]
        )

        with patch.object(backfill, "fetch_tradingview_candles_for_requests", side_effect=AssertionError("should not call TradingView")):
            updated, appended = backfill.build_perp_incremental(
                combos=combos,
                existing_perp=existing,
                direct_perp_rows=direct,
                skip_perp_backfill=False,
                sleep_seconds=0,
                max_gap_seconds=1800.0,
                perp_resolution="1",
                perp_max_bars_per_call=5000,
            )

        self.assertEqual(len(updated), 1)
        self.assertEqual(len(appended), 1)
        self.assertEqual(updated.iloc[0]["perp_futures_mark_price"], 101000.0)
        self.assertEqual(updated.iloc[0]["data_source"], "deribit_snapshot")

    def test_skip_backfill_still_ingests_direct_snapshot(self) -> None:
        combos = pd.DataFrame(
            [
                {
                    "timestamp": "2026-10-26T08:00:00+00:00",
                    "currency": "ETH",
                    "snapshot_file": "deribit_options_snapshot_20261026T080000Z.csv",
                    "snapshot_timestamp_ms": 1793001600000,
                }
            ]
        )
        direct = pd.DataFrame(
            [
                {
                    "obs_datetime": "2026-10-26T08:00:00+00:00",
                    "coin": "ETH",
                    "instrument_name": "ETH-PERPETUAL",
                    "perp_futures_mark_price": 4200.0,
                    "perp_futures_best_bid": 4199.5,
                    "perp_futures_best_ask": 4200.5,
                    "data_source": "deribit_snapshot",
                    "snapshot_file": "deribit_options_snapshot_20261026T080000Z.csv",
                    "snapshot_timestamp_ms": 1793001600000,
                }
            ]
        )

        with patch.object(backfill, "fetch_tradingview_candles_for_requests", side_effect=AssertionError("should not call TradingView")):
            updated, _ = backfill.build_perp_incremental(
                combos=combos,
                existing_perp=pd.DataFrame(),
                direct_perp_rows=direct,
                skip_perp_backfill=True,
                sleep_seconds=0,
                max_gap_seconds=1800.0,
                perp_resolution="1",
                perp_max_bars_per_call=5000,
            )

        self.assertEqual(len(updated), 1)
        self.assertEqual(updated.iloc[0]["data_source"], "deribit_snapshot")
        self.assertEqual(updated.iloc[0]["perp_futures_mark_price"], 4200.0)


if __name__ == "__main__":
    unittest.main()
