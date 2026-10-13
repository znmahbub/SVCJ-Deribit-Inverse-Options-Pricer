from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tests._path import ROOT  # noqa: F401
from src.backfill_futures_marks import build_perp_incremental
from src.hedging_analysis import load_perp_history, make_notes, HedgingConfig


class PerpetualPriceProvenanceTests(unittest.TestCase):
    def test_new_proxy_schema_is_preferred_and_metadata_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {
                    "obs_datetime": ["2026-01-01T00:00:00Z"],
                    "coin": ["BTC"],
                    "perp_futures_price_proxy": [100_000.0],
                    "price_source": ["deribit_tradingview_chart"],
                    "price_kind": ["nearest_1m_candle_open_close"],
                    "candle_price_choice": ["open"],
                    "candle_price_gap_seconds": [11.0],
                    "candle_resolution": ["1"],
                }
            ).to_csv(root / "perpetual_futures_prices.csv", index=False)

            out, used = load_perp_history(root)

        self.assertEqual(used, "perpetual_futures_prices.csv")
        self.assertAlmostEqual(float(out.loc[0, "close_price"]), 100_000.0)
        self.assertEqual(out.loc[0, "price_source"], "deribit_tradingview_chart")
        self.assertEqual(out.loc[0, "price_kind"], "nearest_1m_candle_open_close")
        self.assertAlmostEqual(float(out.loc[0, "proxy_gap_seconds"]), 11.0)
        self.assertEqual(out.loc[0, "proxy_price_choice"], "open")
        self.assertEqual(str(out.loc[0, "proxy_resolution"]), "1")
        self.assertFalse(bool(out.loc[0, "legacy_price_schema"]))

    def test_legacy_mark_schema_loads_with_warning_and_is_labeled_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                {
                    "obs_datetime": ["2026-01-01T00:00:00Z"],
                    "coin": ["ETH"],
                    "perp_futures_mark_price": [3_000.0],
                    "data_source": ["tradingview_chart_open_close_nearest"],
                    "candle_price_choice": ["close"],
                    "candle_price_gap_seconds": [7.0],
                    "candle_resolution": ["1"],
                }
            ).to_csv(root / "perpetual_futures_prices.csv", index=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out, _ = load_perp_history(root)

        self.assertTrue(any("legacy" in str(w.message).lower() for w in caught))
        self.assertAlmostEqual(float(out.loc[0, "close_price"]), 3_000.0)
        self.assertTrue(bool(out.loc[0, "legacy_price_schema"]))
        self.assertEqual(out.loc[0, "price_kind"], "legacy_mark_named_candle_proxy")

    def test_incremental_builder_emits_neutral_proxy_columns_even_when_skipped(self) -> None:
        combos = pd.DataFrame(
            {
                "timestamp": ["2026-01-01T00:00:00Z"],
                "currency": ["BTC"],
                "snapshot_file": ["deribit_options_snapshot_20260101T000000Z.csv"],
                "snapshot_timestamp_ms": [1767225600000],
            }
        )
        updated, new = build_perp_incremental(
            combos=combos,
            existing_perp=pd.DataFrame(),
            skip_perp_backfill=True,
            sleep_seconds=0.0,
            max_gap_seconds=1800.0,
            perp_resolution="1",
            perp_max_bars_per_call=100,
        )
        for frame in (updated, new):
            self.assertIn("perp_futures_price_proxy", frame.columns)
            self.assertIn("price_source", frame.columns)
            self.assertIn("price_kind", frame.columns)
        self.assertEqual(new.loc[0, "price_source"], "skipped")
        self.assertEqual(new.loc[0, "price_kind"], "nearest_1m_candle_open_close")
        self.assertTrue(pd.isna(new.loc[0, "perp_futures_price_proxy"]))

    def test_hedge_notes_record_proxy_kind_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = HedgingConfig(
                calibration_xlsx=root / "calibration.xlsx",
                data_dir=root,
                output_xlsx=root / "hedging.xlsx",
            )
            perp = pd.DataFrame(
                {
                    "price_source": ["deribit_tradingview_chart", "deribit_tradingview_chart"],
                    "price_kind": ["nearest_1m_candle_open_close", "nearest_1m_candle_open_close"],
                    "legacy_price_schema": [False, False],
                }
            )
            notes = make_notes(
                cfg,
                pd.DataFrame(),
                perp,
                pd.DataFrame(),
                "perpetual_futures_prices.csv",
                None,
            ).set_index("item")["value"]

        self.assertEqual(notes["hedge_price_kind"], "nearest_1m_candle_open_close")
        self.assertEqual(notes["hedge_price_source"], "deribit_tradingview_chart")
        self.assertEqual(notes["hedge_price_legacy_schema"], False)


if __name__ == "__main__":
    unittest.main()
