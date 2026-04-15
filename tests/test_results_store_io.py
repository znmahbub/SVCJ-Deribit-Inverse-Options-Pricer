from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests._path import ROOT  # noqa: F401  (ensures src is importable)

from src.results_store import (  # noqa: E402
    PARAM_COLS_BLACK,
    PARAM_SHEET_BLACK,
    TEST_SHEET,
    TRAIN_SHEET,
    flush_workbook_atomic,
    init_empty_workbook,
    load_existing_workbook,
)


class TestResultsStoreIO(unittest.TestCase):
    def test_flush_and_reload_workbook_smoke(self) -> None:
        wb = init_empty_workbook()

        # Minimal param row for Black sheet
        row = {c: None for c in PARAM_COLS_BLACK}
        row.update(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "currency": "BTC",
                "success": True,
                "message": "ok",
                "nfev": 1,
                "sigma": 0.5,
            }
        )
        wb[PARAM_SHEET_BLACK] = pd.DataFrame([row])
        wb[TRAIN_SHEET] = pd.DataFrame(
            [
                {
                    "snapshot_ts": "2026-01-01T00:00:00Z",
                    "currency": "BTC",
                    "instrument_name": "BTC-TEST",
                    "option_type": "call",
                    "strike": 10_000.0,
                    "expiry_datetime": "2026-06-01T00:00:00Z",
                    "time_to_maturity": 0.25,
                    "futures_price": 10_000.0,
                    "bid_price": 0.01,
                    "ask_price": 0.02,
                    "mid_price_clean": 0.015,
                    "rel_spread": 0.5,
                    "open_interest": 10.0,
                    "vega": 1.0,
                    "random_seed": 123,
                    "price_black": 0.01,
                }
            ]
        )
        wb[TEST_SHEET] = wb[TRAIN_SHEET].copy()

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "calibration_results.xlsx"
            flush_workbook_atomic(wb, out)
            self.assertTrue(out.exists())

            reloaded = load_existing_workbook(out)
            self.assertIn(PARAM_SHEET_BLACK, reloaded)
            self.assertIn(TRAIN_SHEET, reloaded)
            self.assertIn(TEST_SHEET, reloaded)
            self.assertTrue(set(PARAM_COLS_BLACK).issubset(set(reloaded[PARAM_SHEET_BLACK].columns)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

