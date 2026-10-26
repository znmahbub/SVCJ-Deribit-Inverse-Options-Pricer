from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests._path import ROOT  # noqa: F401
from src.results_store import (
    PARAM_COLS_BLACK,
    PARAM_SHEET_BLACK,
    PARAM_SHEET_HESTON,
    PARAM_SHEET_SVCJ,
    TEST_SHEET,
    TRAIN_SHEET,
    flush_workbook_atomic,
    init_empty_workbook,
    load_existing_workbook,
)


class TestResultsStoreIO(unittest.TestCase):
    def test_new_schema_has_only_train_test_and_no_validation_metrics(self) -> None:
        wb = init_empty_workbook()
        self.assertEqual(
            set(wb),
            {PARAM_SHEET_BLACK, PARAM_SHEET_HESTON, PARAM_SHEET_SVCJ, TRAIN_SHEET, TEST_SHEET},
        )
        self.assertNotIn("rmse_validation", PARAM_COLS_BLACK)
        self.assertNotIn("mae_validation", PARAM_COLS_BLACK)
        self.assertNotIn("n_validation", PARAM_COLS_BLACK)

    def test_flush_and_reload_workbook_smoke(self) -> None:
        wb = init_empty_workbook()
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
        wb[TRAIN_SHEET] = pd.DataFrame([self._option_row("train")])
        wb[TEST_SHEET] = pd.DataFrame([self._option_row("test")])

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "calibration_results.xlsx"
            flush_workbook_atomic(wb, out)
            reloaded = load_existing_workbook(out)
            self.assertEqual(
                set(reloaded),
                {PARAM_SHEET_BLACK, PARAM_SHEET_HESTON, PARAM_SHEET_SVCJ, TRAIN_SHEET, TEST_SHEET},
            )
            self.assertTrue(set(PARAM_COLS_BLACK).issubset(set(reloaded[PARAM_SHEET_BLACK].columns)))

    @staticmethod
    def _option_row(split: str) -> dict:
        return {
            "snapshot_ts": "2026-01-01T00:00:00Z",
            "currency": "BTC",
            "instrument_name": f"BTC-{split}",
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
            "split": split,
            "price_black": 0.01,
        }


if __name__ == "__main__":
    unittest.main()
