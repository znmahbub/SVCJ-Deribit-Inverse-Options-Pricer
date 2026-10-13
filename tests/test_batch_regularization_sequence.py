from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pandas.testing as pdt

from tests._path import ROOT  # noqa: F401
from src.batch_runner import BatchConfig, run_currency_to_excel
from src.results_store import (
    PARAM_SHEET_BLACK,
    PARAM_SHEET_HESTON,
    PARAM_SHEET_SVCJ,
    init_empty_workbook,
)


PARAMS = {
    "black": lambda value: {"sigma": value},
    "heston": lambda value: {"kappa": value, "theta": 0.04, "sigma_v": 0.30, "rho": -0.50, "v0": 0.04},
    "svcj": lambda value: {
        "kappa": value,
        "theta": 0.04,
        "sigma_v": 0.30,
        "rho": -0.50,
        "v0": 0.04,
        "lam": 0.20,
        "ell_y": -0.05,
        "sigma_y": 0.20,
        "ell_v": 0.10,
        "rho_j": -0.30,
    },
}


def _param_row(model: str, ts_iso: str, value: float) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "timestamp": ts_iso,
            "currency": "BTC",
            "success": True,
            "message": "ok",
            "nfev": 1,
            **PARAMS[model](value),
        }
    ])


class TestBatchRegularizationSequence(unittest.TestCase):
    def _make_files(self, root: Path, n: int = 4) -> list[Path]:
        data = root / "data"
        data.mkdir(parents=True)
        files = []
        for hour in range(n):
            p = data / f"deribit_options_snapshot_20260101T{hour:02d}0000Z.csv"
            p.write_text("unused\n", encoding="utf-8")
            files.append(p)
        return files

    def _cfg(self, root: Path, *, smoke: int | None = None) -> BatchConfig:
        return BatchConfig(
            project_root=root,
            data_dir=root / "data",
            output_xlsx=root / "out.xlsx",
            resume=True,
            smoke_test_max_files_per_currency=smoke,
            min_options_after_filter=1,
            l2_prev_strength=10.0,
            n_workers=4,
            currencies=("BTC",),
            save_every_n_files=100,
        )

    @staticmethod
    def _fake_processor(call_log: list[dict]):
        def fake(path: Path, **kwargs):
            ts = pd.to_datetime(path.stem.split("_")[-1], format="%Y%m%dT%H%M%SZ", utc=True)
            ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            value = float(ts.hour + 1)
            warm = kwargs.get("warm_start") or {}
            call_log.append({"hour": ts.hour, "warm": {k: dict(v) for k, v in warm.items()}})
            warm_next = {k: dict(v) for k, v in warm.items()}
            for model in ("black", "heston", "svcj"):
                warm_next[model] = PARAMS[model](value)
            return {
                "timestamp_iso": ts_iso,
                "timestamp": ts,
                "currency": "BTC",
                "param_rows": {
                    model: _param_row(model, ts_iso, value)
                    for model in ("black", "heston", "svcj")
                },
                "train_df": pd.DataFrame(),
                "validation_df": pd.DataFrame(),
                "test_df": pd.DataFrame(),
                "warm_next": warm_next,
            }
        return fake

    def test_regularized_run_uses_previous_success_even_when_multiple_workers_requested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_files(root)
            calls: list[dict] = []
            with patch("src.batch_runner.process_snapshot_to_payload", side_effect=self._fake_processor(calls)), patch(
                "src.batch_runner.flush_workbook_atomic", return_value=None
            ):
                run_currency_to_excel("BTC", cfg=self._cfg(root), workbook=init_empty_workbook())

            self.assertEqual([c["hour"] for c in calls], [0, 1, 2, 3])
            self.assertEqual(calls[0]["warm"], {})
            for hour in (1, 2, 3):
                prev = float(hour)
                self.assertAlmostEqual(calls[hour]["warm"]["black"]["sigma"], prev)
                self.assertAlmostEqual(calls[hour]["warm"]["heston"]["kappa"], prev)
                self.assertAlmostEqual(calls[hour]["warm"]["svcj"]["kappa"], prev)

    def test_resume_chain_matches_uninterrupted_chain(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_files(root)

            full_calls: list[dict] = []
            resumed_calls: list[dict] = []
            with patch("src.batch_runner.flush_workbook_atomic", return_value=None):
                with patch("src.batch_runner.process_snapshot_to_payload", side_effect=self._fake_processor(full_calls)):
                    wb_full = run_currency_to_excel("BTC", cfg=self._cfg(root), workbook=init_empty_workbook())

                wb_resume = init_empty_workbook()
                with patch("src.batch_runner.process_snapshot_to_payload", side_effect=self._fake_processor(resumed_calls)):
                    wb_resume = run_currency_to_excel("BTC", cfg=self._cfg(root, smoke=2), workbook=wb_resume)
                    wb_resume = run_currency_to_excel("BTC", cfg=self._cfg(root, smoke=None), workbook=wb_resume)

            self.assertEqual([c["hour"] for c in full_calls], [0, 1, 2, 3])
            self.assertEqual([c["hour"] for c in resumed_calls], [0, 1, 2, 3])
            for sheet in (PARAM_SHEET_BLACK, PARAM_SHEET_HESTON, PARAM_SHEET_SVCJ):
                left = wb_full[sheet].reset_index(drop=True)
                right = wb_resume[sheet].reset_index(drop=True)
                pdt.assert_frame_equal(left, right, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
