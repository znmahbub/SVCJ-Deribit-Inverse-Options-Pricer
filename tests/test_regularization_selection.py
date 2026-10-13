from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from tests._path import ROOT  # noqa: F401
from notebooks._lib import chapter3_final


PARAM_COLS = [
    "kappa",
    "theta",
    "sigma_v",
    "rho",
    "v0",
    "lam",
    "ell_y",
    "sigma_y",
    "ell_v",
    "rho_j",
]


def _svcj_frame(*, validation_rmse: float, test_rmse: float) -> pd.DataFrame:
    rows = []
    for currency in ("BTC", "ETH"):
        for i in range(3):
            row = {
                "timestamp": f"2026-01-0{i+1}T00:00:00Z",
                "currency": currency,
                "success": True,
                "nfev": 10 + i,
                "rmse_validation": validation_rmse + i * 1e-6,
                "mae_validation": 0.5 * validation_rmse + i * 1e-6,
                # Deliberately conflicting test fields. Selection must ignore them.
                "rmse_test": test_rmse + i * 1e-6,
                "mae_test": 0.5 * test_rmse + i * 1e-6,
                "feller_gap": 0.01,
                "feller_violated": False,
                "svcj_martingale_margin": 0.8,
                "svcj_martingale_violated": False,
                "svcj_contour_margin": 0.5,
                "svcj_contour_violated": False,
                "kappa": 2.0 + 0.01 * i,
                "theta": 0.04 + 0.001 * i,
                "sigma_v": 0.3 + 0.001 * i,
                "rho": -0.5 + 0.001 * i,
                "v0": 0.04 + 0.001 * i,
                "lam": 0.2 + 0.001 * i,
                "ell_y": -0.05 + 0.001 * i,
                "sigma_y": 0.2 + 0.001 * i,
                "ell_v": 0.1 + 0.001 * i,
                "rho_j": -0.3 + 0.001 * i,
            }
            rows.append(row)
    return pd.DataFrame(rows)


class RegularizationSelectionTests(unittest.TestCase):
    def test_validation_preference_wins_even_when_test_prefers_another_lambda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = root / "calibration_results_reg_1.xlsx"
            p10 = root / "calibration_results_reg_10.xlsx"

            # Validation prefers lambda=1; test deliberately prefers lambda=10.
            _svcj_frame(validation_rmse=0.0020, test_rmse=0.0100).to_excel(
                p1, sheet_name="svcj_params", index=False
            )
            _svcj_frame(validation_rmse=0.0030, test_rmse=0.0010).to_excel(
                p10, sheet_name="svcj_params", index=False
            )

            summary = chapter3_final.compute_regularisation_summary(
                [p1, p10], root, evaluation_split="validation"
            )
            selected = chapter3_final.select_regularisation_candidate(summary)

            self.assertEqual(selected, 1)
            self.assertIn("rmse_validation_mean", summary.columns)
            self.assertIn("mae_validation_mean", summary.columns)
            self.assertNotIn("rmse_test_mean", summary.columns)
            self.assertNotIn("mae_test_mean", summary.columns)

    def test_summary_does_not_require_test_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "calibration_results_reg_25.xlsx"
            frame = _svcj_frame(validation_rmse=0.0025, test_rmse=0.9).drop(
                columns=["rmse_test", "mae_test"]
            )
            frame.to_excel(path, sheet_name="svcj_params", index=False)

            summary = chapter3_final.compute_regularisation_summary(
                [path], root, evaluation_split="validation"
            )
            self.assertAlmostEqual(float(summary.loc[0, "rmse_validation_mean"]), 0.002501)

    def test_selection_is_deterministic_on_validation_metrics(self) -> None:
        summary = pd.DataFrame(
            {
                "reg": [100, 10, 1],
                "rmse_validation_mean": [0.002, 0.002, 0.002],
                "mae_validation_mean": [0.0012, 0.0010, 0.0010],
                "roughness_mean": [0.05, 0.07, 0.08],
                "nfev_mean": [10.0, 12.0, 8.0],
            }
        )
        # RMSE ties -> MAE tie between 10 and 1 -> lower roughness wins (10).
        self.assertEqual(chapter3_final.select_regularisation_candidate(summary), 10)

    def test_pending_lock_blocks_final_reporting_and_locked_record_parses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regularization_selection.md"
            path.write_text(
                "# Selection\n\n- Status: PENDING\n- Selected lambda_TS: TBD\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "not locked"):
                chapter3_final.load_locked_regularisation(path)

            path.write_text(
                "# Selection\n\n- Status: LOCKED\n- Selected lambda_TS: 25\n",
                encoding="utf-8",
            )
            self.assertEqual(chapter3_final.load_locked_regularisation(path), 25)

    def test_selection_notebook_contains_no_test_split_access(self) -> None:
        path = ROOT / "notebooks" / "regularization_selection.ipynb"
        nb = nbformat.read(path, as_version=4)
        code = "\n".join(
            "".join(cell.source) for cell in nb.cells if cell.cell_type == "code"
        ).lower()
        forbidden = ["test_data", "rmse_test", "mae_test", 'sheet_name="test_data"', "sheet_name='test_data'"]
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, code)
        self.assertIn('evaluation_split="validation"', code)

    def test_final_notebook_reads_locked_lambda_and_explicit_test_split(self) -> None:
        path = ROOT / "notebooks" / "calibration_results_final.ipynb"
        nb = nbformat.read(path, as_version=4)
        code = "\n".join(
            "".join(cell.source) for cell in nb.cells if cell.cell_type == "code"
        )
        self.assertIn("load_locked_regularisation", code)
        self.assertIn('evaluation_split="test"', code)
        self.assertNotIn("FINAL_REG = 100", code)
        self.assertNotIn("compute_regularisation_summary", code)
        self.assertNotIn("build_regularisation_tradeoff_figure", code)


if __name__ == "__main__":
    unittest.main()
