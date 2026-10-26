from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests._path import ROOT


NOTEBOOKS = {
    "00_check_data.ipynb": ["inspect_snapshot_data", "assert_snapshot_data_ready"],
    "01_run_calibration.ipynb": ["build_calibration_configs", "run_calibration_suite"],
    "02_pricing_analysis.ipynb": ["run_pricing_analysis"],
}


def _load(name: str) -> dict:
    path = ROOT / "notebooks" / name
    if not path.exists():
        raise AssertionError(f"missing canonical notebook: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _source(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in nb.get("cells", [])
    )


class Stage2NotebookTests(unittest.TestCase):
    def test_three_canonical_notebooks_exist_and_are_clean(self) -> None:
        for name in NOTEBOOKS:
            with self.subTest(name=name):
                nb = _load(name)
                self.assertEqual(nb.get("nbformat"), 4)
                for cell in nb.get("cells", []):
                    if cell.get("cell_type") == "code":
                        self.assertIsNone(cell.get("execution_count"))
                        self.assertEqual(cell.get("outputs", []), [])

    def test_notebooks_are_thin_runners_over_shared_stage2_helper(self) -> None:
        for name, required_calls in NOTEBOOKS.items():
            with self.subTest(name=name):
                text = _source(_load(name))
                self.assertIn("_lib.stage2_workflow", text)
                for call in required_calls:
                    self.assertIn(call, text)

    def test_calibration_notebook_defaults_to_full_data_and_fixed_specs(self) -> None:
        text = _source(_load("01_run_calibration.ipynb"))
        self.assertIn("START_DT = None", text)
        self.assertIn("END_DT = None", text)
        self.assertIn("calibration_wls.xlsx", text)
        self.assertIn("calibration_pwls.xlsx", text)
        self.assertNotIn("validation_frac", text)
        self.assertNotIn("validation_data", text)
        self.assertNotIn("reg_1000", text)

    def test_pricing_notebook_analyzes_test_holdout_only(self) -> None:
        text = _source(_load("02_pricing_analysis.ipynb")).lower()
        self.assertIn("test holdout", text)
        self.assertIn("calibration_wls.xlsx", text)
        self.assertIn("calibration_pwls.xlsx", text)
        self.assertNotIn("regularization_selection", text)
        self.assertNotIn("validation_data", text)
        self.assertNotIn("evaluation_split=\"validation\"", text)


if __name__ == "__main__":
    unittest.main()
