from __future__ import annotations

import json
import unittest

from tests._path import ROOT


NOTEBOOKS = {
    "03_run_hedging.ipynb": ["build_hedging_configs", "run_hedging_suite", "run_hedging_reporting"],
    "04_final_analysis.ipynb": ["assemble_final_package"],
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


class Stage3NotebookTests(unittest.TestCase):
    def test_canonical_stage3_notebooks_exist_and_are_clean(self) -> None:
        for name in NOTEBOOKS:
            with self.subTest(name=name):
                nb = _load(name)
                self.assertEqual(nb.get("nbformat"), 4)
                for cell in nb.get("cells", []):
                    if cell.get("cell_type") == "code":
                        self.assertIsNone(cell.get("execution_count"))
                        self.assertEqual(cell.get("outputs", []), [])
                        self.assertTrue(cell.get("id"))

    def test_notebooks_are_thin_runners_over_shared_stage3_helper(self) -> None:
        for name, required in NOTEBOOKS.items():
            with self.subTest(name=name):
                text = _source(_load(name))
                self.assertIn("_lib.stage3_workflow", text)
                for symbol in required:
                    self.assertIn(symbol, text)

    def test_hedging_notebook_defaults_to_pwls_test_only_full_dataset(self) -> None:
        text = _source(_load("03_run_hedging.ipynb"))
        self.assertIn("START_DT = None", text)
        self.assertIn("END_DT = None", text)
        self.assertIn("RUN_WLS = False", text)
        self.assertIn("RUN_PWLS = True", text)
        self.assertIn("hedging_pwls.xlsx", text)
        self.assertNotIn("validation", text.lower())

    def test_final_notebook_only_assembles_existing_outputs(self) -> None:
        text = _source(_load("04_final_analysis.ipynb"))
        self.assertIn("assemble_final_package", text)
        self.assertNotIn("run_hedging_analysis", text)
        self.assertNotIn("run_all_snapshots_to_excel", text)


if __name__ == "__main__":
    unittest.main()
