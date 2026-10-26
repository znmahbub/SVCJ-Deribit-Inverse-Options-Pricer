from __future__ import annotations

import json
import unittest

from tests._path import ROOT


CANONICAL_NOTEBOOKS = [
    "00_check_data.ipynb",
    "01_run_calibration.ipynb",
    "02_pricing_analysis.ipynb",
    "03_run_hedging.ipynb",
    "04_final_analysis.ipynb",
]

FORBIDDEN_CANONICAL_TOKENS = [
    "validation_data",
    "validation_frac",
    "regularization_selection",
    "reg_1000",
    "lambda sweep",
    "selected lambda",
    "discounting",
    "discount_factor",
    "discounting_promotion_required",
]


def _load(name: str) -> dict:
    path = ROOT / "notebooks" / name
    if not path.exists():
        raise AssertionError(f"missing canonical notebook: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _source(nb: dict) -> str:
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        source = cell.get("source", [])
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


class CanonicalNotebookIntegrityTests(unittest.TestCase):
    def test_all_canonical_notebooks_parse_cleanly_with_unique_cell_ids(self) -> None:
        for name in CANONICAL_NOTEBOOKS:
            with self.subTest(name=name):
                nb = _load(name)
                self.assertEqual(nb.get("nbformat"), 4)
                ids = [cell.get("id") for cell in nb.get("cells", [])]
                self.assertTrue(ids)
                self.assertTrue(all(ids), f"{name} contains a cell without an id")
                self.assertEqual(len(ids), len(set(ids)), f"{name} contains duplicate cell ids")
                for cell in nb.get("cells", []):
                    if cell.get("cell_type") == "code":
                        self.assertIsNone(cell.get("execution_count"))
                        self.assertEqual(cell.get("outputs", []), [])

    def test_canonical_notebooks_contain_no_superseded_selection_or_discounting_machinery(self) -> None:
        for name in CANONICAL_NOTEBOOKS:
            text = _source(_load(name)).lower()
            for token in FORBIDDEN_CANONICAL_TOKENS:
                with self.subTest(name=name, token=token):
                    self.assertNotIn(token, text)

    def test_final_notebook_is_artifact_assembly_only(self) -> None:
        text = _source(_load("04_final_analysis.ipynb"))
        self.assertIn("assemble_final_package", text)
        for forbidden_call in (
            "calibrate_model",
            "run_all_snapshots_to_excel",
            "run_calibration_suite",
            "run_hedging_analysis",
            "run_hedging_suite",
        ):
            with self.subTest(call=forbidden_call):
                self.assertNotIn(forbidden_call, text)


if __name__ == "__main__":
    unittest.main()
