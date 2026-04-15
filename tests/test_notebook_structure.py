from __future__ import annotations

import unittest

import nbformat

from notebooks.sync_regularization_notebooks import NOTEBOOKS as REG_NOTEBOOKS
from notebooks.sync_regularization_notebooks import code_cells
from tests._path import ROOT


class NotebookStructureTests(unittest.TestCase):
    def test_regularization_notebooks_match_sync_template(self) -> None:
        for reg_val in REG_NOTEBOOKS:
            rel_path = ROOT / "notebooks" / f"calibration_analysis_complete_reg_{reg_val}.ipynb"
            with self.subTest(notebook=rel_path.name):
                nb = nbformat.read(rel_path, as_version=4)
                code_cells_in_nb = [cell for cell in nb.cells if cell.cell_type == "code"]
                expected_sources = code_cells(reg_val)
                self.assertEqual(len(code_cells_in_nb), len(expected_sources))
                self.assertEqual(["".join(cell.source) for cell in code_cells_in_nb], expected_sources)

    def test_notebook_bootstrap_uses_shared_path_resolution(self) -> None:
        targets = [
            "notebooks/calibration_results_final.ipynb",
            "notebooks/hedging_analytics.ipynb",
            "notebooks/futures_plotly_viewer.ipynb",
            "notebooks/calibrate_all_to_excel.ipynb",
        ]
        for rel_path in targets:
            with self.subTest(notebook=rel_path):
                nb = nbformat.read(ROOT / rel_path, as_version=4)
                first_code = next(cell for cell in nb.cells if cell.cell_type == "code")
                source = "".join(first_code.source)
                self.assertIn("locate_notebook_paths", source)
                self.assertIn("NOTEBOOK_DIR", source)


if __name__ == "__main__":
    unittest.main()
