from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._path import ROOT
from tests.notebook_utils import execute_notebook_in_memory, execute_notebook_with_replacements


NOTEBOOK_DIR = ROOT / "notebooks"


def code_output_count(notebook) -> int:
    return sum(len(cell.get("outputs", [])) for cell in notebook.cells if cell.cell_type == "code")


class NotebookSmokeTests(unittest.TestCase):
    def _execute_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except PermissionError as exc:
            self.skipTest(f"Notebook execution is not permitted in this environment: {exc}")

    def test_regularization_notebook_executes(self) -> None:
        notebook = self._execute_or_skip(
            execute_notebook_in_memory,
            NOTEBOOK_DIR / "calibration_analysis_complete_reg_100.ipynb",
            working_directory=NOTEBOOK_DIR,
            timeout=1200,
        )
        self.assertGreater(code_output_count(notebook), 20)

    def test_chapter3_final_notebook_executes(self) -> None:
        notebook = self._execute_or_skip(
            execute_notebook_in_memory,
            NOTEBOOK_DIR / "calibration_results_final.ipynb",
            working_directory=NOTEBOOK_DIR,
            timeout=1200,
        )
        self.assertGreater(code_output_count(notebook), 10)

    def test_futures_viewer_notebook_executes(self) -> None:
        notebook = self._execute_or_skip(
            execute_notebook_in_memory,
            NOTEBOOK_DIR / "futures_plotly_viewer.ipynb",
            working_directory=NOTEBOOK_DIR,
            timeout=600,
        )
        self.assertGreater(code_output_count(notebook), 4)

    def test_hedging_notebook_executes_saved_output_mode(self) -> None:
        notebook = self._execute_or_skip(
            execute_notebook_with_replacements,
            NOTEBOOK_DIR / "hedging_analytics.ipynb",
            {"RUN_ENGINE = None": "RUN_ENGINE = False"},
            working_directory=NOTEBOOK_DIR,
            timeout=1200,
        )
        self.assertGreater(code_output_count(notebook), 10)

    def test_calibration_execution_notebook_smoke_runs(self) -> None:
        tmp_output = Path(tempfile.gettempdir()) / "calibration_results_reg_smoke_notebook.xlsx"
        notebook = self._execute_or_skip(
            execute_notebook_with_replacements,
            NOTEBOOK_DIR / "calibrate_all_to_excel.ipynb",
            {
                'OUTPUT_XLSX = ROOT / "excel files/calibration_results_reg_1000.xlsx"': f'OUTPUT_XLSX = Path(r"{tmp_output}")',
                "smoke_test_max_files_per_currency=None": "smoke_test_max_files_per_currency=1",
                "max_nfev=dict(black=250, heston=250, svcj=250)": "max_nfev=dict(black=10, heston=10, svcj=10)",
                "runtime_top_expiries_by_oi=None": "runtime_top_expiries_by_oi=2",
                "runtime_max_options=None": "runtime_max_options=50",
                "n_workers=int(os.cpu_count() - 2)": "n_workers=1",
                "verbose=True": "verbose=False",
            },
            working_directory=NOTEBOOK_DIR,
            timeout=1800,
        )
        self.assertTrue(tmp_output.exists())
        self.assertGreater(code_output_count(notebook), 1)


if __name__ == "__main__":
    unittest.main()
