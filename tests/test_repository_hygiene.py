from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from tests._path import ROOT
from notebooks._lib import chapter3_final


class RepositoryHygieneTests(unittest.TestCase):
    def test_notebook_surface_is_canonical_only(self) -> None:
        notebooks = ROOT / "notebooks"
        names = {path.name for path in notebooks.iterdir()}
        self.assertEqual(
            names,
            {
                "00_check_data.ipynb",
                "01_run_calibration.ipynb",
                "02_pricing_analysis.ipynb",
                "03_run_hedging.ipynb",
                "04_final_analysis.ipynb",
                "_lib",
            },
        )
        self.assertFalse((notebooks / "_lib" / "chapter3_analysis.py").exists())

    def test_stage_branch_has_no_workflows(self) -> None:
        workflows = ROOT / ".github" / "workflows"
        names = (
            {path.name for path in workflows.iterdir() if path.is_file()}
            if workflows.exists()
            else set()
        )
        self.assertEqual(names, set())

    def test_pricing_loader_has_no_split_selection_api(self) -> None:
        self.assertNotIn(
            "evaluation_split",
            inspect.signature(chapter3_final.load_baseline_workbook).parameters,
        )
        for name in (
            "find_calibration_files",
            "reg_from_path",
            "compute_regularisation_summary",
            "select_regularisation_candidate",
            "build_regularisation_tradeoff_figure",
            "load_locked_regularisation",
        ):
            self.assertFalse(hasattr(chapter3_final, name), name)

    def test_active_code_contains_no_superseded_selection_or_discounting_tokens(self) -> None:
        forbidden = (
            "validation_data",
            "validation_frac",
            "regularization_selection",
            "lambda sweep",
            "selected lambda",
            "selected_lambda",
            "reg_1000",
            "discount_factor",
            "DISCOUNTING_PROMOTION_REQUIRED",
            "rhoj_boundary",
            "rate robustness",
        )
        paths: list[Path] = []
        paths.extend((ROOT / "src").glob("*.py"))
        paths.extend((ROOT / "notebooks" / "_lib").glob("*.py"))
        paths.extend((ROOT / "notebooks").glob("0[0-4]_*.ipynb"))
        offenders: list[str] = []
        for path in paths:
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                if token.lower() in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {token}")
        self.assertEqual(offenders, [])

    def test_removed_remediation_paths_are_absent(self) -> None:
        absent = [
            ROOT / "scripts" / "task13_candidate.py",
            ROOT / "scripts" / "task13_prepare_selection.py",
            ROOT / "docs" / "reproducibility" / "regularization_selection.md",
            ROOT / "notebooks" / "regularization_selection.ipynb",
            ROOT / "notebooks" / "calibration_results_final.ipynb",
            ROOT / "notebooks" / "sync_regularization_notebooks.py",
        ]
        self.assertEqual([str(path.relative_to(ROOT)) for path in absent if path.exists()], [])


if __name__ == "__main__":
    unittest.main()
