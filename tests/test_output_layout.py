from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._path import ROOT  # noqa: F401
from notebooks._lib import stage2_workflow as stage2
from notebooks._lib import stage3_workflow as stage3


class CanonicalOutputLayoutTests(unittest.TestCase):
    def test_all_canonical_outputs_live_under_root_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()

            calibration = stage2.canonical_calibration_paths(root)
            hedging = stage3.canonical_hedging_paths(root)

            canonical_root = root / "outputs" / "canonical"
            self.assertEqual(calibration["wls"], canonical_root / "calibration_wls.xlsx")
            self.assertEqual(calibration["pwls"], canonical_root / "calibration_pwls.xlsx")
            self.assertEqual(hedging["wls"], canonical_root / "hedging_wls.xlsx")
            self.assertEqual(hedging["pwls"], canonical_root / "hedging_pwls.xlsx")
            self.assertEqual(stage2.CANONICAL_PRICING_OUTPUT_SUBDIR, Path("outputs") / "canonical" / "pricing")
            self.assertEqual(stage3.CANONICAL_HEDGING_REPORT_SUBDIR, Path("outputs") / "canonical" / "hedging")
            self.assertEqual(stage3.CANONICAL_FINAL_OUTPUT_SUBDIR, Path("outputs") / "canonical" / "final")

    def test_canonical_notebooks_reference_root_output_paths_only(self) -> None:
        forbidden = (
            "excel files/canonical",
            "pricing_outputs/canonical",
            "hedging_outputs/canonical",
            "final_outputs/canonical",
        )
        for name in (
            "01_run_calibration.ipynb",
            "02_pricing_analysis.ipynb",
            "03_run_hedging.ipynb",
            "04_final_analysis.ipynb",
        ):
            notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
            text = "\n".join(
                "".join(cell.get("source", []))
                if isinstance(cell.get("source", []), list)
                else str(cell.get("source", ""))
                for cell in notebook.get("cells", [])
            )
            for token in forbidden:
                with self.subTest(notebook=name, token=token):
                    self.assertNotIn(token, text)

    def test_outputs_directory_is_gitignored(self) -> None:
        ignored = {
            line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("outputs/", ignored)


if __name__ == "__main__":
    unittest.main()
