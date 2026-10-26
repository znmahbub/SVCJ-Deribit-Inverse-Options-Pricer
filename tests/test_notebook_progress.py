from __future__ import annotations

import unittest

from tests._path import ROOT


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class NotebookProgressTests(unittest.TestCase):
    def test_stage2_workflows_expose_progress_bars(self) -> None:
        stage2 = _read("notebooks/_lib/stage2_workflow.py")
        batch = _read("src/batch_runner.py")

        self.assertIn("from tqdm.auto import tqdm", stage2)
        self.assertIn('desc="Inspecting snapshots"', stage2)
        self.assertIn('desc="Pricing specifications"', stage2)
        self.assertNotIn('desc="Calibration specifications"', stage2)

        self.assertIn("show_progress=True", stage2)
        self.assertIn("progress_label=spec.label", stage2)
        self.assertIn("progress_position_offset=index * len(currencies)", stage2)
        self.assertIn('desc=f"{label} {currency}"', batch)

    def test_stage3_and_final_assembly_expose_progress_bars(self) -> None:
        stage3 = _read("notebooks/_lib/stage3_workflow.py")
        self.assertIn("from tqdm.auto import tqdm", stage3)
        self.assertIn('desc="Hedging specifications"', stage3)
        self.assertIn('desc="Hedging reports"', stage3)
        self.assertIn('desc=f"Copying {artifact_type}"', stage3)

        engine = _read("src/hedging_analysis.py")
        self.assertIn("from tqdm.auto import tqdm", engine)
        self.assertIn('inner = _make_progress(n_blocks, "model blocks")', engine)


if __name__ == "__main__":
    unittest.main()
