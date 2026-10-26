from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tests._path import ROOT  # noqa: F401
from notebooks._lib import stage3_workflow as workflow


class Stage3WorkflowTests(unittest.TestCase):
    def test_default_hedging_config_runs_pwls_test_only_on_full_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            (root / "outputs" / "canonical").mkdir(parents=True)
            configs = workflow.build_hedging_configs(root)

        self.assertEqual(list(configs), ["pwls"])
        cfg = configs["pwls"]
        self.assertEqual(cfg.calibration_xlsx.name, "calibration_pwls.xlsx")
        self.assertEqual(cfg.output_xlsx.name, "hedging_pwls.xlsx")
        self.assertEqual(cfg.evaluation_splits, ("test",))
        self.assertIsNone(cfg.analysis_start)
        self.assertIsNone(cfg.analysis_end)

    def test_wls_hedging_is_optional_but_uses_same_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            (root / "outputs" / "canonical").mkdir(parents=True)
            configs = workflow.build_hedging_configs(root, run_wls=True, run_pwls=True)

        self.assertEqual(list(configs), ["wls", "pwls"])
        self.assertEqual(configs["wls"].calibration_xlsx.name, "calibration_wls.xlsx")
        self.assertEqual(configs["wls"].output_xlsx.name, "hedging_wls.xlsx")
        self.assertEqual(configs["wls"].evaluation_splits, ("test",))

    def test_hedging_suite_discards_large_return_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            (root / "outputs" / "canonical").mkdir(parents=True)
            configs = workflow.build_hedging_configs(root)
            configs["pwls"].calibration_xlsx.touch()
            with patch.object(workflow, "run_hedging_analysis", return_value={"large": pd.DataFrame({"x": range(10)})}) as mocked:
                paths = workflow.run_hedging_suite(configs)

        self.assertEqual(paths, {"pwls": configs["pwls"].output_xlsx})
        mocked.assert_called_once()

    def test_final_package_copies_existing_outputs_without_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pricing = root / "pricing"
            hedging = root / "hedging"
            final = root / "final"
            (pricing / "wls").mkdir(parents=True)
            (hedging / "pwls").mkdir(parents=True)
            pd.DataFrame({"metric": ["rmse"], "value": [1.0]}).to_csv(pricing / "overall_metrics_by_specification.csv", index=False)
            pd.DataFrame({"currency": ["BTC"], "rmse_reduction": [0.1]}).to_csv(hedging / "pwls" / "headline_test.csv", index=False)
            (pricing / "wls" / "figure.html").write_text("<html></html>", encoding="utf-8")
            (hedging / "pwls" / "hedge.html").write_text("<html></html>", encoding="utf-8")

            manifest = workflow.assemble_final_package(pricing, hedging, final)

            self.assertTrue((final / "tables" / "pricing" / "overall_metrics_by_specification.csv").exists())
            self.assertTrue((final / "tables" / "hedging" / "pwls" / "headline_test.csv").exists())
            self.assertTrue((final / "figures" / "pricing" / "wls" / "figure.html").exists())
            self.assertTrue((final / "figures" / "hedging" / "pwls" / "hedge.html").exists())
            self.assertTrue((final / "manifest.csv").exists())
            self.assertEqual(len(manifest), 4)


if __name__ == "__main__":
    unittest.main()
