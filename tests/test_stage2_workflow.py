from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tests._path import ROOT  # noqa: F401
from notebooks._lib import stage2_workflow as workflow
from src.batch_runner import PWLS_LAMBDA_TS, WLS_LAMBDA_TS


class Stage2WorkflowTests(unittest.TestCase):
    def test_canonical_specs_are_fixed_wls_and_pwls(self) -> None:
        specs = workflow.canonical_calibration_specs()
        self.assertEqual(list(specs), ["wls", "pwls"])
        self.assertEqual(specs["wls"].lambda_ts, WLS_LAMBDA_TS)
        self.assertEqual(specs["pwls"].lambda_ts, PWLS_LAMBDA_TS)
        self.assertEqual(specs["wls"].workbook_name, "calibration_wls.xlsx")
        self.assertEqual(specs["pwls"].workbook_name, "calibration_pwls.xlsx")

    def test_build_calibration_configs_defaults_to_full_dataset_and_70_30(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            (root / "excel files").mkdir()
            configs = workflow.build_calibration_configs(root)

        self.assertEqual(set(configs), {"wls", "pwls"})
        for cfg in configs.values():
            self.assertIsNone(cfg.analysis_start)
            self.assertIsNone(cfg.analysis_end)
            self.assertAlmostEqual(cfg.train_frac, 0.70)
            self.assertAlmostEqual(cfg.test_frac, 0.30)
            self.assertEqual(cfg.smoke_test_max_files_per_currency, None)
            self.assertEqual(cfg.currencies, ("BTC", "ETH"))
        self.assertEqual(configs["wls"].l2_prev_strength, WLS_LAMBDA_TS)
        self.assertEqual(configs["pwls"].l2_prev_strength, PWLS_LAMBDA_TS)
        self.assertEqual(configs["wls"].output_xlsx.name, "calibration_wls.xlsx")
        self.assertEqual(configs["pwls"].output_xlsx.name, "calibration_pwls.xlsx")
        self.assertEqual(configs["wls"].output_xlsx.parent.name, "canonical")

    def test_explicit_date_bounds_remain_optional_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            (root / "excel files").mkdir()
            configs = workflow.build_calibration_configs(
                root,
                start_dt="2026-01-01",
                end_dt="2026-01-31",
            )
        for cfg in configs.values():
            self.assertEqual(str(cfg.analysis_start), "2026-01-01 00:00:00+00:00")
            self.assertEqual(str(cfg.analysis_end), "2026-01-31 23:59:59.999999999+00:00")

    def test_snapshot_inventory_reports_schema_and_currency_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            columns = {
                "currency": ["BTC", "ETH", "BTC", "ETH"],
                "option_type": ["call", "call", "put", "put"],
                "strike": [100, 10, 90, 9],
                "time_to_maturity": [0.1, 0.1, 0.1, 0.1],
                "bid_price": [0.1, 0.1, 0.1, 0.1],
                "ask_price": [0.2, 0.2, 0.2, 0.2],
                "futures_price": [100, 10, 100, 10],
                "vega": [1, 1, 1, 1],
                "open_interest": [1, 1, 1, 1],
                "expiry_datetime": ["2026-02-01"] * 4,
            }
            for stamp in ["20260101T000000Z", "20260101T010000Z"]:
                pd.DataFrame(columns).to_csv(data_dir / f"deribit_options_snapshot_{stamp}.csv", index=False)

            inventory = workflow.inspect_snapshot_data(data_dir)
            summary = workflow.summarize_snapshot_data(inventory)
            workflow.assert_snapshot_data_ready(inventory)

        self.assertEqual(len(inventory), 2)
        self.assertTrue(inventory["schema_ok"].all())
        self.assertTrue((inventory["n_BTC"] == 2).all())
        self.assertTrue((inventory["n_ETH"] == 2).all())
        self.assertEqual(int(summary.loc[0, "n_snapshots"]), 2)
        self.assertEqual(int(summary.loc[0, "n_schema_failures"]), 0)
        self.assertEqual(int(summary.loc[0, "n_missing_BTC"]), 0)
        self.assertEqual(int(summary.loc[0, "n_missing_ETH"]), 0)

    def test_calibration_suite_returns_paths_not_large_workbook_objects(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            (root / "excel files").mkdir()
            configs = workflow.build_calibration_configs(root, verbose=False)
            large_sentinel = {"train_data": pd.DataFrame({"x": range(1000)})}
            with patch.object(workflow, "run_all_snapshots_to_excel", return_value=large_sentinel) as mocked:
                outputs = workflow.run_calibration_suite(configs)

        self.assertEqual(outputs, {key: cfg.output_xlsx for key, cfg in configs.items()})
        self.assertEqual(mocked.call_count, 2)
        self.assertTrue(all(isinstance(path, Path) for path in outputs.values()))

    def test_pricing_loader_is_hard_wired_to_test_data(self) -> None:
        self.assertNotIn(
            "evaluation_split",
            inspect.signature(workflow.chapter3.load_baseline_workbook).parameters,
        )
        sentinel = (
            pd.DataFrame({"timestamp": []}),
            pd.DataFrame({"timestamp": []}),
            pd.DataFrame({"timestamp": []}),
            pd.DataFrame({"snapshot_ts": []}),
        )
        with patch.object(workflow.chapter3, "load_baseline_workbook", return_value=sentinel) as mocked:
            loaded = workflow.load_pricing_inputs(Path("example.xlsx"))
        self.assertEqual(loaded, sentinel)
        mocked.assert_called_once_with(Path("example.xlsx"))


if __name__ == "__main__":
    unittest.main()