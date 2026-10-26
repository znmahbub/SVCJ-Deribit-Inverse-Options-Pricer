from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tests._path import ROOT

if str(ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(ROOT / "notebooks"))

from _lib import stage2_workflow as stage2_mod
import src.batch_runner as batch_mod
from src.batch_runner import BatchConfig
from src.results_store import SHEET_NAMES, init_empty_workbook


class ParallelCheckpointExecutionTests(unittest.TestCase):
    def _cfg(self, root: Path, *, name: str, lam: float, n_workers: int = 2) -> BatchConfig:
        return BatchConfig(
            project_root=root,
            data_dir=root / "data",
            output_xlsx=root / f"{name}.xlsx",
            resume=False,
            save_every_n_files=1,
            l2_prev_strength=lam,
            n_workers=n_workers,
            currencies=("BTC", "ETH"),
            verbose=False,
        )

    def test_pwls_currency_parallelism_plans_two_independent_chains(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(Path(td), name="pwls", lam=1.0, n_workers=2)
            self.assertTrue(hasattr(batch_mod, "_currency_process_count"))
            self.assertEqual(batch_mod._currency_process_count(cfg, available_cpus=4), 2)

    def test_wls_and_pwls_overlap_when_cpu_budget_is_sufficient(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            configs = {
                "wls": self._cfg(root, name="wls", lam=0.0, n_workers=2),
                "pwls": self._cfg(root, name="pwls", lam=1.0, n_workers=2),
            }
            active = 0
            max_active = 0
            lock = threading.Lock()
            second_started = threading.Event()

            def fake_run(cfg):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                    if active >= 2:
                        second_started.set()
                second_started.wait(timeout=0.35)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return {}

            with patch.object(stage2_mod.os, "cpu_count", return_value=4), patch.object(
                stage2_mod, "run_all_snapshots_to_excel", side_effect=fake_run
            ):
                outputs = stage2_mod.run_calibration_suite(configs)

            self.assertEqual(max_active, 2)
            self.assertEqual(set(outputs), {"wls", "pwls"})

    def test_currency_runner_checkpoints_to_parquet_not_excel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            data.mkdir()
            for stamp in ("20260101T000000", "20260101T010000"):
                (data / f"deribit_options_snapshot_{stamp}Z.csv").write_text("x\n1\n", encoding="utf-8")

            cfg = self._cfg(root, name="pwls", lam=1.0, n_workers=2)

            counter = {"value": 0}

            def fake_process(path, **kwargs):
                counter["value"] += 1
                idx = counter["value"]
                ts = batch_mod.timestamp_from_filename(path)
                ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                base = pd.DataFrame([{"timestamp": ts_iso, "currency": "BTC", "success": True}])
                return {
                    "timestamp_iso": ts_iso,
                    "timestamp": ts,
                    "currency": "BTC",
                    "param_rows": {"black": base, "heston": base, "svcj": base},
                    "train_df": pd.DataFrame([{"snapshot_ts": ts_iso, "currency": "BTC", "split": "train"}]),
                    "test_df": pd.DataFrame([{"snapshot_ts": ts_iso, "currency": "BTC", "split": "test"}]),
                    "warm_next": {"black": {"sigma": float(idx)}},
                }

            with patch.object(batch_mod, "process_snapshot_to_payload", side_effect=fake_process), patch.object(
                batch_mod, "flush_workbook_atomic"
            ) as excel_flush, patch.object(
                batch_mod, "flush_currency_checkpoint", create=True
            ) as parquet_flush:
                batch_mod.run_currency_to_excel("BTC", cfg=cfg, workbook=init_empty_workbook())

            self.assertGreaterEqual(parquet_flush.call_count, 2)
            excel_flush.assert_not_called()

    def test_parquet_checkpoint_roundtrip_preserves_all_logical_sheets(self) -> None:
        self.assertTrue(hasattr(batch_mod, "flush_currency_checkpoint"))
        self.assertTrue(hasattr(batch_mod, "load_currency_checkpoint"))

        with tempfile.TemporaryDirectory() as td:
            checkpoint_dir = Path(td) / "BTC"
            workbook = {}
            for i, sheet in enumerate(SHEET_NAMES):
                workbook[sheet] = pd.DataFrame(
                    {
                        "timestamp": ["2026-01-01T00:00:00Z"],
                        "currency": ["BTC"],
                        "marker": [i],
                    }
                )

            batch_mod.flush_currency_checkpoint(workbook, checkpoint_dir)
            restored = batch_mod.load_currency_checkpoint(checkpoint_dir)

            self.assertEqual(set(restored), set(SHEET_NAMES))
            for sheet in SHEET_NAMES:
                self.assertTrue((checkpoint_dir / f"{sheet}.parquet").exists())
                pd.testing.assert_frame_equal(
                    workbook[sheet].reset_index(drop=True),
                    restored[sheet].reset_index(drop=True),
                    check_dtype=False,
                )


if __name__ == "__main__":
    unittest.main()
