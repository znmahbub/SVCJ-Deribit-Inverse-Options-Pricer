from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tests._path import ROOT  # noqa: F401
from notebooks._lib import stage2_workflow
from src import batch_runner
from src.results_store import init_empty_workbook


class _FakeBar:
    def __init__(self, *, total, desc, position, **kwargs):
        self.total = total
        self.desc = desc
        self.position = position
        self.n = 0
        self.closed = False
        self.refresh_count = 0

    def reset(self, *, total=None):
        self.n = 0
        if total is not None:
            self.total = total

    def update(self, n=1):
        self.n += n

    def refresh(self):
        self.refresh_count += 1

    def close(self):
        self.closed = True


class CalibrationProgressTests(unittest.TestCase):
    def test_canonical_configs_reserve_four_distinct_progress_positions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            configs = stage2_workflow.build_calibration_configs(root, verbose=True)

        self.assertTrue(configs["wls"].show_progress)
        self.assertTrue(configs["pwls"].show_progress)
        self.assertEqual(configs["wls"].progress_label, "WLS")
        self.assertEqual(configs["pwls"].progress_label, "PWLS")
        self.assertEqual(configs["wls"].progress_position_offset, 0)
        self.assertEqual(configs["pwls"].progress_position_offset, 2)

    def test_progress_display_tracks_each_currency_independently(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = batch_runner.BatchConfig(
                project_root=Path(td),
                show_progress=True,
                progress_label="WLS",
                progress_position_offset=0,
                currencies=("BTC", "ETH"),
            )
            created: list[_FakeBar] = []

            def _factory(**kwargs):
                bar = _FakeBar(**kwargs)
                created.append(bar)
                return bar

            with patch.object(batch_runner, "tqdm", side_effect=_factory):
                display = batch_runner._ProgressDisplay(cfg, ("BTC", "ETH"))
                display.handle(("total", "BTC", 5))
                display.handle(("total", "ETH", 4))
                display.handle(("advance", "BTC", 1))
                display.handle(("advance", "BTC", 1))
                display.handle(("advance", "ETH", 1))
                display.close()

        self.assertEqual([(bar.desc, bar.position) for bar in created], [("WLS BTC", 0), ("WLS ETH", 1)])
        self.assertEqual(created[0].total, 5)
        self.assertEqual(created[0].n, 2)
        self.assertEqual(created[1].total, 4)
        self.assertEqual(created[1].n, 1)
        self.assertTrue(all(bar.closed for bar in created))

    def test_currency_runner_emits_snapshot_progress_and_suppresses_worker_chatter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data"
            data_dir.mkdir()
            files = []
            for stamp in ("20260101T000000Z", "20260101T010000Z"):
                path = data_dir / f"deribit_options_snapshot_{stamp}.csv"
                path.write_text("x\n1\n", encoding="utf-8")
                files.append(path)

            cfg = batch_runner.BatchConfig(
                project_root=root,
                data_dir=data_dir,
                output_xlsx=root / "out.xlsx",
                resume=False,
                save_every_n_files=10,
                n_workers=1,
                currencies=("BTC", "ETH"),
                show_progress=True,
                progress_label="WLS",
                verbose=True,
            )
            events = []
            observed_verbose = []

            def fake_process(path, **kwargs):
                observed_verbose.append(kwargs["verbose"])
                ts = batch_runner.timestamp_from_filename(path)
                ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                base = pd.DataFrame([{"timestamp": ts_iso, "currency": "BTC", "success": True}])
                return {
                    "timestamp_iso": ts_iso,
                    "timestamp": ts,
                    "currency": "BTC",
                    "param_rows": {"black": base, "heston": base, "svcj": base},
                    "train_df": pd.DataFrame(),
                    "test_df": pd.DataFrame(),
                    "warm_next": {},
                }

            with patch.object(batch_runner, "process_snapshot_to_payload", side_effect=fake_process), patch.object(
                batch_runner, "flush_currency_checkpoint"
            ):
                batch_runner.run_currency_to_excel(
                    "BTC",
                    cfg=cfg,
                    workbook=init_empty_workbook(),
                    progress_callback=events.append,
                )

        self.assertEqual(events[0], ("total", "BTC", 2))
        self.assertEqual(events.count(("advance", "BTC", 1)), 2)
        self.assertEqual(observed_verbose, [False, False])

    def test_outer_specification_progress_bar_is_not_rendered(self) -> None:
        import inspect

        source = inspect.getsource(stage2_workflow.run_calibration_suite)
        self.assertNotIn("Calibration specifications", source)


if __name__ == "__main__":
    unittest.main()
