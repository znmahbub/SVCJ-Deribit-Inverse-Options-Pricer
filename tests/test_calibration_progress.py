from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._path import ROOT  # noqa: F401
from notebooks._lib import stage2_workflow
from src import batch_runner


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

    def test_outer_specification_progress_bar_is_not_rendered(self) -> None:
        import inspect

        source = inspect.getsource(stage2_workflow.run_calibration_suite)
        self.assertNotIn("Calibration specifications", source)


if __name__ == "__main__":
    unittest.main()
