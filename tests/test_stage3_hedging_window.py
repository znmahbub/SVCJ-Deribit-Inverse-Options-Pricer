from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests._path import ROOT  # noqa: F401
from src.hedging_analysis import HedgingConfig, filter_options_for_run, list_snapshot_files


class Stage3HedgingWindowTests(unittest.TestCase):
    def _cfg(self, root: Path, **kwargs) -> HedgingConfig:
        return HedgingConfig(
            calibration_xlsx=root / "calibration.xlsx",
            data_dir=root / "data",
            output_xlsx=root / "hedging.xlsx",
            **kwargs,
        )

    def test_hedging_config_normalizes_optional_analysis_window(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(
                Path(td),
                analysis_start="2026-01-02",
                analysis_end="2026-01-03",
            )
        self.assertEqual(str(cfg.analysis_start), "2026-01-02 00:00:00+00:00")
        self.assertEqual(str(cfg.analysis_end), "2026-01-03 23:59:59.999999999+00:00")

    def test_validation_split_is_not_part_of_canonical_hedging_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                self._cfg(Path(td), evaluation_splits=("validation",))

    def test_option_filter_applies_same_analysis_window_as_snapshot_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._cfg(
                root,
                analysis_start="2026-01-02T00:00:00Z",
                analysis_end="2026-01-03T00:00:00Z",
            )
            options = pd.DataFrame(
                {
                    "snapshot_ts": pd.to_datetime(
                        [
                            "2026-01-01T00:00:00Z",
                            "2026-01-02T00:00:00Z",
                            "2026-01-03T00:00:00Z",
                            "2026-01-04T00:00:00Z",
                        ],
                        utc=True,
                    ),
                    "currency": ["BTC"] * 4,
                    "instrument_name": ["A", "B", "C", "D"],
                }
            )
            filtered = filter_options_for_run(options, cfg)
        self.assertEqual(filtered["instrument_name"].tolist(), ["B", "C"])

    def test_snapshot_lookup_source_is_filtered_before_forward_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            for stamp in [
                "20260101T000000Z",
                "20260102T000000Z",
                "20260103T000000Z",
                "20260104T000000Z",
            ]:
                (data_dir / f"deribit_options_snapshot_{stamp}.csv").write_text("x\n1\n", encoding="utf-8")

            index = list_snapshot_files(
                data_dir,
                analysis_start="2026-01-02T00:00:00Z",
                analysis_end="2026-01-03T00:00:00Z",
            )

        self.assertEqual(
            index["snapshot_ts"].dt.strftime("%Y-%m-%d").tolist(),
            ["2026-01-02", "2026-01-03"],
        )


if __name__ == "__main__":
    unittest.main()
