from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from tests._path import PROJECT_ROOT  # noqa: F401  (ensures project import path)
from src.analysis_window import (
    filter_paths_by_timestamp,
    normalize_analysis_window,
    normalize_utc_boundary,
)
from src.snapshot_job import timestamp_from_filename


class TestAnalysisWindow(unittest.TestCase):
    def setUp(self) -> None:
        self.files = [
            Path("deribit_options_snapshot_20251229T235959Z.csv"),
            Path("deribit_options_snapshot_20251230T000000Z.csv"),
            Path("deribit_options_snapshot_20260115T120000Z.csv"),
            Path("deribit_options_snapshot_20260408T100119Z.csv"),
            Path("deribit_options_snapshot_20260408T235959Z.csv"),
            Path("deribit_options_snapshot_20260409T000000Z.csv"),
        ]

    def test_none_window_preserves_all_files(self) -> None:
        actual = filter_paths_by_timestamp(
            self.files,
            timestamp_from_filename,
            start=None,
            end=None,
        )
        self.assertEqual(actual, self.files)

    def test_date_window_is_inclusive_and_end_date_means_full_day(self) -> None:
        actual = filter_paths_by_timestamp(
            self.files,
            timestamp_from_filename,
            start="2025-12-30",
            end="2026-04-08",
        )
        self.assertEqual(
            actual,
            self.files[1:5],
        )

    def test_exact_timestamp_boundaries_are_included(self) -> None:
        actual = filter_paths_by_timestamp(
            self.files,
            timestamp_from_filename,
            start=pd.Timestamp("2025-12-30T00:00:00Z"),
            end=pd.Timestamp("2026-04-08T10:01:19Z"),
        )
        self.assertEqual(actual, self.files[1:4])

    def test_date_only_end_normalizes_to_end_of_day(self) -> None:
        end = normalize_utc_boundary("2026-04-08", is_end=True)
        self.assertEqual(end, pd.Timestamp("2026-04-08T23:59:59.999999999Z"))

    def test_invalid_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_analysis_window("2026-04-09", "2026-04-08")


if __name__ == "__main__":
    unittest.main()
