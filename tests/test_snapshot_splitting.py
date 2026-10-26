from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests._path import ROOT  # noqa: F401
from src.calibration import CalibrationResult, WeightConfig
from src.hedging_analysis import combine_option_sheets
from src.inverse_fft_pricer import FFTParams
from src.results_store import TEST_SHEET, TRAIN_SHEET, init_empty_workbook
from src.snapshot_job import process_snapshot_to_payload
from src.splitting import SplitConfig, assign_cross_sectional_split


class TestSnapshotSplitting(unittest.TestCase):
    def _fixture(self, n: int = 40) -> pd.DataFrame:
        expiries = pd.to_datetime(
            ["2026-03-27T08:00:00Z", "2026-06-26T08:00:00Z"], utc=True
        )
        return pd.DataFrame(
            {
                "row_id": np.arange(n),
                "instrument_name": [f"BTC-TEST-{i}" for i in range(n)],
                "expiry_datetime": [expiries[i % 2] for i in range(n)],
                "option_type": ["call" if i % 2 == 0 else "put" for i in range(n)],
                "strike": 90_000.0 + 500.0 * np.arange(n),
            }
        )

    def test_default_split_is_deterministic_70_30_train_test_only(self) -> None:
        df = self._fixture(40)
        cfg = SplitConfig(random_seed=123)
        a = assign_cross_sectional_split(df, cfg)
        b = assign_cross_sectional_split(df.assign(dummy_model="svcj"), cfg)
        pd.testing.assert_series_equal(a, b)
        self.assertEqual(set(a.unique()), {"train", "test"})
        self.assertEqual(int((a == "train").sum()), 28)
        self.assertEqual(int((a == "test").sum()), 12)

    def test_fraction_validation(self) -> None:
        with self.assertRaises(ValueError):
            SplitConfig(train_frac=0.80, test_frac=0.30, random_seed=1)
        with self.assertRaises(ValueError):
            SplitConfig(train_frac=0.0, test_frac=1.0, random_seed=1)

    def test_tiny_strata_never_drop_rows_and_train_is_nonempty(self) -> None:
        for n in range(1, 5):
            with self.subTest(n=n):
                df = pd.DataFrame(
                    {
                        "row_id": np.arange(n),
                        "expiry_datetime": pd.to_datetime(["2026-03-27T08:00:00Z"] * n, utc=True),
                        "option_type": ["call"] * n,
                    }
                )
                split = assign_cross_sectional_split(df, SplitConfig(random_seed=17))
                self.assertEqual(len(split), n)
                self.assertFalse(split.isna().any())
                self.assertTrue(set(split.unique()).issubset({"train", "test"}))
                self.assertGreaterEqual(int((split == "train").sum()), 1)

    def test_feasible_strata_receive_train_and_test(self) -> None:
        df = pd.DataFrame(
            {
                "row_id": np.arange(12),
                "expiry_datetime": pd.to_datetime(
                    ["2026-03-27T08:00:00Z"] * 6 + ["2026-06-26T08:00:00Z"] * 6,
                    utc=True,
                ),
                "option_type": ["call"] * 6 + ["put"] * 6,
            }
        )
        split = assign_cross_sectional_split(df, SplitConfig(random_seed=5))
        for _, idx in df.groupby(["expiry_datetime", "option_type"]).groups.items():
            self.assertEqual(set(split.loc[list(idx)].unique()), {"train", "test"})

    def test_new_workbook_schema_contains_only_train_and_test_option_sheets(self) -> None:
        wb = init_empty_workbook()
        self.assertIn(TRAIN_SHEET, wb)
        self.assertIn(TEST_SHEET, wb)
        self.assertNotIn("validation_data", wb)

    def test_hedging_combiner_defaults_to_test(self) -> None:
        wb = {
            TRAIN_SHEET: pd.DataFrame([self._option_row("train")]),
            TEST_SHEET: pd.DataFrame([self._option_row("test")]),
        }
        got = combine_option_sheets(wb)
        self.assertEqual(set(got["split"]), {"test"})

    @staticmethod
    def _option_row(name: str) -> dict:
        return {
            "snapshot_ts": "2026-01-01T00:00:00Z",
            "currency": "BTC",
            "instrument_name": f"BTC-{name}",
            "option_type": "call",
            "strike": 100_000.0,
            "expiry_datetime": "2026-03-27T08:00:00Z",
            "time_to_maturity": 0.25,
            "mid_price_clean": 0.05,
            "price_black": 0.05,
            "price_heston": 0.05,
            "price_svcj": 0.05,
            "F0": 100_000.0,
        }

    def test_snapshot_job_calibrates_once_on_train_and_persists_two_splits(self) -> None:
        n = 20
        source = pd.DataFrame(
            {
                "row_id": np.arange(n),
                "currency": ["BTC"] * n,
                "instrument_name": [f"BTC-X-{i}" for i in range(n)],
                "expiry_datetime": ["2026-03-27T08:00:00Z"] * n,
                "option_type": ["call" if i % 2 == 0 else "put" for i in range(n)],
                "strike": 90_000.0 + 1_000.0 * np.arange(n),
                "mid_price_clean": np.linspace(0.01, 0.08, n),
                "open_interest": [10.0] * n,
            }
        )
        calibrated_train_ids: list[tuple[str, tuple[int, ...]]] = []

        def fake_calibrate(train: pd.DataFrame, model: str, **kwargs):
            calibrated_train_ids.append((model, tuple(sorted(train["row_id"].astype(int).tolist()))))
            params = {
                "black": {"sigma": 0.5},
                "heston": {"kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.5, "v0": 0.04},
                "svcj": {"kappa": 2.0, "theta": 0.04, "sigma_v": 0.3, "rho": -0.5, "v0": 0.04, "lam": 0.2, "ell_y": -0.05, "sigma_y": 0.2, "ell_v": 0.1, "rho_j": -0.3},
            }[model]
            return CalibrationResult(model=model, params=params, success=True, message="ok", nfev=1, rmse=0.0, mae=0.0)

        def fake_price(df: pd.DataFrame, *args, **kwargs):
            return df["mid_price_clean"].to_numpy(dtype=float)

        with tempfile.TemporaryDirectory() as td:
            csv = Path(td) / "deribit_options_snapshot_20260101T000000Z.csv"
            source.to_csv(csv, index=False)
            with patch("src.snapshot_job.filter_liquid_options", side_effect=lambda df, **kwargs: df.copy()), patch(
                "src.snapshot_job.calibrate_model", side_effect=fake_calibrate
            ), patch("src.snapshot_job.price_dataframe", side_effect=fake_price):
                payload = process_snapshot_to_payload(
                    csv,
                    currency="BTC",
                    filter_rules={},
                    weight_config=WeightConfig(use_spread=False, use_vega=False, use_open_interest=False),
                    fft_base=FFTParams(N=64, alpha=1.5, eta=0.2, b=-5.0, use_simpson=True),
                    max_nfev={"black": 1, "heston": 1, "svcj": 1},
                    train_frac=0.70,
                    test_frac=0.30,
                    random_seed=123,
                    runtime_top_expiries_by_oi=None,
                    runtime_max_options=None,
                    min_options_after_filter=1,
                    warm_start=None,
                    l2_prev_strength=0.0,
                    verbose=False,
                )

        self.assertEqual([m for m, _ in calibrated_train_ids], ["black", "heston", "svcj"])
        self.assertEqual(len({ids for _, ids in calibrated_train_ids}), 1)
        self.assertNotIn("validation_df", payload)
        frames = [payload["train_df"], payload["test_df"]]
        self.assertEqual(sum(len(x) for x in frames), n)
        self.assertEqual({x["split"].iloc[0] for x in frames if len(x)}, {"train", "test"})
        row = payload["param_rows"]["black"].iloc[0]
        self.assertNotIn("rmse_validation", row.index)
        self.assertNotIn("n_validation", row.index)
        self.assertEqual(int(row["n_train"]), 14)
        self.assertEqual(int(row["n_test"]), 6)


if __name__ == "__main__":
    unittest.main()
