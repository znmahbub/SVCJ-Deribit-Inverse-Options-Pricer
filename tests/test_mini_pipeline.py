from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests._path import ROOT  # noqa: F401  (ensures repo imports are available)
from notebooks._lib.stage2_workflow import run_single_pricing_analysis  # noqa: E402
from notebooks._lib.stage3_workflow import assemble_final_package, build_hedging_report  # noqa: E402
from src.batch_runner import BatchConfig, run_all_snapshots_to_excel  # noqa: E402
from src.hedging_analysis import HedgingConfig, run_hedging_analysis  # noqa: E402
from src.inverse_fft_pricer import FFTParams  # noqa: E402


def _black_inverse_price(F0: float, K: float, T: float, sigma: float, option_type: str) -> float:
    vol = sigma * np.sqrt(T)
    d1 = (np.log(F0 / K) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    call_usd = F0 * norm.cdf(d1) - K * norm.cdf(d2)
    call_coin = call_usd / F0
    if option_type == "call":
        return float(call_coin)
    return float(max(call_coin - (1.0 - K / F0), 0.0))


def _write_snapshot(data_dir: Path, stamp: str, *, F0: float, hours_from_start: int) -> None:
    sigma = 0.55
    expiries = [
        ("2026-02-15T00:00:00Z", 45.0 / 365.25),
        ("2026-04-01T00:00:00Z", 90.0 / 365.25),
    ]
    strikes = [8500.0, 9250.0, 10000.0, 10750.0, 11500.0]
    rows: list[dict[str, object]] = []
    for expiry_idx, (expiry, base_t) in enumerate(expiries):
        T = base_t - hours_from_start / (365.25 * 24.0)
        for option_type in ("call", "put"):
            side = "C" if option_type == "call" else "P"
            for K in strikes:
                mid = _black_inverse_price(F0, K, T, sigma, option_type)
                half_spread = max(2.0e-5, 0.025 * max(mid, 1.0e-4))
                bid = max(mid - half_spread, 1.0e-6)
                ask = max(mid + half_spread, bid + 1.0e-6)
                rows.append(
                    {
                        "timestamp": pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(hours=hours_from_start),
                        "currency": "BTC",
                        "instrument_name": f"BTC-E{expiry_idx}-{int(K)}-{side}",
                        "option_type": option_type,
                        "strike": K,
                        "time_to_maturity": T,
                        "bid_price": bid,
                        "ask_price": ask,
                        "futures_price": F0,
                        "vega": 1.0,
                        "open_interest": 100.0,
                        "expiry_datetime": expiry,
                    }
                )
    pd.DataFrame(rows).to_csv(data_dir / f"deribit_options_snapshot_{stamp}.csv", index=False)


def _write_perp_proxy(data_dir: Path) -> None:
    pd.DataFrame(
        {
            "obs_datetime": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "coin": ["BTC", "BTC"],
            "perp_futures_price_proxy": [9990.0, 10015.0],
            "price_source": ["synthetic_stage4_smoke", "synthetic_stage4_smoke"],
            "price_kind": ["nearest_1m_candle_open_close", "nearest_1m_candle_open_close"],
            "candle_price_gap_seconds": [0.0, 0.0],
            "candle_price_choice": ["open", "close"],
            "candle_resolution": ["1", "1"],
        }
    ).to_csv(data_dir / "perpetual_futures_prices.csv", index=False)


class TestMiniCanonicalPipeline(unittest.TestCase):
    def test_synthetic_snapshot_to_final_artifact_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            _write_snapshot(data_dir, "20260101T000000Z", F0=10_000.0, hours_from_start=0)
            _write_snapshot(data_dir, "20260101T010000Z", F0=10_020.0, hours_from_start=1)
            _write_perp_proxy(data_dir)

            calibration_xlsx = root / "excel files" / "canonical" / "calibration_pwls.xlsx"
            cfg = BatchConfig(
                project_root=root,
                data_dir=data_dir,
                output_xlsx=calibration_xlsx,
                resume=False,
                smoke_test_max_files_per_currency=1,
                save_every_n_files=10,
                min_options_after_filter=10,
                filter_rules=dict(
                    require_bid_ask=True,
                    min_time_to_maturity=1 / 365,
                    max_time_to_maturity=None,
                    min_open_interest=1.0,
                    min_vega=0.0,
                    max_rel_spread=0.50,
                    moneyness_range=(0.5, 2.0),
                    drop_synthetic_underlyings=False,
                ),
                fft_base=FFTParams(N=256, eta=0.20, alpha=1.5, b=-10.0, use_simpson=True),
                train_frac=0.70,
                test_frac=0.30,
                global_random_seed=123,
                max_nfev=dict(black=40, heston=60, svcj=80),
                l2_prev_strength=1.0,
                n_workers=1,
                limit_internal_threads=True,
                internal_num_threads=1,
                currencies=("BTC",),
                verbose=False,
            )
            workbook = run_all_snapshots_to_excel(cfg)

            self.assertTrue(calibration_xlsx.exists())
            self.assertEqual(set(workbook), {"black_params", "heston_params", "svcj_params", "train_data", "test_data"})
            self.assertEqual(len(workbook["train_data"]) + len(workbook["test_data"]), 20)
            self.assertEqual(set(workbook["train_data"]["split"].unique()), {"train"})
            self.assertEqual(set(workbook["test_data"]["split"].unique()), {"test"})
            self.assertTrue(bool(workbook["black_params"]["success"].iloc[0]), str(workbook["black_params"]["message"].iloc[0]))
            self.assertTrue(bool(workbook["heston_params"]["success"].iloc[0]), str(workbook["heston_params"]["message"].iloc[0]))
            self.assertGreater(int(workbook["svcj_params"]["nfev"].iloc[0]), 0)
            self.assertTrue(np.isfinite(pd.to_numeric(workbook["svcj_params"]["kappa"], errors="coerce").iloc[0]))
            self.assertEqual(float(workbook["svcj_params"]["l2_prev_strength"].iloc[0]), 1.0)

            sheet_names = set(pd.ExcelFile(calibration_xlsx, engine="openpyxl").sheet_names)
            self.assertIn("train_data", sheet_names)
            self.assertIn("test_data", sheet_names)
            self.assertNotIn("validation_data", sheet_names)

            pricing_dir = root / "notebooks" / "pricing_outputs" / "canonical" / "pwls"
            pricing = run_single_pricing_analysis(
                calibration_xlsx,
                pricing_dir,
                n_buckets=2,
                make_figures=False,
            )
            self.assertFalse(pricing["overall"].empty)
            self.assertFalse(pricing["test_data"].empty)

            hedging_xlsx = root / "excel files" / "canonical" / "hedging_pwls.xlsx"
            hedge_cfg = HedgingConfig(
                calibration_xlsx=calibration_xlsx,
                data_dir=data_dir,
                output_xlsx=hedging_xlsx,
                analysis_start="2026-01-01T00:00:00Z",
                analysis_end="2026-01-01T01:00:00Z",
                verify_prices=True,
                rebalance_every_n_snapshots=1,
                currencies=("BTC",),
                evaluation_splits=("test",),
                exclude_expiry_crossing=True,
                delta_u_max=80.0,
                delta_quad_n=32,
                fft_params=FFTParams(N=256, eta=0.20, alpha=1.5, b=-10.0, use_simpson=True),
            )
            hedge_result = run_hedging_analysis(hedge_cfg)
            panel = hedge_result["hedge_interval_panel"]
            price_verification = hedge_result["price_verification"]
            self.assertTrue(hedging_xlsx.exists())
            self.assertFalse(panel.empty)
            self.assertEqual(set(panel["split"].astype(str).str.lower()), {"test"})
            self.assertTrue(panel["eval_snapshot_ts"].dropna().le(pd.Timestamp("2026-01-01T01:00:00Z")).all())
            self.assertGreater(int(panel["is_summary_eligible"].fillna(False).sum()), 0)
            valid_price_checks = pd.to_numeric(price_verification["n_valid_diff"], errors="coerce").fillna(0) > 0
            self.assertTrue(valid_price_checks.any())
            self.assertLess(
                float(pd.to_numeric(price_verification.loc[valid_price_checks, "max_abs_price_diff"], errors="coerce").max()),
                1e-10,
            )

            hedging_report_dir = root / "notebooks" / "hedging_outputs" / "canonical" / "pwls"
            report = build_hedging_report(
                hedging_xlsx,
                hedging_report_dir,
                q=2,
                make_figures=False,
            )
            self.assertFalse(report["tables"]["headline_test"].empty)

            final_dir = root / "notebooks" / "final_outputs" / "canonical"
            manifest = assemble_final_package(
                pricing_dir.parent,
                hedging_report_dir.parent,
                final_dir,
            )
            self.assertFalse(manifest.empty)
            self.assertTrue((final_dir / "manifest.csv").exists())
            self.assertIn("pricing_table", set(manifest["artifact_type"]))
            self.assertIn("hedging_table", set(manifest["artifact_type"]))


if __name__ == "__main__":
    unittest.main()
