from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests._path import ROOT  # noqa: F401
from notebooks._lib import chapter3_final


class Chapter3MetricTests(unittest.TestCase):
    def _fixture(self) -> pd.DataFrame:
        rows = [
            # BTC snapshot 1: errors +1, +3 coin for Black.
            dict(snapshot_ts="2026-01-01T00:00:00Z", currency="BTC", F0=100.0, bid_price=9.0, ask_price=11.0, mid_price_clean=10.0, price_black=11.0, price_heston=10.5, price_svcj=10.0),
            dict(snapshot_ts="2026-01-01T00:00:00Z", currency="BTC", F0=100.0, bid_price=8.0, ask_price=12.0, mid_price_clean=10.0, price_black=13.0, price_heston=11.0, price_svcj=10.0),
            # BTC snapshot 2: one zero-error quote. This makes quote-weighted and snapshot-mean RMSE differ.
            dict(snapshot_ts="2026-01-02T00:00:00Z", currency="BTC", F0=200.0, bid_price=9.0, ask_price=11.0, mid_price_clean=10.0, price_black=10.0, price_heston=10.0, price_svcj=10.0),
            # ETH uses different coin units and forward level.
            dict(snapshot_ts="2026-01-01T00:00:00Z", currency="ETH", F0=10.0, bid_price=4.0, ask_price=6.0, mid_price_clean=5.0, price_black=6.0, price_heston=5.5, price_svcj=5.0),
        ]
        frame = pd.DataFrame(rows)
        frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True)
        return chapter3_final.add_common_unit_errors(frame)

    def test_option_weighted_and_snapshot_mean_rmse_are_distinct(self) -> None:
        df = self._fixture().query("currency == 'BTC'")
        quote_rmse = chapter3_final.option_weighted_rmse(df["error_coin_black"])
        snapshot_mean = chapter3_final.mean_snapshot_rmse(
            df, error_col="error_coin_black", snapshot_col="snapshot_ts"
        )
        self.assertAlmostEqual(quote_rmse, np.sqrt((1.0 + 9.0 + 0.0) / 3.0))
        self.assertAlmostEqual(snapshot_mean, (np.sqrt(5.0) + 0.0) / 2.0)
        self.assertNotAlmostEqual(quote_rmse, snapshot_mean)

    def test_common_unit_errors_match_hand_calculation(self) -> None:
        df = self._fixture()
        btc0 = df.iloc[0]
        self.assertAlmostEqual(float(btc0["error_usd_black"]), 100.0)
        self.assertAlmostEqual(float(btc0["spread_coin"]), 2.0)
        self.assertAlmostEqual(float(btc0["error_spread_units_black"]), 0.5)

        eth = df.query("currency == 'ETH'").iloc[0]
        self.assertAlmostEqual(float(eth["error_usd_black"]), 10.0)
        self.assertAlmostEqual(float(eth["error_spread_units_black"]), 0.5)

    def test_overall_table_has_no_pooled_raw_coin_rmse(self) -> None:
        df = self._fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = chapter3_final.compute_overall_baseline_metrics(df, Path(tmp))

        pooled = out.loc[out["scope"] == "Pooled"].iloc[0]
        self.assertNotIn("rmse_coin_black", pooled.dropna().index)
        self.assertAlmostEqual(
            float(pooled["rmse_usd_black"]),
            np.sqrt((100.0**2 + 300.0**2 + 0.0**2 + 10.0**2) / 4.0),
        )
        self.assertAlmostEqual(
            float(pooled["rmse_spread_units_black"]),
            np.sqrt((0.5**2 + 0.75**2 + 0.0**2 + 0.5**2) / 4.0),
        )
        self.assertAlmostEqual(float(pooled["within_spread_black"]), 0.75)

        btc = out.loc[out["scope"] == "BTC"].iloc[0]
        self.assertAlmostEqual(
            float(btc["rmse_coin_black"]), np.sqrt((1.0 + 9.0 + 0.0) / 3.0)
        )
        self.assertIn("snapshot_mean_rmse_coin_black", btc.index)

    def test_metric_names_encode_units(self) -> None:
        df = self._fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out = chapter3_final.compute_overall_baseline_metrics(df, Path(tmp))
        forbidden = {"pooled_rmse", "rmse_black", "mae_black", "rmse_heston", "rmse_svcj"}
        self.assertTrue(forbidden.isdisjoint(out.columns))


if __name__ == "__main__":
    unittest.main()
