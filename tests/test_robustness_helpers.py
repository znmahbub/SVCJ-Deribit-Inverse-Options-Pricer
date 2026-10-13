from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests._path import ROOT  # noqa: F401
from notebooks._lib.robustness import (
    DEFAULT_SCENARIO_RATES,
    apply_discounting_to_inverse_price,
    assess_discounting_promotion,
    build_discounting_sensitivity_table,
    discount_factor,
    parameter_correlation_matrix,
    parameter_identification_summary,
)


class RobustnessHelperTests(unittest.TestCase):
    def test_discount_factor_is_exponential_and_zero_rate_is_identity(self) -> None:
        T = np.array([0.0, 0.25, 1.0])
        got = discount_factor(T, 0.05)
        np.testing.assert_allclose(got, np.exp(-0.05 * T), rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(discount_factor(T, 0.0), np.ones_like(T))

    def test_zero_rate_reproduces_current_inverse_prices_exactly(self) -> None:
        price = np.array([0.01, 0.05, 0.20])
        T = np.array([0.1, 0.5, 1.0])
        np.testing.assert_array_equal(apply_discounting_to_inverse_price(price, T, 0.0), price)

    def test_positive_rate_price_sensitivity_grows_with_maturity(self) -> None:
        price = np.ones(3)
        T = np.array([0.1, 0.5, 1.0])
        discounted = apply_discounting_to_inverse_price(price, T, 0.075)
        shift = np.abs(discounted - price)
        self.assertTrue(np.all(np.diff(shift) > 0.0))

    def test_sensitivity_table_has_all_scenario_rates_and_zero_rate_baseline(self) -> None:
        quotes = pd.DataFrame(
            {
                "currency": ["BTC", "BTC"],
                "time_to_maturity": [0.25, 1.0],
                "bid_price": [0.09, 0.18],
                "ask_price": [0.11, 0.22],
                "mid_price_clean": [0.10, 0.20],
                "price_black": [0.11, 0.21],
                "price_heston": [0.105, 0.205],
                "price_svcj": [0.10, 0.20],
            }
        )
        table = build_discounting_sensitivity_table(quotes)
        self.assertEqual(set(table["rate"]), set(DEFAULT_SCENARIO_RATES))
        self.assertEqual(set(table["model"]), {"black", "heston", "svcj"})
        baseline = table.loc[(table["rate"] == 0.0) & (table["model"] == "black")].iloc[0]
        self.assertAlmostEqual(float(baseline["rmse_coin"]), np.sqrt((0.01**2 + 0.01**2) / 2.0))
        self.assertAlmostEqual(float(baseline["mean_abs_price_shift_coin"]), 0.0)

    def test_discounting_promotion_must_be_decided_within_currency(self) -> None:
        table = pd.DataFrame(
            [
                {"scope": scope, "model": model, "rate": rate, "rmse_coin": rmse, "within_spread_share": within}
                for scope in ("BTC", "ETH", "All")
                for rate, rmse, within in ((0.0, 1.0, 0.5), (0.05, 1.01, 0.5))
                for model in ("black", "heston", "svcj")
            ]
        )
        with self.assertRaisesRegex(ValueError, "cross-asset"):
            assess_discounting_promotion(table, scope="All")
        decision = assess_discounting_promotion(table, scope="BTC")
        self.assertEqual(decision["scope"], "BTC")
        self.assertFalse(bool(decision["promote_discounting_to_baseline"]))

    def test_identification_summary_treats_rho_j_as_unbounded_loading(self) -> None:
        params = pd.DataFrame(
            {
                "currency": ["ETH", "ETH", "BTC"],
                "success": [True, True, True],
                "kappa": [2.0, 1.0, 2.0],
                "theta": [0.04, 0.04, 0.04],
                "sigma_v": [0.30, 0.50, 0.30],
                "rho": [-0.99, -0.2, -0.5],
                "v0": [0.04, 0.04, 0.04],
                "lam": [0.2, 1e-6, 0.2],
                "ell_y": [-0.05, -0.1, -0.05],
                "sigma_y": [0.20, 1e-4, 0.20],
                "ell_v": [0.1, 0.1, 0.1],
                "rho_j": [-2.0, 1.25, -0.3],
                "feller_gap": [0.07, -0.17, 0.07],
                "feller_violated": [False, True, False],
                "svcj_martingale_margin": [1.2, 0.875, 1.03],
                "svcj_contour_margin": [0.8, 0.3, 0.9],
            }
        )
        summary = parameter_identification_summary(params)
        eth = summary.loc[summary["currency"] == "ETH"].iloc[0]
        self.assertAlmostEqual(float(eth["feller_violation_share"]), 0.5)
        self.assertAlmostEqual(float(eth["rho_boundary_share"]), 0.5)
        self.assertAlmostEqual(float(eth["rho_j_abs_gt_one_share"]), 1.0)
        self.assertNotIn("rho_j_boundary_share", summary.columns)
        self.assertAlmostEqual(float(eth["sigma_y_near_lower_bound_share"]), 0.5)
        self.assertAlmostEqual(float(eth["lambda_near_lower_bound_share"]), 0.5)
        self.assertAlmostEqual(float(eth["martingale_margin_min"]), 0.875)
        self.assertAlmostEqual(float(eth["contour_margin_min"]), 0.3)

    def test_parameter_correlation_matrix_is_currency_specific(self) -> None:
        params = pd.DataFrame(
            {
                "currency": ["ETH", "ETH", "ETH", "BTC", "BTC"],
                "success": [True] * 5,
                "kappa": [1.0, 2.0, 3.0, 1.0, 2.0],
                "theta": [2.0, 4.0, 6.0, 5.0, 4.0],
                "rho_j": [-2.0, -1.0, 0.0, -0.5, -0.4],
            }
        )
        corr = parameter_correlation_matrix(params, currency="ETH", columns=["kappa", "theta", "rho_j"])
        self.assertAlmostEqual(float(corr.loc["kappa", "theta"]), 1.0)
        self.assertAlmostEqual(float(corr.loc["kappa", "rho_j"]), 1.0)


if __name__ == "__main__":
    unittest.main()
