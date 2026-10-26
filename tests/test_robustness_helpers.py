from __future__ import annotations

import unittest

import pandas as pd

from tests._path import ROOT  # noqa: F401
from notebooks._lib.robustness import (
    parameter_correlation_matrix,
    parameter_identification_summary,
)


class RobustnessHelperTests(unittest.TestCase):
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
