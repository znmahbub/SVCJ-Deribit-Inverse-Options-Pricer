from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd

from tests._path import ROOT  # noqa: F401
from src.calibration import (
    WeightConfig,
    _constraint_diagnostics,
    _constraint_residuals,
    _regularization_residual,
    _residual_scales,
)


class TestCalibrationObjective(unittest.TestCase):
    def test_spread_scaling_implies_inverse_spread_squared_objective(self) -> None:
        df = pd.DataFrame(
            {
                "spread": [0.09, 0.19],
                "vega": [1.0, 1.0],
                "open_interest": [1.0, 1.0],
            }
        )
        cfg = WeightConfig(
            use_spread=True,
            use_vega=False,
            use_open_interest=False,
            spread_power=1.0,
            eps_spread=0.01,
            cap=None,
        )
        scales = _residual_scales(df, cfg)
        npt.assert_allclose(scales, np.array([10.0, 5.0]), rtol=0.0, atol=1e-14)

        errors = np.array([0.02, -0.02])
        residuals = scales * errors
        expected_sse = (0.02 / 0.10) ** 2 + (-0.02 / 0.20) ** 2
        self.assertAlmostEqual(float(np.sum(residuals**2)), expected_sse, places=14)

    def test_regularization_residual_is_in_packed_coordinate_space(self) -> None:
        x = np.array([1.0, -0.5, 2.0])
        x_prev = np.array([0.5, -0.25, 1.0])
        got = _regularization_residual(x, x_prev, strength=4.0)
        npt.assert_allclose(got, 2.0 * (x - x_prev), rtol=0.0, atol=0.0)
        self.assertEqual(_regularization_residual(x, None, strength=4.0).size, 0)
        self.assertEqual(_regularization_residual(x, x_prev, strength=0.0).size, 0)

    def test_feller_violation_is_a_finite_soft_penalty(self) -> None:
        params = {"kappa": 1.0, "theta": 0.04, "sigma_v": 1.0, "rho": -0.5, "v0": 0.04}
        penalty = _constraint_residuals(
            "heston",
            params,
            constraint_penalty=100.0,
            feller_eps=0.0,
            svcj_moment_eps=1e-6,
            carr_madan_q_values=(),
        )
        self.assertEqual(penalty.shape, (1,))
        self.assertTrue(np.isfinite(penalty[0]))
        self.assertGreater(float(penalty[0]), 0.0)

        diag = _constraint_diagnostics(
            "heston",
            params,
            feller_eps=0.0,
            svcj_moment_eps=1e-6,
            carr_madan_q_values=(),
        )
        self.assertTrue(diag["feller_penalty_active"])
        self.assertTrue(diag["feller_violated"])
        self.assertLess(diag["feller_gap"], 0.0)

    def test_svcj_moment_diagnostics_record_active_and_violated_flags(self) -> None:
        params = {
            "kappa": 2.0,
            "theta": 0.04,
            "sigma_v": 0.30,
            "rho": -0.50,
            "v0": 0.04,
            "lam": 0.20,
            "ell_y": -0.05,
            "sigma_y": 0.20,
            "ell_v": 0.50,
            "rho_j": 1.0,
        }
        diag = _constraint_diagnostics(
            "svcj",
            params,
            feller_eps=0.0,
            svcj_moment_eps=1e-6,
            carr_madan_q_values=(2.5,),
        )
        self.assertTrue(diag["svcj_moment_penalty_active"])
        self.assertFalse(diag["svcj_martingale_violated"])
        self.assertTrue(diag["svcj_contour_violated"])
        self.assertAlmostEqual(diag["svcj_martingale_margin"], 0.5)
        self.assertAlmostEqual(diag["svcj_contour_margin"], -0.25)


if __name__ == "__main__":
    unittest.main()
