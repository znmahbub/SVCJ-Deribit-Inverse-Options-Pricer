from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt

from tests._path import ROOT  # noqa: F401
from src.hedge_mapping import hedge_notional_usd, inverse_perp_pnl


class TestHedgeMapping(unittest.TestCase):
    def test_equal_term_and_perp_price_reduces_to_H_delta_for_short_option(self) -> None:
        H = np.array([10_000.0, 12_000.0])
        delta = np.array([0.4, -0.3])
        npt.assert_allclose(hedge_notional_usd(delta, H, H), H * delta, rtol=0.0, atol=0.0)

    def test_nonzero_basis_uses_H_squared_over_F(self) -> None:
        F = np.array([10_000.0, 10_000.0])
        H = np.array([9_800.0, 10_300.0])
        delta = np.array([0.4, -0.3])
        npt.assert_allclose(
            hedge_notional_usd(delta, F, H),
            (H ** 2 / F) * delta,
            rtol=0.0,
            atol=0.0,
        )

    def test_option_position_controls_sign(self) -> None:
        F, H, delta = 10_000.0, 9_800.0, 0.5
        short = hedge_notional_usd(delta, F, H, option_position=-1.0)
        long = hedge_notional_usd(delta, F, H, option_position=1.0)
        self.assertGreater(float(short), 0.0)
        self.assertLess(float(long), 0.0)
        self.assertEqual(float(long), -float(short))

    def test_regular_and_net_rules_use_identical_basis_scale(self) -> None:
        F = np.array([10_000.0, 10_500.0])
        H = np.array([9_700.0, 10_800.0])
        scale = hedge_notional_usd(np.ones(2), F, H)
        npt.assert_allclose(scale, H ** 2 / F, rtol=0.0, atol=0.0)

    def test_local_basis_derivative_scales_notional(self) -> None:
        F, H, delta = 10_000.0, 9_800.0, 0.5
        npt.assert_allclose(
            hedge_notional_usd(delta, F, H, dF_dH=0.8),
            0.8 * (H ** 2 / F) * delta,
        )

    def test_inverse_perpetual_pnl_definition(self) -> None:
        N = np.array([5_000.0, -2_500.0])
        H0 = np.array([10_000.0, 2_000.0])
        H1 = np.array([10_100.0, 1_950.0])
        npt.assert_allclose(
            inverse_perp_pnl(N, H0, H1),
            N * (1.0 / H0 - 1.0 / H1),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
