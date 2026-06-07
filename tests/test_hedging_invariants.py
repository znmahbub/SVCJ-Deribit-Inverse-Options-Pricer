from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd

from tests._path import ROOT  # noqa: F401  (ensures src is importable)

from src.hedging_analysis import compute_prices_and_deltas  # noqa: E402


class TestHedgingInvariants(unittest.TestCase):
    def test_net_delta_identity_holds_exactly(self) -> None:
        block = pd.DataFrame(
            {
                "expiry_datetime": ["2026-06-01T00:00:00Z"] * 4,
                "option_type": ["call", "put", "call", "put"],
                "strike": [9_000.0, 9_000.0, 12_000.0, 12_000.0],
                "time_to_maturity": [0.25, 0.25, 0.25, 0.0],  # include one expired row
                "F0": [10_000.0, 10_000.0, 10_000.0, 10_000.0],
            }
        )

        params = {
            "black": {"sigma": 0.5},
            "heston": {"kappa": 2.0, "theta": 0.04, "sigma_v": 0.30, "rho": -0.50, "v0": 0.04},
            "svcj": {
                "kappa": 2.0,
                "theta": 0.04,
                "sigma_v": 0.30,
                "rho": -0.50,
                "v0": 0.04,
                "lam": 0.20,
                "ell_y": -0.05,
                "sigma_y": 0.20,
                "ell_v": 0.10,
                "rho_j": -0.30,
            },
        }

        from src.inverse_fft_pricer import FFTParams  # local import to keep test import-time light

        fft_params = FFTParams(N=2**10, alpha=1.5, eta=0.10, b=-10.0, use_simpson=True)

        for model in ["black", "heston", "svcj"]:
            prices, reg_delta, net_delta = compute_prices_and_deltas(
                model=model,
                block=block,
                params=params[model],
                fft_params=fft_params,
                dynamic_b=True,
                delta_u_max=200.0,
                delta_quad_n=64,
            )
            npt.assert_array_equal(net_delta, reg_delta - prices)


    def test_net_delta_identity_svcj_all_moneyness(self) -> None:
        """Δ_net = Δ_reg − V holds for SVCJ across a wide strike range and maturities."""
        block = pd.DataFrame(
            {
                "expiry_datetime": ["2026-06-01T00:00:00Z"] * 8,
                "option_type": [
                    "call", "call", "call", "call",
                    "put",  "put",  "put",  "put",
                ],
                "strike": [5_000.0, 8_000.0, 10_000.0, 15_000.0,
                           5_000.0, 8_000.0, 10_000.0, 15_000.0],
                "time_to_maturity": [0.10, 0.25, 0.50, 0.25,
                                     0.10, 0.25, 0.50, 0.25],
                "F0": [10_000.0] * 8,
            }
        )

        svcj_params = {
            "kappa": 3.0, "theta": 0.09, "sigma_v": 0.50, "rho": -0.70, "v0": 0.09,
            "lam": 0.50, "ell_y": -0.10, "sigma_y": 0.30, "ell_v": 0.15, "rho_j": -0.50,
        }

        from src.inverse_fft_pricer import FFTParams

        fft_params = FFTParams(N=2**11, alpha=1.5, eta=0.10, b=-10.0, use_simpson=True)

        prices, reg_delta, net_delta = compute_prices_and_deltas(
            model="svcj",
            block=block,
            params=svcj_params,
            fft_params=fft_params,
            dynamic_b=True,
            delta_u_max=200.0,
            delta_quad_n=64,
        )
        npt.assert_array_equal(net_delta, reg_delta - prices)

    def test_net_delta_identity_heston_deep_itm_otm(self) -> None:
        """Δ_net = Δ_reg − V holds for Heston even for deep ITM/OTM options."""
        block = pd.DataFrame(
            {
                "expiry_datetime": ["2026-09-01T00:00:00Z"] * 4,
                "option_type": ["call", "call", "put", "put"],
                "strike": [4_000.0, 20_000.0, 4_000.0, 20_000.0],
                "time_to_maturity": [0.5, 0.5, 0.5, 0.5],
                "F0": [10_000.0] * 4,
            }
        )

        heston_params = {
            "kappa": 2.0, "theta": 0.04, "sigma_v": 0.30, "rho": -0.50, "v0": 0.04,
        }

        from src.inverse_fft_pricer import FFTParams

        fft_params = FFTParams(N=2**12, alpha=1.5, eta=0.10, b=-12.0, use_simpson=True)

        prices, reg_delta, net_delta = compute_prices_and_deltas(
            model="heston",
            block=block,
            params=heston_params,
            fft_params=fft_params,
            dynamic_b=True,
            delta_u_max=200.0,
            delta_quad_n=128,
        )
        npt.assert_array_equal(net_delta, reg_delta - prices)

    def test_net_delta_sign_convention(self) -> None:
        """Regular delta is in (0, 1) for calls and (−1, 0) for puts (Black model)."""
        block = pd.DataFrame(
            {
                "expiry_datetime": ["2026-06-01T00:00:00Z"] * 4,
                "option_type": ["call", "call", "put", "put"],
                "strike": [9_000.0, 11_000.0, 9_000.0, 11_000.0],
                "time_to_maturity": [0.25, 0.25, 0.25, 0.25],
                "F0": [10_000.0] * 4,
            }
        )

        from src.inverse_fft_pricer import FFTParams

        fft_params = FFTParams(N=2**12, alpha=1.5, eta=0.10, b=-10.0, use_simpson=True)

        _, reg_delta, _ = compute_prices_and_deltas(
            model="black",
            block=block,
            params={"sigma": 0.50},
            fft_params=fft_params,
            dynamic_b=True,
        )
        call_deltas = reg_delta[:2]
        put_deltas  = reg_delta[2:]

        for d in call_deltas:
            if np.isfinite(d):
                self.assertGreater(d, 0.0, f"Call delta should be positive, got {d}")
                self.assertLessEqual(d, 1.0 + 1e-6, f"Call delta should be ≤ 1, got {d}")
        for d in put_deltas:
            if np.isfinite(d):
                self.assertLess(d, 0.0, f"Put delta should be negative, got {d}")
                self.assertGreaterEqual(d, -1.0 - 1e-6, f"Put delta should be ≥ −1, got {d}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

