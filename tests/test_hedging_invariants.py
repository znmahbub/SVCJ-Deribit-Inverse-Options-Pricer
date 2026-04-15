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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

