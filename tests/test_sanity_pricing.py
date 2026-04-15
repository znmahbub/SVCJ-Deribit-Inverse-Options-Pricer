from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt

from tests._path import ROOT  # noqa: F401  (ensures src is importable)
from tests.reference_inputs import build_reference_case  # noqa: E402

from src.calibration import clear_fft_cache, price_dataframe  # noqa: E402
from src.inverse_fft_pricer import (  # noqa: E402
    FFTParams,
    black76_call_price,
    black76_put_price,
    cf_black,
    cf_heston,
    cf_svcj,
    heston_call_price,
    price_inverse_option,
)


class TestSanityPricing(unittest.TestCase):
    def test_characteristic_functions_martingale(self) -> None:
        F0 = 10_000.0
        T = 0.5

        # Black
        sigma = 0.6
        phi0 = cf_black(np.array([0.0 + 0.0j]), T, F0, sigma)[0]
        phiminusi = cf_black(np.array([-1j]), T, F0, sigma)[0]
        npt.assert_allclose(phi0, 1.0 + 0.0j, rtol=0.0, atol=0.0)
        npt.assert_allclose(phiminusi, F0 + 0.0j, rtol=1e-12, atol=1e-10)

        # Heston
        h = dict(kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.50, v0=0.04)
        phi0 = cf_heston(np.array([0.0 + 0.0j]), T, F0, **h)[0]
        phiminusi = cf_heston(np.array([-1j]), T, F0, **h)[0]
        npt.assert_allclose(phi0, 1.0 + 0.0j, rtol=0.0, atol=0.0)
        npt.assert_allclose(phiminusi, F0 + 0.0j, rtol=1e-8, atol=1e-6)

        # SVCJ
        s = dict(
            **h,
            lam=0.20,
            ell_y=-0.05,
            sigma_y=0.20,
            ell_v=0.10,
            rho_j=-0.30,
        )
        phi0 = cf_svcj(np.array([0.0 + 0.0j]), T, F0, **s)[0]
        phiminusi = cf_svcj(np.array([-1j]), T, F0, **s)[0]
        npt.assert_allclose(phi0, 1.0 + 0.0j, rtol=0.0, atol=0.0)
        # Quadrature makes this slightly approximate; keep a tight but realistic tolerance.
        npt.assert_allclose(phiminusi, F0 + 0.0j, rtol=1e-6, atol=1e-4)

    def test_black_fft_matches_black76_reasonably(self) -> None:
        F0 = 10_000.0
        T = 0.25
        sigma = 0.50
        K = np.array([8_000.0, 10_000.0, 12_000.0], dtype=float)

        N = 2**12
        eta = 0.10
        lam = 2.0 * np.pi / (float(N) * float(eta))
        b = float(np.log(np.median(K))) - 0.5 * float(N) * lam
        fftp = FFTParams(N=N, alpha=1.5, eta=eta, b=b, use_simpson=True)

        clear_fft_cache()
        call_coin = price_inverse_option("black", K=K, T=T, F0=F0, params={"sigma": sigma}, option_type="call", fft_params=fftp, use_cache=False)
        put_coin = price_inverse_option("black", K=K, T=T, F0=F0, params={"sigma": sigma}, option_type="put", fft_params=fftp, use_cache=False)

        call_usd = call_coin * F0
        put_usd = put_coin * F0

        call_ref = black76_call_price(F0, K, T, sigma)
        put_ref = black76_put_price(F0, K, T, sigma)

        # FFT introduces small discretization/interpolation error; keep the check tight but realistic.
        npt.assert_allclose(call_usd, call_ref, rtol=5e-4, atol=3e-1)
        npt.assert_allclose(put_usd, put_ref, rtol=5e-4, atol=3e-1)

    def test_heston_fft_matches_semi_analytic_reasonably(self) -> None:
        F0 = 10_000.0
        T = 0.5
        params = dict(kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.50, v0=0.04)
        K = np.array([8_000.0, 10_000.0, 12_000.0], dtype=float)

        N = 2**11
        eta = 0.10
        lam = 2.0 * np.pi / (float(N) * float(eta))
        b = float(np.log(np.median(K))) - 0.5 * float(N) * lam
        fftp = FFTParams(N=N, alpha=1.5, eta=eta, b=b, use_simpson=True)

        clear_fft_cache()
        call_coin = price_inverse_option("heston", K=K, T=T, F0=F0, params=params, option_type="call", fft_params=fftp, use_cache=False)
        call_usd_fft = call_coin * F0

        call_usd_ref = heston_call_price(
            F0,
            K,
            T,
            params["kappa"],
            params["theta"],
            params["sigma_v"],
            params["rho"],
            params["v0"],
            u_max=200.0,
            n=128,
        )

        # Two independent numerical methods; allow a modest tolerance.
        npt.assert_allclose(call_usd_fft, call_usd_ref, rtol=1e-2, atol=5e-1)

    def test_put_call_parity_coin_where_floor_inactive(self) -> None:
        F0 = 10_000.0
        T = 0.25
        K = np.array([12_000.0, 15_000.0], dtype=float)  # K > F0 => floor should not bind
        params = {"sigma": 0.50}

        N = 2**10
        eta = 0.10
        lam = 2.0 * np.pi / (float(N) * float(eta))
        b = float(np.log(np.median(K))) - 0.5 * float(N) * lam
        fftp = FFTParams(N=N, alpha=1.5, eta=eta, b=b, use_simpson=True)

        clear_fft_cache()
        c = price_inverse_option("black", K=K, T=T, F0=F0, params=params, option_type="call", fft_params=fftp, use_cache=False)
        p = price_inverse_option("black", K=K, T=T, F0=F0, params=params, option_type="put", fft_params=fftp, use_cache=False)

        lhs = c - p
        rhs = 1.0 - K / F0
        npt.assert_allclose(lhs, rhs, rtol=0.0, atol=1e-12)

    def test_price_dataframe_matches_manual_per_expiry(self) -> None:
        case = build_reference_case()

        clear_fft_cache()
        got = price_dataframe(
            case.df,
            model="black",
            params=case.black_params,
            fft_params_base=case.fft_params,
            dynamic_b=False,
            fft_params_by_expiry=case.fft_params_by_expiry,
            use_cache=False,
        )

        manual = np.empty(len(case.df), dtype=float)
        # This mirrors the logic in src.calibration._price_with_plan
        for exp, idx in case.df.groupby("expiry_datetime", sort=False).groups.items():
            block = case.df.loc[list(idx)]
            K = block["strike"].to_numpy(dtype=float)
            put_mask = (block["option_type"].astype(str).str.lower().to_numpy() == "put")
            T = float(np.median(block["time_to_maturity"].to_numpy(dtype=float)))
            F0 = float(np.median(block["F0"].to_numpy(dtype=float)))
            fftp = case.fft_params_by_expiry[exp]

            call_coin = price_inverse_option(
                "black",
                K=K,
                T=T,
                F0=F0,
                params=case.black_params,
                option_type="call",
                fft_params=fftp,
                use_cache=False,
            )
            manual[list(idx)] = call_coin
            if np.any(put_mask):
                K_put = K[put_mask]
                p_put = call_coin[put_mask] - (1.0 - K_put / F0)
                p_put = np.maximum(p_put, 0.0)
                manual[np.array(list(idx), dtype=int)[put_mask]] = p_put

        npt.assert_array_equal(got, manual)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
