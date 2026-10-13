from __future__ import annotations

from functools import partial
import unittest

import numpy as np
import numpy.testing as npt

from tests._path import ROOT  # noqa: F401
from src.inverse_fft_pricer import _leggauss_cached, _norm_cdf, cf_black, cf_heston, cf_svcj
from src.greeks import (
    black_regular_delta,
    call_p1_from_logprice_cf,
    net_delta,
    regular_delta_from_call_p1,
)


class TestMeasureDeltaIdentities(unittest.TestCase):
    def test_black_closed_form_measure_and_delta_identities(self) -> None:
        F0 = 10_000.0
        K = np.array([8_000.0, 10_000.0, 12_000.0])
        T = 0.35
        sigma = 0.55

        phi_minus_i = cf_black(np.array([-1j]), T, F0, sigma)[0]
        npt.assert_allclose(phi_minus_i, F0, rtol=1e-12, atol=1e-10)

        vol = sigma * np.sqrt(T)
        d1 = (np.log(F0 / K) + 0.5 * vol * vol) / vol
        d2 = d1 - vol

        # Direct Q probability Q(F_T > K) from phi_Q must reduce to N(d2).
        nodes, weights = _leggauss_cached(256)
        u = 100.0 * (nodes + 1.0)
        w = 100.0 * weights
        phi_u = cf_black(u, T, F0, sigma)
        direct = 0.5 + (1.0 / np.pi) * np.sum(
            w[:, None]
            * np.real(
                np.exp(-1j * u[:, None] * np.log(K)[None, :])
                * phi_u[:, None]
                / ((1j * u)[:, None])
            ),
            axis=0,
        )
        npt.assert_allclose(direct, _norm_cdf(d2), rtol=2e-5, atol=2e-5)

        # Shifted characteristic function is the Q^F probability and equals N(d1).
        cf = partial(cf_black, T=T, F0=F0, sigma=sigma)
        p1 = call_p1_from_logprice_cf(K, F0, cf, quad_u_max=200.0, quad_n=256)
        npt.assert_allclose(p1, _norm_cdf(d1), rtol=2e-5, atol=2e-5)

        call_delta = black_regular_delta(F0, K, T, sigma, "call")
        put_delta = black_regular_delta(F0, K, T, sigma, "put")
        npt.assert_allclose(call_delta, _norm_cdf(d1), rtol=2e-7, atol=2e-7)
        npt.assert_allclose(put_delta, _norm_cdf(d1) - 1.0, rtol=2e-7, atol=2e-7)

        inverse_prices = np.array([0.25, 0.10, 0.04])
        npt.assert_array_equal(net_delta(call_delta, inverse_prices), call_delta - inverse_prices)

    def test_heston_and_svcj_shifted_cf_sanity(self) -> None:
        F0 = 10_000.0
        T = 0.50
        K = np.array([8_000.0, 10_000.0, 12_000.0])
        heston = dict(kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.50, v0=0.04)
        svcj = dict(**heston, lam=0.20, ell_y=-0.05, sigma_y=0.20, ell_v=0.10, rho_j=-0.30)

        for func, params in ((cf_heston, heston), (cf_svcj, svcj)):
            npt.assert_allclose(
                func(np.array([-1j]), T, F0, **params)[0],
                F0,
                rtol=1e-6,
                atol=1e-4,
            )
            cf = partial(func, T=T, F0=F0, **params)
            p1 = call_p1_from_logprice_cf(K, F0, cf, quad_u_max=160.0, quad_n=128)
            self.assertTrue(np.all(np.isfinite(p1)))
            self.assertTrue(np.all((p1 >= 0.0) & (p1 <= 1.0)))
            put_delta = regular_delta_from_call_p1(p1, np.array(["put"] * 3))
            npt.assert_array_equal(put_delta, p1 - 1.0)


if __name__ == "__main__":
    unittest.main()
