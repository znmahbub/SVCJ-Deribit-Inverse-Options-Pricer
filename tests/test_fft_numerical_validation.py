from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt
from scipy.stats import norm

from tests._path import ROOT  # noqa: F401  (ensures src is importable)
from src.inverse_fft_pricer import (  # noqa: E402
    FFTParams,
    center_fft_params_on_strikes,
    clear_fft_cache,
    heston_call_price,
    price_inverse_option,
)


CANONICAL_FFT = FFTParams(N=4096, eta=0.10, alpha=1.5, use_simpson=True)


def _black76_call_independent(F0: float, K: np.ndarray, T: float, sigma: float) -> np.ndarray:
    """Independent Black-76 reference using SciPy's normal CDF."""
    K = np.asarray(K, dtype=float)
    vol = float(sigma) * np.sqrt(float(T))
    d1 = (np.log(float(F0) / K) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    return float(F0) * norm.cdf(d1) - K * norm.cdf(d2)


def _canonical_params(strikes: np.ndarray) -> FFTParams:
    return center_fft_params_on_strikes(CANONICAL_FFT, np.asarray(strikes, dtype=float))


class TestCanonicalFftNumerics(unittest.TestCase):
    def test_black_canonical_grid_matches_independent_black76(self) -> None:
        F0 = 10_000.0
        sigma = 0.50

        # Medium maturity deliberately contains off-grid ITM/OTM strikes so this
        # test covers both FFT quadrature and interpolation at the production grid.
        cases = [
            (0.01, np.array([10_000.0])),
            (0.25, np.array([8_000.0, 10_000.0, 12_000.0])),
            (1.00, np.array([10_000.0])),
        ]

        for T, strikes in cases:
            with self.subTest(T=T, strikes=strikes.tolist()):
                fft_params = _canonical_params(strikes)
                clear_fft_cache()
                got_coin = price_inverse_option(
                    "black",
                    K=strikes,
                    T=T,
                    F0=F0,
                    params={"sigma": sigma},
                    option_type="call",
                    fft_params=fft_params,
                    use_cache=False,
                )
                got_usd = np.asarray(got_coin, dtype=float) * F0
                ref_usd = _black76_call_independent(F0, strikes, T, sigma)

                # At N=4096, eta=0.10 the dominant off-grid error is strike-grid
                # interpolation. Observed canonical error is below $0.28 here.
                npt.assert_allclose(got_usd, ref_usd, rtol=5e-4, atol=0.35)

    def test_heston_canonical_grid_matches_converged_semi_analytic_reference(self) -> None:
        F0 = 10_000.0
        T = 0.50
        strikes = np.array([8_000.0, 10_000.0, 12_000.0])
        params = dict(kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.50, v0=0.04)

        # First establish that the independent Gil-Pelaez reference itself is
        # converged well inside the FFT acceptance tolerance.
        ref_256 = np.asarray(
            heston_call_price(F0, strikes, T, **params, u_max=300.0, n=256),
            dtype=float,
        )
        ref_512 = np.asarray(
            heston_call_price(F0, strikes, T, **params, u_max=300.0, n=512),
            dtype=float,
        )
        npt.assert_allclose(ref_512, ref_256, rtol=1e-8, atol=1e-6)

        fft_params = _canonical_params(strikes)
        clear_fft_cache()
        got_coin = price_inverse_option(
            "heston",
            K=strikes,
            T=T,
            F0=F0,
            params=params,
            option_type="call",
            fft_params=fft_params,
            use_cache=False,
        )
        got_usd = np.asarray(got_coin, dtype=float) * F0

        # The canonical grid's largest absolute error on this benchmark is about
        # $0.18; the OTM option has the largest relative error because its price
        # is only about $42.5. These bounds are intentionally above that measured
        # error while remaining much tighter than the old generic Heston sanity test.
        npt.assert_allclose(got_usd, ref_512, rtol=5e-3, atol=0.25)


if __name__ == "__main__":
    unittest.main()
