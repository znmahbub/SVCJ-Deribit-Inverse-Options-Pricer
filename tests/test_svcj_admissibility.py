from __future__ import annotations

import unittest

import numpy as np

from tests._path import ROOT  # noqa: F401
from src.calibration import _default_bounds, _pack_svcj, _unpack_svcj


BASE = dict(
    kappa=2.0,
    theta=0.04,
    sigma_v=0.30,
    rho=-0.50,
    v0=0.04,
    lam=0.20,
    ell_y=-0.05,
    sigma_y=0.20,
    ell_v=0.10,
    rho_j=0.0,
)


class TestSVCJAdmissibility(unittest.TestCase):
    def test_pack_unpack_preserves_unbounded_jump_loading(self) -> None:
        for rho_j in (-2.0, 0.0, 1.25):
            params = {**BASE, "rho_j": rho_j}
            got = _unpack_svcj(_pack_svcj(params))
            self.assertAlmostEqual(got["rho_j"], rho_j, places=12)

    def test_default_bounds_do_not_bound_jump_loading(self) -> None:
        lb, ub = _default_bounds("svcj")
        self.assertTrue(np.isneginf(lb[9]))
        self.assertTrue(np.isposinf(ub[9]))

    def test_contour_counterexample_is_rejected_before_pricing(self) -> None:
        from src.inverse_fft_pricer import (
            FFTParams,
            SVCJAdmissibilityError,
            price_inverse_option,
            svcj_carr_madan_admissible,
            svcj_martingale_moment_ok,
        )

        # First moment exists: 1 - ell_v*rho_j = 0.5 > 0.
        # Carr-Madan q=alpha+1=2.5 fails already at s=0:
        # 1 - q*ell_v*rho_j = -0.25.
        ell_v, rho_j, alpha = 0.50, 1.0, 1.5
        self.assertTrue(svcj_martingale_moment_ok(ell_v, rho_j, 1e-12))
        self.assertFalse(
            svcj_carr_madan_admissible(
                kappa=2.0,
                sigma_v=0.30,
                rho=-0.50,
                ell_v=ell_v,
                rho_j=rho_j,
                alpha=alpha,
                T=0.25,
                N=256,
                eta=0.10,
                quad_nodes=24,
                eps=1e-12,
            )
        )

        bad = {**BASE, "ell_v": ell_v, "rho_j": rho_j}
        fftp = FFTParams(N=256, alpha=alpha, eta=0.10, b=-10.0, use_simpson=True)
        with self.assertRaises(SVCJAdmissibilityError):
            price_inverse_option(
                "svcj",
                K=np.array([10_000.0]),
                T=0.25,
                F0=10_000.0,
                params=bad,
                option_type="call",
                fft_params=fftp,
                use_cache=False,
            )

    def test_materially_negative_loading_is_not_clipped_and_cf_is_finite(self) -> None:
        from src.inverse_fft_pricer import cf_svcj

        params = {**BASE, "rho_j": -2.0}
        self.assertEqual(_unpack_svcj(_pack_svcj(params))["rho_j"], -2.0)
        phi = cf_svcj(
            np.array([0.5 + 0j, 1.0 + 0j, -1j]),
            0.25,
            10_000.0,
            **params,
        )
        self.assertTrue(np.all(np.isfinite(phi)))


if __name__ == "__main__":
    unittest.main()
