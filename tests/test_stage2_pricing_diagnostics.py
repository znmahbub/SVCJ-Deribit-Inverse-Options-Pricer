from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests._path import ROOT  # noqa: F401
from notebooks._lib import chapter3_final


class Stage2PricingDiagnosticTests(unittest.TestCase):
    def _svcj_params(self) -> pd.DataFrame:
        rows = []
        for i, rho_j in enumerate([-1.4, 1.2, -0.7]):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i),
                    "currency": "BTC",
                    "kappa": 2.0 + 0.1 * i,
                    "theta": 0.04,
                    "sigma_v": 0.30,
                    "rho": -0.5,
                    "v0": 0.04,
                    "lam": 0.2,
                    "ell_y": -0.05,
                    "sigma_y": 0.20,
                    "ell_v": 0.10,
                    "rho_j": rho_j,
                }
            )
        return pd.DataFrame(rows)

    def test_parameter_summary_does_not_report_a_rhoj_boundary_share(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = chapter3_final.compute_parameter_summary(self._svcj_params(), Path(td))
        self.assertNotIn("rhoj_boundary_share", out.columns)

    def test_parameter_plot_only_draws_correlation_bounds_for_rho(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fig = chapter3_final.plot_parameter_paths(
                self._svcj_params(),
                currency="BTC",
                stem="params",
                title="",
                output_dir=Path(td),
            )
        self.assertEqual(len(fig.layout.shapes or []), 2)


if __name__ == "__main__":
    unittest.main()
