from __future__ import annotations

import unittest

import numpy as np
import numpy.testing as npt

from tests.reference_inputs import REFERENCE_NPZ_PATH, build_reference_case

from tests._path import ROOT  # noqa: F401  (ensures src is importable)

from src.calibration import clear_fft_cache, price_dataframe  # noqa: E402
from src.inverse_fft_pricer import (  # noqa: E402
    carr_madan_call_fft,
    cf_black,
    price_inverse_option,
)


class TestReferenceOutputsRegression(unittest.TestCase):
    def setUp(self) -> None:
        if not REFERENCE_NPZ_PATH.exists():
            self.fail(
                f"Missing reference fixture: {REFERENCE_NPZ_PATH}. "
                "Generate it via: python tests/generate_reference_outputs.py"
            )

    @staticmethod
    def _assert_reference_price_array(actual: np.ndarray, expected: np.ndarray) -> None:
        """Require user-facing prices to match at near-machine precision."""
        npt.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)

    def test_reference_outputs_match_machine_precision(self) -> None:
        case = build_reference_case()
        ref = np.load(REFERENCE_NPZ_PATH, allow_pickle=False)

        clear_fft_cache()

        # User-facing inverse-option prices: retain a near-machine-precision
        # regression guard across supported NumPy/pocketfft builds.
        for model, params in [
            ("black", case.black_params),
            ("heston", case.heston_params),
            ("svcj", case.svcj_params),
        ]:
            got_call = price_inverse_option(
                model=model,
                K=case.K,
                T=case.T,
                F0=case.F0,
                params=params,
                option_type="call",
                fft_params=case.fft_params,
                use_cache=False,
            )
            got_put = price_inverse_option(
                model=model,
                K=case.K,
                T=case.T,
                F0=case.F0,
                params=params,
                option_type="put",
                fft_params=case.fft_params,
                use_cache=False,
            )
            self._assert_reference_price_array(got_call, ref[f"{model}_call_coin"])
            self._assert_reference_price_array(got_put, ref[f"{model}_put_coin"])

        # price_dataframe outputs use the same strict pricing tolerance.
        for model, params in [
            ("black", case.black_params),
            ("heston", case.heston_params),
            ("svcj", case.svcj_params),
        ]:
            got_df = price_dataframe(
                case.df,
                model=model,
                params=params,
                fft_params_base=case.fft_params,
                dynamic_b=False,
                fft_params_by_expiry=case.fft_params_by_expiry,
                use_cache=False,
            )
            self._assert_reference_price_array(got_df, ref[f"df_{model}_coin"])

        # The full raw Carr-Madan grid extends many orders of magnitude away
        # from the economically relevant strike region. At those extreme grid
        # points the inverse FFT is dominated by cancellation and last-bit
        # pocketfft differences can be magnified enormously. Preserve a raw-grid
        # regression check on the economically meaningful region instead; the
        # user-facing interpolated prices above remain guarded much more tightly.
        K_grid, C_grid = carr_madan_call_fft(
            cf_black,
            case.T,
            case.F0,
            (case.black_params["sigma"],),
            case.fft_params,
        )
        ref_K = ref["black_fft_K_grid"]
        ref_C = ref["black_fft_C_grid_usd"]
        npt.assert_allclose(K_grid, ref_K, rtol=1e-14, atol=1e-15)

        economic = (ref_K >= 0.25 * case.F0) & (ref_K <= 4.0 * case.F0)
        self.assertGreater(int(np.sum(economic)), 10)
        npt.assert_allclose(
            C_grid[economic],
            ref_C[economic],
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
