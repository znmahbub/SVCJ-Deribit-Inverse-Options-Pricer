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

    def test_reference_outputs_match_exactly(self) -> None:
        case = build_reference_case()
        ref = np.load(REFERENCE_NPZ_PATH, allow_pickle=False)

        clear_fft_cache()

        # price_inverse_option outputs
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
            npt.assert_array_equal(got_call, ref[f"{model}_call_coin"])
            npt.assert_array_equal(got_put, ref[f"{model}_put_coin"])

        # price_dataframe outputs
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
            npt.assert_array_equal(got_df, ref[f"df_{model}_coin"])

        # Raw black FFT grid outputs
        K_grid, C_grid = carr_madan_call_fft(
            cf_black,
            case.T,
            case.F0,
            (case.black_params["sigma"],),
            case.fft_params,
        )
        npt.assert_array_equal(K_grid, ref["black_fft_K_grid"])
        npt.assert_array_equal(C_grid, ref["black_fft_C_grid_usd"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

