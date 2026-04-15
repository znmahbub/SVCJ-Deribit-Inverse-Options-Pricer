#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.reference_inputs import FIXTURES_DIR, REFERENCE_NPZ_PATH, build_reference_case  # noqa: E402

from src.calibration import clear_fft_cache, price_dataframe  # noqa: E402
from src.inverse_fft_pricer import (  # noqa: E402
    carr_madan_call_fft,
    cf_black,
    price_inverse_option,
)


def main() -> int:
    case = build_reference_case()

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    clear_fft_cache()

    # --- Scalar/array pricing regression: price_inverse_option
    out = {}
    for model, params in [
        ("black", case.black_params),
        ("heston", case.heston_params),
        ("svcj", case.svcj_params),
    ]:
        out[f"{model}_call_coin"] = price_inverse_option(
            model=model,
            K=case.K,
            T=case.T,
            F0=case.F0,
            params=params,
            option_type="call",
            fft_params=case.fft_params,
            use_cache=False,
        )
        out[f"{model}_put_coin"] = price_inverse_option(
            model=model,
            K=case.K,
            T=case.T,
            F0=case.F0,
            params=params,
            option_type="put",
            fft_params=case.fft_params,
            use_cache=False,
        )

    # --- Vectorized pricing regression: price_dataframe
    for model, params in [
        ("black", case.black_params),
        ("heston", case.heston_params),
        ("svcj", case.svcj_params),
    ]:
        out[f"df_{model}_coin"] = price_dataframe(
            case.df,
            model=model,
            params=params,
            fft_params_base=case.fft_params,
            dynamic_b=False,
            fft_params_by_expiry=case.fft_params_by_expiry,
            use_cache=False,
        )

    # --- Optional: verify the raw FFT grid for Black is unchanged
    K_grid, C_grid = carr_madan_call_fft(
        cf_black,
        case.T,
        case.F0,
        (case.black_params["sigma"],),
        case.fft_params,
    )
    out["black_fft_K_grid"] = K_grid
    out["black_fft_C_grid_usd"] = C_grid

    np.savez(REFERENCE_NPZ_PATH, **out)
    print(f"Wrote reference fixture: {REFERENCE_NPZ_PATH} ({len(out)} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
