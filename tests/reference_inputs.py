from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tests._path import ROOT

# Import after ROOT is on sys.path
from src.inverse_fft_pricer import FFTParams  # noqa: E402


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REFERENCE_NPZ_PATH = FIXTURES_DIR / "reference_outputs.npz"


@dataclass(frozen=True)
class ReferenceCase:
    """Deterministic reference case used for regression tests."""

    F0: float
    T: float
    K: np.ndarray
    fft_params: FFTParams
    black_params: dict[str, float]
    heston_params: dict[str, float]
    svcj_params: dict[str, float]

    df: pd.DataFrame
    fft_params_by_expiry: dict


def _center_b_for_strikes(*, N: int, eta: float, strikes: np.ndarray) -> float:
    lam = 2.0 * np.pi / (float(N) * float(eta))
    logK_center = float(np.log(np.median(np.asarray(strikes, dtype=float))))
    return logK_center - 0.5 * float(N) * lam


def build_reference_case() -> ReferenceCase:
    # --- Base scalar inputs
    F0 = 10_000.0
    T = 0.25
    K = np.array([5_000.0, 8_000.0, 10_000.0, 12_000.0, 15_000.0], dtype=float)

    # --- FFT params (small N for fast but representative pricing)
    N = 2**10
    eta = 0.10
    alpha = 1.5
    b = _center_b_for_strikes(N=N, eta=eta, strikes=K)
    fft_params = FFTParams(N=N, alpha=alpha, eta=eta, b=b, use_simpson=True)

    # --- Model params (chosen to be stable / within typical domains)
    black_params = {"sigma": 0.50}

    heston_params = {
        "kappa": 2.0,
        "theta": 0.04,
        "sigma_v": 0.30,
        "rho": -0.50,
        "v0": 0.04,
    }

    svcj_params = {
        **heston_params,
        "lam": 0.20,
        "ell_y": -0.05,
        "sigma_y": 0.20,
        "ell_v": 0.10,
        "rho_j": -0.30,
    }

    # --- DataFrame case for vectorized pricing (2 expiries, mixed calls/puts)
    exp1 = "2026-06-01T00:00:00Z"
    exp2 = "2026-07-01T00:00:00Z"

    strikes_1 = np.array([8_000.0, 10_000.0, 12_000.0], dtype=float)
    strikes_2 = np.array([9_000.0, 10_500.0, 13_000.0], dtype=float)

    df = pd.DataFrame(
        {
            "expiry_datetime": [exp1] * 6 + [exp2] * 6,
            "option_type": ["call"] * 3 + ["put"] * 3 + ["call"] * 3 + ["put"] * 3,
            "strike": np.concatenate([strikes_1, strikes_1, strikes_2, strikes_2]),
            "time_to_maturity": [0.10] * 6 + [0.20] * 6,
            "F0": [F0] * 6 + [10_500.0] * 6,
        }
    )

    # Precompute stable per-expiry FFTParams with the same centering rule used in the pipeline.
    fft_params_by_expiry = {
        exp1: FFTParams(
            N=N,
            alpha=alpha,
            eta=eta,
            b=_center_b_for_strikes(N=N, eta=eta, strikes=strikes_1),
            use_simpson=True,
        ),
        exp2: FFTParams(
            N=N,
            alpha=alpha,
            eta=eta,
            b=_center_b_for_strikes(N=N, eta=eta, strikes=strikes_2),
            use_simpson=True,
        ),
    }

    return ReferenceCase(
        F0=F0,
        T=T,
        K=K,
        fft_params=fft_params,
        black_params=black_params,
        heston_params=heston_params,
        svcj_params=svcj_params,
        df=df,
        fft_params_by_expiry=fft_params_by_expiry,
    )

