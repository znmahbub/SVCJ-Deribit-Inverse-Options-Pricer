from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src import calibration
from src.calibration import WeightConfig, calibrate_model
from src.snapshot_job import process_snapshot_to_payload


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mid_price_clean": [0.1, 0.12],
            "spread": [0.01, 0.02],
            "vega": [0.5, 0.6],
            "open_interest": [10.0, 15.0],
            "expiry_datetime": ["2026-01-31", "2026-01-31"],
            "option_type": ["call", "put"],
            "strike": [100.0, 110.0],
            "time_to_maturity": [0.1, 0.1],
            "F0": [105.0, 105.0],
        }
    )


def _stub_least_squares_capture_len(captured: dict):
    def _impl(fun, x0, **kwargs):
        captured["residual_len"] = len(fun(np.array(x0, dtype=float)))
        return SimpleNamespace(x=np.array(x0, dtype=float), success=True, message="ok", nfev=1)

    return _impl


def _stub_price_with_plan(*args, **kwargs):
    out_prices = kwargs["out_prices"]
    out_prices[:] = 0.0


def test_l2_prev_strength_zero_is_baseline(monkeypatch, tiny_df):
    captured = {}
    monkeypatch.setattr(calibration, "least_squares", _stub_least_squares_capture_len(captured))
    monkeypatch.setattr(calibration, "_price_with_plan", _stub_price_with_plan)

    calibrate_model(tiny_df, "black", weight_config=WeightConfig(use_vega=False, use_open_interest=False), verbose=0)

    assert captured["residual_len"] == len(tiny_df)


def test_l2_prev_strength_positive_with_previous_params_adds_regularization(monkeypatch, tiny_df):
    captured = {}
    monkeypatch.setattr(calibration, "least_squares", _stub_least_squares_capture_len(captured))
    monkeypatch.setattr(calibration, "_price_with_plan", _stub_price_with_plan)

    calibrate_model(
        tiny_df,
        "black",
        weight_config=WeightConfig(use_vega=False, use_open_interest=False),
        l2_prev_strength=1.0,
        previous_params={"sigma": 0.4},
        verbose=0,
    )

    assert captured["residual_len"] == len(tiny_df) + 1


def test_l2_prev_strength_positive_without_previous_params_adds_no_regularization(monkeypatch, tiny_df):
    captured = {}
    monkeypatch.setattr(calibration, "least_squares", _stub_least_squares_capture_len(captured))
    monkeypatch.setattr(calibration, "_price_with_plan", _stub_price_with_plan)

    calibrate_model(
        tiny_df,
        "black",
        weight_config=WeightConfig(use_vega=False, use_open_interest=False),
        l2_prev_strength=1.0,
        previous_params=None,
        verbose=0,
    )

    assert captured["residual_len"] == len(tiny_df)


def test_negative_l2_prev_strength_raises_value_error(tiny_df):
    with pytest.raises(ValueError, match="non-negative"):
        calibrate_model(tiny_df, "black", l2_prev_strength=-0.1, verbose=0)


def test_snapshot_uses_warm_start_params_as_regularization_anchor(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "deribit_options_snapshot_20260101T000000Z.csv"
    df = pd.DataFrame(
        {
            "currency": ["BTC", "BTC", "BTC"],
            "option_type": ["call", "put", "call"],
            "strike": [100.0, 110.0, 120.0],
            "time_to_maturity": [0.1, 0.1, 0.1],
            "bid_price": [0.1, 0.1, 0.1],
            "ask_price": [0.11, 0.11, 0.11],
            "futures_price": [105.0, 105.0, 105.0],
            "vega": [0.5, 0.5, 0.5],
            "open_interest": [5.0, 5.0, 5.0],
            "expiry_datetime": ["2026-01-31", "2026-01-31", "2026-01-31"],
        }
    )
    df.to_csv(csv_path, index=False)

    calls = []

    def _stub_calibrate_model(train, model, **kwargs):
        calls.append((model, kwargs.get("previous_params"), kwargs.get("l2_prev_strength")))
        params = {"sigma": 0.6} if model == "black" else {"kappa": 2.0, "theta": 0.2, "sigma_v": 0.4, "rho": -0.3, "v0": 0.2}
        if model == "svcj":
            params.update({"lam": 0.1, "ell_y": -0.05, "sigma_y": 0.2, "ell_v": 0.1, "rho_j": 0.0})
        return SimpleNamespace(success=True, message="ok", nfev=1, rmse=0.0, mae=0.0, params=params)

    monkeypatch.setattr("src.snapshot_job.calibrate_model", _stub_calibrate_model)
    monkeypatch.setattr("src.snapshot_job.price_dataframe", lambda *args, **kwargs: np.zeros(len(args[0]), dtype=float))

    warm = {
        "black": {"sigma": 0.5},
        "heston": {"kappa": 1.0, "theta": 0.1, "sigma_v": 0.2, "rho": -0.2, "v0": 0.1},
        "svcj": {"kappa": 1.0, "theta": 0.1, "sigma_v": 0.2, "rho": -0.2, "v0": 0.1, "lam": 0.1, "ell_y": -0.05, "sigma_y": 0.2, "ell_v": 0.1, "rho_j": 0.0},
    }

    process_snapshot_to_payload(
        csv_path,
        currency="BTC",
        filter_rules=dict(min_open_interest=1.0, min_vega=0.0, min_time_to_maturity=0.0),
        weight_config=WeightConfig(),
        fft_base=calibration.FFTParams(),
        max_nfev={"black": 1, "heston": 1, "svcj": 1},
        train_frac=0.67,
        random_seed=0,
        runtime_top_expiries_by_oi=None,
        runtime_max_options=None,
        min_options_after_filter=1,
        warm_start=warm,
        l2_prev_strength=3.0,
        verbose=False,
    )

    assert calls == [
        ("black", warm["black"], 3.0),
        ("heston", warm["heston"], 3.0),
        ("svcj", warm["svcj"], 3.0),
    ]
