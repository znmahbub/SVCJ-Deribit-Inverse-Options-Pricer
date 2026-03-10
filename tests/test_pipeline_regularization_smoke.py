from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import WeightConfig
from src.inverse_fft_pricer import FFTParams
from src.snapshot_job import process_snapshot_to_payload


def _distance_to_anchor(row: pd.Series, anchor: dict[str, float]) -> float:
    keys = [k for k in anchor.keys() if k in row.index and pd.notna(row[k])]
    if not keys:
        return float("nan")
    return float(np.sqrt(sum((float(row[k]) - float(anchor[k])) ** 2 for k in keys)))


def test_pipeline_small_subset_regularization_pulls_params_toward_anchor() -> None:
    """Smoke test the snapshot pipeline on a tiny subset (~5 options).

    We run two consecutive snapshots:
      1) first snapshot to produce warm-start parameters,
      2) second snapshot twice (no-regularization vs strong regularization),
    and verify that strong regularization keeps fitted params closer to anchor.
    """

    first = Path("data/deribit_options_snapshot_20260101T091835Z.csv")
    second = Path("data/deribit_options_snapshot_20260101T211918Z.csv")
    assert first.exists() and second.exists()

    common = dict(
        currency="BTC",
        filter_rules=dict(
            require_bid_ask=True,
            min_time_to_maturity=1 / 365,
            max_time_to_maturity=None,
            min_open_interest=1.0,
            min_vega=0.0,
            max_rel_spread=0.50,
            moneyness_range=(0.5, 2.0),
            drop_synthetic_underlyings=False,
        ),
        weight_config=WeightConfig(use_spread=True, use_vega=False, use_open_interest=False),
        fft_base=FFTParams(N=2**10, eta=0.15, alpha=1.5, b=-10.0, use_simpson=True),
        max_nfev={"black": 20, "heston": 30, "svcj": 30},
        train_frac=0.7,
        random_seed=123,
        runtime_top_expiries_by_oi=2,
        runtime_max_options=5,
        min_options_after_filter=1,
        verbose=False,
    )

    p1 = process_snapshot_to_payload(first, warm_start=None, l2_prev_strength=0.0, **common)
    warm = p1["warm_next"]

    p2_no = process_snapshot_to_payload(second, warm_start=warm, l2_prev_strength=0.0, **common)
    p2_rg = process_snapshot_to_payload(second, warm_start=warm, l2_prev_strength=1_000.0, **common)

    # Black anchor should exist and regularized fit should be no farther from it.
    assert "black" in warm
    d_no_black = _distance_to_anchor(p2_no["param_rows"]["black"].iloc[0], warm["black"])
    d_rg_black = _distance_to_anchor(p2_rg["param_rows"]["black"].iloc[0], warm["black"])
    assert d_rg_black <= d_no_black + 1e-12

    # Heston often succeeds here as well; if anchor exists, check same property.
    if "heston" in warm:
        d_no_heston = _distance_to_anchor(p2_no["param_rows"]["heston"].iloc[0], warm["heston"])
        d_rg_heston = _distance_to_anchor(p2_rg["param_rows"]["heston"].iloc[0], warm["heston"])
        assert d_rg_heston <= d_no_heston + 1e-12
