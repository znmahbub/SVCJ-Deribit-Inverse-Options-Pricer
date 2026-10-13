from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from notebooks._lib.chapter3_final import compute_regularisation_summary
from src.batch_runner import BatchConfig, run_all_snapshots_to_excel
from src.calibration import WeightConfig
from src.inverse_fft_pricer import FFTParams


ANALYSIS_START = "2025-12-30T17:31:15Z"
ANALYSIS_END = "2026-04-08T05:03:27Z"
EXPECTED_PER_CURRENCY = 388
EXPECTED_PARAM_ROWS = 776

SELECTION_COLUMNS = [
    "timestamp",
    "currency",
    "success",
    "nfev",
    "rmse_validation",
    "mae_validation",
    "kappa",
    "theta",
    "sigma_v",
    "rho",
    "v0",
    "lam",
    "ell_y",
    "sigma_y",
    "ell_v",
    "rho_j",
    "feller_gap",
    "feller_violated",
    "svcj_martingale_margin",
    "svcj_martingale_violated",
    "svcj_contour_margin",
    "svcj_contour_violated",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    reg = int(os.environ["REG"])
    root = ROOT
    work_dir = root / "_task13" / f"reg_{reg}"
    export_dir = work_dir / "candidate_export"
    work_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = work_dir / f"calibration_results_reg_{reg}.xlsx"

    cfg = BatchConfig(
        project_root=root,
        data_dir=root / "data",
        output_xlsx=workbook_path,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        resume=False,
        save_every_n_files=1000,
        smoke_test_max_files_per_currency=None,
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
        min_options_after_filter=50,
        weight_config=WeightConfig(
            use_spread=True,
            use_vega=False,
            use_open_interest=False,
            spread_power=1.0,
            vega_power=0.5,
            oi_power=0.5,
            eps_spread=1e-6,
            eps_other=1e-12,
            cap=1e6,
        ),
        fft_base=FFTParams(N=2**12, eta=0.10, alpha=1.5, b=-10.0, use_simpson=True),
        train_frac=0.60,
        validation_frac=0.20,
        test_frac=0.20,
        global_random_seed=123,
        max_nfev=dict(black=250, heston=250, svcj=250),
        l2_prev_strength=float(reg),
        runtime_top_expiries_by_oi=None,
        runtime_max_options=None,
        n_workers=min(4, max(1, (os.cpu_count() or 2) - 1)),
        limit_internal_threads=True,
        internal_num_threads=1,
        currencies=("BTC", "ETH"),
        verbose=False,
    )

    wb = run_all_snapshots_to_excel(cfg)
    svcj = wb["svcj_params"].copy()
    counts = svcj.groupby(svcj["currency"].astype(str).str.upper()).size().to_dict()
    if len(svcj) != EXPECTED_PARAM_ROWS:
        raise RuntimeError(f"lambda={reg}: expected {EXPECTED_PARAM_ROWS} SVCJ parameter rows, got {len(svcj)}")
    for currency in ("BTC", "ETH"):
        if int(counts.get(currency, 0)) != EXPECTED_PER_CURRENCY:
            raise RuntimeError(
                f"lambda={reg}: expected {EXPECTED_PER_CURRENCY} {currency} rows, got {counts.get(currency, 0)}"
            )

    timestamps = pd.to_datetime(svcj["timestamp"], utc=True, errors="raise")
    if timestamps.min() != pd.Timestamp(ANALYSIS_START):
        raise RuntimeError(f"lambda={reg}: unexpected first timestamp {timestamps.min()}")
    if timestamps.max() != pd.Timestamp(ANALYSIS_END):
        raise RuntimeError(f"lambda={reg}: unexpected last timestamp {timestamps.max()}")

    split_counts = {
        name: int(len(wb[name]))
        for name in ("train_data", "validation_data", "test_data")
    }
    if any(value <= 0 for value in split_counts.values()):
        raise RuntimeError(f"lambda={reg}: one or more split sheets are empty: {split_counts}")

    missing = [col for col in SELECTION_COLUMNS if col not in svcj.columns]
    if missing:
        raise RuntimeError(f"lambda={reg}: selection columns missing from svcj_params: {missing}")
    selection_only = svcj[SELECTION_COLUMNS].copy()
    selection_only.to_csv(export_dir / f"svcj_params_reg_{reg}.csv", index=False)

    summary = compute_regularisation_summary(
        [workbook_path],
        export_dir,
        evaluation_split="validation",
    )

    success = pd.to_numeric(svcj["success"], errors="coerce")
    qc = {
        "lambda_TS": reg,
        "analysis_start": ANALYSIS_START,
        "analysis_end": ANALYSIS_END,
        "svcj_param_rows": int(len(svcj)),
        "param_rows_by_currency": {str(k): int(v) for k, v in counts.items()},
        "split_row_counts": split_counts,
        "svcj_success_rate": float(success.mean()),
        "validation_rmse_mean": float(summary.iloc[0]["rmse_validation_mean"]),
        "validation_mae_mean": float(summary.iloc[0]["mae_validation_mean"]),
        "workbook_sha256": sha256(workbook_path),
        "workbook_bytes": int(workbook_path.stat().st_size),
    }
    (export_dir / f"candidate_qc_reg_{reg}.json").write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({k: v for k, v in qc.items() if k != "workbook_sha256"}, indent=2, sort_keys=True))
    del wb, svcj, selection_only
    gc.collect()


if __name__ == "__main__":
    main()
