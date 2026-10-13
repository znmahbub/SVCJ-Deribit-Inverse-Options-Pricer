from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep numerical libraries single-threaded: candidate-level parallelism is
# provided by the CI matrix, while each temporal path is deliberately sequential.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

from src.analysis_window import filter_paths_by_timestamp
from src.batch_runner import list_snapshot_files
from src.calibration import WeightConfig
from src.inverse_fft_pricer import FFTParams
from src.snapshot_job import process_snapshot_to_payload, timestamp_from_filename


ANALYSIS_START = pd.Timestamp("2025-12-30T17:31:15Z")
ANALYSIS_END = pd.Timestamp("2026-04-08T10:01:19Z")
EXPECTED_SNAPSHOTS = 388
GLOBAL_SEED = 123
CURRENCIES = ("BTC", "ETH")

FILTER_RULES = dict(
    require_bid_ask=True,
    min_time_to_maturity=1 / 365,
    max_time_to_maturity=None,
    min_open_interest=1.0,
    min_vega=0.0,
    max_rel_spread=0.50,
    moneyness_range=(0.5, 2.0),
    drop_synthetic_underlyings=False,
)

WEIGHTS = WeightConfig(
    use_spread=True,
    use_vega=False,
    use_open_interest=False,
    spread_power=1.0,
    vega_power=0.5,
    oi_power=0.5,
    eps_spread=1e-6,
    eps_other=1e-12,
    cap=1e6,
)

FFT_BASE = FFTParams(N=2 ** 12, eta=0.10, alpha=1.5, b=-10.0, use_simpson=True)
MAX_NFEV = dict(black=250, heston=250, svcj=250)


def run_candidate(project_root: Path, lam: float, output: Path) -> None:
    data_dir = project_root / "data"
    files = sorted(list_snapshot_files(data_dir), key=timestamp_from_filename)
    files = filter_paths_by_timestamp(
        files,
        timestamp_from_filename,
        start=ANALYSIS_START,
        end=ANALYSIS_END,
    )
    if len(files) != EXPECTED_SNAPSHOTS:
        raise RuntimeError(
            f"Frozen sample expected {EXPECTED_SNAPSHOTS} snapshots, found {len(files)}"
        )

    rows: dict[str, list[pd.DataFrame]] = {"black": [], "heston": [], "svcj": []}

    for currency_index, currency in enumerate(CURRENCIES):
        warm: dict[str, dict[str, float]] = {}
        for file_index, path in enumerate(files):
            seed = GLOBAL_SEED + 10_000 * currency_index + file_index
            payload = process_snapshot_to_payload(
                path,
                currency=currency,
                filter_rules=FILTER_RULES,
                weight_config=WEIGHTS,
                fft_base=FFT_BASE,
                max_nfev=MAX_NFEV,
                train_frac=0.60,
                validation_frac=0.20,
                test_frac=0.20,
                random_seed=seed,
                runtime_top_expiries_by_oi=None,
                runtime_max_options=None,
                min_options_after_filter=50,
                warm_start=warm,
                l2_prev_strength=float(lam),
                verbose=False,
            )
            for model in rows:
                frame = payload.get("param_rows", {}).get(model)
                if frame is not None and not frame.empty:
                    rows[model].append(frame.copy())
            warm = payload.get("warm_next") or warm

            if (file_index + 1) % 50 == 0 or file_index + 1 == len(files):
                print(
                    f"lambda={lam:g} {currency}: {file_index + 1}/{len(files)} snapshots",
                    flush=True,
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for model, sheet in (
            ("black", "black_params"),
            ("heston", "heston_params"),
            ("svcj", "svcj_params"),
        ):
            frame = pd.concat(rows[model], ignore_index=True) if rows[model] else pd.DataFrame()
            frame.to_excel(writer, sheet_name=sheet, index=False)

    svcj = pd.concat(rows["svcj"], ignore_index=True)
    print(
        f"Wrote {output} | lambda={lam:g} | rows={len(svcj)} | "
        f"success={pd.to_numeric(svcj['success'], errors='coerce').mean():.3%} | "
        f"validation RMSE={pd.to_numeric(svcj['rmse_validation'], errors='coerce').mean():.8f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-ts", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_candidate(args.project_root.resolve(), args.lambda_ts, args.output.resolve())


if __name__ == "__main__":
    main()
