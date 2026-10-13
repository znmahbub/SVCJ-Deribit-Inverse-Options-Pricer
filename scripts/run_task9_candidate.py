from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Candidate/currency jobs are parallelized by the CI matrix; each temporal path
# itself must remain strictly chronological.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.analysis_window import filter_paths_by_timestamp
from src.batch_runner import list_snapshot_files
from src.calibration import WeightConfig, calibrate_model, filter_liquid_options, price_dataframe
from src.inverse_fft_pricer import FFTParams, center_fft_params_on_strikes
from src.snapshot_job import compute_errors, timestamp_from_filename, timestamp_to_iso_z
from src.splitting import SplitConfig, assign_cross_sectional_split


ANALYSIS_START = pd.Timestamp("2025-12-30T17:31:15Z")
ANALYSIS_END = pd.Timestamp("2026-04-08T05:03:27Z")
EXPECTED_SNAPSHOTS = 388
GLOBAL_SEED = 123
CURRENCY_INDEX = {"BTC": 0, "ETH": 1}

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
MAX_NFEV = 250
MIN_OPTIONS_AFTER_FILTER = 50

SVCJ_PARAM_NAMES = (
    "kappa", "theta", "sigma_v", "rho", "v0",
    "lam", "ell_y", "sigma_y", "ell_v", "rho_j",
)
DIAGNOSTIC_NAMES = (
    "regularization_active",
    "l2_prev_strength",
    "feller_penalty_active",
    "feller_gap",
    "feller_violated",
    "svcj_moment_penalty_active",
    "svcj_martingale_margin",
    "svcj_martingale_violated",
    "svcj_contour_margin",
    "svcj_contour_violated",
)


def _frozen_snapshot_files(project_root: Path) -> list[Path]:
    """Return the exact 388-snapshot submitted-thesis sample."""
    data_dir = project_root / "data"
    combos = pd.read_csv(data_dir / "snapshot_currency_combos.csv", usecols=["snapshot_file"])
    membership = set(combos["snapshot_file"].dropna().astype(str))

    files = sorted(list_snapshot_files(data_dir), key=timestamp_from_filename)
    files = filter_paths_by_timestamp(
        files,
        timestamp_from_filename,
        start=ANALYSIS_START,
        end=ANALYSIS_END,
    )
    files = [path for path in files if path.name in membership]

    if len(files) != EXPECTED_SNAPSHOTS:
        raise RuntimeError(
            f"Frozen submitted-thesis sample expected {EXPECTED_SNAPSHOTS} snapshots, found {len(files)}"
        )
    return files


def _svcj_seed_from_black_sigma(sigma: float) -> dict[str, float]:
    """Match ``snapshot_job._init_svcj_from_black`` when no SVCJ warm start exists."""
    sigma = float(sigma)
    sigma2 = sigma * sigma
    return {
        "kappa": 2.0,
        "theta": 0.5 * sigma2,
        "sigma_v": 0.75 * sigma,
        "rho": -0.5,
        "v0": 0.5 * sigma2,
        "lam": 0.10,
        "ell_y": -0.05,
        "sigma_y": 0.15,
        "ell_v": 0.10,
        "rho_j": -0.3,
    }


def _failure_row(*, ts_iso: str, currency: str, seed: int, lam: float, message: str, n_total: int) -> dict:
    row = {
        "timestamp": ts_iso,
        "currency": currency,
        "success": False,
        "message": message,
        "nfev": 0,
        "rmse_validation": np.nan,
        "mae_validation": np.nan,
        "n_options_total": int(n_total),
        "n_train": 0,
        "n_validation": 0,
        "n_test": 0,
        "random_seed": int(seed),
        "regularization_active": False,
        "l2_prev_strength": float(lam),
        "feller_penalty_active": False,
        "feller_gap": np.nan,
        "feller_violated": False,
        "svcj_moment_penalty_active": False,
        "svcj_martingale_margin": np.nan,
        "svcj_martingale_violated": False,
        "svcj_contour_margin": np.nan,
        "svcj_contour_violated": False,
    }
    row.update({name: np.nan for name in SVCJ_PARAM_NAMES})
    return row


def _prepare_snapshot(path: Path, *, currency: str, seed: int):
    raw = pd.read_csv(path)
    if "currency" not in raw.columns:
        raise ValueError("Missing required column: currency")
    ccy = raw[raw["currency"].astype(str).str.strip().str.upper() == currency].copy()
    filt = filter_liquid_options(ccy, currency=currency, **FILTER_RULES)
    if len(filt) < MIN_OPTIONS_AFTER_FILTER:
        raise ValueError(
            f"too few options after filtering: n={len(filt)} < {MIN_OPTIONS_AFTER_FILTER}"
        )

    filt = filt.reset_index(drop=True).copy()
    split_cfg = SplitConfig(
        train_frac=0.60,
        validation_frac=0.20,
        test_frac=0.20,
        random_seed=int(seed),
    )
    filt["split"] = assign_cross_sectional_split(filt, split_cfg).to_numpy()
    train = filt.loc[filt["split"] == "train"].copy()
    validation = filt.loc[filt["split"] == "validation"].copy()
    test = filt.loc[filt["split"] == "test"].copy()

    fft_params_by_expiry = {}
    for exp, group in filt.groupby("expiry_datetime", sort=False):
        fft_params_by_expiry[exp] = center_fft_params_on_strikes(
            FFT_BASE, group["strike"].to_numpy(dtype=float)
        )
    return filt, train, validation, test, fft_params_by_expiry


def _black_sigma_seed(
    train: pd.DataFrame,
    fft_params_by_expiry: dict,
    *,
    previous_black: dict[str, float] | None,
    lam: float,
):
    """Replicate the full pipeline's Black path until SVCJ first succeeds.

    Heston never feeds SVCJ initialization. Black does, but only when there is
    no successful previous SVCJ vector. The Black fit itself is warm-started and
    receives the same temporal-regularization strength in the full pipeline, so
    we preserve that behavior exactly here.
    """
    result = calibrate_model(
        train,
        "black",
        weight_config=WEIGHTS,
        fft_params_base=FFT_BASE,
        dynamic_b=False,
        fft_params_by_expiry=fft_params_by_expiry,
        use_cache_in_optimization=False,
        initial_params=previous_black,
        previous_params=previous_black,
        l2_prev_strength=float(lam),
        max_nfev=MAX_NFEV,
        verbose=0,
        clear_cache_before=False,
    )
    sigma = 0.6
    next_black = previous_black
    if result.success and np.isfinite(result.params.get("sigma", np.nan)):
        sigma = float(result.params["sigma"])
        next_black = dict(result.params)
    elif previous_black and np.isfinite(previous_black.get("sigma", np.nan)):
        # This branch only supplies a seed if the current Black optimizer fails;
        # it matches snapshot_job's retained warm-start state.
        sigma = float(previous_black["sigma"])
    return sigma, next_black


def run_candidate(project_root: Path, lam: float, currency: str, output: Path) -> None:
    currency = str(currency).upper()
    if currency not in CURRENCY_INDEX:
        raise ValueError(f"currency must be one of {sorted(CURRENCY_INDEX)}")

    files = _frozen_snapshot_files(project_root)
    currency_index = CURRENCY_INDEX[currency]
    warm_black: dict[str, float] | None = None
    warm_svcj: dict[str, float] | None = None
    rows: list[dict] = []

    for file_index, path in enumerate(files):
        ts = timestamp_from_filename(path)
        ts_iso = timestamp_to_iso_z(ts)
        seed = GLOBAL_SEED + 10_000 * currency_index + file_index

        try:
            filt, train, validation, test, fft_params_by_expiry = _prepare_snapshot(
                path, currency=currency, seed=seed
            )
        except Exception as exc:
            rows.append(
                _failure_row(
                    ts_iso=ts_iso,
                    currency=currency,
                    seed=seed,
                    lam=lam,
                    message=f"Preparation exception: {exc!r}",
                    n_total=0,
                )
            )
            continue

        if warm_svcj is None:
            sigma_seed, warm_black = _black_sigma_seed(
                train,
                fft_params_by_expiry,
                previous_black=warm_black,
                lam=lam,
            )
            initial_params = _svcj_seed_from_black_sigma(sigma_seed)
        else:
            initial_params = dict(warm_svcj)

        try:
            result = calibrate_model(
                train,
                "svcj",
                weight_config=WEIGHTS,
                fft_params_base=FFT_BASE,
                dynamic_b=False,
                fft_params_by_expiry=fft_params_by_expiry,
                use_cache_in_optimization=False,
                initial_params=initial_params,
                previous_params=warm_svcj,
                l2_prev_strength=float(lam),
                max_nfev=MAX_NFEV,
                verbose=0,
                clear_cache_before=False,
            )
        except Exception as exc:
            rows.append(
                _failure_row(
                    ts_iso=ts_iso,
                    currency=currency,
                    seed=seed,
                    lam=lam,
                    message=f"Calibration exception: {exc!r}",
                    n_total=len(filt),
                )
            )
            continue

        # Keep the calibration-success flag and warm-state semantics identical
        # to process_snapshot_to_payload. Validation repricing is an evaluation
        # step; a pricing NaN must not retroactively turn a successful optimizer
        # result into a failed calibration or alter the next warm start.
        calibration_success = bool(result.success)
        message = str(result.message)
        rmse_validation = np.nan
        mae_validation = np.nan
        if calibration_success:
            try:
                validation_prices = price_dataframe(
                    validation,
                    "svcj",
                    dict(result.params),
                    fft_params_base=FFT_BASE,
                    dynamic_b=False,
                    fft_params_by_expiry=fft_params_by_expiry,
                    use_cache=True,
                )
                validation_errors = compute_errors(
                    validation["mid_price_clean"].to_numpy(dtype=float),
                    validation_prices,
                )
                rmse_validation = float(validation_errors["rmse"])
                mae_validation = float(validation_errors["mae"])
            except Exception as exc:
                message += f" (validation pricing exception: {exc!r})"

        row = {
            "timestamp": ts_iso,
            "currency": currency,
            "success": calibration_success,
            "message": message,
            "nfev": int(result.nfev),
            "rmse_validation": rmse_validation,
            "mae_validation": mae_validation,
            "n_options_total": int(len(filt)),
            "n_train": int(len(train)),
            "n_validation": int(len(validation)),
            "n_test": int(len(test)),
            "random_seed": int(seed),
        }
        row.update({name: float(result.params[name]) for name in SVCJ_PARAM_NAMES})
        row.update({name: getattr(result, name) for name in DIAGNOSTIC_NAMES})
        rows.append(row)

        if calibration_success:
            warm_svcj = dict(result.params)

        if (file_index + 1) % 25 == 0 or file_index + 1 == len(files):
            completed = pd.DataFrame(rows)
            ok = completed.loc[completed["success"] == True]
            mean_rmse = pd.to_numeric(ok["rmse_validation"], errors="coerce").mean()
            print(
                f"lambda={lam:g} {currency}: {file_index + 1}/{len(files)} | "
                f"success={len(ok)}/{len(completed)} | validation_rmse={mean_rmse:.8f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_SNAPSHOTS:
        raise RuntimeError(
            f"Expected {EXPECTED_SNAPSHOTS} parameter rows for {currency}, found {len(frame)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="svcj_params", index=False)

    ok = frame.loc[frame["success"] == True]
    print(
        f"Wrote {output} | lambda={lam:g} | currency={currency} | rows={len(frame)} | "
        f"success={len(ok) / len(frame):.3%} | "
        f"validation RMSE={pd.to_numeric(ok['rmse_validation'], errors='coerce').mean():.8f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-ts", type=float, required=True)
    parser.add_argument("--currency", choices=sorted(CURRENCY_INDEX), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_candidate(
        args.project_root.resolve(),
        args.lambda_ts,
        args.currency,
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
