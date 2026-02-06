"""src.snapshot_job

"Process one snapshot" routine used by the batch calibration pipeline.

The public entrypoint :func:`process_snapshot_to_payload` takes a single
``deribit_options_snapshot_*.csv`` file and a currency (BTC/ETH), then:

1. Filters liquid options (via :func:`src.calibration.filter_liquid_options`).
2. Builds fixed, per-expiry FFT grids for the snapshot (one FFT per expiry).
3. Splits deterministically into train/test.
4. Calibrates Black, Heston, and SVCJ on the train split (warm-started).
5. Prices *all* filtered options once per model and adds three price columns.
6. Returns a payload suitable for persistence into an Excel workbook.

This module intentionally contains no Excel logic; see :mod:`src.results_store`
for persistence.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .calibration import WeightConfig, calibrate_model, filter_liquid_options, price_dataframe
from .inverse_fft_pricer import FFTParams


# ---------------------------------------------------------------------------
# Snapshot filename timestamps
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"deribit_options_snapshot_(\d{8}T\d{6})Z\.csv$")


def timestamp_from_filename(path: Path) -> pd.Timestamp:
    """Parse UTC timestamp from Deribit snapshot filename."""

    m = _TS_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse timestamp from filename: {path.name}")
    return pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S", utc=True)


def timestamp_to_iso_z(ts: pd.Timestamp) -> str:
    ts = pd.to_datetime(ts, utc=True)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def compute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Simple unweighted error metrics."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(keep):
        return {"mse": float("nan"), "mae": float("nan")}
    err = y_pred[keep] - y_true[keep]
    return {"mse": float(np.mean(err * err)), "mae": float(np.mean(np.abs(err)))}


def restrict_for_runtime(
    df: pd.DataFrame,
    *,
    top_expiries: Optional[int],
    max_options: Optional[int],
    random_state: int,
) -> pd.DataFrame:
    """Optional throttles for long batch runs.

    - Keep only the top ``top_expiries`` expiries by total open interest.
    - Cap the total number of rows to ``max_options`` by random sampling.
    """

    out = df

    if top_expiries is not None and top_expiries > 0:
        oi_by_exp = out.groupby("expiry_datetime")["open_interest"].sum().sort_values(ascending=False)
        keep_exp = set(oi_by_exp.head(int(top_expiries)).index)
        out = out[out["expiry_datetime"].isin(keep_exp)].copy()

    if max_options is not None and max_options > 0 and len(out) > int(max_options):
        out = out.sample(n=int(max_options), random_state=int(random_state)).reset_index(drop=True)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------


def process_snapshot_to_payload(
    csv_path: Path,
    *,
    currency: str,
    filter_rules: dict,
    weight_config: WeightConfig,
    fft_base: FFTParams,
    max_nfev: dict,
    train_frac: float,
    random_seed: int,
    runtime_top_expiries_by_oi: Optional[int],
    runtime_max_options: Optional[int],
    min_options_after_filter: int,
    warm_start: Optional[dict[str, dict[str, float]]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Process one snapshot CSV and return an Excel-ready payload.

    The payload contains:
      - ``timestamp_iso``: ISO string from filename
      - ``currency``
      - ``param_rows``: dict(model -> 1-row DataFrame)
      - ``train_df`` / ``test_df``: priced option rows (with price_* columns)
      - ``warm_next``: dict(model -> params) for warm-starting the next snapshot
    """

    ts = timestamp_from_filename(csv_path)
    ts_iso = timestamp_to_iso_z(ts)

    warm_start = warm_start or {}
    warm_next: dict[str, dict[str, float]] = {k: dict(v) for k, v in warm_start.items()}

    empty_tt = pd.DataFrame()

    def _make_param_row(
        model: str,
        *,
        success: bool,
        message: str,
        nfev: int,
        rmse_fit: float,
        mae_fit: float,
        rmse_train: float,
        mae_train: float,
        rmse_test: float,
        mae_test: float,
        n_total: int,
        n_train: int,
        n_test: int,
        params: Optional[dict[str, float]],
    ) -> pd.DataFrame:
        base = dict(
            timestamp=ts_iso,
            currency=currency,
            success=bool(success),
            message=str(message),
            nfev=int(nfev),
            rmse_fit=float(rmse_fit) if rmse_fit is not None else float("nan"),
            mae_fit=float(mae_fit) if mae_fit is not None else float("nan"),
            rmse_train=float(rmse_train) if rmse_train is not None else float("nan"),
            mae_train=float(mae_train) if mae_train is not None else float("nan"),
            rmse_test=float(rmse_test) if rmse_test is not None else float("nan"),
            mae_test=float(mae_test) if mae_test is not None else float("nan"),
            n_options_total=int(n_total),
            n_train=int(n_train),
            n_test=int(n_test),
            random_seed=int(random_seed),
        )
        params = params or {}
        base.update(params)
        return pd.DataFrame([base])

    # --- Load & filter
    try:
        df_raw = pd.read_csv(csv_path)
    except Exception as e:
        msg = f"Read CSV exception: {repr(e)}"
        black_row = _make_param_row(
            "black",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=0,
            n_train=0,
            n_test=0,
            params={"sigma": np.nan},
        )
        heston_row = _make_param_row(
            "heston",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=0,
            n_train=0,
            n_test=0,
            params={"kappa": np.nan, "theta": np.nan, "sigma_v": np.nan, "rho": np.nan, "v0": np.nan},
        )
        svcj_row = _make_param_row(
            "svcj",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=0,
            n_train=0,
            n_test=0,
            params={
                "kappa": np.nan,
                "theta": np.nan,
                "sigma_v": np.nan,
                "rho": np.nan,
                "v0": np.nan,
                "lam": np.nan,
                "ell_y": np.nan,
                "sigma_y": np.nan,
                "ell_v": np.nan,
                "rho_j": np.nan,
            },
        )
        return {
            "timestamp_iso": ts_iso,
            "timestamp": ts,
            "currency": currency,
            "param_rows": {"black": black_row, "heston": heston_row, "svcj": svcj_row},
            "train_df": empty_tt,
            "test_df": empty_tt,
            "warm_next": warm_next,
        }

    if "currency" not in df_raw.columns:
        msg = "Missing required column: 'currency'"
        black_row = _make_param_row(
            "black",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=int(len(df_raw)),
            n_train=0,
            n_test=0,
            params={"sigma": np.nan},
        )
        heston_row = _make_param_row(
            "heston",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=int(len(df_raw)),
            n_train=0,
            n_test=0,
            params={"kappa": np.nan, "theta": np.nan, "sigma_v": np.nan, "rho": np.nan, "v0": np.nan},
        )
        svcj_row = _make_param_row(
            "svcj",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=int(len(df_raw)),
            n_train=0,
            n_test=0,
            params={
                "kappa": np.nan,
                "theta": np.nan,
                "sigma_v": np.nan,
                "rho": np.nan,
                "v0": np.nan,
                "lam": np.nan,
                "ell_y": np.nan,
                "sigma_y": np.nan,
                "ell_v": np.nan,
                "rho_j": np.nan,
            },
        )
        return {
            "timestamp_iso": ts_iso,
            "timestamp": ts,
            "currency": currency,
            "param_rows": {"black": black_row, "heston": heston_row, "svcj": svcj_row},
            "train_df": empty_tt,
            "test_df": empty_tt,
            "warm_next": warm_next,
        }

    df_ccy = df_raw[df_raw["currency"].astype(str) == str(currency)].copy()

    if df_ccy.empty:
        msg = f"No rows found for currency={currency} in this snapshot."
        black_row = _make_param_row(
            "black",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=0,
            n_train=0,
            n_test=0,
            params={"sigma": np.nan},
        )
        heston_row = _make_param_row(
            "heston",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=0,
            n_train=0,
            n_test=0,
            params={"kappa": np.nan, "theta": np.nan, "sigma_v": np.nan, "rho": np.nan, "v0": np.nan},
        )
        svcj_row = _make_param_row(
            "svcj",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=0,
            n_train=0,
            n_test=0,
            params={
                "kappa": np.nan,
                "theta": np.nan,
                "sigma_v": np.nan,
                "rho": np.nan,
                "v0": np.nan,
                "lam": np.nan,
                "ell_y": np.nan,
                "sigma_y": np.nan,
                "ell_v": np.nan,
                "rho_j": np.nan,
            },
        )
        return {
            "timestamp_iso": ts_iso,
            "timestamp": ts,
            "currency": currency,
            "param_rows": {"black": black_row, "heston": heston_row, "svcj": svcj_row},
            "train_df": empty_tt,
            "test_df": empty_tt,
            "warm_next": warm_next,
        }

    try:
        df_filt = filter_liquid_options(df_ccy, currency=currency, **filter_rules)
    except Exception as e:
        msg = f"Filtering exception: {repr(e)}"
        black_row = _make_param_row(
            "black",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=0,
            n_train=0,
            n_test=0,
            params={"sigma": np.nan},
        )
        heston_row = _make_param_row(
            "heston",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=0,
            n_train=0,
            n_test=0,
            params={"kappa": np.nan, "theta": np.nan, "sigma_v": np.nan, "rho": np.nan, "v0": np.nan},
        )
        svcj_row = _make_param_row(
            "svcj",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=0,
            n_train=0,
            n_test=0,
            params={
                "kappa": np.nan,
                "theta": np.nan,
                "sigma_v": np.nan,
                "rho": np.nan,
                "v0": np.nan,
                "lam": np.nan,
                "ell_y": np.nan,
                "sigma_y": np.nan,
                "ell_v": np.nan,
                "rho_j": np.nan,
            },
        )
        return {
            "timestamp_iso": ts_iso,
            "timestamp": ts,
            "currency": currency,
            "param_rows": {"black": black_row, "heston": heston_row, "svcj": svcj_row},
            "train_df": empty_tt,
            "test_df": empty_tt,
            "warm_next": warm_next,
        }

    if len(df_filt) < int(min_options_after_filter):
        msg = f"Skipped: too few options after filtering: n={len(df_filt)} < {min_options_after_filter}"
        black_row = _make_param_row(
            "black",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=int(len(df_filt)),
            n_train=0,
            n_test=0,
            params={"sigma": np.nan},
        )
        heston_row = _make_param_row(
            "heston",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=int(len(df_filt)),
            n_train=0,
            n_test=0,
            params={"kappa": np.nan, "theta": np.nan, "sigma_v": np.nan, "rho": np.nan, "v0": np.nan},
        )
        svcj_row = _make_param_row(
            "svcj",
            success=False,
            message=msg,
            nfev=0,
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=np.nan,
            mae_train=np.nan,
            rmse_test=np.nan,
            mae_test=np.nan,
            n_total=int(len(df_filt)),
            n_train=0,
            n_test=0,
            params={
                "kappa": np.nan,
                "theta": np.nan,
                "sigma_v": np.nan,
                "rho": np.nan,
                "v0": np.nan,
                "lam": np.nan,
                "ell_y": np.nan,
                "sigma_y": np.nan,
                "ell_v": np.nan,
                "rho_j": np.nan,
            },
        )
        return {
            "timestamp_iso": ts_iso,
            "timestamp": ts,
            "currency": currency,
            "param_rows": {"black": black_row, "heston": heston_row, "svcj": svcj_row},
            "train_df": empty_tt,
            "test_df": empty_tt,
            "warm_next": warm_next,
        }

    # Optional runtime restriction
    df_filt = restrict_for_runtime(
        df_filt,
        top_expiries=runtime_top_expiries_by_oi,
        max_options=runtime_max_options,
        random_state=random_seed,
    )

    # Precompute per-expiry FFTParams using *all* filtered options in this snapshot.
    def _fft_params_for_expiry(strikes: np.ndarray) -> FFTParams:
        N = fft_base.N
        eta = fft_base.eta
        lam = 2.0 * np.pi / (N * eta)
        logK_center = float(np.log(np.median(strikes)))
        b = logK_center - 0.5 * N * lam
        return FFTParams(N=N, alpha=fft_base.alpha, eta=eta, b=b, use_simpson=fft_base.use_simpson)

    fft_params_by_expiry: dict = {}
    for exp, g in df_filt.groupby("expiry_datetime", sort=False):
        K_all = g["strike"].to_numpy(dtype=float)
        fft_params_by_expiry[exp] = _fft_params_for_expiry(K_all)

    # Deterministic split (shuffled)
    df_filt = df_filt.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    n_train = int(np.floor(train_frac * len(df_filt)))
    train = df_filt.iloc[:n_train].copy()
    test = df_filt.iloc[n_train:].copy()

    # Warm-start initializations
    init_black = warm_start.get("black", None)

    def _init_heston_from_black(sigma: float, prev: Optional[dict[str, float]]) -> dict[str, float]:
        sigma2 = float(sigma) * float(sigma)
        init = dict(prev) if prev else {}
        init.setdefault("kappa", 2.0)
        init.setdefault("theta", 0.5 * sigma2)
        init.setdefault("sigma_v", 0.75 * sigma)
        init.setdefault("rho", -0.5)
        init.setdefault("v0", 0.5 * sigma2)
        return init

    def _init_svcj_from_black(sigma: float, prev: Optional[dict[str, float]]) -> dict[str, float]:
        sigma2 = float(sigma) * float(sigma)
        init = dict(prev) if prev else {}
        init.setdefault("kappa", 2.0)
        init.setdefault("theta", 0.5 * sigma2)
        init.setdefault("sigma_v", 0.75 * sigma)
        init.setdefault("rho", -0.5)
        init.setdefault("v0", 0.5 * sigma2)
        # Jump defaults (mild)
        init.setdefault("lam", 0.10)
        init.setdefault("ell_y", -0.05)
        init.setdefault("sigma_y", 0.15)
        init.setdefault("ell_v", 0.10)
        init.setdefault("rho_j", -0.3)
        return init

    models = ["black", "heston", "svcj"]
    results: dict[str, Any] = {}
    sigma_seed: Optional[float] = None

    for model in models:
        if verbose:
            print(f"[{currency}] {ts_iso} | {model}: calibrating on n_train={len(train)}")

        if model == "black":
            init_params = init_black
        elif model == "heston":
            if sigma_seed is None:
                if init_black and "sigma" in init_black:
                    sigma_seed = float(init_black["sigma"])
                else:
                    sigma_seed = 0.6
            init_params = _init_heston_from_black(sigma_seed, warm_start.get("heston", None))
        else:
            if sigma_seed is None:
                if init_black and "sigma" in init_black:
                    sigma_seed = float(init_black["sigma"])
                else:
                    sigma_seed = 0.6
            init_params = _init_svcj_from_black(sigma_seed, warm_start.get("svcj", None))

        try:
            res = calibrate_model(
                train,
                model,
                weight_config=weight_config,
                fft_params_base=fft_base,
                dynamic_b=False,
                fft_params_by_expiry=fft_params_by_expiry,
                use_cache_in_optimization=False,
                initial_params=init_params,
                max_nfev=int(max_nfev[model]),
                verbose=0,
                clear_cache_before=False,
            )
        except Exception as e:
            results[model] = {"success": False, "message": f"Exception: {repr(e)}", "nfev": 0, "params": {}}
            continue

        results[model] = res
        if getattr(res, "success", False):
            warm_next[model] = dict(res.params)
            if model == "black":
                try:
                    sigma_seed = float(res.params.get("sigma", np.nan))
                except Exception:
                    sigma_seed = sigma_seed

    # Reprice full snapshot once per model (then split train/test for output & errors)
    df_out = df_filt.copy()
    df_out["snapshot_ts"] = ts_iso
    df_out["currency"] = currency
    df_out["random_seed"] = int(random_seed)

    price_cols = {"black": "price_black", "heston": "price_heston", "svcj": "price_svcj"}
    for model in models:
        col = price_cols[model]
        if model in results and hasattr(results[model], "params") and getattr(results[model], "success", False):
            try:
                p = price_dataframe(
                    df_out,
                    model,
                    dict(results[model].params),
                    fft_params_base=fft_base,
                    dynamic_b=False,
                    fft_params_by_expiry=fft_params_by_expiry,
                    use_cache=True,
                )
                df_out[col] = p
            except Exception as e:
                df_out[col] = np.nan
                if verbose:
                    print(f"[{currency}] {ts_iso} | {model}: pricing exception: {repr(e)}")
        else:
            df_out[col] = np.nan

    train_out = df_out.iloc[:n_train].copy()
    test_out = df_out.iloc[n_train:].copy()

    # Errors (unweighted)
    y_train = train_out["mid_price_clean"].to_numpy(dtype=float) if len(train_out) else np.array([])
    y_test = test_out["mid_price_clean"].to_numpy(dtype=float) if len(test_out) else np.array([])

    errs: dict[str, dict[str, float]] = {}
    for model in models:
        col = price_cols[model]
        p_train = train_out[col].to_numpy(dtype=float) if len(train_out) else np.array([])
        p_test = test_out[col].to_numpy(dtype=float) if len(test_out) else np.array([])
        e_tr = compute_errors(y_train, p_train) if len(train_out) else {"mse": np.nan, "mae": np.nan}
        e_te = compute_errors(y_test, p_test) if len(test_out) else {"mse": np.nan, "mae": np.nan}
        errs[model] = dict(
            rmse_train=float(math.sqrt(e_tr["mse"])) if np.isfinite(e_tr["mse"]) else float("nan"),
            mae_train=float(e_tr["mae"]),
            rmse_test=float(math.sqrt(e_te["mse"])) if np.isfinite(e_te["mse"]) else float("nan"),
            mae_test=float(e_te["mae"]),
        )

    n_total = int(len(df_out))
    n_test = int(len(test_out))

    # Parameter rows
    if isinstance(results.get("black"), dict):
        msg = results["black"].get("message", "failed")
        black_row = _make_param_row(
            "black",
            success=False,
            message=msg,
            nfev=results["black"].get("nfev", 0),
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=errs["black"]["rmse_train"],
            mae_train=errs["black"]["mae_train"],
            rmse_test=errs["black"]["rmse_test"],
            mae_test=errs["black"]["mae_test"],
            n_total=n_total,
            n_train=n_train,
            n_test=n_test,
            params={"sigma": np.nan},
        )
    else:
        resb = results.get("black", None)
        if resb is None:
            black_row = _make_param_row(
                "black",
                success=False,
                message="failed",
                nfev=0,
                rmse_fit=np.nan,
                mae_fit=np.nan,
                rmse_train=errs["black"]["rmse_train"],
                mae_train=errs["black"]["mae_train"],
                rmse_test=errs["black"]["rmse_test"],
                mae_test=errs["black"]["mae_test"],
                n_total=n_total,
                n_train=n_train,
                n_test=n_test,
                params={"sigma": np.nan},
            )
        else:
            black_row = _make_param_row(
                "black",
                success=bool(resb.success),
                message=resb.message,
                nfev=resb.nfev,
                rmse_fit=resb.rmse,
                mae_fit=resb.mae,
                rmse_train=errs["black"]["rmse_train"],
                mae_train=errs["black"]["mae_train"],
                rmse_test=errs["black"]["rmse_test"],
                mae_test=errs["black"]["mae_test"],
                n_total=n_total,
                n_train=n_train,
                n_test=n_test,
                params=dict(resb.params),
            )

    if isinstance(results.get("heston"), dict):
        msg = results["heston"].get("message", "failed")
        heston_row = _make_param_row(
            "heston",
            success=False,
            message=msg,
            nfev=results["heston"].get("nfev", 0),
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=errs["heston"]["rmse_train"],
            mae_train=errs["heston"]["mae_train"],
            rmse_test=errs["heston"]["rmse_test"],
            mae_test=errs["heston"]["mae_test"],
            n_total=n_total,
            n_train=n_train,
            n_test=n_test,
            params={"kappa": np.nan, "theta": np.nan, "sigma_v": np.nan, "rho": np.nan, "v0": np.nan},
        )
    else:
        resh = results.get("heston", None)
        if resh is None:
            heston_row = _make_param_row(
                "heston",
                success=False,
                message="failed",
                nfev=0,
                rmse_fit=np.nan,
                mae_fit=np.nan,
                rmse_train=errs["heston"]["rmse_train"],
                mae_train=errs["heston"]["mae_train"],
                rmse_test=errs["heston"]["rmse_test"],
                mae_test=errs["heston"]["mae_test"],
                n_total=n_total,
                n_train=n_train,
                n_test=n_test,
                params={"kappa": np.nan, "theta": np.nan, "sigma_v": np.nan, "rho": np.nan, "v0": np.nan},
            )
        else:
            heston_row = _make_param_row(
                "heston",
                success=bool(resh.success),
                message=resh.message,
                nfev=resh.nfev,
                rmse_fit=resh.rmse,
                mae_fit=resh.mae,
                rmse_train=errs["heston"]["rmse_train"],
                mae_train=errs["heston"]["mae_train"],
                rmse_test=errs["heston"]["rmse_test"],
                mae_test=errs["heston"]["mae_test"],
                n_total=n_total,
                n_train=n_train,
                n_test=n_test,
                params=dict(resh.params),
            )

    if isinstance(results.get("svcj"), dict):
        msg = results["svcj"].get("message", "failed")
        svcj_row = _make_param_row(
            "svcj",
            success=False,
            message=msg,
            nfev=results["svcj"].get("nfev", 0),
            rmse_fit=np.nan,
            mae_fit=np.nan,
            rmse_train=errs["svcj"]["rmse_train"],
            mae_train=errs["svcj"]["mae_train"],
            rmse_test=errs["svcj"]["rmse_test"],
            mae_test=errs["svcj"]["mae_test"],
            n_total=n_total,
            n_train=n_train,
            n_test=n_test,
            params={
                "kappa": np.nan,
                "theta": np.nan,
                "sigma_v": np.nan,
                "rho": np.nan,
                "v0": np.nan,
                "lam": np.nan,
                "ell_y": np.nan,
                "sigma_y": np.nan,
                "ell_v": np.nan,
                "rho_j": np.nan,
            },
        )
    else:
        ress = results.get("svcj", None)
        if ress is None:
            svcj_row = _make_param_row(
                "svcj",
                success=False,
                message="failed",
                nfev=0,
                rmse_fit=np.nan,
                mae_fit=np.nan,
                rmse_train=errs["svcj"]["rmse_train"],
                mae_train=errs["svcj"]["mae_train"],
                rmse_test=errs["svcj"]["rmse_test"],
                mae_test=errs["svcj"]["mae_test"],
                n_total=n_total,
                n_train=n_train,
                n_test=n_test,
                params={
                    "kappa": np.nan,
                    "theta": np.nan,
                    "sigma_v": np.nan,
                    "rho": np.nan,
                    "v0": np.nan,
                    "lam": np.nan,
                    "ell_y": np.nan,
                    "sigma_y": np.nan,
                    "ell_v": np.nan,
                    "rho_j": np.nan,
                },
            )
        else:
            svcj_row = _make_param_row(
                "svcj",
                success=bool(ress.success),
                message=ress.message,
                nfev=ress.nfev,
                rmse_fit=ress.rmse,
                mae_fit=ress.mae,
                rmse_train=errs["svcj"]["rmse_train"],
                mae_train=errs["svcj"]["mae_train"],
                rmse_test=errs["svcj"]["rmse_test"],
                mae_test=errs["svcj"]["mae_test"],
                n_total=n_total,
                n_train=n_train,
                n_test=n_test,
                params=dict(ress.params),
            )

    return {
        "timestamp_iso": ts_iso,
        "timestamp": ts,
        "currency": currency,
        "param_rows": {"black": black_row, "heston": heston_row, "svcj": svcj_row},
        "train_df": train_out,
        "test_df": test_out,
        "warm_next": warm_next,
    }


__all__ = [
    "process_snapshot_to_payload",
    "timestamp_from_filename",
    "timestamp_to_iso_z",
    "restrict_for_runtime",
    "compute_errors",
]
