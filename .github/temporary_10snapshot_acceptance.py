from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
SELECTION = json.loads((ROOT / "acceptance_selection_10.json").read_text(encoding="utf-8"))
SNAPSHOTS = SELECTION["selected_snapshots"]
RX = re.compile(r"deribit_options_snapshot_(\d{8}T\d{6})Z\.csv")
TIMES = [pd.to_datetime(RX.fullmatch(name).group(1), format="%Y%m%dT%H%M%S", utc=True) for name in SNAPSHOTS]
EXPECTED_BLOCKS = {(ts, ccy) for ts in TIMES for ccy in ("BTC", "ETH")}
SHEETS = ("black_params", "heston_params", "svcj_params", "train_data", "test_data")

REPORT: dict[str, object] = {
    "selection": SELECTION,
    "checks": [],
    "warnings": [],
    "calibration": {},
    "checkpoints": {},
    "pricing": {},
    "hedging": {},
    "final": {},
}


def check(condition, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)
    REPORT["checks"].append(message)


def warn(condition, message: str) -> None:
    if bool(condition):
        REPORT["warnings"].append(message)


def num(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def rmse(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else float("nan")


def keyset(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    out = frame.loc[:, columns].copy()
    for col in columns:
        if "time" in col or col.endswith("_ts"):
            parsed = pd.to_datetime(out[col], utc=True, errors="coerce")
            out[col] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            out[col] = out[col].astype(str)
    return set(map(tuple, out.itertuples(index=False, name=None)))


check(len(SNAPSHOTS) == 10, "exactly 10 real snapshots selected")
check(len(set(TIMES)) == 10, "selected snapshot timestamps are unique")
check(TIMES == sorted(TIMES), "selected snapshots are chronological")

# ------------------------------------------------------------------
# Calibration + checkpoint persistence
# ------------------------------------------------------------------
test_frames: dict[str, pd.DataFrame] = {}
test_keys: dict[str, set[tuple[str, ...]]] = {}
for spec, lam in (("wls", 0.0), ("pwls", 1.0)):
    workbook_path = ROOT / "excel files" / "canonical" / f"calibration_{spec}.xlsx"
    check(workbook_path.exists(), f"{spec}: final calibration workbook exists")
    check(set(pd.ExcelFile(workbook_path).sheet_names) == set(SHEETS), f"{spec}: final workbook has canonical five-sheet schema")

    model_report: dict[str, object] = {}
    for model in ("black", "heston", "svcj"):
        frame = pd.read_excel(workbook_path, sheet_name=f"{model}_params")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["currency"] = frame["currency"].astype(str).str.upper()
        check(set(zip(frame["timestamp"], frame["currency"])) == EXPECTED_BLOCKS, f"{spec} {model}: all 20 timestamp/currency blocks present")
        check(np.allclose(num(frame["l2_prev_strength"]), lam, rtol=0, atol=0), f"{spec} {model}: lambda_TS={lam:g}")
        success = frame["success"].fillna(False).astype(bool)
        check(success.all(), f"{spec} {model}: all 20 calibrations converged")

        if model in {"heston", "svcj"}:
            rho = num(frame["rho"])
            check(np.isfinite(rho).all() and (np.abs(rho) <= 1.0 + 1e-10).all(), f"{spec} {model}: rho remains in correlation bounds")
            for col in ("kappa", "theta", "sigma_v", "v0"):
                values = num(frame[col])
                check(np.isfinite(values).all() and (values >= 0).all(), f"{spec} {model}: {col} finite and non-negative")
        if model == "svcj":
            check(not frame["svcj_martingale_violated"].fillna(False).astype(bool).any(), f"{spec} SVCJ: martingale admissibility holds")
            check(not frame["svcj_contour_violated"].fillna(False).astype(bool).any(), f"{spec} SVCJ: contour admissibility holds")
            for col in ("lam", "sigma_y"):
                values = num(frame[col])
                check(np.isfinite(values).all() and (values >= 0).all(), f"{spec} SVCJ: {col} finite and non-negative")
            feller_rate = float(frame["feller_violated"].fillna(False).astype(bool).mean())
        else:
            feller_rate = float(frame.get("feller_violated", pd.Series(False, index=frame.index)).fillna(False).astype(bool).mean())

        model_report[model] = {
            "rows": int(len(frame)),
            "success_rate": float(success.mean()),
            "mean_test_rmse_coin": float(pd.to_numeric(frame["rmse_test"], errors="coerce").mean()),
            "feller_violation_rate": feller_rate,
        }

    train = pd.read_excel(workbook_path, sheet_name="train_data")
    test = pd.read_excel(workbook_path, sheet_name="test_data")
    for frame in (train, test):
        frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True, errors="coerce")
        frame["currency"] = frame["currency"].astype(str).str.upper()
    check(set(train["split"].astype(str).str.lower()) == {"train"}, f"{spec}: train sheet contains only train rows")
    check(set(test["split"].astype(str).str.lower()) == {"test"}, f"{spec}: test sheet contains only test rows")

    train_shares: list[float] = []
    for ts, ccy in sorted(EXPECTED_BLOCKS):
        tr = train[(train["snapshot_ts"] == ts) & (train["currency"] == ccy)]
        te = test[(test["snapshot_ts"] == ts) & (test["currency"] == ccy)]
        check(len(tr) > 0 and len(te) > 0, f"{spec} {ccy} {ts}: both train/test splits non-empty")
        check(not (set(tr["instrument_name"].astype(str)) & set(te["instrument_name"].astype(str))), f"{spec} {ccy} {ts}: zero train/test instrument overlap")
        train_shares.append(len(tr) / (len(tr) + len(te)))
    check(all(0.65 <= share <= 0.75 for share in train_shares), f"{spec}: every block remains approximately 70/30")

    test_frames[spec] = test.copy()
    test_keys[spec] = keyset(test, ["snapshot_ts", "currency", "instrument_name"])
    REPORT["calibration"][spec] = {
        "models": model_report,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_share_min": float(min(train_shares)),
        "train_share_max": float(max(train_shares)),
    }

    checkpoint_report: dict[str, object] = {}
    for ccy in ("BTC", "ETH"):
        checkpoint_dir = ROOT / "excel files" / "canonical" / ".checkpoints" / f"calibration_{spec}" / ccy
        check(checkpoint_dir.is_dir(), f"{spec} {ccy}: Parquet checkpoint directory exists")
        checkpoint_report[ccy] = {}
        for sheet in SHEETS:
            parquet_path = checkpoint_dir / f"{sheet}.parquet"
            check(parquet_path.exists(), f"{spec} {ccy}: {sheet}.parquet checkpoint exists")
            part = pd.read_parquet(parquet_path)
            final = pd.read_excel(workbook_path, sheet_name=sheet)
            final_ccy = final[final["currency"].astype(str).str.upper() == ccy]
            check(len(part) == len(final_ccy), f"{spec} {ccy} {sheet}: checkpoint row count matches final Excel")
            if sheet.endswith("_params"):
                cols = ["timestamp", "currency"]
            else:
                cols = ["snapshot_ts", "currency", "instrument_name"]
            check(keyset(part, cols) == keyset(final_ccy, cols), f"{spec} {ccy} {sheet}: checkpoint keys match final Excel")
            checkpoint_report[ccy][sheet] = int(len(part))
    REPORT["checkpoints"][spec] = checkpoint_report

check(test_keys["wls"] == test_keys["pwls"], "WLS and PWLS use the identical deterministic holdout")

# ------------------------------------------------------------------
# Pricing math and economic diagnostics
# ------------------------------------------------------------------
pricing_root = ROOT / "notebooks" / "pricing_outputs" / "canonical"
for spec in ("wls", "pwls"):
    overall = pd.read_csv(pricing_root / spec / "baseline_overall_metrics.csv")
    check(set(overall["scope"]) == {"BTC", "ETH", "Pooled"}, f"{spec}: pricing report contains BTC/ETH/Pooled scopes")
    pooled = overall[overall["scope"] == "Pooled"].iloc[0]
    test = test_frames[spec]
    F = num(test["F0"])
    mid = num(test["mid_price_clean"])
    metrics: dict[str, object] = {}
    for model in ("black", "heston", "svcj"):
        prices = num(test[f"price_{model}"])
        check(np.isfinite(prices).all() and (prices >= -1e-12).all(), f"{spec} {model}: all test prices finite and non-negative")
        check(pd.isna(pooled[f"rmse_coin_{model}"]) and pd.isna(pooled[f"mae_coin_{model}"]), f"{spec} {model}: pooled raw-coin errors remain undefined")
        err = prices - mid
        direct_usd = rmse(F * err)
        reported_usd = float(pooled[f"rmse_usd_{model}"])
        check(np.isclose(direct_usd, reported_usd, rtol=1e-10, atol=1e-12), f"{spec} {model}: pooled USD RMSE reconstructs exactly")
        within = float(pooled[f"within_spread_{model}"])
        check(0.0 <= within <= 1.0, f"{spec} {model}: within-spread rate is a valid probability")
        for ccy in ("BTC", "ETH"):
            mask = test["currency"].astype(str).str.upper().to_numpy() == ccy
            reported_coin = float(overall.loc[overall["scope"] == ccy, f"rmse_coin_{model}"].iloc[0])
            check(np.isclose(rmse(err[mask]), reported_coin, rtol=1e-10, atol=1e-12), f"{spec} {ccy} {model}: coin RMSE reconstructs")
        metrics[model] = {"rmse_usd": reported_usd, "within_spread": within}
    REPORT["pricing"][spec] = metrics

    best_rmse = min(value["rmse_usd"] for value in metrics.values())
    warn(metrics["svcj"]["rmse_usd"] > 1.5 * best_rmse, f"{spec}: SVCJ pooled USD RMSE is >50% above the best model on this 10-snapshot window")

# ------------------------------------------------------------------
# Hedging math and economic diagnostics
# ------------------------------------------------------------------
hedging_root = ROOT / "notebooks" / "hedging_outputs" / "canonical"
for spec in ("wls", "pwls"):
    workbook_path = ROOT / "excel files" / "canonical" / f"hedging_{spec}.xlsx"
    check(workbook_path.exists(), f"{spec}: hedging workbook exists")
    test = pd.read_excel(workbook_path, sheet_name="test_data")
    panel = pd.read_excel(workbook_path, sheet_name="hedge_interval_panel")
    verify = pd.read_excel(workbook_path, sheet_name="price_verification")
    check(set(test["split"].astype(str).str.lower()) == {"test"}, f"{spec}: hedging option sheet is test-only")
    check(set(panel["split"].astype(str).str.lower()) <= {"test"}, f"{spec}: hedge interval panel is test-only")
    eligible = panel["is_summary_eligible"].fillna(False).astype(bool).to_numpy()
    check(int(eligible.sum()) > 0, f"{spec}: hedge-eligible interval sample is non-empty")

    price_diag: dict[str, float] = {}
    for model in ("black", "heston", "svcj"):
        rows = verify[verify["model"].astype(str).str.lower() == model]
        check(len(rows) == 20, f"{spec} {model}: price verification covers all 20 blocks")
        check(rows["error"].isna().all(), f"{spec} {model}: no hedging repricing errors")
        check((pd.to_numeric(rows["n_valid_prices"]) == pd.to_numeric(rows["n_rows"])).all(), f"{spec} {model}: all hedge-stage reprices finite")
        max_diff = float(pd.to_numeric(rows["max_abs_price_diff"], errors="coerce").max())
        check(max_diff < 1e-10, f"{spec} {model}: hedge repricing matches calibration price below 1e-10 coin")
        regular = num(test[f"regular_delta_{model}"])
        net = num(test[f"net_delta_{model}"])
        price = num(test[f"price_{model}"])
        mask = np.isfinite(regular) & np.isfinite(net) & np.isfinite(price)
        check(np.allclose(net[mask], regular[mask] - price[mask], rtol=1e-10, atol=1e-10), f"{spec} {model}: net delta = regular delta - V_coin")
        price_diag[model] = max_diff

    keys = ["snapshot_ts", "currency", "instrument_name"]
    delta_cols = [f"{kind}_delta_{model}" for model in ("black", "heston", "svcj") for kind in ("regular", "net")]
    merged = panel.merge(test[keys + delta_cols], on=keys, how="left", validate="many_to_one")
    elig = merged["is_summary_eligible"].fillna(False).astype(bool).to_numpy()
    F = num(merged["F_term"])
    H0 = num(merged["H_perp"])
    H1 = num(merged["perp_close_t1"])
    scale = num(merged["hedge_notional_scale_usd_per_delta"])
    valid_scale = np.isfinite(F) & np.isfinite(H0) & np.isfinite(scale) & (F > 0) & (H0 > 0)
    check(np.allclose(scale[valid_scale], H0[valid_scale] ** 2 / F[valid_scale], rtol=1e-12, atol=1e-12), f"{spec}: inverse-perpetual hedge scale H^2/F")

    unhedged = num(merged["option_pnl_coin_short"])
    hedge_metrics: dict[str, object] = {}
    improvements: list[float] = []
    for model in ("black", "heston", "svcj"):
        hedge_metrics[model] = {}
        for hedge_type, delta_col in (("delta", f"regular_delta_{model}"), ("net_delta", f"net_delta_{model}")):
            delta = num(merged[delta_col])
            notional = num(merged[f"hedge_notional_usd_{model}_{hedge_type}"])
            actual = num(merged[f"net_pnl_coin_{model}_{hedge_type}"])
            mask = elig & np.isfinite(delta) & np.isfinite(notional) & np.isfinite(scale)
            check(mask.any(), f"{spec} {model} {hedge_type}: eligible finite hedge rows exist")
            check(np.allclose(notional[mask], scale[mask] * delta[mask], rtol=1e-10, atol=1e-10), f"{spec} {model} {hedge_type}: N_USD=(H^2/F)delta")
            direct = unhedged + notional * (1.0 / H0 - 1.0 / H1)
            pnl_mask = mask & np.isfinite(H1) & (H1 > 0) & np.isfinite(unhedged) & np.isfinite(actual)
            check(np.allclose(actual[pnl_mask], direct[pnl_mask], rtol=1e-10, atol=1e-12), f"{spec} {model} {hedge_type}: inverse-perpetual hedge P&L identity")
            base_rmse = rmse(unhedged[pnl_mask])
            hedge_rmse = rmse(actual[pnl_mask])
            reduction = float(1.0 - hedge_rmse / base_rmse) if base_rmse > 0 else float("nan")
            improvements.append(reduction)
            hedge_metrics[model][hedge_type] = {
                "n": int(pnl_mask.sum()),
                "unhedged_rmse_coin": base_rmse,
                "hedged_rmse_coin": hedge_rmse,
                "rmse_reduction": reduction,
            }

    warn(not any(np.isfinite(value) and value > 0 for value in improvements), f"{spec}: none of the six hedge variants reduces RMSE on this 10-snapshot window")
    headline = pd.read_csv(hedging_root / spec / "headline_test.csv")
    check(len(headline) > 0, f"{spec}: headline hedge report is non-empty")
    check(set(headline["model"].astype(str).str.lower()) == {"black", "heston", "svcj"}, f"{spec}: headline includes all three models")
    check(set(headline["hedge_type"].astype(str).str.lower()) == {"delta", "net_delta"}, f"{spec}: headline includes both hedge rules")
    REPORT["hedging"][spec] = {
        "panel_rows": int(len(panel)),
        "eligible_intervals": int(elig.sum()),
        "max_repricing_diff_by_model": price_diag,
        "rmse_diagnostics": hedge_metrics,
    }

# ------------------------------------------------------------------
# Final package
# ------------------------------------------------------------------
final_root = ROOT / "notebooks" / "final_outputs" / "canonical"
manifest = pd.read_csv(final_root / "manifest.csv")
check(len(manifest) > 0, "final artifact manifest is non-empty")
check({"pricing_table", "pricing_figure", "hedging_table", "hedging_figure"} <= set(manifest["artifact_type"]), "final manifest contains all four artifact classes")
check(all(Path(path).exists() for path in manifest["destination"].astype(str)), "every final manifest destination exists")
check(manifest["destination"].astype(str).str.contains("/hedging/wls/").any(), "final package contains WLS hedging artifacts")
check(manifest["destination"].astype(str).str.contains("/hedging/pwls/").any(), "final package contains PWLS hedging artifacts")
REPORT["final"] = {
    "manifest_rows": int(len(manifest)),
    "artifact_counts": {str(k): int(v) for k, v in manifest["artifact_type"].value_counts().to_dict().items()},
}

warnings = pd.DataFrame({"warning": REPORT["warnings"]})
warnings.to_csv(ROOT / "acceptance_warnings_10.csv", index=False)
(ROOT / "acceptance_report_10.json").write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
print("TEN_SNAPSHOT_ACCEPTANCE_SUCCESS")
print(json.dumps(REPORT, indent=2, default=str))
