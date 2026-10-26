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

report = {"selection": SELECTION, "calibration": {}, "pricing": {}, "hedging": {}, "checks": []}


def check(cond, msg):
    if not bool(cond):
        raise AssertionError(msg)
    report["checks"].append(msg)


def as_num(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def keys(frame, cols):
    x = frame.loc[:, cols].copy()
    for col in cols:
        if col.endswith("_ts") or col == "timestamp":
            x[col] = pd.to_datetime(x[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            x[col] = x[col].astype(str)
    return set(map(tuple, x.itertuples(index=False, name=None)))


check(len(SNAPSHOTS) == 10 and len(set(TIMES)) == 10, "exactly 10 unique snapshots")
holdout_keys = {}

for spec, lam in (("wls", 0.0), ("pwls", 1.0)):
    book = ROOT / "excel files" / "canonical" / f"calibration_{spec}.xlsx"
    check(book.exists(), f"{spec}: calibration workbook exists")
    check(set(pd.ExcelFile(book).sheet_names) == set(SHEETS), f"{spec}: canonical five-sheet schema")
    stats = {}

    for model in ("black", "heston", "svcj"):
        df = pd.read_excel(book, sheet_name=f"{model}_params")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["currency"] = df["currency"].astype(str).str.upper()
        check(set(zip(df["timestamp"], df["currency"])) == EXPECTED_BLOCKS, f"{spec} {model}: all 20 blocks present")
        check(np.allclose(as_num(df["l2_prev_strength"]), lam, atol=0, rtol=0), f"{spec} {model}: lambda_TS={lam:g}")
        success = df["success"].fillna(False).astype(bool)
        if model == "svcj":
            check(int(success.sum()) >= 18, f"{spec} SVCJ: at least 18/20 calibrations succeed")
            fail_cols = [c for c in ("timestamp", "currency", "message", "nfev", "rmse_fit", "svcj_martingale_violated", "svcj_contour_violated") if c in df.columns]
            failed = df.loc[~success, fail_cols].copy()
            failed.to_csv(ROOT / f"acceptance_failed_svcj_{spec}.csv", index=False)
            if len(failed):
                if "svcj_martingale_violated" in failed:
                    check(not failed["svcj_martingale_violated"].fillna(False).astype(bool).any(), f"{spec} failed SVCJ blocks remain martingale-admissible")
                if "svcj_contour_violated" in failed:
                    check(not failed["svcj_contour_violated"].fillna(False).astype(bool).any(), f"{spec} failed SVCJ blocks remain contour-admissible")
        else:
            check(success.all(), f"{spec} {model}: all 20 calibrations succeed")
        stats[model] = {"success": int(success.sum()), "total": int(len(df))}

    train = pd.read_excel(book, sheet_name="train_data")
    test = pd.read_excel(book, sheet_name="test_data")
    for frame in (train, test):
        frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True, errors="coerce")
        frame["currency"] = frame["currency"].astype(str).str.upper()
    check(set(train["split"].astype(str).str.lower()) == {"train"}, f"{spec}: train sheet is train-only")
    check(set(test["split"].astype(str).str.lower()) == {"test"}, f"{spec}: test sheet is test-only")
    for ts, ccy in EXPECTED_BLOCKS:
        tr = train[(train["snapshot_ts"] == ts) & (train["currency"] == ccy)]
        te = test[(test["snapshot_ts"] == ts) & (test["currency"] == ccy)]
        check(len(tr) > 0 and len(te) > 0, f"{spec} {ccy} {ts}: non-empty train/test")
        check(not (set(tr["instrument_name"].astype(str)) & set(te["instrument_name"].astype(str))), f"{spec} {ccy} {ts}: zero train/test overlap")
    holdout_keys[spec] = keys(test, ["snapshot_ts", "currency", "instrument_name"])

    cp_rows = {}
    for ccy in ("BTC", "ETH"):
        cpdir = ROOT / "excel files" / "canonical" / ".checkpoints" / f"calibration_{spec}" / ccy
        check(cpdir.is_dir(), f"{spec} {ccy}: checkpoint directory exists")
        cp_rows[ccy] = {}
        for sheet in SHEETS:
            pq = cpdir / f"{sheet}.parquet"
            check(pq.exists(), f"{spec} {ccy}: {sheet}.parquet exists")
            part = pd.read_parquet(pq)
            final = pd.read_excel(book, sheet_name=sheet)
            final = final[final["currency"].astype(str).str.upper() == ccy]
            key_cols = ["timestamp", "currency"] if sheet.endswith("_params") else ["snapshot_ts", "currency", "instrument_name"]
            check(len(part) == len(final) and keys(part, key_cols) == keys(final, key_cols), f"{spec} {ccy} {sheet}: checkpoint matches final Excel")
            cp_rows[ccy][sheet] = int(len(part))

    report["calibration"][spec] = {"models": stats, "checkpoint_rows": cp_rows, "train_rows": int(len(train)), "test_rows": int(len(test))}

check(holdout_keys["wls"] == holdout_keys["pwls"], "WLS and PWLS use identical deterministic holdout keys")

pricing_root = ROOT / "notebooks" / "pricing_outputs" / "canonical"
for spec in ("wls", "pwls"):
    overall = pd.read_csv(pricing_root / spec / "baseline_overall_metrics.csv")
    check(set(overall["scope"]) == {"BTC", "ETH", "Pooled"}, f"{spec}: pricing scopes present")
    pooled = overall.loc[overall["scope"] == "Pooled"].iloc[0]
    out = {}
    for model in ("black", "heston", "svcj"):
        check(pd.isna(pooled[f"rmse_coin_{model}"]) and pd.isna(pooled[f"mae_coin_{model}"]), f"{spec} {model}: pooled raw-coin metrics undefined")
        rmse_usd = float(pooled[f"rmse_usd_{model}"])
        within = float(pooled[f"within_spread_{model}"])
        check(np.isfinite(rmse_usd), f"{spec} {model}: pooled USD RMSE finite")
        check(0.0 <= within <= 1.0, f"{spec} {model}: within-spread rate valid")
        out[model] = {"rmse_usd": rmse_usd, "within_spread": within}
    report["pricing"][spec] = out

for spec in ("wls", "pwls"):
    book = ROOT / "excel files" / "canonical" / f"hedging_{spec}.xlsx"
    check(book.exists(), f"{spec}: hedging workbook exists")
    test = pd.read_excel(book, sheet_name="test_data")
    panel = pd.read_excel(book, sheet_name="hedge_interval_panel")
    check(set(test["split"].astype(str).str.lower()) == {"test"}, f"{spec}: hedging data is test-only")
    eligible = panel["is_summary_eligible"].fillna(False).astype(bool).to_numpy()
    check(int(eligible.sum()) > 0, f"{spec}: eligible hedge panel non-empty")

    for model in ("black", "heston", "svcj"):
        regular = as_num(test[f"regular_delta_{model}"])
        net = as_num(test[f"net_delta_{model}"])
        price = as_num(test[f"price_{model}"])
        good = np.isfinite(regular) & np.isfinite(net) & np.isfinite(price)
        check(np.allclose(net[good], regular[good] - price[good], rtol=1e-10, atol=1e-10), f"{spec} {model}: net-delta identity")

    F = as_num(panel["F_term"])
    H0 = as_num(panel["H_perp"])
    H1 = as_num(panel["perp_close_t1"])
    scale = as_num(panel["hedge_notional_scale_usd_per_delta"])
    good_scale = np.isfinite(F) & np.isfinite(H0) & np.isfinite(scale) & (F > 0) & (H0 > 0)
    check(np.allclose(scale[good_scale], H0[good_scale] ** 2 / F[good_scale], rtol=1e-12, atol=1e-12), f"{spec}: hedge scale H^2/F")

    unhedged = as_num(panel["option_pnl_coin_short"])
    reductions = []
    for model in ("black", "heston", "svcj"):
        for hedge_type in ("delta", "net_delta"):
            notional = as_num(panel[f"hedge_notional_usd_{model}_{hedge_type}"])
            actual = as_num(panel[f"net_pnl_coin_{model}_{hedge_type}"])
            good = eligible & np.isfinite(notional) & np.isfinite(H0) & np.isfinite(H1) & (H0 > 0) & (H1 > 0) & np.isfinite(unhedged) & np.isfinite(actual)
            check(good.any(), f"{spec} {model} {hedge_type}: finite eligible rows")
            direct = unhedged + notional * (1.0 / H0 - 1.0 / H1)
            check(np.allclose(actual[good], direct[good], rtol=1e-10, atol=1e-12), f"{spec} {model} {hedge_type}: inverse-perpetual P&L identity")
            base = float(np.sqrt(np.mean(unhedged[good] ** 2)))
            hedged = float(np.sqrt(np.mean(actual[good] ** 2)))
            reductions.append(1.0 - hedged / base)
    check(any(x > 0 for x in reductions), f"{spec}: at least one hedge reduces RMSE")
    report["hedging"][spec] = {"eligible_intervals": int(eligible.sum()), "best_rmse_reduction": float(max(reductions))}

manifest = pd.read_csv(ROOT / "notebooks" / "final_outputs" / "canonical" / "manifest.csv")
check(len(manifest) > 0, "final manifest non-empty")
check({"pricing_table", "pricing_figure", "hedging_table", "hedging_figure"} <= set(manifest["artifact_type"]), "final manifest has all artifact classes")
check(all(Path(p).exists() for p in manifest["destination"].astype(str)), "all final artifact destinations exist")

(ROOT / "acceptance_report_10_v2.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print("TEN_SNAPSHOT_ACCEPTANCE_V2_SUCCESS")
print(json.dumps(report, indent=2, default=str))
