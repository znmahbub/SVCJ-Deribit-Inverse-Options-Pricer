from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
SELECTION = json.loads((ROOT / "acceptance_selection_10.json").read_text(encoding="utf-8"))
NAMES = SELECTION["selected_snapshots"]
RX = re.compile(r"deribit_options_snapshot_(\d{8}T\d{6})Z\.csv")
TIMES = [pd.to_datetime(RX.fullmatch(name).group(1), format="%Y%m%dT%H%M%S", utc=True) for name in NAMES]
BLOCKS = {(ts, ccy) for ts in TIMES for ccy in ("BTC", "ETH")}
SHEETS = ("black_params", "heston_params", "svcj_params", "train_data", "test_data")
REPORT = {"selection": SELECTION, "checks": [], "calibration": {}, "pricing": {}, "hedging": {}, "svcj_failures": {}}


def check(ok: bool, msg: str) -> None:
    if not bool(ok):
        raise AssertionError(msg)
    REPORT["checks"].append(msg)


def num(s) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def keyset(df: pd.DataFrame, cols: list[str]) -> set[tuple[str, ...]]:
    x = df.loc[:, cols].copy()
    for col in cols:
        if col == "timestamp" or col.endswith("_ts"):
            x[col] = pd.to_datetime(x[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            x[col] = x[col].astype(str)
    return set(map(tuple, x.itertuples(index=False, name=None)))


check(len(NAMES) == 10 and len(set(TIMES)) == 10 and TIMES == sorted(TIMES), "10 unique chronological snapshots")
holdout_keys = {}

for spec, lam in (("wls", 0.0), ("pwls", 1.0)):
    book = ROOT / "excel files" / "canonical" / f"calibration_{spec}.xlsx"
    check(book.exists(), f"{spec}: final calibration workbook exists")
    check(set(pd.ExcelFile(book).sheet_names) == set(SHEETS), f"{spec}: canonical five-sheet workbook")
    model_stats = {}

    for model in ("black", "heston", "svcj"):
        df = pd.read_excel(book, sheet_name=f"{model}_params")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["currency"] = df["currency"].astype(str).str.upper()
        check(set(zip(df["timestamp"], df["currency"])) == BLOCKS, f"{spec} {model}: all 20 timestamp/currency blocks")
        check(np.allclose(num(df["l2_prev_strength"]), lam, rtol=0, atol=0), f"{spec} {model}: lambda_TS={lam:g}")
        success = df["success"].fillna(False).astype(bool)
        if model == "svcj":
            check(int(success.sum()) >= 18, f"{spec} SVCJ: at least 18/20 optimizer successes")
            check(not df["svcj_martingale_violated"].fillna(False).astype(bool).any(), f"{spec} SVCJ: martingale admissibility")
            check(not df["svcj_contour_violated"].fillna(False).astype(bool).any(), f"{spec} SVCJ: contour admissibility")
            failed_cols = [c for c in ("timestamp", "currency", "message", "nfev", "rmse_fit", "svcj_martingale_violated", "svcj_contour_violated") if c in df.columns]
            failed = df.loc[~success, failed_cols].copy()
            failed.to_csv(ROOT / f"acceptance_failed_svcj_{spec}.csv", index=False)
            REPORT["svcj_failures"][spec] = failed.astype(object).where(pd.notna(failed), None).to_dict("records")
        else:
            check(success.all(), f"{spec} {model}: 20/20 optimizer successes")
        model_stats[model] = {"success": int(success.sum()), "total": int(len(df))}

    train = pd.read_excel(book, sheet_name="train_data")
    test = pd.read_excel(book, sheet_name="test_data")
    for frame in (train, test):
        frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True, errors="coerce")
        frame["currency"] = frame["currency"].astype(str).str.upper()
    check(set(train["split"].astype(str).str.lower()) == {"train"}, f"{spec}: train sheet train-only")
    check(set(test["split"].astype(str).str.lower()) == {"test"}, f"{spec}: test sheet test-only")
    shares = []
    for ts, ccy in sorted(BLOCKS):
        tr = train[(train["snapshot_ts"] == ts) & (train["currency"] == ccy)]
        te = test[(test["snapshot_ts"] == ts) & (test["currency"] == ccy)]
        check(len(tr) > 0 and len(te) > 0, f"{spec} {ccy} {ts}: non-empty train/test")
        check(not (set(tr["instrument_name"].astype(str)) & set(te["instrument_name"].astype(str))), f"{spec} {ccy} {ts}: zero split overlap")
        shares.append(len(tr) / (len(tr) + len(te)))
    check(all(0.65 <= x <= 0.75 for x in shares), f"{spec}: all blocks approximately 70/30")
    holdout_keys[spec] = keyset(test, ["snapshot_ts", "currency", "instrument_name"])

    checkpoint_rows = {}
    for ccy in ("BTC", "ETH"):
        cpdir = ROOT / "excel files" / "canonical" / ".checkpoints" / f"calibration_{spec}" / ccy
        check(cpdir.is_dir(), f"{spec} {ccy}: checkpoint directory")
        checkpoint_rows[ccy] = {}
        for sheet in SHEETS:
            pq = cpdir / f"{sheet}.parquet"
            check(pq.exists(), f"{spec} {ccy}: {sheet}.parquet exists")
            part = pd.read_parquet(pq)
            final = pd.read_excel(book, sheet_name=sheet)
            final = final[final["currency"].astype(str).str.upper() == ccy]
            cols = ["timestamp", "currency"] if sheet.endswith("_params") else ["snapshot_ts", "currency", "instrument_name"]
            check(len(part) == len(final) and keyset(part, cols) == keyset(final, cols), f"{spec} {ccy} {sheet}: Parquet matches final Excel")
            checkpoint_rows[ccy][sheet] = int(len(part))

    REPORT["calibration"][spec] = {
        "models": model_stats,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_share_min": float(min(shares)),
        "train_share_max": float(max(shares)),
        "checkpoint_rows": checkpoint_rows,
    }

check(holdout_keys["wls"] == holdout_keys["pwls"], "WLS/PWLS deterministic holdout keys identical")

pricing_root = ROOT / "notebooks" / "pricing_outputs" / "canonical"
for spec in ("wls", "pwls"):
    overall = pd.read_csv(pricing_root / spec / "baseline_overall_metrics.csv")
    check(set(overall["scope"]) == {"BTC", "ETH", "Pooled"}, f"{spec}: pricing scopes BTC/ETH/Pooled")
    pooled = overall.loc[overall["scope"] == "Pooled"].iloc[0]
    metrics = {}
    for model in ("black", "heston", "svcj"):
        check(pd.isna(pooled[f"rmse_coin_{model}"]) and pd.isna(pooled[f"mae_coin_{model}"]), f"{spec} {model}: pooled raw-coin errors undefined")
        value = float(pooled[f"rmse_usd_{model}"])
        within = float(pooled[f"within_spread_{model}"])
        check(np.isfinite(value), f"{spec} {model}: pooled USD RMSE finite")
        check(0.0 <= within <= 1.0, f"{spec} {model}: within-spread probability valid")
        metrics[model] = {"rmse_usd": value, "within_spread": within}
    REPORT["pricing"][spec] = metrics

for spec in ("wls", "pwls"):
    book = ROOT / "excel files" / "canonical" / f"hedging_{spec}.xlsx"
    check(book.exists(), f"{spec}: hedging workbook exists")
    test = pd.read_excel(book, sheet_name="test_data")
    panel = pd.read_excel(book, sheet_name="hedge_interval_panel")
    check(set(test["split"].astype(str).str.lower()) == {"test"}, f"{spec}: hedging option data test-only")
    eligible = panel["is_summary_eligible"].fillna(False).astype(bool).to_numpy()
    check(int(eligible.sum()) > 0, f"{spec}: non-empty hedge-eligible sample")

    for model in ("black", "heston", "svcj"):
        reg = num(test[f"regular_delta_{model}"])
        net = num(test[f"net_delta_{model}"])
        price = num(test[f"price_{model}"])
        good = np.isfinite(reg) & np.isfinite(net) & np.isfinite(price)
        check(np.allclose(net[good], reg[good] - price[good], rtol=1e-10, atol=1e-10), f"{spec} {model}: net delta identity")

    F = num(panel["F_term"])
    H0 = num(panel["H_perp"])
    H1 = num(panel["perp_close_t1"])
    scale = num(panel["hedge_notional_scale_usd_per_delta"])
    valid_scale = np.isfinite(F) & np.isfinite(H0) & np.isfinite(scale) & (F > 0) & (H0 > 0)
    check(np.allclose(scale[valid_scale], H0[valid_scale] ** 2 / F[valid_scale], rtol=1e-12, atol=1e-12), f"{spec}: H^2/F hedge scale")

    unhedged = num(panel["option_pnl_coin_short"])
    reductions = []
    for model in ("black", "heston", "svcj"):
        for hedge_type in ("delta", "net_delta"):
            notional = num(panel[f"hedge_notional_usd_{model}_{hedge_type}"])
            actual = num(panel[f"net_pnl_coin_{model}_{hedge_type}"])
            good = eligible & np.isfinite(notional) & np.isfinite(H0) & np.isfinite(H1) & (H0 > 0) & (H1 > 0) & np.isfinite(unhedged) & np.isfinite(actual)
            check(good.any(), f"{spec} {model} {hedge_type}: finite eligible rows")
            direct = unhedged + notional * (1.0 / H0 - 1.0 / H1)
            check(np.allclose(actual[good], direct[good], rtol=1e-10, atol=1e-12), f"{spec} {model} {hedge_type}: inverse-perpetual P&L identity")
            base = float(np.sqrt(np.mean(unhedged[good] ** 2)))
            hedged = float(np.sqrt(np.mean(actual[good] ** 2)))
            reductions.append(1.0 - hedged / base)
    check(any(x > 0 for x in reductions), f"{spec}: at least one hedge reduces RMSE")
    REPORT["hedging"][spec] = {"eligible_intervals": int(eligible.sum()), "best_rmse_reduction": float(max(reductions))}

manifest = pd.read_csv(ROOT / "notebooks" / "final_outputs" / "canonical" / "manifest.csv")
check(len(manifest) > 0, "final manifest non-empty")
check({"pricing_table", "pricing_figure", "hedging_table", "hedging_figure"} <= set(manifest["artifact_type"]), "final manifest has all artifact classes")
check(all(Path(p).exists() for p in manifest["destination"].astype(str)), "all final artifact destinations exist")

(ROOT / "acceptance_report_10.json").write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
print("TEN_SNAPSHOT_ACCEPTANCE_SUCCESS")
print(json.dumps(REPORT, indent=2, default=str))
