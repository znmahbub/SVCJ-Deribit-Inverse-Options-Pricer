from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SELECTED = json.loads((ROOT / "acceptance_selection.json").read_text())
FILES = SELECTED["selected_snapshots"]
RX = re.compile(r"deribit_options_snapshot_(\d{8}T\d{6})Z\.csv")
TIMES = [pd.to_datetime(RX.fullmatch(x).group(1), format="%Y%m%dT%H%M%S", utc=True) for x in FILES]
EXPECTED = {(t, c) for t in TIMES for c in ("BTC", "ETH")}
N_EXPECTED_BLOCKS = len(EXPECTED)
HARD_CHECKS: list[str] = []
WARNINGS: list[str] = []
REPORT: dict[str, object] = {
    "selection": {k: v for k, v in SELECTED.items() if k != "selected_snapshots"},
    "hard_checks": HARD_CHECKS,
    "warnings": WARNINGS,
    "calibration": {},
    "pricing": {},
    "hedging": {},
    "economic_diagnostics": {},
    "final": {},
}


def check(cond, msg: str) -> None:
    if not bool(cond):
        raise AssertionError(msg)
    HARD_CHECKS.append(msg)


def warn(cond, msg: str) -> None:
    if not bool(cond):
        WARNINGS.append(msg)


def num(x) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy(float)


def rmse(x) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if len(a) else np.nan


def qstats(x) -> dict[str, float | int]:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    qs = np.quantile(a, [0, .01, .05, .5, .95, .99, 1])
    return {
        "n": int(len(a)),
        "min": float(qs[0]), "p01": float(qs[1]), "p05": float(qs[2]),
        "median": float(qs[3]), "p95": float(qs[4]), "p99": float(qs[5]), "max": float(qs[6]),
    }


def transformed_path_movement(df: pd.DataFrame, params: list[str]) -> dict[str, float | int]:
    vals: list[float] = []
    for _, g in df.sort_values("timestamp").groupby("currency"):
        g = g.sort_values("timestamp")
        for p in params:
            x = pd.to_numeric(g[p], errors="coerce").to_numpy(float)
            if p in {"kappa", "theta", "sigma_v", "v0", "lam", "sigma_y", "ell_v", "sigma"}:
                x = np.log(np.clip(x, 1e-15, None))
            elif p == "rho":
                x = np.arctanh(np.clip(x, -.999999, .999999))
            d = np.abs(np.diff(x))
            vals.extend(d[np.isfinite(d)].tolist())
    return qstats(vals)


check(len(FILES) == 100, "exactly 100 consecutive snapshots selected")
check(all(t2 > t1 for t1, t2 in zip(TIMES, TIMES[1:])), "snapshot timestamps strictly increasing")

progress_markers = {
    "00_check_data.ipynb": ["Inspecting snapshots"],
    "01_run_calibration.ipynb": ["Calibration specifications"],
    "02_pricing_analysis.ipynb": ["Pricing specifications"],
    "03_run_hedging.ipynb": ["Hedging specifications", "Hedging reports"],
    "04_final_analysis.ipynb": ["Copying pricing_table", "Copying hedging_table"],
}
for name, markers in progress_markers.items():
    nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    errors, rendered = [], []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append(out.get("evalue", "error"))
            if "text" in out:
                rendered.append(str(out["text"]))
            for value in out.get("data", {}).values():
                rendered.append(str(value))
    check(not errors, f"{name}: no execution errors")
    blob = "\n".join(rendered)
    check(all(m in blob for m in markers), f"{name}: expected progress/status output emitted")

TEST_FRAMES: dict[str, pd.DataFrame] = {}
HOLDOUT_KEYS: dict[str, set] = {}
PARAM_FRAMES: dict[tuple[str, str], pd.DataFrame] = {}
BOUNDS = {
    "black": {"sigma": (1e-4, 5.0)},
    "heston": {"kappa": (1e-4, 50.0), "theta": (1e-6, 5.0), "sigma_v": (1e-4, 10.0), "rho": (-1, 1), "v0": (1e-6, 5.0)},
    "svcj": {"kappa": (1e-4, 50.0), "theta": (1e-6, 5.0), "sigma_v": (1e-4, 10.0), "rho": (-1, 1), "v0": (1e-6, 5.0), "lam": (1e-6, 10.0), "ell_y": (-5, 5), "sigma_y": (1e-4, 5.0), "ell_v": (1e-6, 10.0)},
}

for spec, lam in [("wls", 0.0), ("pwls", 1.0)]:
    path = ROOT / "excel files" / "canonical" / f"calibration_{spec}.xlsx"
    check(path.exists(), f"{spec}: calibration workbook exists")
    check(set(pd.ExcelFile(path).sheet_names) == {"black_params", "heston_params", "svcj_params", "train_data", "test_data"}, f"{spec}: canonical calibration schema")
    spec_info: dict[str, object] = {"models": {}}
    for model in ("black", "heston", "svcj"):
        d = pd.read_excel(path, sheet_name=f"{model}_params")
        d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
        d["currency"] = d.currency.astype(str).str.upper()
        PARAM_FRAMES[(spec, model)] = d.copy()
        check(len(d) == N_EXPECTED_BLOCKS, f"{spec} {model}: exactly {N_EXPECTED_BLOCKS} BTC/ETH calibration blocks")
        check(set(zip(d.timestamp, d.currency)) == EXPECTED, f"{spec} {model}: every expected timestamp/currency block present")
        check(np.allclose(num(d.l2_prev_strength), lam, rtol=0, atol=0), f"{spec} {model}: lambda_TS={lam:g}")
        success = d.success.fillna(False).astype(bool)
        success_rate = float(success.mean())
        check(success_rate >= .95, f"{spec} {model}: calibration success rate >=95% ({success_rate:.1%})")
        params = list(BOUNDS[model]) + (["rho_j"] if model == "svcj" else [])
        for p in params:
            vals = pd.to_numeric(d.loc[success, p], errors="coerce")
            check(float(np.isfinite(vals).mean()) == 1.0, f"{spec} {model} {p}: finite on successful blocks")
        if model == "svcj":
            check(not d.loc[success, "svcj_martingale_violated"].fillna(False).astype(bool).any(), f"{spec} SVCJ: no martingale violations among successful blocks")
            check(not d.loc[success, "svcj_contour_violated"].fillna(False).astype(bool).any(), f"{spec} SVCJ: no Carr-Madan contour violations among successful blocks")
        proximity = {}
        for p, (lo, hi) in BOUNDS[model].items():
            x = pd.to_numeric(d.loc[success, p], errors="coerce").to_numpy(float)
            x = x[np.isfinite(x)]
            share = np.nan if not len(x) else float(np.mean((x <= lo + .01 * (hi - lo)) | (x >= hi - .01 * (hi - lo))))
            proximity[p] = share
            if np.isfinite(share):
                warn(share < .25, f"{spec} {model} {p}: {share:.1%} of successful estimates lie within 1% of a hard bound")
        feller = float(d.loc[success, "feller_violated"].fillna(False).astype(bool).mean()) if "feller_violated" in d else np.nan
        spec_info["models"][model] = {
            "rows": int(len(d)), "success": int(success.sum()), "success_rate": success_rate,
            "feller_violation_share": feller, "boundary_proximity_share": proximity,
        }
        warn(success_rate >= .99, f"{spec} {model}: >1% calibration failures ({1-success_rate:.1%})")

    tr = pd.read_excel(path, sheet_name="train_data")
    te = pd.read_excel(path, sheet_name="test_data")
    TEST_FRAMES[spec] = te.copy()
    for f in (tr, te):
        f["snapshot_ts"] = pd.to_datetime(f.snapshot_ts, utc=True)
        f["currency"] = f.currency.astype(str).str.upper()
    check(set(tr.split.astype(str).str.lower()) == {"train"}, f"{spec}: train sheet pure")
    check(set(te.split.astype(str).str.lower()) == {"test"}, f"{spec}: test sheet pure")
    shares = []
    for ts, c in sorted(EXPECTED):
        a = tr[(tr.snapshot_ts == ts) & (tr.currency == c)]
        b = te[(te.snapshot_ts == ts) & (te.currency == c)]
        check(len(a) > 0 and len(b) > 0, f"{spec} {c} {ts}: train and test are both nonempty")
        check(not (set(a.instrument_name.astype(str)) & set(b.instrument_name.astype(str))), f"{spec} {c} {ts}: no train/test instrument overlap")
        shares.append(len(a) / (len(a) + len(b)))
    check(all(.65 <= x <= .75 for x in shares), f"{spec}: every block is approximately 70/30")
    HOLDOUT_KEYS[spec] = set(map(tuple, te[["snapshot_ts", "currency", "instrument_name"]].astype(str).itertuples(index=False, name=None)))
    spec_info.update({"train_rows": int(len(tr)), "test_rows": int(len(te)), "train_share_stats": qstats(shares)})
    REPORT["calibration"][spec] = spec_info

check(HOLDOUT_KEYS["wls"] == HOLDOUT_KEYS["pwls"], "WLS/PWLS use identical deterministic test holdout")

movement = {}
param_map = {
    "black": ["sigma"],
    "heston": ["kappa", "theta", "sigma_v", "rho", "v0"],
    "svcj": ["kappa", "theta", "sigma_v", "rho", "v0", "lam", "ell_y", "sigma_y", "ell_v", "rho_j"],
}
for model, params in param_map.items():
    movement[model] = {spec: transformed_path_movement(PARAM_FRAMES[(spec, model)], params) for spec in ("wls", "pwls")}
    w, p = movement[model]["wls"].get("median", np.nan), movement[model]["pwls"].get("median", np.nan)
    if np.isfinite(w) and w > 0 and np.isfinite(p):
        ratio = float(p / w)
        movement[model]["pwls_to_wls_median_movement_ratio"] = ratio
        warn(ratio <= 1.25, f"PWLS {model} transformed parameter path is unexpectedly less smooth than WLS (median movement ratio {ratio:.2f})")
REPORT["economic_diagnostics"]["parameter_path_movement"] = movement

PROOT = ROOT / "notebooks" / "pricing_outputs" / "canonical"
pricing_diag = {}
for spec in ("wls", "pwls"):
    overall = pd.read_csv(PROOT / spec / "baseline_overall_metrics.csv")
    snap = pd.read_csv(PROOT / spec / "snapshot_time_series_metrics.csv")
    wins = pd.read_csv(PROOT / spec / "snapshot_model_win_rates.csv")
    check(set(overall.scope) == {"BTC", "ETH", "Pooled"}, f"{spec}: pricing scopes BTC/ETH/Pooled")
    pooled = overall[overall.scope == "Pooled"].iloc[0]
    te = TEST_FRAMES[spec]
    F, mid = num(te.F0), num(te.mid_price_clean)
    metrics = {}
    for model in ("black", "heston", "svcj"):
        prices = num(te[f"price_{model}"])
        check(np.isfinite(prices).all(), f"{spec} {model}: all persisted test prices finite")
        check(((prices >= -1e-12) & (prices <= 1 + 1e-12)).all(), f"{spec} {model}: inverse option prices lie in [0,1]")
        check(pd.isna(pooled[f"rmse_coin_{model}"]) and pd.isna(pooled[f"mae_coin_{model}"]), f"{spec} {model}: pooled BTC/ETH raw-coin errors are undefined")
        direct = rmse(F * (prices - mid))
        reported = float(pooled[f"rmse_usd_{model}"])
        check(np.isclose(direct, reported, rtol=1e-10, atol=1e-12), f"{spec} {model}: pooled USD RMSE reconstructs exactly")
        within = float(pooled[f"within_spread_{model}"])
        check(0 <= within <= 1, f"{spec} {model}: within-spread rate is a probability")
        metrics[model] = {"rmse_usd": reported, "mae_usd": float(pooled[f"mae_usd_{model}"]), "within_spread": within}

        pairs = te[["snapshot_ts", "currency", "expiry_datetime", "strike", "F0", "option_type", f"price_{model}"]].copy()
        pairs["option_type"] = pairs.option_type.astype(str).str.lower()
        calls = pairs[pairs.option_type == "call"].rename(columns={f"price_{model}": "call"})
        puts = pairs[pairs.option_type == "put"].rename(columns={f"price_{model}": "put"})
        keys = ["snapshot_ts", "currency", "expiry_datetime", "strike", "F0"]
        pp = calls[keys + ["call"]].merge(puts[keys + ["put"]], on=keys, how="inner")
        if len(pp):
            parity = np.abs(num(pp.call) - num(pp.put) - (1 - num(pp.strike) / num(pp.F0)))
            check(float(np.nanmax(parity)) < 1e-8, f"{spec} {model}: inverse put-call parity holds on matched test pairs")

    ranking = sorted(metrics, key=lambda m: metrics[m]["rmse_usd"])
    warn(metrics["heston"]["rmse_usd"] < metrics["black"]["rmse_usd"], f"{spec}: Heston does not improve pooled USD RMSE over Black")
    warn(metrics["svcj"]["rmse_usd"] < metrics["black"]["rmse_usd"], f"{spec}: SVCJ does not improve pooled USD RMSE over Black")
    pricing_diag[spec] = {
        "overall": metrics,
        "rmse_ranking_best_to_worst": ranking,
        "snapshot_win_rates": wins.to_dict(orient="records"),
        "snapshot_rmse_stats": {m: qstats(pd.to_numeric(snap[f"rmse_{m}"], errors="coerce")) for m in ("black", "heston", "svcj")},
    }
REPORT["pricing"] = pricing_diag

HROOT = ROOT / "notebooks" / "hedging_outputs" / "canonical"
hedge_diag = {}
for spec in ("wls", "pwls"):
    path = ROOT / "excel files" / "canonical" / f"hedging_{spec}.xlsx"
    check(path.exists(), f"{spec}: hedging workbook exists")
    te = pd.read_excel(path, sheet_name="test_data")
    panel = pd.read_excel(path, sheet_name="hedge_interval_panel")
    pv = pd.read_excel(path, sheet_name="price_verification")
    check(set(te.split.astype(str).str.lower()) == {"test"}, f"{spec}: hedging option sheet is test-only")
    check(set(panel.split.astype(str).str.lower()) <= {"test"}, f"{spec}: hedge interval panel is test-only")
    panel["snapshot_ts"] = pd.to_datetime(panel.snapshot_ts, utc=True, errors="coerce")
    eligible = panel.is_summary_eligible.fillna(False).astype(bool)
    check(int(eligible.sum()) > 0, f"{spec}: nonempty hedge-eligible sample")
    nonterminal = panel.snapshot_ts < max(TIMES)
    coverage = float(eligible[nonterminal].mean()) if nonterminal.any() else 0.0
    warn(coverage >= .60, f"{spec}: hedge eligibility among nonterminal rows is only {coverage:.1%}")

    price_diag = {}
    for model in ("black", "heston", "svcj"):
        q = pv[pv.model.astype(str).str.lower() == model]
        check(len(q) == N_EXPECTED_BLOCKS, f"{spec} {model}: price verification has all {N_EXPECTED_BLOCKS} blocks")
        check(q.error.isna().all(), f"{spec} {model}: no hedging repricing errors")
        check((pd.to_numeric(q.n_valid_prices) == pd.to_numeric(q.n_rows)).all(), f"{spec} {model}: all test prices finite in hedging repricer")
        maxdiff = float(pd.to_numeric(q.max_abs_price_diff, errors="coerce").max())
        check(maxdiff < 1e-10, f"{spec} {model}: hedging repricing matches calibration prices <1e-10 coin")
        reg, net, price = num(te[f"regular_delta_{model}"]), num(te[f"net_delta_{model}"]), num(te[f"price_{model}"])
        m = np.isfinite(reg) & np.isfinite(net) & np.isfinite(price)
        check(np.allclose(net[m], reg[m] - price[m], rtol=1e-10, atol=1e-10), f"{spec} {model}: net delta = regular delta - V_coin")
        price_diag[model] = maxdiff

    F, H0, H1 = num(panel.F_term), num(panel.H_perp), num(panel.perp_close_t1)
    scale = num(panel.hedge_notional_scale_usd_per_delta)
    valid = np.isfinite(F) & np.isfinite(H0) & np.isfinite(scale) & (F > 0) & (H0 > 0)
    check(np.allclose(scale[valid], H0[valid] ** 2 / F[valid], rtol=1e-12, atol=1e-12), f"{spec}: hedge scale H^2/F")
    keys = ["snapshot_ts", "currency", "instrument_name"]
    delta_cols = [f"{kind}_delta_{m}" for m in ("black", "heston", "svcj") for kind in ("regular", "net")]
    te["snapshot_ts"] = pd.to_datetime(te.snapshot_ts, utc=True, errors="coerce")
    merged = panel.merge(te[keys + delta_cols], on=keys, how="left", validate="many_to_one")
    elig = merged.is_summary_eligible.fillna(False).astype(bool).to_numpy()
    for model in ("black", "heston", "svcj"):
        for hedge_type, delta_col in [("delta", f"regular_delta_{model}"), ("net_delta", f"net_delta_{model}")]:
            delta = num(merged[delta_col])
            N = num(merged[f"hedge_notional_usd_{model}_{hedge_type}"])
            mask = elig & np.isfinite(delta) & np.isfinite(N) & np.isfinite(scale)
            check(mask.any(), f"{spec} {model} {hedge_type}: eligible finite rows exist")
            check(np.allclose(N[mask], scale[mask] * delta[mask], rtol=1e-10, atol=1e-10), f"{spec} {model} {hedge_type}: N_USD=(H^2/F)delta")
            un = num(merged.option_pnl_coin_short)
            actual = num(merged[f"net_pnl_coin_{model}_{hedge_type}"])
            direct = un + N * (1 / H0 - 1 / H1)
            pm = mask & np.isfinite(H1) & (H1 > 0) & np.isfinite(un) & np.isfinite(actual)
            check(np.allclose(actual[pm], direct[pm], rtol=1e-10, atol=1e-12), f"{spec} {model} {hedge_type}: inverse-perpetual hedge-error identity")

    headline = pd.read_csv(HROOT / spec / "headline_test.csv")
    sample = pd.read_csv(HROOT / spec / "sample_construction.csv")
    check(len(headline) > 0, f"{spec}: headline hedge table nonempty")
    check(set(headline.model.astype(str).str.lower()) == {"black", "heston", "svcj"}, f"{spec}: headline includes all models")
    check(set(headline.hedge_type.astype(str).str.lower()) == {"delta", "net_delta"}, f"{spec}: headline includes both hedge rules")
    pos_rmse = float((pd.to_numeric(headline.rmse_reduction, errors="coerce") > 0).mean())
    pos_mae = float((pd.to_numeric(headline.mae_reduction, errors="coerce") > 0).mean())
    warn(pos_rmse >= .50, f"{spec}: fewer than half of currency/model/hedge cells reduce RMSE ({pos_rmse:.1%})")
    warn(pos_mae >= .50, f"{spec}: fewer than half of currency/model/hedge cells reduce MAE ({pos_mae:.1%})")
    dt, basis = num(panel.loc[eligible, "dt_hours"]), num(panel.loc[eligible, "basis_abs"])
    notionals: list[float] = []
    for model in ("black", "heston", "svcj"):
        for ht in ("delta", "net_delta"):
            notionals.extend(np.abs(num(panel.loc[eligible, f"hedge_notional_usd_{model}_{ht}"])).tolist())
    warn(qstats(dt).get("p99", 0) < 48, f"{spec}: 99th percentile hedge interval exceeds 48h")
    warn(qstats(basis).get("p99", 0) < .20, f"{spec}: 99th percentile absolute perp/term basis exceeds 20%")
    hedge_diag[spec] = {
        "panel_rows": int(len(panel)), "eligible_intervals": int(eligible.sum()), "nonterminal_eligibility_rate": coverage,
        "max_repricing_diff_by_model": price_diag, "headline": headline.to_dict(orient="records"),
        "sample_construction": sample.to_dict(orient="records"), "dt_hours": qstats(dt), "basis_abs": qstats(basis),
        "abs_hedge_notional_usd": qstats(notionals), "positive_rmse_reduction_share": pos_rmse,
        "positive_mae_reduction_share": pos_mae,
    }
REPORT["hedging"] = hedge_diag

final = ROOT / "notebooks" / "final_outputs" / "canonical"
manifest = pd.read_csv(final / "manifest.csv")
check(len(manifest) > 0, "final manifest nonempty")
check({"pricing_table", "pricing_figure", "hedging_table", "hedging_figure"} <= set(manifest.artifact_type), "final manifest contains all artifact classes")
check(all(Path(x).exists() for x in manifest.destination.astype(str)), "every final manifest destination exists")
check(manifest.destination.astype(str).str.contains("/hedging/wls/").any(), "final package includes WLS hedging")
check(manifest.destination.astype(str).str.contains("/hedging/pwls/").any(), "final package includes PWLS hedging")
REPORT["final"] = {"manifest_rows": int(len(manifest)), "artifact_counts": manifest.artifact_type.value_counts().to_dict()}

REPORT["verdict"] = "PASS" if not WARNINGS else "PASS_WITH_WARNINGS"
(ROOT / "acceptance_report_100.json").write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
pd.DataFrame([{"warning": w} for w in WARNINGS]).to_csv(ROOT / "acceptance_warnings_100.csv", index=False)
print("FINAL_100_SNAPSHOT_ACCEPTANCE_COMPLETE")
print("VERDICT:", REPORT["verdict"])
print("HARD_CHECKS:", len(HARD_CHECKS))
print("WARNINGS:", len(WARNINGS))
for w in WARNINGS:
    print("WARNING:", w)
print("SELECTION:", REPORT["selection"])
print("PRICING:", json.dumps(REPORT["pricing"], indent=2, default=str))
print("HEDGING:", json.dumps(REPORT["hedging"], indent=2, default=str))
print("PARAMETER_MOVEMENT:", json.dumps(REPORT["economic_diagnostics"]["parameter_path_movement"], indent=2, default=str))
