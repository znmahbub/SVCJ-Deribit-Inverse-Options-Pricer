from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_block(text: str, start_marker: str, end_marker: str, replacement: str, label: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"{label}: start marker not found: {start_marker!r}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"{label}: end marker not found: {end_marker!r}")
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def append_once(text: str, marker: str, block: str) -> str:
    if marker in text:
        return text
    return text.rstrip() + "\n\n" + block.strip() + "\n"


# ---------------------------------------------------------------------------
# Task 10 — unit-consistent pricing metrics
# ---------------------------------------------------------------------------
chapter3_path = ROOT / "notebooks/_lib/chapter3_final.py"
chapter3 = chapter3_path.read_text(encoding="utf-8")

metric_block = r'''
def option_weighted_rmse(errors) -> float:
    """RMSE over individual option quotes, after dropping non-finite errors."""
    values = pd.to_numeric(pd.Series(errors), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values * values))) if len(values) else float("nan")


def option_weighted_mae(errors) -> float:
    """MAE over individual option quotes, after dropping non-finite errors."""
    values = pd.to_numeric(pd.Series(errors), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(np.abs(values))) if len(values) else float("nan")


def mean_snapshot_rmse(
    frame: pd.DataFrame,
    *,
    error_col: str,
    snapshot_col: str = "snapshot_ts",
) -> float:
    """Equal-weighted mean of snapshot-specific quote RMSE values."""
    if frame.empty or error_col not in frame.columns or snapshot_col not in frame.columns:
        return float("nan")
    values: list[float] = []
    for _, group in frame.groupby(snapshot_col, dropna=False):
        rmse = option_weighted_rmse(group[error_col])
        if np.isfinite(rmse):
            values.append(rmse)
    return float(np.mean(values)) if values else float("nan")


def add_common_unit_errors(data: pd.DataFrame) -> pd.DataFrame:
    """Add explicit coin, USD-equivalent, and spread-normalised pricing errors.

    BTC and ETH coin premiums are economically different units, so raw coin
    errors are retained only for asset-specific reporting. Cross-asset pooled
    reporting uses ``F0 * error_coin`` (USD-equivalent at the observation
    forward) and the dimensionless error measured in full quoted spreads.
    """
    out = data.copy()
    required = {"mid_price_clean", "bid_price", "ask_price"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"pricing metrics missing required columns: {sorted(missing)}")

    if "F0" not in out.columns:
        if "futures_price" in out.columns:
            out["F0"] = pd.to_numeric(out["futures_price"], errors="coerce")
        else:
            raise ValueError("pricing metrics require F0 or futures_price for common-unit reporting")

    for col in ["F0", "mid_price_clean", "bid_price", "ask_price"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    spread = out["ask_price"] - out["bid_price"]
    valid_spread = np.isfinite(spread) & (spread > 0.0)
    out["spread_coin"] = np.where(valid_spread, spread, np.nan)

    for model in ["black", "heston", "svcj"]:
        price_col = f"price_{model}"
        if price_col not in out.columns:
            continue
        out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
        err = out[price_col] - out["mid_price_clean"]
        out[f"error_coin_{model}"] = err
        out[f"abs_error_coin_{model}"] = err.abs()
        out[f"sq_error_coin_{model}"] = err * err
        out[f"error_usd_{model}"] = out["F0"] * err
        out[f"error_spread_units_{model}"] = err / out["spread_coin"]
        valid_quote = (
            np.isfinite(out[price_col])
            & np.isfinite(out["bid_price"])
            & np.isfinite(out["ask_price"])
            & (out["ask_price"] >= out["bid_price"])
        )
        within = np.where(
            valid_quote,
            ((out[price_col] >= out["bid_price"]) & (out[price_col] <= out["ask_price"])).astype(float),
            np.nan,
        )
        out[f"within_spread_{model}"] = within

        # Backward-compatible aliases used by existing bucket/time-series code.
        out[f"abs_err_{model}"] = out[f"abs_error_coin_{model}"]
        out[f"sq_err_{model}"] = out[f"sq_error_coin_{model}"]
        out[f"within_{model}"] = out[f"within_spread_{model}"]
    return out


def load_baseline_workbook(
    baseline_path: Path,
    *,
    evaluation_split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    black_params = pd.read_excel(baseline_path, sheet_name="black_params")
    heston_params = pd.read_excel(baseline_path, sheet_name="heston_params")
    svcj_params = pd.read_excel(baseline_path, sheet_name="svcj_params")
    for frame in [black_params, heston_params, svcj_params]:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    evaluation_split = str(evaluation_split).lower()
    if evaluation_split not in {"train", "validation", "test"}:
        raise ValueError("evaluation_split must be train, validation, or test")
    test_data = pd.read_excel(baseline_path, sheet_name=f"{evaluation_split}_data")
    required = {
        "snapshot_ts",
        "currency",
        "time_to_maturity",
        "bid_price",
        "ask_price",
        "mid_price_clean",
        "price_black",
        "price_heston",
        "price_svcj",
    }
    missing = required - set(test_data.columns)
    if missing:
        raise ValueError(f"{evaluation_split}_data missing required columns: {sorted(missing)}")
    test_data["snapshot_ts"] = pd.to_datetime(test_data["snapshot_ts"], utc=True)
    test_data["currency"] = test_data["currency"].astype(str).str.upper()
    test_data["evaluation_split"] = evaluation_split
    test_data = add_common_unit_errors(test_data)
    return black_params, heston_params, svcj_params, test_data


def compute_overall_baseline_metrics(test_data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Build asset-level coin metrics and unit-coherent pooled metrics.

    Asset rows report coin-unit errors as well as USD-equivalent and spread-unit
    errors. The pooled BTC+ETH row intentionally leaves raw coin RMSE/MAE blank
    because BTC and ETH are different economic units.
    """
    data = add_common_unit_errors(test_data)
    rows: list[dict[str, object]] = []
    scopes = [
        ("BTC", data.loc[data["currency"] == "BTC"]),
        ("ETH", data.loc[data["currency"] == "ETH"]),
        ("Pooled", data),
    ]
    for scope, frame in scopes:
        row: dict[str, object] = {"scope": scope, "n_test_options": int(len(frame))}
        for model in ["black", "heston", "svcj"]:
            coin_col = f"error_coin_{model}"
            usd_col = f"error_usd_{model}"
            spread_col = f"error_spread_units_{model}"

            if scope != "Pooled":
                row[f"rmse_coin_{model}"] = option_weighted_rmse(frame[coin_col])
                row[f"mae_coin_{model}"] = option_weighted_mae(frame[coin_col])
                row[f"snapshot_mean_rmse_coin_{model}"] = mean_snapshot_rmse(frame, error_col=coin_col)
            else:
                row[f"rmse_coin_{model}"] = np.nan
                row[f"mae_coin_{model}"] = np.nan
                row[f"snapshot_mean_rmse_coin_{model}"] = np.nan

            row[f"rmse_usd_{model}"] = option_weighted_rmse(frame[usd_col])
            row[f"mae_usd_{model}"] = option_weighted_mae(frame[usd_col])
            row[f"snapshot_mean_rmse_usd_{model}"] = mean_snapshot_rmse(frame, error_col=usd_col)
            row[f"rmse_spread_units_{model}"] = option_weighted_rmse(frame[spread_col])
            row[f"mae_spread_units_{model}"] = option_weighted_mae(frame[spread_col])
            row[f"snapshot_mean_rmse_spread_units_{model}"] = mean_snapshot_rmse(frame, error_col=spread_col)
            row[f"within_spread_{model}"] = float(pd.to_numeric(frame[f"within_spread_{model}"], errors="coerce").mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "baseline_overall_metrics.csv", index=False)
    return out
'''

chapter3 = replace_block(
    chapter3,
    "def load_baseline_workbook(",
    "def compute_snapshot_metrics(",
    metric_block,
    "chapter3 metric/load block",
)
chapter3_path.write_text(chapter3, encoding="utf-8")

analysis_path = ROOT / "notebooks/_lib/chapter3_analysis.py"
analysis = analysis_path.read_text(encoding="utf-8")
analysis = replace_once(
    analysis,
    "HOLDOUT_INFERENCE_LABEL = (\n    \"snapshot-level cross-sectional holdout loss differential with HAC standard errors\"\n)\n",
    "HOLDOUT_INFERENCE_LABEL = (\n    \"snapshot-level cross-sectional holdout loss differential with HAC standard errors\"\n)\n\nPRICING_ERROR_UNIT_NOTE = (\n    \"Snapshot RMSE/MAE in this module are coin-denominated and therefore interpreted \"\n    \"within currency. Cross-asset pooled headline metrics must use the common-unit \"\n    \"helpers in chapter3_final (USD-equivalent or spread units).\"\n)\n",
    "chapter3 analysis metric-unit note",
)
# Preserve existing schema while exposing explicit aliases for notebook consumers.
analysis = replace_once(
    analysis,
    "        agg[\"rmse\"] = np.sqrt(agg[\"mse\"])\n        agg[\"rmse_over_mean_price\"] = agg[\"rmse\"] / (agg[\"mean_price\"].abs() + EPS)\n",
    "        agg[\"rmse\"] = np.sqrt(agg[\"mse\"])\n        agg[\"rmse_coin\"] = agg[\"rmse\"]\n        agg[\"mae_coin\"] = agg[\"mae\"]\n        agg[\"rmse_over_mean_price\"] = agg[\"rmse\"] / (agg[\"mean_price\"].abs() + EPS)\n",
    "chapter3 analysis unit aliases",
)
analysis_path.write_text(analysis, encoding="utf-8")

final_nb_path = ROOT / "notebooks/calibration_results_final.ipynb"
nb = json.loads(final_nb_path.read_text(encoding="utf-8"))
for cell in nb.get("cells", []):
    if cell.get("id") == "82c47e29":
        cell["source"] = [
            "## 2. Load the provisional penalized baseline (`lambda_TS = 100`) and build test-holdout pricing diagnostics\n",
            "\n",
            "BTC and ETH retain their own coin-denominated RMSE/MAE. The combined BTC+ETH headline row does **not** pool raw coin units; it reports USD-equivalent errors (`F0 × coin error`), spread-normalised errors, and within-spread rates. Option-weighted RMSE and the equal-snapshot mean RMSE are exported under separate names. Formal lambda selection is deferred; `lambda_TS = 100` is the provisional penalized specification."
        ]
final_nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Task 11 — neutral perpetual-price proxy provenance
# ---------------------------------------------------------------------------
backfill_path = ROOT / "src/backfill_futures_marks.py"
backfill = backfill_path.read_text(encoding="utf-8")

perp_block = r'''
def _perp_price_kind(resolution: str) -> str:
    res = str(resolution).strip()
    return "nearest_1m_candle_open_close" if res == "1" else f"nearest_{res}_candle_open_close"


def _migrate_legacy_perp_schema(existing: pd.DataFrame) -> pd.DataFrame:
    """Normalize historical files without pretending a candle proxy was a mark."""
    out = existing.copy()
    if out.empty:
        return out
    if "timestamp" in out.columns and "obs_datetime" not in out.columns:
        out["obs_datetime"] = out["timestamp"]
    if "currency" in out.columns and "coin" not in out.columns:
        out["coin"] = out["currency"]
    if "perp_futures_price_proxy" not in out.columns and "perp_futures_mark_price" in out.columns:
        out["perp_futures_price_proxy"] = pd.to_numeric(out["perp_futures_mark_price"], errors="coerce")
        out["price_kind"] = "legacy_mark_named_candle_proxy"
        if "price_source" not in out.columns:
            if "data_source" in out.columns:
                out["price_source"] = out["data_source"]
            else:
                out["price_source"] = "legacy_unknown"
    return out


def build_perp_incremental(
    combos: pd.DataFrame,
    existing_perp: pd.DataFrame,
    skip_perp_backfill: bool,
    sleep_seconds: float,
    max_gap_seconds: Optional[float],
    perp_resolution: str,
    perp_max_bars_per_call: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if combos.empty:
        return _migrate_legacy_perp_schema(existing_perp), pd.DataFrame()

    perp_requests = combos.copy()
    perp_requests["obs_datetime"] = perp_requests["timestamp"]
    perp_requests["coin"] = perp_requests["currency"]
    perp_requests["instrument_name"] = perp_requests["coin"].astype(str) + "-PERPETUAL"
    key_cols = ["obs_datetime", "coin"]

    existing_norm = _migrate_legacy_perp_schema(existing_perp)
    needs_refresh = perp_requests.copy()
    if not existing_norm.empty and all(c in existing_norm.columns for c in key_cols):
        existing_subset = select_rows_by_keys(existing_norm, perp_requests, key_cols)
        if not existing_subset.empty:
            merged = perp_requests.merge(existing_subset, on=key_cols, how="left", suffixes=("", "__existing"))
            existing_price = pd.to_numeric(merged.get("perp_futures_price_proxy"), errors="coerce")
            existing_ok = existing_price.notna()
            needs_refresh = merged.loc[~existing_ok, perp_requests.columns].copy()

    if needs_refresh.empty:
        return existing_norm.copy(), pd.DataFrame()

    price_kind = _perp_price_kind(perp_resolution)
    if skip_perp_backfill:
        perp_new = needs_refresh.copy()
        perp_new["perp_futures_price_proxy"] = pd.NA
        perp_new["perp_futures_best_bid"] = pd.NA
        perp_new["perp_futures_best_ask"] = pd.NA
        perp_new["price_source"] = "skipped"
        perp_new["price_kind"] = price_kind
        for col in [
            "candle_found",
            "candle_tick_ms",
            "candle_open_time_ms",
            "candle_close_time_ms",
            "candle_gap_open_seconds",
            "candle_gap_close_seconds",
            "candle_price_choice",
            "candle_price_timestamp_ms",
            "candle_price_gap_seconds",
            "candle_open",
            "candle_high",
            "candle_low",
            "candle_close",
            "candle_volume",
            "candle_cost",
            "accepted_under_max_gap",
            "candle_resolution",
            "candle_price_used",
        ]:
            perp_new[col] = pd.NA
    else:
        perp_new = fetch_tradingview_candles_for_requests(
            request_rows=needs_refresh,
            instrument_col="instrument_name",
            timestamp_ms_col="snapshot_timestamp_ms",
            resolution=perp_resolution,
            sleep_seconds=sleep_seconds,
            max_gap_seconds=max_gap_seconds,
            max_bars_per_call=perp_max_bars_per_call,
            progress_desc="Fetching perp candles",
        )
        perp_new["perp_futures_price_proxy"] = np.where(
            perp_new["accepted_under_max_gap"].fillna(False),
            pd.to_numeric(perp_new["candle_price_used"], errors="coerce"),
            np.nan,
        )
        perp_new["perp_futures_best_bid"] = pd.NA
        perp_new["perp_futures_best_ask"] = pd.NA
        perp_new["price_source"] = "deribit_tradingview_chart"
        perp_new["price_kind"] = price_kind

    if "timestamp" in perp_new.columns:
        perp_new = perp_new.drop(columns=["timestamp"], errors="ignore")
    if "currency" in perp_new.columns:
        perp_new = perp_new.drop(columns=["currency"], errors="ignore")

    existing_keep = exclude_rows_by_keys(existing_norm, needs_refresh, key_cols)
    updated_perp = concat_dedup_sort(
        existing_df=existing_keep,
        new_df=perp_new,
        key_cols=key_cols,
        sort_cols=["obs_datetime", "coin"],
    )
    return updated_perp, perp_new
'''

backfill = replace_block(
    backfill,
    "def build_perp_incremental(",
    "# -----------------------------\n# Main",
    perp_block,
    "perpetual incremental builder",
)
backfill_path.write_text(backfill, encoding="utf-8")

hedging_path = ROOT / "src/hedging_analysis.py"
hedging = hedging_path.read_text(encoding="utf-8")
if "import warnings\n" not in hedging:
    hedging = replace_once(hedging, "import re\n", "import re\nimport warnings\n", "hedging warnings import")

load_perp_block = r'''
def load_perp_history(data_dir: Path, filename: str | None = None) -> tuple[pd.DataFrame, str]:
    path, used = _resolve_data_file(
        data_dir,
        filename,
        ("perpetual_futures_prices_history.csv", "perpetual_futures_prices.csv"),
    )
    df = pd.read_csv(path)
    base_required = {"obs_datetime", "coin"}
    if not base_required.issubset(df.columns):
        raise ValueError(f"{path} must contain obs_datetime and coin.")

    out = df.copy()
    out["currency"] = out["coin"].astype(str).str.upper()
    out["timestamp"] = pd.to_datetime(out["obs_datetime"], utc=True, errors="coerce")

    if "perp_futures_price_proxy" in out.columns:
        out["close_price"] = pd.to_numeric(out["perp_futures_price_proxy"], errors="coerce")
        if "price_source" not in out.columns:
            out["price_source"] = "unknown_proxy_source"
        if "price_kind" not in out.columns:
            out["price_kind"] = "unspecified_price_proxy"
        out["legacy_price_schema"] = False
    elif "perp_futures_mark_price" in out.columns:
        warnings.warn(
            f"{path.name} uses the legacy perp_futures_mark_price name for a candle-derived proxy; "
            "load is supported for backward compatibility only.",
            UserWarning,
            stacklevel=2,
        )
        out["close_price"] = pd.to_numeric(out["perp_futures_mark_price"], errors="coerce")
        if "data_source" in out.columns:
            out["price_source"] = out["data_source"].astype(str)
        else:
            out["price_source"] = "legacy_unknown"
        out["price_kind"] = "legacy_mark_named_candle_proxy"
        out["legacy_price_schema"] = True
    else:
        raise ValueError(
            f"{path} must contain perp_futures_price_proxy (preferred) or the legacy perp_futures_mark_price column."
        )

    # Fallback only when the canonical/legacy proxy itself is missing.
    if "perp_futures_best_bid" in out.columns and "perp_futures_best_ask" in out.columns:
        bid = pd.to_numeric(out["perp_futures_best_bid"], errors="coerce")
        ask = pd.to_numeric(out["perp_futures_best_ask"], errors="coerce")
        valid_mid = np.isfinite(bid) & np.isfinite(ask) & (ask >= bid)
        missing_close = ~np.isfinite(out["close_price"])
        out.loc[missing_close & valid_mid, "close_price"] = 0.5 * (bid[missing_close & valid_mid] + ask[missing_close & valid_mid])
        out.loc[missing_close & valid_mid, "price_kind"] = "best_bid_ask_mid_fallback"
    if "trade_price" in out.columns:
        trade_price = pd.to_numeric(out["trade_price"], errors="coerce")
        missing_close = ~np.isfinite(out["close_price"])
        out.loc[missing_close, "close_price"] = trade_price[missing_close]
        out.loc[missing_close & np.isfinite(trade_price), "price_kind"] = "trade_price_fallback"

    out["proxy_gap_seconds"] = pd.to_numeric(out.get("candle_price_gap_seconds", np.nan), errors="coerce")
    out["proxy_price_choice"] = out.get("candle_price_choice", pd.Series(pd.NA, index=out.index))
    out["proxy_resolution"] = out.get("candle_resolution", pd.Series(pd.NA, index=out.index))

    out = out.dropna(subset=["currency", "timestamp", "close_price"]).copy()
    out = out.sort_values(["currency", "timestamp"]).drop_duplicates(
        subset=["currency", "timestamp"], keep="first"
    ).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path} contains no usable perpetual-futures proxy rows after parsing.")
    keep = [
        "currency",
        "timestamp",
        "close_price",
        "price_source",
        "price_kind",
        "proxy_gap_seconds",
        "proxy_price_choice",
        "proxy_resolution",
        "legacy_price_schema",
    ]
    return out[keep], used
'''

hedging = replace_block(
    hedging,
    "def load_perp_history(",
    "def load_funding_history(",
    load_perp_block,
    "perp history loader",
)

old_notes = '''        ("hedge_price_file", used_perp_file),
        ("funding_file", used_funding_file),'''
new_notes = '''        ("hedge_price_file", used_perp_file),
        ("hedge_price_source", ",".join(sorted(perp["price_source"].dropna().astype(str).unique())) if "price_source" in perp.columns else None),
        ("hedge_price_kind", ",".join(sorted(perp["price_kind"].dropna().astype(str).unique())) if "price_kind" in perp.columns else None),
        ("hedge_price_legacy_schema", bool(perp["legacy_price_schema"].fillna(False).astype(bool).any()) if "legacy_price_schema" in perp.columns else None),
        ("funding_file", used_funding_file),'''
hedging = replace_once(hedging, old_notes, new_notes, "hedge proxy notes")
hedging_path.write_text(hedging, encoding="utf-8")

viewer_path = ROOT / "notebooks/_lib/futures_viewer.py"
viewer = viewer_path.read_text(encoding="utf-8")
viewer = replace_once(
    viewer,
    '    perp_price_col = "perp_futures_mark_price" if "perp_futures_mark_price" in perp.columns else "candle_price_used"\n',
    '    if "perp_futures_price_proxy" in perp.columns:\n        perp_price_col = "perp_futures_price_proxy"\n    elif "perp_futures_mark_price" in perp.columns:\n        perp_price_col = "perp_futures_mark_price"  # legacy compatibility\n    else:\n        perp_price_col = "candle_price_used"\n',
    "futures viewer proxy preference",
)
viewer_path.write_text(viewer, encoding="utf-8")

readme_path = ROOT / "README.md"
readme = readme_path.read_text(encoding="utf-8")
readme = append_once(
    readme,
    "## Perpetual hedge price provenance",
    r'''
## Perpetual hedge price provenance

The historical hedge input in `data/perpetual_futures_prices.csv` is a **price proxy**, not an exchange-provided historical Deribit mark series. New backfills use the neutral field `perp_futures_price_proxy`: the closer in timestamp of the open or close of the matched Deribit TradingView candle (one-minute resolution in the thesis baseline), subject to the configured maximum-gap filter. `price_source`, `price_kind`, `candle_price_choice`, `candle_price_gap_seconds`, and `candle_resolution` preserve the provenance/alignment diagnostics. `src.hedging_analysis.load_perp_history` still reads the old `perp_futures_mark_price` field for legacy workbooks, but explicitly labels and warns about that schema.
''',
)
readme_path.write_text(readme, encoding="utf-8")

code_doc_path = ROOT / "docs/code_documentation.md"
code_doc = code_doc_path.read_text(encoding="utf-8")
code_doc = append_once(
    code_doc,
    "### Perpetual hedge price provenance",
    r'''
### Perpetual hedge price provenance

New perpetual-history rows use `perp_futures_price_proxy` rather than the misleading `perp_futures_mark_price` label. The thesis baseline proxy comes from Deribit's TradingView chart endpoint and chooses the closer of the one-minute candle open or close to the option-snapshot timestamp, subject to the maximum-gap rule. Source/kind, chosen candle endpoint, timestamp gap, and resolution are retained through the hedge loader and written into `hedge_notes`. Legacy files with `perp_futures_mark_price` remain readable, but the loader emits a warning and tags them as `legacy_mark_named_candle_proxy`.
''',
)
code_doc_path.write_text(code_doc, encoding="utf-8")


# ---------------------------------------------------------------------------
# Task 12 — robustness notebook (helpers live in notebooks/_lib/robustness.py)
# ---------------------------------------------------------------------------
robustness_nb_path = ROOT / "notebooks/robustness_checks.ipynb"
robustness_nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Discounting and SVCJ identification robustness\n",
                "\n",
                "This notebook is diagnostic. It does **not** tune `lambda_TS` and does not alter the baseline calibration. For the current remediation pass the fixed specifications are ordinary WLS (`lambda_TS = 0`) and the provisional penalized WLS baseline (`lambda_TS = 100`). Deterministic discount rates 0%, 2.5%, 5%, and 7.5% are applied to already-computed regular USD futures-option values before conversion to coin units.\n",
                "\n",
                "Promotion rule before Task 13: move discounting into the baseline only if a plausible positive-rate scenario reverses the model RMSE ranking, changes a model RMSE by at least 5% relative, or changes within-spread share by at least 2 percentage points. Identification diagnostics treat `rho_j` as an unbounded jump-size loading; `|rho_j|>1` is descriptive, not a boundary violation."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from __future__ import annotations\n",
                "from pathlib import Path\n",
                "import sys\n",
                "import pandas as pd\n",
                "\n",
                "ROOT = Path.cwd().resolve()\n",
                "if (ROOT / 'notebooks' / '_lib').exists():\n",
                "    PROJECT_ROOT = ROOT\n",
                "    NOTEBOOK_DIR = ROOT / 'notebooks'\n",
                "elif (ROOT / '_lib').exists():\n",
                "    NOTEBOOK_DIR = ROOT\n",
                "    PROJECT_ROOT = ROOT.parent\n",
                "else:\n",
                "    raise FileNotFoundError('Could not locate notebooks/_lib')\n",
                "if str(NOTEBOOK_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(NOTEBOOK_DIR))\n",
                "from _lib import chapter3_final, robustness\n",
                "\n",
                "REMEDIATION_DIR = PROJECT_ROOT / 'excel files' / 'remediation'\n",
                "OUTPUT_DIR = NOTEBOOK_DIR / 'chapter3_outputs' / 'remediation' / 'robustness'\n",
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
                "REGULARISATIONS = (0, 100)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "available = {}\n",
                "for reg in REGULARISATIONS:\n",
                "    path = REMEDIATION_DIR / f'calibration_results_reg_{reg}.xlsx'\n",
                "    if path.exists():\n",
                "        available[reg] = path\n",
                "    else:\n",
                "        print(f'Pending Task 13: {path.name} does not exist yet')\n",
                "available\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "discount_tables = []\n",
                "promotion_rows = []\n",
                "identification_tables = []\n",
                "for reg, path in available.items():\n",
                "    _, _, svcj_params, test_data = chapter3_final.load_baseline_workbook(path, evaluation_split='test')\n",
                "    discount = robustness.build_discounting_sensitivity_table(test_data)\n",
                "    discount['lambda_TS'] = reg\n",
                "    discount_tables.append(discount)\n",
                "    decision = robustness.assess_discounting_promotion(discount, scope='All')\n",
                "    decision['lambda_TS'] = reg\n",
                "    promotion_rows.append(decision)\n",
                "    ident = robustness.parameter_identification_summary(svcj_params)\n",
                "    ident['lambda_TS'] = reg\n",
                "    identification_tables.append(ident)\n",
                "    for currency in ('BTC', 'ETH'):\n",
                "        corr = robustness.parameter_correlation_matrix(svcj_params, currency=currency)\n",
                "        corr.to_csv(OUTPUT_DIR / f'svcj_parameter_correlations_reg_{reg}_{currency}.csv')\n",
                "\n",
                "discount_sensitivity = pd.concat(discount_tables, ignore_index=True) if discount_tables else pd.DataFrame()\n",
                "promotion_decisions = pd.DataFrame(promotion_rows)\n",
                "identification_summary = pd.concat(identification_tables, ignore_index=True) if identification_tables else pd.DataFrame()\n",
                "if not discount_sensitivity.empty:\n",
                "    discount_sensitivity.to_csv(OUTPUT_DIR / 'discounting_sensitivity.csv', index=False)\n",
                "if not promotion_decisions.empty:\n",
                "    promotion_decisions.to_csv(OUTPUT_DIR / 'discounting_promotion_decisions.csv', index=False)\n",
                "if not identification_summary.empty:\n",
                "    identification_summary.to_csv(OUTPUT_DIR / 'svcj_identification_summary.csv', index=False)\n",
                "promotion_decisions\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Legacy-versus-corrected identification comparison\n",
                "\n",
                "The comparison is intentionally conditional on a corrected `lambda_TS=100` workbook existing. Until Task 13 reruns calibration with unbounded `rho_j` and the corrected admissibility rules, the notebook does not manufacture a 'new' parameter path."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "legacy_path = PROJECT_ROOT / 'excel files' / 'calibration_results_reg_100.xlsx'\n",
                "if 100 in available and legacy_path.exists():\n",
                "    corrected_svcj = pd.read_excel(available[100], sheet_name='svcj_params')\n",
                "    legacy_svcj = pd.read_excel(legacy_path, sheet_name='svcj_params')\n",
                "    legacy_vs_corrected = robustness.compare_identification_summaries(corrected_svcj, legacy_svcj)\n",
                "    legacy_vs_corrected.to_csv(OUTPUT_DIR / 'svcj_legacy_vs_corrected_identification.csv', index=False)\n",
                "else:\n",
                "    legacy_vs_corrected = pd.DataFrame()\n",
                "legacy_vs_corrected\n",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
robustness_nb_path.write_text(json.dumps(robustness_nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

code_doc = code_doc_path.read_text(encoding="utf-8")
code_doc = append_once(
    code_doc,
    "### Discounting and identification robustness",
    r'''
### Discounting and identification robustness

`notebooks/_lib/robustness.py` provides deterministic discount-factor scenarios at 0%, 2.5%, 5%, and 7.5% without changing the calibration itself. Under the maintained futures-option convention it multiplies the regular USD option value by `exp(-rT)` before inverse conversion; equivalently the already-computed coin premium is multiplied by the same deterministic factor. The pre-Task-13 promotion rule is fixed ex ante: promote discounting into the baseline if a plausible positive-rate scenario reverses the model RMSE ranking, changes a model RMSE by at least 5% relative, or moves within-spread share by at least 2 percentage points.

The same module reports SVCJ Feller-gap frequencies, Brownian-`rho` boundary clustering, `sigma_y`/jump-intensity near-bound behavior, martingale/FFT-contour moment margins, descriptive `|rho_j|>1` frequency, and parameter correlations. `rho_j` has no boundary-hit statistic because it is an unbounded conditional jump-size loading. `notebooks/robustness_checks.ipynb` is prepared to compare the legacy and corrected parameter paths once Task 13 has generated the corrected WLS (`lambda_TS=0`) and provisional penalized-WLS (`lambda_TS=100`) workbooks.
''',
)
code_doc_path.write_text(code_doc, encoding="utf-8")

print("Applied Tasks 10-12 production changes.")
