
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib.util
import math
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_ORDER = ["black", "heston", "svcj"]
HEDGE_ORDER = ["delta", "net_delta"]
HEDGE_LABELS = {"delta": "Delta", "net_delta": "Net delta"}
CURRENCY_ORDER = ["BTC", "ETH"]
MATURITY_ORDER = ["0-7d", "7-14d", "14-30d", "30-90d", "90d+"]
SERIES_ORDER = ["Unhedged", "Delta", "Net delta"]

MODEL_LABELS = {"black": "Black", "heston": "Heston", "svcj": "SVCJ"}
MODEL_COLORS = {"black": "#1f4e79", "heston": "#2f7d32", "svcj": "#8c2d19"}
SERIES_COLORS = {"Unhedged": "#7f8c8d", "Delta": "#3b82f6", "Net delta": "#dc2626"}
HEDGE_COLORS = {"Delta": "#3b82f6", "Net delta": "#dc2626"}

PLOTLY_TEMPLATE_NAME = "hedging_thesis_white"
if PLOTLY_TEMPLATE_NAME not in pio.templates:
    thesis_template = go.layout.Template()
    thesis_template.layout = go.Layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=["#1f4e79", "#dc2626", "#2f7d32", "#7f8c8d", "#9467bd"],
        font=dict(family="Arial, Helvetica, sans-serif", size=14, color="#1f2937"),
        title=dict(font=dict(size=20, color="#111827"), x=0.02, xanchor="left"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
        ),
        margin=dict(l=70, r=30, t=90, b=60),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1.0,
            linecolor="#D1D5DB",
            ticks="outside",
            tickcolor="#D1D5DB",
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(17,24,39,0.08)",
            showline=True,
            linewidth=1.0,
            linecolor="#D1D5DB",
            ticks="outside",
            tickcolor="#D1D5DB",
            zeroline=False,
        ),
        hoverlabel=dict(font=dict(size=13)),
    )
    pio.templates[PLOTLY_TEMPLATE_NAME] = thesis_template

pd.options.display.max_columns = 120
pd.options.display.float_format = lambda x: f"{x:,.6f}"

INTERVAL_LONG_BASE_COLUMNS = [
    "snapshot_ts",
    "eval_snapshot_ts",
    "currency",
    "split",
    "instrument_name",
    "option_type",
    "strike",
    "expiry_datetime",
    "time_to_maturity",
    "maturity_bucket",
    "moneyness_ratio",
    "moneyness_bucket",
    "basis_signed",
    "basis_abs",
    "option_price_coin_t",
    "option_price_coin_t1",
    "dt_years",
    "dt_hours",
    "is_summary_eligible",
]
INTERVAL_LONG_EXTRA_COLUMNS = [
    "model",
    "hedge_type",
    "unhedged_pnl_coin",
    "hedged_pnl_coin",
    "hedge_notional_usd",
    "abs_unhedged_pnl_coin",
    "abs_hedged_pnl_coin",
    "abs_improvement_coin",
    "time_to_maturity_days",
    "cheap_option_flag",
]
POOLED_METRIC_COLUMNS = [
    "n_intervals",
    "mean_unhedged_coin",
    "mean_hedged_coin",
    "rmse_unhedged_coin",
    "rmse_hedged_coin",
    "rmse_reduction",
    "mae_unhedged_coin",
    "mae_hedged_coin",
    "mae_reduction",
    "variance_reduction",
    "q05_unhedged_coin",
    "q05_hedged_coin",
    "es05_unhedged_coin",
    "es05_hedged_coin",
    "hit_rate_abs_improvement",
    "mean_abs_hedge_notional_usd",
    "median_abs_improvement_coin",
    "median_basis_abs",
    "median_dt_hours",
]


@dataclass(frozen=True)
class AnalysisPaths:
    project_root: Path
    notebook_dir: Path
    calibration_xlsx: Path
    hedging_engine_py: Path
    analytics_module_py: Path
    data_dir: Path
    perp_history_csv: Path
    output_xlsx: Path


def resolve_project_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    if start.name == "notebooks":
        return start.parent
    return start


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so decorators (e.g., @dataclass) can resolve
    # module-level symbols through sys.modules during import-time introspection.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Avoid keeping a half-initialized module cached under this alias.
        sys.modules.pop(name, None)
        raise
    return module


def resolve_default_paths(
    reg_val: int = 100,
    notebook_dir: Path | None = None,
    output_filename: str = "hedging_results.xlsx",
) -> AnalysisPaths:
    notebook_dir = Path.cwd().resolve() if notebook_dir is None else Path(notebook_dir).resolve()
    project_root = resolve_project_root(notebook_dir)

    calibration_candidates = [
        project_root / "excel files" / f"calibration_results_reg_{reg_val}.xlsx",
        project_root / f"calibration_results_reg_{reg_val}.xlsx",
        notebook_dir / f"calibration_results_reg_{reg_val}.xlsx",
        Path("/mnt/data") / f"calibration_results_reg_{reg_val}.xlsx",
    ]
    engine_candidates = [
        project_root / "src" / "hedging_analysis.py",
        project_root / "hedging_analysis.py",
        notebook_dir / "src" / "hedging_analysis.py",
        notebook_dir / "hedging_analysis.py",
        Path("/mnt/data") / "hedging_analysis.py",
    ]
    analytics_candidates = [
        notebook_dir / "hedging_analytics.py",
        project_root / "src" / "hedging_analytics.py",
        project_root / "hedging_analytics.py",
        Path("/mnt/data") / "hedging_analytics.py",
    ]
    data_candidates = [
        project_root / "data",
        notebook_dir / "data",
        Path("/mnt/data"),
    ]
    data_dir = first_existing(data_candidates)
    perp_candidates = [
        data_dir / "perpetual_futures_prices.csv",
        data_dir / "perpetual_futures_prices_history.csv",
        notebook_dir / "perpetual_futures_prices.csv",
        Path("/mnt/data") / "perpetual_futures_prices.csv",
    ]
    output_candidates = [
        project_root / "excel files" / output_filename,
        project_root / output_filename,
        notebook_dir / output_filename,
        Path("/mnt/data") / output_filename,
    ]
    return AnalysisPaths(
        project_root=project_root,
        notebook_dir=notebook_dir,
        calibration_xlsx=first_existing(calibration_candidates),
        hedging_engine_py=first_existing(engine_candidates),
        analytics_module_py=first_existing(analytics_candidates),
        data_dir=data_dir,
        perp_history_csv=first_existing(perp_candidates),
        output_xlsx=output_candidates[0] if output_candidates[0].parent.exists() else output_candidates[-1],
    )


def parse_datetime_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    explicit_datetime_cols = {
        "snapshot_ts",
        "eval_snapshot_ts",
        "next_snapshot_ts",
        "timestamp",
        "obs_datetime",
        "expiry_datetime",
    }

    for col in out.columns:
        name = str(col).lower()
        should_parse = (
            name in explicit_datetime_cols
            or name.endswith("_ts")
            or name.endswith("_datetime")
            or "timestamp" in name
        )
        if should_parse:
            try:
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
            except Exception:
                pass
    return out


def normalize_perp_history_to_seconds(perp_path: Path, output_path: Path | None = None) -> Path:
    """Normalize perp timestamps to second precision for the hedging backbone merge."""
    perp_path = Path(perp_path)
    perp = pd.read_csv(perp_path)
    if "obs_datetime" not in perp.columns:
        if {"timestamp", "currency", "close_price"}.issubset(perp.columns):
            out = perp.copy()
            out["obs_datetime"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.floor("s")
            out["coin"] = out["currency"]
            out["perp_futures_mark_price"] = pd.to_numeric(out["close_price"], errors="coerce")
            perp = out[["obs_datetime", "coin", "perp_futures_mark_price"]].copy()
        else:
            raise ValueError("Perp file must contain either obs_datetime/coin/perp_futures_mark_price or timestamp/currency/close_price columns.")
    else:
        perp = perp.copy()
        perp["obs_datetime"] = pd.to_datetime(perp["obs_datetime"], utc=True, errors="coerce").dt.floor("s")
        perp["coin"] = perp["coin"].astype(str).str.upper()
        perp = perp.dropna(subset=["obs_datetime", "coin"])
        perp = perp.sort_values(["coin", "obs_datetime"]).drop_duplicates(subset=["coin", "obs_datetime"], keep="first")
    output_path = perp_path.with_name(perp_path.stem + "_seconds.csv") if output_path is None else Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    perp.to_csv(output_path, index=False)
    return output_path


def run_engine_with_normalized_perp(
    hedging_engine_py: Path,
    calibration_xlsx: Path,
    data_dir: Path,
    output_xlsx: Path,
    perp_history_csv: Path,
    rebalance_every_n_snapshots: int = 1,
    max_snapshot_groups: int | None = None,
    currencies: tuple[str, ...] | None = None,
    exclude_expiry_crossing: bool = True,
) -> dict[str, pd.DataFrame]:
    hm = import_module_from_path("hedging_backbone", Path(hedging_engine_py))
    normalized_perp = normalize_perp_history_to_seconds(
        perp_history_csv,
        Path(data_dir) / "perpetual_futures_prices_seconds.csv",
    )
    cfg = hm.HedgingConfig(
        calibration_xlsx=Path(calibration_xlsx),
        data_dir=Path(data_dir),
        output_xlsx=Path(output_xlsx),
        hedge_price_file=normalized_perp.name,
        rebalance_every_n_snapshots=rebalance_every_n_snapshots,
        max_snapshot_groups=max_snapshot_groups,
        currencies=currencies,
        exclude_expiry_crossing=exclude_expiry_crossing,
    )
    return hm.run_hedging_analysis(cfg)


def load_output_workbook(path: Path) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path, engine="openpyxl")
    out = {sheet: parse_datetime_like_columns(pd.read_excel(path, sheet_name=sheet, engine="openpyxl")) for sheet in xls.sheet_names}
    return out


def ensure_analysis_columns(panel: pd.DataFrame) -> pd.DataFrame:
    out = parse_datetime_like_columns(panel)
    if "time_to_maturity" in out.columns:
        out["time_to_maturity"] = pd.to_numeric(out["time_to_maturity"], errors="coerce")
        out["time_to_maturity_days"] = 365.25 * out["time_to_maturity"]
    else:
        out["time_to_maturity_days"] = np.nan

    if "maturity_bucket" not in out.columns:
        out["maturity_bucket"] = pd.cut(
            out["time_to_maturity_days"],
            bins=[-np.inf, 7, 14, 30, 90, np.inf],
            labels=MATURITY_ORDER,
        ).astype("object")

    if "moneyness_ratio" not in out.columns:
        if "moneyness" in out.columns:
            out["moneyness_ratio"] = pd.to_numeric(out["moneyness"], errors="coerce")
        elif {"strike", "F0"}.issubset(out.columns):
            out["moneyness_ratio"] = pd.to_numeric(out["strike"], errors="coerce") / pd.to_numeric(out["F0"], errors="coerce")
        else:
            out["moneyness_ratio"] = np.nan

    if "moneyness_bucket" not in out.columns:
        out["moneyness_bucket"] = pd.cut(
            out["moneyness_ratio"],
            bins=[-np.inf, 0.95, 1.05, np.inf],
            labels=["OTM (K/F<0.95)", "ATM (0.95-1.05)", "ITM (K/F>1.05)"],
        ).astype("object")

    out["dt_hours"] = pd.to_numeric(out.get("dt_hours"), errors="coerce")
    if {"snapshot_ts", "eval_snapshot_ts"}.issubset(out.columns):
        out["snapshot_ts"] = pd.to_datetime(out["snapshot_ts"], utc=True, errors="coerce")
        out["eval_snapshot_ts"] = pd.to_datetime(out["eval_snapshot_ts"], utc=True, errors="coerce")
    if out["dt_hours"].isna().all() and {"snapshot_ts", "eval_snapshot_ts"}.issubset(out.columns):
        out["dt_hours"] = (out["eval_snapshot_ts"] - out["snapshot_ts"]).dt.total_seconds() / 3600.0

    out["basis_signed"] = pd.to_numeric(out.get("basis_signed"), errors="coerce")
    if out["basis_signed"].isna().all() and {"perp_close_t", "F0"}.issubset(out.columns):
        H = pd.to_numeric(out["perp_close_t"], errors="coerce")
        F = pd.to_numeric(out["F0"], errors="coerce")
        out["basis_signed"] = np.where(np.isfinite(H) & np.isfinite(F) & (F > 0), H / F - 1.0, np.nan)
    out["basis_abs"] = pd.to_numeric(out.get("basis_abs"), errors="coerce")
    if out["basis_abs"].isna().all():
        out["basis_abs"] = np.abs(out["basis_signed"])

    out["cheap_option_flag"] = pd.to_numeric(out.get("option_price_coin_t"), errors="coerce") < 0.005
    if "mid_price_clean" in out.columns:
        out["cheap_option_flag"] = pd.to_numeric(out["mid_price_clean"], errors="coerce") < 0.005
    return out


def empty_interval_long() -> pd.DataFrame:
    return pd.DataFrame(columns=INTERVAL_LONG_BASE_COLUMNS + INTERVAL_LONG_EXTRA_COLUMNS)


def empty_pooled_summary(group_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*group_cols, *POOLED_METRIC_COLUMNS])


def make_interval_long(panel: pd.DataFrame) -> pd.DataFrame:
    panel = ensure_analysis_columns(panel)
    missing_base = [c for c in INTERVAL_LONG_BASE_COLUMNS if c not in panel.columns]
    if missing_base:
        raise KeyError(f"interval_panel is missing required columns: {missing_base}")

    needed_model_cols: list[str] = []
    for model in MODEL_ORDER:
        for hedge_type in HEDGE_ORDER:
            needed_model_cols.extend([f"net_pnl_coin_{model}_{hedge_type}", f"hedge_notional_usd_{model}_{hedge_type}"])
    missing_model_cols = [c for c in ["option_pnl_coin_short", *needed_model_cols] if c not in panel.columns]
    if missing_model_cols:
        raise KeyError(f"interval_panel is missing required hedging columns: {missing_model_cols}")

    if panel.empty:
        return empty_interval_long()

    frames: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        for hedge_type in HEDGE_ORDER:
            tmp = panel[INTERVAL_LONG_BASE_COLUMNS].copy()
            tmp["model"] = model
            tmp["hedge_type"] = hedge_type
            tmp["unhedged_pnl_coin"] = pd.to_numeric(panel["option_pnl_coin_short"], errors="coerce")
            tmp["hedged_pnl_coin"] = pd.to_numeric(panel[f"net_pnl_coin_{model}_{hedge_type}"], errors="coerce")
            tmp["hedge_notional_usd"] = pd.to_numeric(panel[f"hedge_notional_usd_{model}_{hedge_type}"], errors="coerce")
            tmp["abs_unhedged_pnl_coin"] = np.abs(tmp["unhedged_pnl_coin"])
            tmp["abs_hedged_pnl_coin"] = np.abs(tmp["hedged_pnl_coin"])
            tmp["abs_improvement_coin"] = tmp["abs_unhedged_pnl_coin"] - tmp["abs_hedged_pnl_coin"]
            tmp["time_to_maturity_days"] = pd.to_numeric(panel["time_to_maturity_days"], errors="coerce")
            tmp["cheap_option_flag"] = panel["cheap_option_flag"].fillna(False).astype(bool)
            frames.append(tmp)
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def q05(x: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan
    return float(np.quantile(vals, 0.05))


def es05(x: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan
    q = np.quantile(vals, 0.05)
    tail = vals[vals <= q]
    return float(np.mean(tail)) if len(tail) else np.nan


def _summarize_long_group(g: pd.DataFrame) -> dict[str, Any]:
    if g.empty:
        return {
            "n_intervals": 0,
            "mean_unhedged_coin": np.nan,
            "mean_hedged_coin": np.nan,
            "rmse_unhedged_coin": np.nan,
            "rmse_hedged_coin": np.nan,
            "rmse_reduction": np.nan,
            "mae_unhedged_coin": np.nan,
            "mae_hedged_coin": np.nan,
            "mae_reduction": np.nan,
            "variance_reduction": np.nan,
            "q05_unhedged_coin": np.nan,
            "q05_hedged_coin": np.nan,
            "es05_unhedged_coin": np.nan,
            "es05_hedged_coin": np.nan,
            "hit_rate_abs_improvement": np.nan,
            "mean_abs_hedge_notional_usd": np.nan,
            "median_abs_improvement_coin": np.nan,
            "median_basis_abs": np.nan,
            "median_dt_hours": np.nan,
        }
    un = pd.to_numeric(g["unhedged_pnl_coin"], errors="coerce").to_numpy(dtype=float)
    he = pd.to_numeric(g["hedged_pnl_coin"], errors="coerce").to_numpy(dtype=float)
    notion = pd.to_numeric(g["hedge_notional_usd"], errors="coerce").to_numpy(dtype=float)
    abs_imp = pd.to_numeric(g["abs_improvement_coin"], errors="coerce").to_numpy(dtype=float)
    basis_abs = pd.to_numeric(g.get("basis_abs"), errors="coerce").to_numpy(dtype=float)
    dt_hours = pd.to_numeric(g.get("dt_hours"), errors="coerce").to_numpy(dtype=float)

    rmse_un = float(np.sqrt(np.mean(un ** 2)))
    rmse_he = float(np.sqrt(np.mean(he ** 2)))
    mae_un = float(np.mean(np.abs(un)))
    mae_he = float(np.mean(np.abs(he)))
    var_un = float(np.var(un, ddof=1)) if len(un) > 1 else np.nan
    var_he = float(np.var(he, ddof=1)) if len(he) > 1 else np.nan
    return {
        "n_intervals": int(len(g)),
        "mean_unhedged_coin": float(np.mean(un)),
        "mean_hedged_coin": float(np.mean(he)),
        "rmse_unhedged_coin": rmse_un,
        "rmse_hedged_coin": rmse_he,
        "rmse_reduction": 1.0 - (rmse_he / rmse_un) if rmse_un > 0 else np.nan,
        "mae_unhedged_coin": mae_un,
        "mae_hedged_coin": mae_he,
        "mae_reduction": 1.0 - (mae_he / mae_un) if mae_un > 0 else np.nan,
        "variance_reduction": 1.0 - (var_he / var_un) if np.isfinite(var_un) and var_un > 0 and np.isfinite(var_he) else np.nan,
        "q05_unhedged_coin": q05(un),
        "q05_hedged_coin": q05(he),
        "es05_unhedged_coin": es05(un),
        "es05_hedged_coin": es05(he),
        "hit_rate_abs_improvement": float(np.mean(np.abs(he) < np.abs(un))),
        "mean_abs_hedge_notional_usd": float(np.nanmean(np.abs(notion))),
        "median_abs_improvement_coin": float(np.nanmedian(abs_imp)) if len(abs_imp) else np.nan,
        "median_basis_abs": float(np.nanmedian(basis_abs)) if len(basis_abs) else np.nan,
        "median_dt_hours": float(np.nanmedian(dt_hours)) if len(dt_hours) else np.nan,
    }


def pooled_summary(interval_long: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if interval_long.empty:
        return empty_pooled_summary(group_cols)
    eligible = interval_long.loc[interval_long["is_summary_eligible"].fillna(False)].copy()
    if eligible.empty:
        return empty_pooled_summary(group_cols)

    rows = []
    for keys, g in eligible.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(_summarize_long_group(g))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return empty_pooled_summary(group_cols)
    return out.sort_values(group_cols).reset_index(drop=True)


def representative_panel(interval_long: pd.DataFrame, model: str = "black", split: str = "test") -> pd.DataFrame:
    out = interval_long.copy()
    out = out[out["model"].astype(str).str.lower() == str(model).lower()].copy()
    out = out[out["split"].astype(str).str.lower() == str(split).lower()].copy()
    return out


def unhedged_vs_hedged_panel(rep_panel: pd.DataFrame) -> pd.DataFrame:
    if rep_panel.empty:
        return pd.DataFrame(columns=["currency", "series", "pnl_coin", "abs_pnl_coin", "time_to_maturity_days", "basis_abs", "dt_hours"])
    frames = []
    base_cols = ["currency", "time_to_maturity_days", "basis_abs", "dt_hours"]
    un = rep_panel[rep_panel["hedge_type"] == "delta"][base_cols + ["unhedged_pnl_coin"]].copy()
    un = un.rename(columns={"unhedged_pnl_coin": "pnl_coin"})
    un["series"] = "Unhedged"
    frames.append(un)
    for hedge_type, label in [("delta", "Delta"), ("net_delta", "Net delta")]:
        he = rep_panel[rep_panel["hedge_type"] == hedge_type][base_cols + ["hedged_pnl_coin"]].copy()
        he = he.rename(columns={"hedged_pnl_coin": "pnl_coin"})
        he["series"] = label
        frames.append(he)
    out = pd.concat(frames, ignore_index=True)
    out["abs_pnl_coin"] = np.abs(pd.to_numeric(out["pnl_coin"], errors="coerce"))
    return out


def _coerce_group_cols(group_cols: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(group_cols, str):
        return [group_cols]
    return list(group_cols)


def _ordered_bucket_labels(labels: pd.Series) -> list[str]:
    vals = [str(x) for x in labels.dropna().unique().tolist()]
    def _key(x: str):
        tail = ''.join(ch for ch in x if ch.isdigit())
        return (int(tail) if tail else 10**9, x)
    return sorted(vals, key=_key)


def _bucket_rank_from_label(label: object) -> int:
    s = str(label)
    tail = ''.join(ch for ch in s if ch.isdigit())
    return int(tail) if tail else 10**9


def _format_bucket_range_label(vmin: float, vmax: float, value_kind: str | None = None) -> str:
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return ""
    if value_kind == "maturity_days":
        return f"{vmin:.1f}–{vmax:.1f}d"
    if value_kind == "moneyness":
        return f"{vmin:.3f}–{vmax:.3f} K/F"
    if value_kind == "basis_abs":
        return f"[{100*vmin:.2f}%, {100*vmax:.2f}%]"
    return f"[{vmin:.4g}, {vmax:.4g}]"


def attach_bucket_range_labels(
    df: pd.DataFrame,
    ranges: pd.DataFrame,
    bucket_col: str,
    output_col: str | None = None,
    value_kind: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    output_col = output_col or f"{bucket_col}_label"
    if out.empty or bucket_col not in out.columns or ranges.empty:
        out[output_col] = out.get(bucket_col)
        return out

    join_cols = [c for c in ["currency", bucket_col] if c in out.columns and c in ranges.columns]
    if bucket_col not in join_cols:
        join_cols.append(bucket_col)

    label_map = ranges.copy()
    label_map[output_col] = [
        _format_bucket_range_label(vmin, vmax, value_kind=value_kind)
        for vmin, vmax in zip(
            pd.to_numeric(label_map["bucket_min"], errors="coerce"),
            pd.to_numeric(label_map["bucket_max"], errors="coerce"),
        )
    ]
    label_map[f"{bucket_col}__rank"] = label_map[bucket_col].map(_bucket_rank_from_label)
    keep_cols = [*join_cols, output_col, f"{bucket_col}__rank"]
    label_map = label_map[keep_cols].drop_duplicates(join_cols, keep="first")

    out = out.merge(label_map, on=join_cols, how="left")
    out[output_col] = out[output_col].fillna(out[bucket_col].astype(str))
    if f"{bucket_col}__rank" not in out.columns:
        out[f"{bucket_col}__rank"] = out[bucket_col].map(_bucket_rank_from_label)
    else:
        out[f"{bucket_col}__rank"] = pd.to_numeric(out[f"{bucket_col}__rank"], errors="coerce").fillna(
            out[bucket_col].map(_bucket_rank_from_label)
        )
    return out


def _ordered_bucket_display_labels(df: pd.DataFrame, bucket_col: str, label_col: str) -> list[str]:
    if df.empty or label_col not in df.columns:
        return []
    tmp = df[[c for c in [bucket_col, label_col, f"{bucket_col}__rank"] if c in df.columns]].dropna(subset=[label_col]).copy()
    if tmp.empty:
        return []
    if f"{bucket_col}__rank" not in tmp.columns:
        tmp[f"{bucket_col}__rank"] = tmp[bucket_col].map(_bucket_rank_from_label)
    tmp = tmp.sort_values([f"{bucket_col}__rank", label_col]).drop_duplicates(label_col, keep="first")
    return tmp[label_col].astype(str).tolist()


def add_groupwise_equal_count_buckets(
    df: pd.DataFrame,
    value_col: str,
    group_cols: str | list[str] | tuple[str, ...] = ("currency",),
    q: int = 5,
    output_col: str | None = None,
    label_prefix: str = "Q",
) -> pd.DataFrame:
    out = df.copy()
    output_col = output_col or f"{value_col}_q{q}"
    out[output_col] = pd.Series(pd.NA, index=out.index, dtype="object")
    group_cols_list = _coerce_group_cols(group_cols)

    for _, idx in out.groupby(group_cols_list, dropna=False).groups.items():
        s = pd.to_numeric(out.loc[list(idx), value_col], errors="coerce")
        valid = s.notna()
        n_valid = int(valid.sum())
        if n_valid < q:
            continue
        ranks = s[valid].rank(method="first")
        try:
            bucket_codes = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
        except ValueError:
            continue
        bucket_codes = pd.Series(bucket_codes, index=s[valid].index)
        n_bins = int(bucket_codes.nunique())
        if n_bins == 0:
            continue
        labels = {i: f"{label_prefix}{i + 1}" for i in range(n_bins)}
        out.loc[bucket_codes.index, output_col] = bucket_codes.map(labels).astype("object")
    return out


def add_groupwise_qcut(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "currency",
    q: int = 5,
    output_col: str | None = None,
    label_prefix: str = "Q",
) -> pd.DataFrame:
    return add_groupwise_equal_count_buckets(
        df=df,
        value_col=value_col,
        group_cols=[group_col],
        q=q,
        output_col=output_col,
        label_prefix=label_prefix,
    )


def build_equal_count_bucket_range_table(
    df: pd.DataFrame,
    bucket_col: str,
    value_col: str,
    group_cols: str | list[str] | tuple[str, ...] = ("currency",),
) -> pd.DataFrame:
    group_cols_list = _coerce_group_cols(group_cols)
    needed = [*group_cols_list, bucket_col, value_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for equal-count bucket range table: {missing}")
    rows = []
    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp[tmp[bucket_col].notna() & tmp[value_col].notna()].copy()
    if tmp.empty:
        return pd.DataFrame(columns=[*group_cols_list, bucket_col, "n_intervals", "bucket_min", "bucket_median", "bucket_max"])
    for keys, g in tmp.groupby([*group_cols_list, bucket_col], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip([*group_cols_list, bucket_col], keys)}
        row.update({
            "n_intervals": int(len(g)),
            "bucket_min": float(g[value_col].min()),
            "bucket_median": float(g[value_col].median()),
            "bucket_max": float(g[value_col].max()),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values([*group_cols_list, bucket_col]).reset_index(drop=True)


def build_equal_count_bucket_summary(
    interval_long: pd.DataFrame,
    value_col: str,
    bucket_col: str,
    q: int = 5,
    group_cols: str | list[str] | tuple[str, ...] = ("currency",),
    label_prefix: str = "Q",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bucketed = add_groupwise_equal_count_buckets(
        interval_long,
        value_col=value_col,
        group_cols=group_cols,
        q=q,
        output_col=bucket_col,
        label_prefix=label_prefix,
    )
    summary_group_cols = [*_coerce_group_cols(group_cols), bucket_col, "hedge_type"]
    if "model" in bucketed.columns:
        summary_group_cols.append("model")
    summary = pooled_summary(bucketed, summary_group_cols)
    ranges = build_equal_count_bucket_range_table(bucketed, bucket_col=bucket_col, value_col=value_col, group_cols=group_cols)
    return bucketed, summary, ranges


def build_sample_construction(interval_panel: pd.DataFrame) -> pd.DataFrame:
    panel = ensure_analysis_columns(interval_panel)
    rows = []
    for (currency, split), g in panel.groupby(["currency", "split"], dropna=False):
        eligible = g["is_summary_eligible"].fillna(False)
        has_eval = pd.to_numeric(g.get("option_price_coin_t1", g.get("eval_market_price")), errors="coerce").notna()
        has_perp_t = pd.to_numeric(g.get("perp_close_t"), errors="coerce").notna()
        has_perp_t1 = pd.to_numeric(g.get("perp_close_t1"), errors="coerce").notna()
        expiry_cross = g.get("expires_before_eval_snapshot", pd.Series(False, index=g.index)).fillna(False)
        rows.append({
            "currency": currency,
            "split": split,
            "n_rows": int(len(g)),
            "n_eligible": int(eligible.sum()),
            "eligible_share": float(eligible.mean()) if len(g) else np.nan,
            "n_unique_instruments": int(g["instrument_name"].nunique()),
            "n_unique_snapshots": int(g["snapshot_ts"].nunique()),
            "median_intervals_per_snapshot": float(g.groupby("snapshot_ts").size().median()) if len(g) else np.nan,
            "median_dt_hours": float(pd.to_numeric(g["dt_hours"], errors="coerce").median()),
            "missing_next_option_share": float((~has_eval).mean()),
            "missing_perp_t_share": float((~has_perp_t).mean()),
            "missing_perp_t1_share": float((~has_perp_t1).mean()),
            "expiry_crossing_share": float(expiry_cross.mean()),
            "cheap_option_share": float(g["cheap_option_flag"].fillna(False).mean()),
        })
    return pd.DataFrame(rows).sort_values(["split", "currency"]).reset_index(drop=True)


def build_sensitivity_table(rep_panel: pd.DataFrame) -> pd.DataFrame:
    rep_panel = rep_panel.copy()
    filters = {
        "Full test sample": np.ones(len(rep_panel), dtype=bool),
        "Exclude <=7d": pd.to_numeric(rep_panel["time_to_maturity_days"], errors="coerce") > 7.0,
        "Exclude <=7d and cheap": (pd.to_numeric(rep_panel["time_to_maturity_days"], errors="coerce") > 7.0) & (~rep_panel["cheap_option_flag"].fillna(False)),
    }
    rows = []
    for label, mask in filters.items():
        sub = rep_panel.loc[mask].copy()
        summ = pooled_summary(sub, ["currency", "hedge_type"])
        if summ.empty:
            continue
        summ["sample"] = label
        rows.append(summ)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    cols = ["sample", "currency", "hedge_type", "n_intervals", "rmse_reduction", "mae_reduction", "variance_reduction", "q05_hedged_coin", "es05_hedged_coin", "hit_rate_abs_improvement"]
    return out[cols].sort_values(["sample", "currency", "hedge_type"]).reset_index(drop=True)


def display_ready(df: pd.DataFrame, percent_cols: list[str] | None = None, round_digits: int = 6) -> pd.DataFrame:
    out = df.copy()
    percent_cols = percent_cols or []
    for col in out.columns:
        if col in percent_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: f"{100*x:.2f}%" if pd.notna(x) else None)
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(round_digits)
    return out


def apply_thesis_layout(fig: go.Figure, title: str | None = None, height: int = 520, width: int | None = None) -> go.Figure:
    fig.update_layout(template=PLOTLY_TEMPLATE_NAME, height=height, width=width)
    if title is not None:
        fig.update_layout(title=title)
    fig.update_annotations(font=dict(size=13, color="#374151"))
    return fig


def figure_pooled_metric(summary_df: pd.DataFrame, metric: str = "rmse_reduction", split: str = "test", height: int = 520) -> go.Figure:
    plot_df = summary_df.copy()
    if "split" in plot_df.columns:
        plot_df = plot_df[plot_df["split"].astype(str).str.lower() == str(split).lower()].copy()
    if plot_df.empty:
        return apply_thesis_layout(go.Figure(), title=f"No data for {metric}", height=height)
    plot_df["model_label"] = plot_df["model"].map(MODEL_LABELS).fillna(plot_df["model"])
    plot_df["hedge_label"] = plot_df["hedge_type"].map(HEDGE_LABELS).fillna(plot_df["hedge_type"])
    plot_df["text"] = pd.to_numeric(plot_df[metric], errors="coerce").map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "")
    fig = px.bar(
        plot_df,
        x="model_label",
        y=metric,
        color="hedge_label",
        facet_col="currency",
        barmode="group",
        text="text",
        category_orders={"model_label": [MODEL_LABELS[m] for m in MODEL_ORDER], "hedge_label": ["Delta", "Net delta"], "currency": CURRENCY_ORDER},
        labels={"model_label": "Model", metric: metric.replace("_", " ").title(), "hedge_label": "Hedge type", "currency": "Currency"},
        color_discrete_map=HEDGE_COLORS,
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_thesis_layout(fig, title=f"Test-sample pooled {metric.replace('_', ' ')}", height=height)


def figure_snapshot_metric(summary_ts: pd.DataFrame, metric: str = "rmse_reduction", representative_model: str = "black", hedge_type: str = "net_delta", height: int = 520) -> go.Figure:
    plot_df = summary_ts.copy()
    plot_df = plot_df[(plot_df["model"].astype(str).str.lower() == representative_model) & (plot_df["hedge_type"].astype(str).str.lower() == hedge_type)].copy()
    if plot_df.empty:
        return apply_thesis_layout(go.Figure(), title="No timestamp summary data", height=height)
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.sort_values(["currency", "snapshot_ts"]).reset_index(drop=True)
    plot_df["rolling"] = plot_df.groupby("currency")[metric].transform(lambda s: s.rolling(7, min_periods=1).median())
    fig = px.line(
        plot_df,
        x="snapshot_ts",
        y=metric,
        color="currency",
        category_orders={"currency": CURRENCY_ORDER},
        color_discrete_sequence=["#1f4e79", "#dc2626"],
        labels={"snapshot_ts": "Snapshot time", metric: metric.replace("_", " ").title(), "currency": "Currency"},
    )
    for currency, g in plot_df.groupby("currency"):
        fig.add_trace(
            go.Scatter(
                x=g["snapshot_ts"],
                y=g["rolling"],
                mode="lines",
                line=dict(width=3),
                name=f"{currency} rolling median",
                legendgroup=currency,
                showlegend=False,
            )
        )
    fig.update_yaxes(tickformat=".0%")
    return apply_thesis_layout(fig, title=f"Snapshot-level {metric.replace('_', ' ')} — {MODEL_LABELS.get(representative_model, representative_model.title())} + {HEDGE_LABELS.get(hedge_type, hedge_type)}", height=height)


def figure_ecdf_abs_pnl(rep_series: pd.DataFrame, height: int = 520) -> go.Figure:
    plot_df = rep_series.copy()
    if plot_df.empty:
        return apply_thesis_layout(go.Figure(), title="No representative series data", height=height)
    plot_df["abs_pnl_coin"] = pd.to_numeric(plot_df["abs_pnl_coin"], errors="coerce")
    plot_df = plot_df.loc[plot_df["abs_pnl_coin"].notna()].copy()
    if plot_df.empty:
        return apply_thesis_layout(go.Figure(), title="No representative series data", height=height)

    positive = plot_df.loc[plot_df["abs_pnl_coin"] > 0, "abs_pnl_coin"]
    eps = float(positive.min()) * 0.5 if len(positive) else 1e-12
    plot_df["abs_pnl_coin_logsafe"] = np.where(plot_df["abs_pnl_coin"] > 0, plot_df["abs_pnl_coin"], eps)

    fig = px.ecdf(
        plot_df,
        x="abs_pnl_coin_logsafe",
        color="series",
        facet_col="currency",
        log_x=True,
        category_orders={"series": SERIES_ORDER, "currency": CURRENCY_ORDER},
        color_discrete_map=SERIES_COLORS,
        labels={"abs_pnl_coin_logsafe": "Absolute interval PnL in coin terms (log scale)", "series": "Series", "currency": "Currency"},
    )
    return apply_thesis_layout(fig, title="ECDF of absolute interval PnL", height=height)


def figure_tail_comparison(rep_series: pd.DataFrame, height: int = 700) -> go.Figure:
    rows = []
    for (currency, series), g in rep_series.groupby(["currency", "series"], dropna=False):
        rows.append({"currency": currency, "series": series, "metric": "5% quantile", "coin_value": q05(g["pnl_coin"]), "n_intervals": int(len(g))})
        rows.append({"currency": currency, "series": series, "metric": "Expected shortfall (5%)", "coin_value": es05(g["pnl_coin"]), "n_intervals": int(len(g))})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return apply_thesis_layout(go.Figure(), title="No tail data", height=height)
    fig = px.bar(
        plot_df,
        x="series",
        y="coin_value",
        color="series",
        facet_row="metric",
        facet_col="currency",
        barmode="group",
        category_orders={"series": SERIES_ORDER, "currency": CURRENCY_ORDER},
        color_discrete_map=SERIES_COLORS,
        labels={"series": "Series", "coin_value": "Coin PnL", "currency": "Currency", "metric": ""},
    )
    fig.update_layout(showlegend=False)
    return apply_thesis_layout(fig, title="Tail-risk comparison", height=height)


def figure_maturity_metric(summary_df: pd.DataFrame, metric: str = "rmse_reduction", representative_model: str = "black", split: str = "test", height: int = 520) -> go.Figure:
    plot_df = summary_df.copy()
    if "split" in plot_df.columns:
        plot_df = plot_df[plot_df["split"].astype(str).str.lower() == split].copy()
    plot_df = plot_df[plot_df["model"].astype(str).str.lower() == representative_model].copy()
    if plot_df.empty:
        return apply_thesis_layout(go.Figure(), title="No maturity summary data", height=height)
    plot_df["hedge_label"] = plot_df["hedge_type"].map(HEDGE_LABELS).fillna(plot_df["hedge_type"])
    plot_df["text"] = pd.to_numeric(plot_df[metric], errors="coerce").map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "")
    fig = px.bar(
        plot_df,
        x="maturity_bucket",
        y=metric,
        color="hedge_label",
        facet_col="currency",
        barmode="group",
        text="text",
        hover_data={"n_intervals": True},
        category_orders={"maturity_bucket": MATURITY_ORDER, "hedge_label": ["Delta", "Net delta"], "currency": CURRENCY_ORDER},
        color_discrete_map=HEDGE_COLORS,
        labels={"maturity_bucket": "Maturity bucket", metric: metric.replace("_", " ").title(), "hedge_label": "Hedge type", "currency": "Currency"},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_thesis_layout(fig, title=f"Maturity-bucket {metric.replace('_', ' ')} — {MODEL_LABELS.get(representative_model, representative_model.title())}", height=height)


def figure_basis_by_maturity(interval_long: pd.DataFrame, split: str = "test", height: int = 520) -> go.Figure:
    plot_df = interval_long.copy()
    plot_df = plot_df[(plot_df["split"].astype(str).str.lower() == split) & (plot_df["hedge_type"].astype(str).str.lower() == "net_delta") & (plot_df["model"].astype(str).str.lower() == "black")].copy()
    if plot_df.empty:
        return apply_thesis_layout(go.Figure(), title="No basis data", height=height)
    rows = []
    for (currency, maturity_bucket), g in plot_df.groupby(["currency", "maturity_bucket"], dropna=False):
        rows.append({"currency": currency, "maturity_bucket": maturity_bucket, "median_basis_abs": float(pd.to_numeric(g["basis_abs"], errors="coerce").median()), "n_intervals": int(len(g))})
    agg = pd.DataFrame(rows)
    agg["text"] = agg["median_basis_abs"].map(lambda x: f"{10000*x:.1f} bps" if pd.notna(x) else "")
    fig = px.bar(
        agg,
        x="maturity_bucket",
        y="median_basis_abs",
        color="currency",
        barmode="group",
        text="text",
        category_orders={"maturity_bucket": MATURITY_ORDER, "currency": CURRENCY_ORDER},
        labels={"maturity_bucket": "Maturity bucket", "median_basis_abs": "Median absolute perp-term basis", "currency": "Currency"},
        color_discrete_sequence=["#1f4e79", "#dc2626"],
    )
    fig.update_yaxes(tickformat=".2%")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_thesis_layout(fig, title="Median absolute perp-term basis by maturity bucket", height=height)


def figure_moneyness_quintile_metric(rep_panel: pd.DataFrame, metric: str = "rmse_reduction", q: int = 5, height: int = 520) -> go.Figure:
    return figure_equal_count_moneyness_metric(rep_panel, metric=metric, q=q, height=height)


def figure_basis_quintile_metric(rep_panel: pd.DataFrame, metric: str = "rmse_reduction", q: int = 5, height: int = 520) -> go.Figure:
    return figure_equal_count_basis_metric(rep_panel, metric=metric, q=q, height=height)


def figure_equal_count_bucket_metric(
    summary_df: pd.DataFrame,
    bucket_col: str,
    metric: str = "rmse_reduction",
    bucket_title: str = "Equal-count bucket",
    height: int = 520,
    bucket_label_col: str | None = None,
) -> go.Figure:
    plot_df = summary_df.copy()
    if plot_df.empty or bucket_col not in plot_df.columns:
        return apply_thesis_layout(go.Figure(), title=f"No {bucket_title.lower()} data", height=height)
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df[plot_df[bucket_col].notna()].copy()
    if plot_df.empty:
        return apply_thesis_layout(go.Figure(), title=f"No {bucket_title.lower()} data", height=height)
    plot_df["hedge_label"] = plot_df["hedge_type"].map(HEDGE_LABELS).fillna(plot_df["hedge_type"])
    plot_df["text"] = plot_df[metric].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "")
    x_col = bucket_label_col if bucket_label_col and bucket_label_col in plot_df.columns else bucket_col
    bucket_order = _ordered_bucket_display_labels(plot_df, bucket_col=bucket_col, label_col=x_col) if x_col != bucket_col else _ordered_bucket_labels(plot_df[bucket_col])
    fig = px.bar(
        plot_df,
        x=x_col,
        y=metric,
        color="hedge_label",
        facet_col="currency",
        barmode="group",
        text="text",
        category_orders={x_col: bucket_order, "hedge_label": ["Delta", "Net delta"], "currency": CURRENCY_ORDER},
        color_discrete_map=HEDGE_COLORS,
        labels={x_col: bucket_title, metric: metric.replace("_", " ").title(), "hedge_label": "Hedge type", "currency": "Currency"},
        hover_data={"n_intervals": True},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_thesis_layout(fig, title=f"{bucket_title} {metric.replace('_', ' ')}", height=height)


def figure_basis_by_bucket(
    bucketed_panel: pd.DataFrame,
    bucket_col: str,
    bucket_title: str = "Equal-count bucket",
    height: int = 520,
    bucket_label_col: str | None = None,
) -> go.Figure:
    plot_df = bucketed_panel.copy()
    if plot_df.empty or bucket_col not in plot_df.columns:
        return apply_thesis_layout(go.Figure(), title=f"No {bucket_title.lower()} basis data", height=height)
    plot_df = plot_df[plot_df[bucket_col].notna()].copy()
    if plot_df.empty:
        return apply_thesis_layout(go.Figure(), title=f"No {bucket_title.lower()} basis data", height=height)
    rows = []
    group_keys = ["currency", bucket_col]
    if bucket_label_col and bucket_label_col in plot_df.columns:
        group_keys.append(bucket_label_col)
    if f"{bucket_col}__rank" in plot_df.columns:
        group_keys.append(f"{bucket_col}__rank")
    for keys, g in plot_df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {
            "currency": keys[0],
            bucket_col: keys[1],
            "median_basis_abs": float(pd.to_numeric(g["basis_abs"], errors="coerce").median()),
            "n_intervals": int(len(g)),
        }
        if bucket_label_col and bucket_label_col in plot_df.columns:
            row[bucket_label_col] = keys[2]
            if f"{bucket_col}__rank" in plot_df.columns:
                row[f"{bucket_col}__rank"] = keys[3]
        elif f"{bucket_col}__rank" in plot_df.columns:
            row[f"{bucket_col}__rank"] = keys[2]
        rows.append(row)
    agg = pd.DataFrame(rows)
    if agg.empty:
        return apply_thesis_layout(go.Figure(), title=f"No {bucket_title.lower()} basis data", height=height)
    x_col = bucket_label_col if bucket_label_col and bucket_label_col in agg.columns else bucket_col
    bucket_order = _ordered_bucket_display_labels(agg, bucket_col=bucket_col, label_col=x_col) if x_col != bucket_col else _ordered_bucket_labels(agg[bucket_col])
    agg["text"] = agg["median_basis_abs"].map(lambda x: f"{10000*x:.1f} bps" if pd.notna(x) else "")
    fig = px.bar(
        agg,
        x=x_col,
        y="median_basis_abs",
        color="currency",
        barmode="group",
        text="text",
        category_orders={x_col: bucket_order, "currency": CURRENCY_ORDER},
        labels={x_col: bucket_title, "median_basis_abs": "Median absolute perp-term basis", "currency": "Currency"},
        color_discrete_sequence=["#1f4e79", "#dc2626"],
        hover_data={"n_intervals": True},
    )
    fig.update_yaxes(tickformat=".2%")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_thesis_layout(fig, title=f"Median absolute perp-term basis by {bucket_title.lower()}", height=height)


def figure_equal_count_maturity_metric(rep_panel: pd.DataFrame, metric: str = "rmse_reduction", q: int = 5, height: int = 520) -> go.Figure:
    _, summary, ranges = build_equal_count_bucket_summary(rep_panel, value_col="time_to_maturity_days", bucket_col="maturity_equal_bucket", q=q, group_cols=("currency",))
    if summary.empty:
        return apply_thesis_layout(go.Figure(), title="No equal-count maturity data", height=height)
    summary = attach_bucket_range_labels(summary, ranges, bucket_col="maturity_equal_bucket", output_col="maturity_equal_bucket_label", value_kind="maturity_days")
    return figure_equal_count_bucket_metric(
        summary,
        bucket_col="maturity_equal_bucket",
        bucket_label_col="maturity_equal_bucket_label",
        metric=metric,
        bucket_title="Maturity range (days)",
        height=height,
    )


def figure_basis_by_equal_count_maturity(rep_panel: pd.DataFrame, q: int = 5, height: int = 520) -> go.Figure:
    bucketed, _, ranges = build_equal_count_bucket_summary(rep_panel, value_col="time_to_maturity_days", bucket_col="maturity_equal_bucket", q=q, group_cols=("currency",))
    bucketed = attach_bucket_range_labels(bucketed, ranges, bucket_col="maturity_equal_bucket", output_col="maturity_equal_bucket_label", value_kind="maturity_days")
    return figure_basis_by_bucket(
        bucketed,
        bucket_col="maturity_equal_bucket",
        bucket_label_col="maturity_equal_bucket_label",
        bucket_title="Maturity range (days)",
        height=height,
    )


def figure_equal_count_moneyness_metric(rep_panel: pd.DataFrame, metric: str = "rmse_reduction", q: int = 5, height: int = 520) -> go.Figure:
    _, summary, ranges = build_equal_count_bucket_summary(rep_panel, value_col="moneyness_ratio", bucket_col="moneyness_equal_bucket", q=q, group_cols=("currency",))
    if summary.empty:
        return apply_thesis_layout(go.Figure(), title="No equal-count moneyness data", height=height)
    summary = attach_bucket_range_labels(summary, ranges, bucket_col="moneyness_equal_bucket", output_col="moneyness_equal_bucket_label", value_kind="moneyness")
    return figure_equal_count_bucket_metric(
        summary,
        bucket_col="moneyness_equal_bucket",
        bucket_label_col="moneyness_equal_bucket_label",
        metric=metric,
        bucket_title="Moneyness range (K/F)",
        height=height,
    )


def figure_equal_count_basis_metric(rep_panel: pd.DataFrame, metric: str = "rmse_reduction", q: int = 5, height: int = 520) -> go.Figure:
    _, summary, _ = build_equal_count_bucket_summary(rep_panel, value_col="basis_abs", bucket_col="basis_equal_bucket", q=q, group_cols=("currency",))
    if summary.empty:
        return apply_thesis_layout(go.Figure(), title="No equal-count basis data", height=height)
    return figure_equal_count_bucket_metric(summary, bucket_col="basis_equal_bucket", metric=metric, bucket_title=f"Equal-count basis bucket (Q={q})", height=height)


def choose_representative_model(headline_test: pd.DataFrame) -> str:
    if headline_test.empty:
        return "black"
    tmp = headline_test.copy()
    tmp = tmp[tmp["hedge_type"].astype(str).str.lower() == "net_delta"].copy()
    if tmp.empty:
        return "black"
    score = tmp.groupby("model", dropna=False)["rmse_reduction"].mean().sort_values(ascending=False)
    return str(score.index[0]) if len(score) else "black"


def export_figures_html(figures: dict[str, go.Figure], export_dir: Path) -> None:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.write_html(export_dir / f"{name}.html")


def export_tables_csv(tables: dict[str, pd.DataFrame], export_dir: Path) -> None:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(export_dir / f"{name}.csv", index=False)
