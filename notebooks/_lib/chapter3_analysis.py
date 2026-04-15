from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import configure_notebook_display


MODEL_SPECS = {
    "Black": {"label": "Black-Scholes", "price_col": "price_black"},
    "Heston": {"label": "Heston", "price_col": "price_heston"},
    "SVCJ": {"label": "SVCJ", "price_col": "price_svcj"},
}

COLORS = {
    "Black": "#636EFA",
    "Heston": "#EF553B",
    "SVCJ": "#00CC96",
}

FIGDIMS = dict(width=1200, height=1100)
EPS = 1e-12
MONEY_BINS = [0.0, 0.05, 0.15, 0.30, np.inf]
MONEY_LABELS = ["|log(K/F)|≤0.05", "0.05–0.15", "0.15–0.30", ">0.30"]
T_BINS = [0.0, 7 / 365, 30 / 365, 90 / 365, np.inf]
T_LABELS = ["≤1w", "1w–1m", "1m–3m", ">3m"]
RHO_LB = np.tanh(-5.0)
RHO_UB = np.tanh(5.0)

BOUNDS = {
    "Black": {"sigma": (1e-4, 5.0)},
    "Heston": {
        "kappa": (1e-4, 50.0),
        "theta": (1e-6, 5.0),
        "sigma_v": (1e-4, 10.0),
        "rho": (RHO_LB, RHO_UB),
        "v0": (1e-6, 5.0),
    },
    "SVCJ": {
        "kappa": (1e-4, 50.0),
        "theta": (1e-6, 5.0),
        "sigma_v": (1e-4, 10.0),
        "rho": (RHO_LB, RHO_UB),
        "v0": (1e-6, 5.0),
        "lam": (1e-6, 10.0),
        "ell_y": (-5.0, 5.0),
        "sigma_y": (1e-4, 5.0),
        "ell_v": (1e-6, 10.0),
        "rho_j": (RHO_LB, RHO_UB),
    },
}


@dataclass
class AnalysisState:
    data_path: Path
    black_params: pd.DataFrame
    heston_params: pd.DataFrame
    svcj_params: pd.DataFrame
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    results_long: pd.DataFrame
    results_ok: pd.DataFrame
    opt_metrics: pd.DataFrame
    bucket_all: pd.DataFrame
    rng: np.random.Generator


def initialize_notebook_defaults() -> np.random.Generator:
    configure_notebook_display(
        max_columns=200,
        width=180,
        template="plotly_white",
        renderer="plotly_mimetype",
    )
    return np.random.default_rng(123)


def load_workbook(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    black_params = pd.read_excel(path, sheet_name="black_params")
    heston_params = pd.read_excel(path, sheet_name="heston_params")
    svcj_params = pd.read_excel(path, sheet_name="svcj_params")
    train_data = pd.read_excel(path, sheet_name="train_data")
    test_data = pd.read_excel(path, sheet_name="test_data")

    for frame in (black_params, heston_params, svcj_params):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    for frame in (train_data, test_data):
        frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True)
        frame["expiry_datetime"] = pd.to_datetime(frame["expiry_datetime"], utc=True)

    return black_params, heston_params, svcj_params, train_data, test_data


def _to_long(df: pd.DataFrame, model_name: str, param_cols: list[str]) -> pd.DataFrame:
    base_cols = [
        "timestamp",
        "currency",
        "success",
        "message",
        "nfev",
        "rmse_fit",
        "mae_fit",
        "rmse_train",
        "mae_train",
        "rmse_test",
        "mae_test",
        "n_options_total",
        "n_train",
        "n_test",
        "random_seed",
    ]
    keep = base_cols + param_cols
    out = df[keep].copy()
    out["model"] = model_name
    return out


def build_results_long(
    black_params: pd.DataFrame,
    heston_params: pd.DataFrame,
    svcj_params: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_long = pd.concat(
        [
            _to_long(black_params, "Black", ["sigma"]),
            _to_long(heston_params, "Heston", ["kappa", "theta", "sigma_v", "rho", "v0"]),
            _to_long(
                svcj_params,
                "SVCJ",
                ["kappa", "theta", "sigma_v", "rho", "v0", "lam", "ell_y", "sigma_y", "ell_v", "rho_j"],
            ),
        ],
        ignore_index=True,
    )
    results_long = results_long.sort_values(["currency", "timestamp", "model"]).reset_index(drop=True)
    results_ok = results_long.loc[results_long["success"] == True].copy()
    return results_long, add_feller_ratio(results_ok)


def compute_snapshot_metrics_from_quotes(df_quotes: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out_all: list[pd.DataFrame] = []
    spread_col = "bid_ask_spread" if "bid_ask_spread" in df_quotes.columns else "spread"

    for model_key, spec in MODEL_SPECS.items():
        price_col = spec["price_col"]
        if price_col not in df_quotes.columns:
            continue

        tmp = df_quotes[
            ["currency", "snapshot_ts", "mid_price_clean", spread_col, "log_moneyness", "time_to_maturity", price_col]
        ].copy()
        tmp = tmp.rename(columns={price_col: "price_model", spread_col: "spread_abs"})
        tmp["mid_price_clean"] = pd.to_numeric(tmp["mid_price_clean"], errors="coerce")
        tmp["price_model"] = pd.to_numeric(tmp["price_model"], errors="coerce")
        tmp["spread_abs"] = pd.to_numeric(tmp["spread_abs"], errors="coerce")
        tmp = tmp[np.isfinite(tmp["mid_price_clean"]) & np.isfinite(tmp["price_model"]) & np.isfinite(tmp["spread_abs"])].copy()
        tmp = tmp.loc[tmp["spread_abs"] > 0].copy()

        err = tmp["price_model"] - tmp["mid_price_clean"]
        abs_err = err.abs()
        tmp["err2"] = err * err
        tmp["abs_err"] = abs_err
        tmp["within_spread"] = (abs_err <= tmp["spread_abs"]).astype(float)
        tmp["within_half_spread"] = (abs_err <= 0.5 * tmp["spread_abs"]).astype(float)
        tmp["abs_err_over_spread"] = abs_err / (tmp["spread_abs"] + EPS)
        tmp["smape"] = 2.0 * abs_err / (tmp["price_model"].abs() + tmp["mid_price_clean"].abs() + EPS)

        grouped = tmp.groupby(["currency", "snapshot_ts"], as_index=False)
        agg = grouped.agg(
            n=("abs_err", "count"),
            mse=("err2", "mean"),
            mae=("abs_err", "mean"),
            within_spread=("within_spread", "mean"),
            within_half_spread=("within_half_spread", "mean"),
            abs_err_over_spread=("abs_err_over_spread", "mean"),
            smape=("smape", "mean"),
            mean_price=("mid_price_clean", "mean"),
        )
        agg["rmse"] = np.sqrt(agg["mse"])
        agg["rmse_over_mean_price"] = agg["rmse"] / (agg["mean_price"].abs() + EPS)
        agg["model"] = model_key
        agg["split"] = split_name
        out_all.append(agg)

    if not out_all:
        return pd.DataFrame()
    return pd.concat(out_all, ignore_index=True).sort_values(["currency", "snapshot_ts", "model", "split"]).reset_index(drop=True)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 3000,
    alpha: float = 0.05,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    if n == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = clean[idx].mean(axis=1)
    lo = np.quantile(boot_means, alpha / 2)
    hi = np.quantile(boot_means, 1 - alpha / 2)
    return clean.mean(), lo, hi


def summarize_snapshot_series(values: pd.Series, *, n_boot: int = 3000, rng: np.random.Generator) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    mean, lo, hi = bootstrap_mean_ci(arr, n_boot=n_boot, rng=rng)
    return {
        "n": len(arr),
        "mean": float(mean),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def add_line(fig: go.Figure, row: int, col: int, df: pd.DataFrame, xcol: str, ycol: str, name: str, color: str) -> None:
    series = df.groupby(xcol, as_index=False)[ycol].mean()
    fig.add_trace(
        go.Scatter(x=series[xcol], y=series[ycol], mode="lines", line=dict(color=color, width=2), name=name, showlegend=False),
        row=row,
        col=col,
    )


def add_box(fig: go.Figure, row: int, col: int, values: np.ndarray, name: str, color: str) -> None:
    fig.add_trace(
        go.Box(
            y=values,
            name=name,
            marker_color=color,
            boxmean=True,
            boxpoints="outliers",
            jitter=0.15,
            pointpos=0.0,
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _subplot_axis_suffix_2x2(row: int, col: int) -> str:
    idx = (row - 1) * 2 + col
    return "" if idx == 1 else str(idx)


def add_subplot_legend(fig: go.Figure, row: int, col: int, model_keys: list[str], *, font_size: int = 12) -> None:
    suffix = _subplot_axis_suffix_2x2(row, col)
    xref = f"x{suffix} domain" if suffix else "x domain"
    yref = f"y{suffix} domain" if suffix else "y domain"
    lines = [
        f"<span style='color:{COLORS[model_key]}'>●</span> {MODEL_SPECS[model_key]['label']}"
        for model_key in model_keys
    ]
    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref=xref,
        yref=yref,
        text="<br>".join(lines),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.70)",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1,
        font=dict(size=font_size),
    )


def plot_error_timeseries(results_long_df: pd.DataFrame, currency: str, *, split: str = "test") -> go.Figure:
    metric_rmse = f"rmse_{split}"
    metric_mae = f"mae_{split}"
    df = results_long_df[(results_long_df["currency"] == currency) & (results_long_df["success"] == True)].copy().sort_values("timestamp")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"RMSE {split.title()}",
            f"MAE {split.title()}",
            f"RMSE {split.title()} – Heston vs SVCJ",
            f"MAE {split.title()} – Heston vs SVCJ",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    for model in ["Black", "Heston", "SVCJ"]:
        sub = df[df["model"] == model]
        add_line(fig, 1, 1, sub, "timestamp", metric_rmse, MODEL_SPECS[model]["label"], COLORS[model])
        add_line(fig, 1, 2, sub, "timestamp", metric_mae, MODEL_SPECS[model]["label"], COLORS[model])

    for model in ["Heston", "SVCJ"]:
        sub = df[df["model"] == model]
        add_line(fig, 2, 1, sub, "timestamp", metric_rmse, MODEL_SPECS[model]["label"], COLORS[model])
        add_line(fig, 2, 2, sub, "timestamp", metric_mae, MODEL_SPECS[model]["label"], COLORS[model])

    add_subplot_legend(fig, 1, 1, ["Black", "Heston", "SVCJ"])
    add_subplot_legend(fig, 1, 2, ["Black", "Heston", "SVCJ"])
    add_subplot_legend(fig, 2, 1, ["Heston", "SVCJ"])
    add_subplot_legend(fig, 2, 2, ["Heston", "SVCJ"])

    fig.update_layout(title=f"{currency} — {split.title()} error time series (snapshot-level)", showlegend=False, **FIGDIMS)
    fig.update_yaxes(title_text="RMSE (coin premium)", row=1, col=1)
    fig.update_yaxes(title_text="MAE (coin premium)", row=1, col=2)
    fig.update_yaxes(title_text="RMSE (coin premium)", row=2, col=1)
    fig.update_yaxes(title_text="MAE (coin premium)", row=2, col=2)
    return fig


def plot_error_boxplots(results_long_df: pd.DataFrame, currency: str, *, split: str = "test") -> go.Figure:
    metric_rmse = f"rmse_{split}"
    metric_mae = f"mae_{split}"
    df = results_long_df[(results_long_df["currency"] == currency) & (results_long_df["success"] == True)].copy()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"RMSE {split.title()} (distribution across snapshots)",
            f"MAE {split.title()} (distribution across snapshots)",
            f"RMSE {split.title()} – Heston vs SVCJ",
            f"MAE {split.title()} – Heston vs SVCJ",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    for model in ["Black", "Heston", "SVCJ"]:
        add_box(fig, 1, 1, df.loc[df["model"] == model, metric_rmse].dropna().values, MODEL_SPECS[model]["label"], COLORS[model])
        add_box(fig, 1, 2, df.loc[df["model"] == model, metric_mae].dropna().values, MODEL_SPECS[model]["label"], COLORS[model])

    for model in ["Heston", "SVCJ"]:
        add_box(fig, 2, 1, df.loc[df["model"] == model, metric_rmse].dropna().values, MODEL_SPECS[model]["label"], COLORS[model])
        add_box(fig, 2, 2, df.loc[df["model"] == model, metric_mae].dropna().values, MODEL_SPECS[model]["label"], COLORS[model])

    add_subplot_legend(fig, 1, 1, ["Black", "Heston", "SVCJ"])
    add_subplot_legend(fig, 1, 2, ["Black", "Heston", "SVCJ"])
    add_subplot_legend(fig, 2, 1, ["Heston", "SVCJ"])
    add_subplot_legend(fig, 2, 2, ["Heston", "SVCJ"])

    fig.update_layout(title=f"{currency} — {split.title()} error boxplots (snapshot-level)", showlegend=False, **FIGDIMS)
    fig.update_yaxes(title_text="RMSE (coin premium)", row=1, col=1)
    fig.update_yaxes(title_text="MAE (coin premium)", row=1, col=2)
    fig.update_yaxes(title_text="RMSE (coin premium)", row=2, col=1)
    fig.update_yaxes(title_text="MAE (coin premium)", row=2, col=2)
    return fig


def error_summary_table(
    results_long_df: pd.DataFrame,
    currency: str,
    *,
    split: str = "test",
    n_boot: int = 3000,
    rng: np.random.Generator,
) -> pd.DataFrame:
    metric_cols = [(f"rmse_{split}", "RMSE"), (f"mae_{split}", "MAE")]
    df = results_long_df[(results_long_df["currency"] == currency) & (results_long_df["success"] == True)].copy().sort_values("timestamp")

    rows: list[dict[str, object]] = []
    for col, metric_name in metric_cols:
        for model in ["Black", "Heston", "SVCJ"]:
            summary = summarize_snapshot_series(df.loc[df["model"] == model, col], n_boot=n_boot, rng=rng)
            rows.append({"currency": currency, "split": split, "metric": metric_name, "item": model, **summary})

        wide = df.pivot_table(index="timestamp", columns="model", values=col, aggfunc="mean")
        if {"Black", "Heston", "SVCJ"}.issubset(wide.columns):
            diff_hb = wide["Black"] - wide["Heston"]
            pct_hb = diff_hb / wide["Black"]
            summary_diff_hb = summarize_snapshot_series(diff_hb, n_boot=n_boot, rng=rng)
            summary_pct_hb = summarize_snapshot_series(pct_hb, n_boot=n_boot, rng=rng)
            win_hb = float((wide["Heston"] < wide["Black"]).mean())
            rows.append(
                {
                    "currency": currency,
                    "split": split,
                    "metric": metric_name,
                    "item": "GAIN: Black→Heston (abs)",
                    **summary_diff_hb,
                    "win_rate": win_hb,
                }
            )
            rows.append(
                {
                    "currency": currency,
                    "split": split,
                    "metric": metric_name,
                    "item": "GAIN: Black→Heston (%)",
                    **summary_pct_hb,
                    "win_rate": win_hb,
                }
            )

            diff_sh = wide["Heston"] - wide["SVCJ"]
            pct_sh = diff_sh / wide["Heston"]
            summary_diff_sh = summarize_snapshot_series(diff_sh, n_boot=n_boot, rng=rng)
            summary_pct_sh = summarize_snapshot_series(pct_sh, n_boot=n_boot, rng=rng)
            win_sh = float((wide["SVCJ"] < wide["Heston"]).mean())
            rows.append(
                {
                    "currency": currency,
                    "split": split,
                    "metric": metric_name,
                    "item": "GAIN: Heston→SVCJ (abs)",
                    **summary_diff_sh,
                    "win_rate": win_sh,
                }
            )
            rows.append(
                {
                    "currency": currency,
                    "split": split,
                    "metric": metric_name,
                    "item": "GAIN: Heston→SVCJ (%)",
                    **summary_pct_sh,
                    "win_rate": win_sh,
                }
            )

    out = pd.DataFrame(rows)
    out["ci_95"] = out.apply(lambda row: f"[{row['ci_low']:.6g}, {row['ci_high']:.6g}]" if pd.notna(row["ci_low"]) else "", axis=1)
    cols = ["currency", "split", "metric", "item", "n", "mean", "ci_95", "median", "q25", "q75", "std", "min", "max", "win_rate"]
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[cols].sort_values(["metric", "item"]).reset_index(drop=True)


def convergence_table(results_long_df: pd.DataFrame) -> pd.DataFrame:
    df = results_long_df.copy()
    df["hit_cap"] = df["message"].astype(str).str.contains("maximum number of function evaluations", case=False, regex=False)
    grouped = df.groupby(["currency", "model"], as_index=False)
    out = grouped.agg(
        n_snapshots=("timestamp", "count"),
        success_rate=("success", "mean"),
        nfev_median=("nfev", "median"),
        nfev_mean=("nfev", "mean"),
        nfev_p90=("nfev", lambda series: float(np.quantile(pd.to_numeric(series, errors="coerce"), 0.90))),
        nfev_max=("nfev", "max"),
        hit_cap_rate=("hit_cap", "mean"),
    )

    msg = (
        df.groupby(["currency", "model", "message"])
        .size()
        .reset_index(name="count")
        .sort_values(["currency", "model", "count"], ascending=[True, True, False])
    )
    top_msgs = msg.groupby(["currency", "model"]).head(3).copy()
    top_msgs["rank"] = top_msgs.groupby(["currency", "model"]).cumcount() + 1
    top_msgs = top_msgs.pivot_table(index=["currency", "model"], columns="rank", values="message", aggfunc="first").reset_index()
    top_msgs = top_msgs.rename(columns={1: "top_message_1", 2: "top_message_2", 3: "top_message_3"})
    return out.merge(top_msgs, on=["currency", "model"], how="left").sort_values(["currency", "model"]).reset_index(drop=True)


def spread_metric_timeseries(opt_metrics_df: pd.DataFrame, currency: str, *, split: str = "test") -> go.Figure:
    df = opt_metrics_df[(opt_metrics_df["currency"] == currency) & (opt_metrics_df["split"] == split)].copy().sort_values("snapshot_ts")
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Within spread (fraction)",
            "Within half-spread (fraction)",
            "Mean |error| / spread",
            "sMAPE",
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )
    for model in ["Black", "Heston", "SVCJ"]:
        sub = df[df["model"] == model]
        add_line(fig, 1, 1, sub, "snapshot_ts", "within_spread", MODEL_SPECS[model]["label"], COLORS[model])
        add_line(fig, 1, 2, sub, "snapshot_ts", "within_half_spread", MODEL_SPECS[model]["label"], COLORS[model])
        add_line(fig, 2, 1, sub, "snapshot_ts", "abs_err_over_spread", MODEL_SPECS[model]["label"], COLORS[model])
        add_line(fig, 2, 2, sub, "snapshot_ts", "smape", MODEL_SPECS[model]["label"], COLORS[model])

    add_subplot_legend(fig, 1, 1, ["Black", "Heston", "SVCJ"])
    add_subplot_legend(fig, 1, 2, ["Black", "Heston", "SVCJ"])
    add_subplot_legend(fig, 2, 1, ["Black", "Heston", "SVCJ"])
    add_subplot_legend(fig, 2, 2, ["Black", "Heston", "SVCJ"])

    fig.update_layout(title=f"{currency} — {split.title()} spread-relative diagnostics (per snapshot)", showlegend=False, width=1200, height=900)
    return fig


def spread_metric_summary_table(
    opt_metrics_df: pd.DataFrame,
    currency: str,
    *,
    split: str = "test",
    n_boot: int = 3000,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = opt_metrics_df[(opt_metrics_df["currency"] == currency) & (opt_metrics_df["split"] == split)].copy()
    rows: list[dict[str, object]] = []
    for model in ["Black", "Heston", "SVCJ"]:
        sub = df[df["model"] == model]
        for col, metric in [
            ("within_spread", "within_spread"),
            ("within_half_spread", "within_half_spread"),
            ("abs_err_over_spread", "abs_err_over_spread"),
            ("smape", "sMAPE"),
            ("rmse_over_mean_price", "rmse_over_mean_price"),
        ]:
            summary = summarize_snapshot_series(sub[col], n_boot=n_boot, rng=rng)
            rows.append(
                {
                    "currency": currency,
                    "split": split,
                    "model": model,
                    "metric": metric,
                    **summary,
                    "ci_95": f"[{summary['ci_low']:.6g}, {summary['ci_high']:.6g}]",
                }
            )
    out = pd.DataFrame(rows)
    return out[
        ["currency", "split", "model", "metric", "n", "mean", "ci_95", "median", "q25", "q75", "std", "min", "max"]
    ].sort_values(["metric", "model"])


def _add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abs_log_moneyness"] = out["log_moneyness"].abs()
    out["m_bucket"] = pd.cut(out["abs_log_moneyness"], bins=MONEY_BINS, labels=MONEY_LABELS, right=True, include_lowest=True)
    out["t_bucket"] = pd.cut(out["time_to_maturity"], bins=T_BINS, labels=T_LABELS, right=True, include_lowest=True)
    return out


def bucket_mae_by_snapshot(df_quotes: pd.DataFrame, currency: str, *, split: str = "test") -> pd.DataFrame:
    df = _add_buckets(df_quotes[df_quotes["currency"] == currency].copy())
    results: list[pd.DataFrame] = []

    for model_key, spec in MODEL_SPECS.items():
        price_col = spec["price_col"]
        if price_col not in df.columns:
            continue
        tmp = df[["snapshot_ts", "m_bucket", "t_bucket", "mid_price_clean", price_col]].copy().rename(columns={price_col: "price_model"})
        tmp["mid_price_clean"] = pd.to_numeric(tmp["mid_price_clean"], errors="coerce")
        tmp["price_model"] = pd.to_numeric(tmp["price_model"], errors="coerce")
        tmp = tmp[np.isfinite(tmp["mid_price_clean"]) & np.isfinite(tmp["price_model"])].copy()
        tmp["abs_err"] = (tmp["price_model"] - tmp["mid_price_clean"]).abs()

        bucket_m = tmp.groupby(["snapshot_ts", "m_bucket"], as_index=False)["abs_err"].mean()
        bucket_m["model"] = model_key
        bucket_m["split"] = split
        bucket_m["bucket_type"] = "moneyness"
        results.append(bucket_m.rename(columns={"m_bucket": "bucket", "abs_err": "mae"}))

        bucket_t = tmp.groupby(["snapshot_ts", "t_bucket"], as_index=False)["abs_err"].mean()
        bucket_t["model"] = model_key
        bucket_t["split"] = split
        bucket_t["bucket_type"] = "maturity"
        results.append(bucket_t.rename(columns={"t_bucket": "bucket", "abs_err": "mae"}))

    out = pd.concat(results, ignore_index=True)
    out["currency"] = currency
    return out


def bucket_summary_table(bucket_df: pd.DataFrame, currency: str, bucket_type: str, *, n_boot: int = 2000, rng: np.random.Generator) -> pd.DataFrame:
    df = bucket_df[(bucket_df["currency"] == currency) & (bucket_df["bucket_type"] == bucket_type)].copy()
    rows: list[dict[str, object]] = []
    for model in ["Black", "Heston", "SVCJ"]:
        sub = df[df["model"] == model]
        for bucket in sub["bucket"].dropna().unique():
            vals = sub[sub["bucket"] == bucket].groupby("snapshot_ts")["mae"].mean()
            summary = summarize_snapshot_series(vals, n_boot=n_boot, rng=rng)
            rows.append(
                {
                    "currency": currency,
                    "bucket_type": bucket_type,
                    "bucket": str(bucket),
                    "model": model,
                    **summary,
                    "ci_95": f"[{summary['ci_low']:.6g}, {summary['ci_high']:.6g}]",
                }
            )
    out = pd.DataFrame(rows)
    return out[["currency", "bucket_type", "bucket", "model", "n", "mean", "ci_95", "median", "q25", "q75"]].sort_values(["bucket", "model"])


def plot_bucket_bars(bucket_df: pd.DataFrame, currency: str, bucket_type: str) -> go.Figure:
    df = bucket_df[(bucket_df["currency"] == currency) & (bucket_df["bucket_type"] == bucket_type)].copy()
    df_mean = df.groupby(["model", "bucket"], as_index=False)["mae"].mean()
    order = MONEY_LABELS if bucket_type == "moneyness" else T_LABELS
    df_mean["bucket"] = pd.Categorical(df_mean["bucket"], categories=order, ordered=True)
    df_mean = df_mean.sort_values("bucket")

    fig = go.Figure()
    for model in ["Black", "Heston", "SVCJ"]:
        sub = df_mean[df_mean["model"] == model]
        fig.add_trace(
            go.Bar(
                x=sub["bucket"].astype(str),
                y=sub["mae"],
                name=MODEL_SPECS[model]["label"],
                marker_color=COLORS[model],
            )
        )
    fig.update_layout(
        title=f"{currency} — Test MAE by {bucket_type} bucket (snapshot-equal-weighted)",
        barmode="group",
        width=1100,
        height=500,
    )
    fig.update_yaxes(title_text="MAE (coin premium)")
    return fig


def add_feller_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"kappa", "theta", "sigma_v"}.issubset(out.columns):
        out["feller_ratio"] = (out["sigma_v"] ** 2) / (2.0 * out["kappa"] * out["theta"] + EPS)
    return out


def near_bound_rates(df: pd.DataFrame, model: str, *, tol: float = 0.02) -> pd.DataFrame:
    bounds = BOUNDS[model]
    sub = df[(df["model"] == model) & (df["success"] == True)].copy()
    out: list[dict[str, float | str]] = []
    for param, (lb, ub) in bounds.items():
        if param not in sub.columns:
            continue
        values = pd.to_numeric(sub[param], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        rng = ub - lb
        out.append(
            {
                "model": model,
                "param": param,
                "lb": lb,
                "ub": ub,
                "near_lb_rate": np.mean((values - lb) <= tol * rng),
                "near_ub_rate": np.mean((ub - values) <= tol * rng),
                "min": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(out).sort_values(["model", "param"]).reset_index(drop=True)


def plot_param_timeseries(results_long_df: pd.DataFrame, currency: str, model: str, params: list[str], title: str) -> go.Figure:
    df = add_feller_ratio(results_long_df[(results_long_df["currency"] == currency) & (results_long_df["model"] == model) & (results_long_df["success"] == True)].copy()).sort_values("timestamp")
    ncols = 2
    nrows = int(np.ceil(len(params) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=params, vertical_spacing=0.10, horizontal_spacing=0.10)
    for idx, param in enumerate(params):
        row = idx // ncols + 1
        col = idx % ncols + 1
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[param],
                mode="lines",
                line=dict(color=COLORS.get(model, "#333333"), width=2),
                name=param,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_layout(title=title, width=1200, height=320 * nrows)
    return fig


def plot_param_boxplots(results_long_df: pd.DataFrame, currency: str, model: str, params: list[str], title: str) -> go.Figure:
    df = add_feller_ratio(results_long_df[(results_long_df["currency"] == currency) & (results_long_df["model"] == model) & (results_long_df["success"] == True)].copy())
    ncols = 2
    nrows = int(np.ceil(len(params) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=params, vertical_spacing=0.12, horizontal_spacing=0.12)
    for idx, param in enumerate(params):
        row = idx // ncols + 1
        col = idx % ncols + 1
        add_box(fig, row, col, df[param].dropna().values, param, COLORS.get(model, "#333333"))
    fig.update_layout(title=title, width=1200, height=320 * nrows, showlegend=False)
    return fig


def build_analysis_state(data_path: Path, *, rng: np.random.Generator | None = None) -> AnalysisState:
    current_rng = rng or initialize_notebook_defaults()
    black_params, heston_params, svcj_params, train_data, test_data = load_workbook(data_path)
    results_long, results_ok = build_results_long(black_params, heston_params, svcj_params)
    opt_metrics = pd.concat(
        [
            compute_snapshot_metrics_from_quotes(train_data, "train"),
            compute_snapshot_metrics_from_quotes(test_data, "test"),
        ],
        ignore_index=True,
    ).sort_values(["currency", "snapshot_ts", "model", "split"]).reset_index(drop=True)
    bucket_all = pd.concat(
        [
            bucket_mae_by_snapshot(test_data, "BTC", split="test"),
            bucket_mae_by_snapshot(test_data, "ETH", split="test"),
        ],
        ignore_index=True,
    )
    return AnalysisState(
        data_path=data_path,
        black_params=black_params,
        heston_params=heston_params,
        svcj_params=svcj_params,
        train_data=train_data,
        test_data=test_data,
        results_long=results_long,
        results_ok=results_ok,
        opt_metrics=opt_metrics,
        bucket_all=bucket_all,
        rng=current_rng,
    )


def run_currency_report(state: AnalysisState, currency: str, *, n_boot: int = 3000) -> None:
    print("=" * 90)
    print(f"REPORT — {currency}")
    print("=" * 90)

    sub = state.results_long[state.results_long["currency"] == currency]
    print("Snapshots:", sub["timestamp"].nunique())
    print("Success rates:")
    from IPython.display import display

    display(sub.groupby("model")["success"].mean().to_frame("success_rate"))

    for split in ["train", "test"]:
        plot_error_timeseries(state.results_long, currency, split=split).show()
        plot_error_boxplots(state.results_long, currency, split=split).show()
        print(f"Summary table — {currency} / {split}")
        display(error_summary_table(state.results_long, currency, split=split, n_boot=n_boot, rng=state.rng))

    for split in ["train", "test"]:
        spread_metric_timeseries(state.opt_metrics, currency, split=split).show()
        print(f"Spread-relative summary — {currency} / {split}")
        display(spread_metric_summary_table(state.opt_metrics, currency, split=split, n_boot=n_boot, rng=state.rng))

    print("Bucket tables (test) — moneyness & maturity")
    display(bucket_summary_table(state.bucket_all, currency, "moneyness", n_boot=max(1000, n_boot // 2), rng=state.rng))
    display(bucket_summary_table(state.bucket_all, currency, "maturity", n_boot=max(1000, n_boot // 2), rng=state.rng))
    plot_bucket_bars(state.bucket_all, currency, "moneyness").show()
    plot_bucket_bars(state.bucket_all, currency, "maturity").show()

    print("Parameter stability — Heston")
    hes_params = ["kappa", "theta", "sigma_v", "rho", "v0", "feller_ratio"]
    plot_param_timeseries(state.results_long, currency, "Heston", hes_params, f"{currency} — Heston parameter time series").show()
    plot_param_boxplots(state.results_long, currency, "Heston", hes_params, f"{currency} — Heston parameter distributions").show()
    display(near_bound_rates(state.results_long[state.results_long["currency"] == currency], "Heston"))

    print("Parameter stability — SVCJ")
    svcj_params = ["kappa", "theta", "sigma_v", "rho", "v0", "lam", "ell_y", "sigma_y", "ell_v", "rho_j", "feller_ratio"]
    plot_param_timeseries(state.results_long, currency, "SVCJ", svcj_params, f"{currency} — SVCJ parameter time series").show()
    plot_param_boxplots(state.results_long, currency, "SVCJ", svcj_params, f"{currency} — SVCJ parameter distributions").show()
    display(near_bound_rates(state.results_long[state.results_long["currency"] == currency], "SVCJ"))

