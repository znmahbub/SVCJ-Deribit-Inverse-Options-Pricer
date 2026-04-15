from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import save_plotly_figure, style_figure


MODEL_LABELS = {"black": "Black", "heston": "Heston", "svcj": "SVCJ"}
MODEL_COLORS = {"black": "#6b7280", "heston": "#2563eb", "svcj": "#dc2626"}


def find_calibration_files(search_roots: list[Path], pattern: str) -> list[Path]:
    found: dict[Path, Path] = {}
    for root in search_roots:
        if root.exists():
            for path in root.glob(pattern):
                found[path.resolve()] = path.resolve()
    files = sorted(found.values(), key=lambda path: int(re.search(r"reg_(\d+)", path.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} were found in {[str(path) for path in search_roots]}")
    return files


def reg_from_path(path: Path) -> int:
    match = re.search(r"reg_(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse regularisation value from {path.name}")
    return int(match.group(1))


def save_figure(fig: go.Figure, output_dir: Path, stem: str) -> None:
    save_plotly_figure(fig, output_dir, stem)


def equal_count_bucket_labels(n_buckets: int) -> list[str]:
    return [f"Q{i}" for i in range(1, n_buckets + 1)]


def bucketize_equal_count(series: pd.Series, *, n_buckets: int = 5) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=n_buckets, labels=equal_count_bucket_labels(n_buckets))


def compute_regularisation_summary(files: list[Path], output_dir: Path) -> pd.DataFrame:
    rows = []
    selected = ["kappa", "theta", "sigma_v", "rho", "v0", "lam", "ell_y", "sigma_y", "ell_v", "rho_j"]

    for path in files:
        reg = reg_from_path(path)
        svcj = pd.read_excel(path, sheet_name="svcj_params")
        svcj["timestamp"] = pd.to_datetime(svcj["timestamp"], utc=True)
        svcj["feller_gap"] = 2 * svcj["kappa"] * svcj["theta"] - svcj["sigma_v"] ** 2

        roughness_terms: list[float] = []
        for _, group in svcj.sort_values("timestamp").groupby("currency"):
            for param in selected:
                series = group[param].astype(float)
                iqr = series.quantile(0.75) - series.quantile(0.25)
                scale = iqr if pd.notna(iqr) and iqr > 1e-12 else series.std()
                if pd.isna(scale) or scale <= 1e-12:
                    continue
                roughness_terms.append(float(series.diff().abs().median() / scale))

        rows.append(
            {
                "reg": reg,
                "rmse_test_mean": float(svcj["rmse_test"].mean()),
                "mae_test_mean": float(svcj["mae_test"].mean()),
                "nfev_mean": float(svcj["nfev"].mean()),
                "nfev_median": float(svcj["nfev"].median()),
                "roughness_mean": float(np.mean(roughness_terms)),
                "feller_violation_share": float((svcj["feller_gap"] < 0).mean()),
                "rho_boundary_share": float((svcj["rho"].abs() > 0.95).mean()),
                "rhoj_boundary_share": float((svcj["rho_j"].abs() > 0.95).mean()),
            }
        )

    out = pd.DataFrame(rows).sort_values("reg").reset_index(drop=True)
    out.to_csv(output_dir / "regularisation_summary.csv", index=False)
    return out


def build_regularisation_tradeoff_figure(reg_summary: pd.DataFrame, *, final_reg: int, output_dir: Path) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.20,
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
    )

    fig.add_trace(
        go.Scatter(
            x=reg_summary["reg"],
            y=reg_summary["rmse_test_mean"],
            mode="lines+markers+text",
            name="Mean test RMSE",
            text=reg_summary["reg"].astype(str),
            textposition="top center",
            textfont=dict(size=10),
            cliponaxis=False,
            line=dict(color="#111827", width=2),
            marker=dict(size=8),
            hovertemplate="λ_TS=%{x}<br>Mean test RMSE=%{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=reg_summary["reg"],
            y=reg_summary["nfev_mean"],
            mode="lines+markers",
            name="Mean function evaluations",
            line=dict(color="#2563eb", width=2, dash="dash"),
            marker=dict(size=8),
            hovertemplate="λ_TS=%{x}<br>Mean function evaluations=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=reg_summary["reg"],
            y=reg_summary["roughness_mean"],
            mode="lines+markers+text",
            name="Roughness index",
            text=reg_summary["reg"].astype(str),
            textposition="top center",
            textfont=dict(size=10),
            cliponaxis=False,
            line=dict(color="#dc2626", width=2),
            marker=dict(size=8),
            hovertemplate="λ_TS=%{x}<br>Roughness index=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    for col in [1, 2]:
        fig.add_vline(x=final_reg, line_width=1.5, line_dash="dot", line_color="#dc2626", row=1, col=col)

    tickvals = reg_summary["reg"].tolist()
    ticktext = [str(value) for value in tickvals]
    fig.update_xaxes(type="log", tickvals=tickvals, ticktext=ticktext, title_text="λ_TS (log scale)", row=1, col=1, showgrid=True)
    fig.update_xaxes(type="log", tickvals=tickvals, ticktext=ticktext, title_text="λ_TS (log scale)", row=1, col=2, showgrid=True)
    fig.update_yaxes(title_text="Mean test RMSE", row=1, col=1, secondary_y=False, tickformat=".5f", showgrid=True)
    fig.update_yaxes(title_text="Mean function evaluations", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="Roughness index", row=1, col=2, showgrid=True)

    style_figure(fig, height=520, title="Regularisation trade-off across the SVCJ sweep", legend_y=1.10)
    fig.update_layout(
        margin=dict(t=95, b=70, l=70, r=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    save_figure(fig, output_dir, "regularisation_tradeoff")
    return fig


def load_baseline_workbook(baseline_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    black_params = pd.read_excel(baseline_path, sheet_name="black_params")
    heston_params = pd.read_excel(baseline_path, sheet_name="heston_params")
    svcj_params = pd.read_excel(baseline_path, sheet_name="svcj_params")
    for frame in [black_params, heston_params, svcj_params]:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    test_cols = [
        "snapshot_ts",
        "currency",
        "time_to_maturity",
        "moneyness",
        "bid_price",
        "ask_price",
        "mid_price_clean",
        "price_black",
        "price_heston",
        "price_svcj",
    ]
    test_data = pd.read_excel(baseline_path, sheet_name="test_data", usecols=test_cols)
    test_data["snapshot_ts"] = pd.to_datetime(test_data["snapshot_ts"], utc=True)
    for model in ["black", "heston", "svcj"]:
        error = test_data[f"price_{model}"] - test_data["mid_price_clean"]
        test_data[f"abs_err_{model}"] = error.abs()
        test_data[f"sq_err_{model}"] = error ** 2
        test_data[f"within_{model}"] = (
            (test_data[f"price_{model}"] >= test_data["bid_price"]) & (test_data[f"price_{model}"] <= test_data["ask_price"])
        ).astype(int)
    return black_params, heston_params, svcj_params, test_data


def compute_overall_baseline_metrics(test_data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for currency, frame in [("BTC", test_data.query("currency == 'BTC'")), ("ETH", test_data.query("currency == 'ETH'")), ("Pooled", test_data)]:
        row = {"currency": currency, "n_test_options": int(len(frame))}
        for model in ["black", "heston", "svcj"]:
            row[f"rmse_{model}"] = float(np.sqrt(frame[f"sq_err_{model}"].mean()))
            row[f"mae_{model}"] = float(frame[f"abs_err_{model}"].mean())
            row[f"within_{model}"] = float(frame[f"within_{model}"].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "baseline_overall_metrics.csv", index=False)
    return out


def compute_snapshot_metrics(test_data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    grouped = test_data.groupby(["currency", "snapshot_ts"], as_index=False)
    for (currency, ts), frame in grouped:
        row = {"currency": currency, "snapshot_ts": ts, "n_options": int(len(frame))}
        for model in ["black", "heston", "svcj"]:
            row[f"rmse_{model}"] = float(np.sqrt(frame[f"sq_err_{model}"].mean()))
            row[f"mae_{model}"] = float(frame[f"abs_err_{model}"].mean())
            row[f"within_{model}"] = float(frame[f"within_{model}"].mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["currency", "snapshot_ts"]).reset_index(drop=True)
    out.to_csv(output_dir / "snapshot_time_series_metrics.csv", index=False)
    return out


def plot_snapshot_rmse_mae_panel(snapshot_metrics: pd.DataFrame, output_dir: Path) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        subplot_titles=("BTC RMSE", "ETH RMSE", "BTC MAE", "ETH MAE"),
    )
    panel_map = {(1, 1): ("BTC", "rmse"), (1, 2): ("ETH", "rmse"), (2, 1): ("BTC", "mae"), (2, 2): ("ETH", "mae")}

    for (row, col), (currency, metric) in panel_map.items():
        frame = snapshot_metrics.query("currency == @currency").sort_values("snapshot_ts")
        for model in ["black", "heston", "svcj"]:
            fig.add_trace(
                go.Scatter(
                    x=frame["snapshot_ts"],
                    y=frame[f"{metric}_{model}"],
                    mode="lines",
                    name=MODEL_LABELS[model],
                    legendgroup=MODEL_LABELS[model],
                    showlegend=(row, col) == (1, 1),
                    line=dict(color=MODEL_COLORS[model], width=2),
                ),
                row=row,
                col=col,
            )

    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)
    fig.update_yaxes(title_text="MAE", row=2, col=1)
    fig.update_yaxes(title_text="MAE", row=2, col=2)
    style_figure(fig, height=650)
    save_figure(fig, output_dir, "time_series_rmse_mae_panel")
    return fig


def plot_snapshot_within_spread(snapshot_metrics: pd.DataFrame, output_dir: Path) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("BTC within-spread proportion", "ETH within-spread proportion"),
    )
    for idx, currency in enumerate(["BTC", "ETH"], start=1):
        frame = snapshot_metrics.query("currency == @currency").sort_values("snapshot_ts")
        for model in ["black", "heston", "svcj"]:
            fig.add_trace(
                go.Scatter(
                    x=frame["snapshot_ts"],
                    y=frame[f"within_{model}"],
                    mode="lines",
                    name=MODEL_LABELS[model],
                    legendgroup=MODEL_LABELS[model],
                    showlegend=(idx == 1),
                    line=dict(color=MODEL_COLORS[model], width=2),
                ),
                row=idx,
                col=1,
            )
        fig.update_yaxes(title_text="Share inside spread", tickformat=".0%", row=idx, col=1)
    style_figure(fig, height=600)
    save_figure(fig, output_dir, "time_series_within_spread_panel")
    return fig


def compute_bucket_metrics(test_data: pd.DataFrame, *, n_buckets: int, output_dir: Path) -> pd.DataFrame:
    rows = []
    for currency, frame in test_data.groupby("currency"):
        frame = frame.copy()
        for dimension in ["time_to_maturity", "moneyness"]:
            bucket_name = f"{dimension}_bucket"
            frame[bucket_name] = bucketize_equal_count(frame[dimension], n_buckets=n_buckets)
            bucket_ranges = frame.groupby(bucket_name, observed=False)[dimension].agg(["min", "max", "median", "count"]).reset_index()
            for bucket, bucket_frame in frame.groupby(bucket_name, observed=False):
                bounds = bucket_ranges.loc[bucket_ranges[bucket_name] == bucket].iloc[0]
                row = {
                    "currency": currency,
                    "dimension": dimension,
                    "bucket": bucket,
                    "n_options": int(len(bucket_frame)),
                    "bucket_min": float(bounds["min"]),
                    "bucket_median": float(bounds["median"]),
                    "bucket_max": float(bounds["max"]),
                }
                for model in ["black", "heston", "svcj"]:
                    row[f"rmse_{model}"] = float(np.sqrt(bucket_frame[f"sq_err_{model}"].mean()))
                    row[f"mae_{model}"] = float(bucket_frame[f"abs_err_{model}"].mean())
                    row[f"within_{model}"] = float(bucket_frame[f"within_{model}"].mean())
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "cross_sectional_bucket_metrics.csv", index=False)
    return out


def make_bucket_axis_labels(frame: pd.DataFrame, dimension: str) -> list[str]:
    if dimension == "time_to_maturity":
        lows = (365.0 * frame["bucket_min"]).round().astype(int)
        highs = (365.0 * frame["bucket_max"]).round().astype(int)
        meds = (365.0 * frame["bucket_median"]).round().astype(int)
        return [f"{low}–{high}d<br>(med {med}d)" for low, high, med in zip(lows, highs, meds)]
    return [f"{low:.2f}–{high:.2f}<br>(med {med:.2f})" for low, high, med in zip(frame["bucket_min"], frame["bucket_max"], frame["bucket_median"])]


def make_bucket_hover_text(frame: pd.DataFrame, dimension: str) -> list[str]:
    if dimension == "time_to_maturity":
        return [
            f"Maturity bucket<br>median T = {med * 365:.1f} days<br>range = [{low * 365:.1f}, {high * 365:.1f}] days"
            for med, low, high in zip(frame["bucket_median"], frame["bucket_min"], frame["bucket_max"])
        ]
    return [
        f"Moneyness bucket<br>median K/F0 = {med:.3f}<br>range = [{low:.3f}, {high:.3f}]"
        for med, low, high in zip(frame["bucket_median"], frame["bucket_min"], frame["bucket_max"])
    ]


def plot_bucket_panel(bucket_metrics: pd.DataFrame, *, dimension: str, stem: str, title: str, output_dir: Path) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12, subplot_titles=("BTC", "ETH"))
    for row_idx, currency in enumerate(["BTC", "ETH"], start=1):
        frame = bucket_metrics.query("currency == @currency and dimension == @dimension").copy().sort_values("bucket_median")
        frame["bucket_label"] = make_bucket_axis_labels(frame, dimension)
        hover_text = make_bucket_hover_text(frame, dimension)
        for model in ["black", "heston", "svcj"]:
            fig.add_trace(
                go.Bar(
                    x=frame["bucket_label"],
                    y=frame[f"rmse_{model}"],
                    name=MODEL_LABELS[model],
                    legendgroup=MODEL_LABELS[model],
                    showlegend=(row_idx == 1),
                    marker_color=MODEL_COLORS[model],
                    hovertext=hover_text,
                    hovertemplate="%{hovertext}<br>RMSE = %{y:.6f}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="RMSE", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=2, col=1)
    fig.update_xaxes(tickangle=0, row=1, col=1)
    fig.update_xaxes(tickangle=0, row=2, col=1)
    style_figure(fig, height=600, title=title)
    save_figure(fig, output_dir, stem)
    return fig


def compute_parameter_summary(svcj_params: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = svcj_params.copy()
    frame["feller_gap"] = 2 * frame["kappa"] * frame["theta"] - frame["sigma_v"] ** 2
    rows = []
    for currency, group in frame.groupby("currency"):
        row = {"currency": currency}
        ordered = group.sort_values("timestamp")
        for param in ["kappa", "theta", "sigma_v", "rho", "v0", "lam", "ell_y", "sigma_y", "ell_v", "rho_j"]:
            row[f"{param}_mean"] = float(group[param].mean())
            row[f"{param}_median"] = float(group[param].median())
            row[f"{param}_std"] = float(group[param].std())
            row[f"{param}_iqr"] = float(group[param].quantile(0.75) - group[param].quantile(0.25))
            row[f"{param}_mad_diff"] = float(ordered[param].diff().abs().median())
        row["rho_boundary_share"] = float((group["rho"].abs() > 0.95).mean())
        row["rhoj_boundary_share"] = float((group["rho_j"].abs() > 0.95).mean())
        row["feller_violation_share"] = float((group["feller_gap"] < 0).mean())
        row["corr_theta_v0"] = float(group[["theta", "v0"]].corr().iloc[0, 1])
        row["corr_kappa_theta"] = float(group[["kappa", "theta"]].corr().iloc[0, 1])
        row["corr_lam_sigmay"] = float(group[["lam", "sigma_y"]].corr().iloc[0, 1])
        row["corr_lam_elly"] = float(group[["lam", "ell_y"]].corr().iloc[0, 1])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "svcj_parameter_summary.csv", index=False)
    return out


def plot_parameter_paths(
    svcj_params: pd.DataFrame,
    *,
    currency: str,
    stem: str,
    title: str,
    output_dir: Path,
    params: list[str] | None = None,
) -> go.Figure:
    params = params or ["v0", "theta", "sigma_v", "rho", "lam", "ell_y", "sigma_y", "rho_j"]
    labels = {"v0": "v0", "theta": "theta", "sigma_v": "sigma_v", "rho": "rho", "lam": "lambda", "ell_y": "ell_y", "sigma_y": "sigma_y", "rho_j": "rho_j"}
    frame = svcj_params.query("currency == @currency").sort_values("timestamp").copy()
    n_rows = math.ceil(len(params) / 2)
    fig = make_subplots(rows=n_rows, cols=2, shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.10, subplot_titles=[labels[param] for param in params])
    color = "#b45309" if currency == "BTC" else "#0f766e"
    for idx, param in enumerate(params):
        row = idx // 2 + 1
        col = idx % 2 + 1
        fig.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame[param],
                mode="lines",
                name=currency,
                showlegend=(idx == 0),
                line=dict(color=color, width=2),
            ),
            row=row,
            col=col,
        )
        if param in {"rho", "rho_j"}:
            fig.add_hline(y=0.95, line_dash="dot", line_color="#9ca3af", row=row, col=col)
            fig.add_hline(y=-0.95, line_dash="dot", line_color="#9ca3af", row=row, col=col)
    style_figure(fig, height=350 * n_rows, title=title, hovermode="x unified")
    save_figure(fig, output_dir, stem)
    return fig
