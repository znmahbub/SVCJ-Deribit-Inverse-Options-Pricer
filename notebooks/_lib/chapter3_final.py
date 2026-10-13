from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import save_plotly_figure, style_figure


# Floor used when normalising roughness by IQR — prevents division by zero for
# parameters that barely move across snapshots.
_EPS_SCALE = 1e-12

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


def compute_regularisation_summary(
    files: list[Path],
    output_dir: Path,
    *,
    evaluation_split: str,
) -> pd.DataFrame:
    """Summarise an SVCJ regularisation sweep on one explicit data split.

    The selection workflow calls this with ``evaluation_split='validation'``.
    Only columns belonging to the requested split are loaded from ``svcj_params``;
    in particular validation selection does not read ``rmse_test``/``mae_test``.
    """
    split = str(evaluation_split).lower()
    if split not in {"train", "validation", "test"}:
        raise ValueError("evaluation_split must be train, validation, or test")

    metric_cols = [f"rmse_{split}", f"mae_{split}"]
    selected = ["kappa", "theta", "sigma_v", "rho", "v0", "lam", "ell_y", "sigma_y", "ell_v", "rho_j"]
    diagnostic_cols = [
        "timestamp",
        "currency",
        "success",
        "nfev",
        *metric_cols,
        *selected,
        "feller_gap",
        "feller_violated",
        "svcj_martingale_margin",
        "svcj_martingale_violated",
        "svcj_contour_margin",
        "svcj_contour_violated",
    ]

    rows = []
    for path in files:
        reg = reg_from_path(path)
        svcj = pd.read_excel(path, sheet_name="svcj_params", usecols=diagnostic_cols)
        svcj["timestamp"] = pd.to_datetime(svcj["timestamp"], utc=True)
        ok = svcj.loc[svcj["success"] == True].copy()

        roughness_terms: list[float] = []
        for _, group in ok.sort_values("timestamp").groupby("currency"):
            for param in selected:
                series = pd.to_numeric(group[param], errors="coerce").dropna()
                if len(series) < 2:
                    continue
                iqr = series.quantile(0.75) - series.quantile(0.25)
                scale = iqr if pd.notna(iqr) and iqr > _EPS_SCALE else series.std()
                if pd.isna(scale) or scale <= _EPS_SCALE:
                    continue
                roughness_terms.append(float(series.diff().abs().median() / scale))

        def _share(column: str) -> float:
            values = pd.to_numeric(ok[column], errors="coerce")
            return float(values.mean()) if len(values) else float("nan")

        def _minimum(column: str) -> float:
            values = pd.to_numeric(ok[column], errors="coerce")
            values = values[np.isfinite(values)]
            return float(values.min()) if len(values) else float("nan")

        rows.append(
            {
                "reg": reg,
                f"rmse_{split}_mean": float(pd.to_numeric(ok[f"rmse_{split}"], errors="coerce").mean()),
                f"mae_{split}_mean": float(pd.to_numeric(ok[f"mae_{split}"], errors="coerce").mean()),
                "success_rate": float(pd.to_numeric(svcj["success"], errors="coerce").mean()),
                "nfev_mean": float(pd.to_numeric(ok["nfev"], errors="coerce").mean()),
                "nfev_median": float(pd.to_numeric(ok["nfev"], errors="coerce").median()),
                "roughness_mean": float(np.mean(roughness_terms)) if roughness_terms else float("nan"),
                "feller_violation_share": _share("feller_violated"),
                "svcj_martingale_violation_share": _share("svcj_martingale_violated"),
                "svcj_contour_violation_share": _share("svcj_contour_violated"),
                "svcj_martingale_margin_min": _minimum("svcj_martingale_margin"),
                "svcj_contour_margin_min": _minimum("svcj_contour_margin"),
                "rho_boundary_share": float((pd.to_numeric(ok["rho"], errors="coerce").abs() > 0.95).mean()),
            }
        )

    out = pd.DataFrame(rows).sort_values("reg").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / f"regularisation_summary_{split}.csv", index=False)
    return out


def select_regularisation_candidate(reg_summary: pd.DataFrame) -> int:
    """Choose lambda_TS using validation metrics only.

    Primary criterion: mean validation RMSE. Exact ties are resolved by mean
    validation MAE, parameter-path roughness, mean optimiser evaluations, then
    the smaller lambda. Test-set columns are deliberately neither required nor
    consulted.
    """
    required = {
        "reg",
        "rmse_validation_mean",
        "mae_validation_mean",
        "roughness_mean",
        "nfev_mean",
    }
    missing = required - set(reg_summary.columns)
    if missing:
        raise ValueError(
            "regularisation selection requires validation-only summary columns; "
            f"missing {sorted(missing)}"
        )

    work = reg_summary[list(required)].copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[np.isfinite(work["reg"]) & np.isfinite(work["rmse_validation_mean"])].copy()
    if work.empty:
        raise ValueError("no finite validation candidates are available")

    # NaN tie-break diagnostics should lose to finite values but must never
    # change the primary validation-RMSE ordering.
    for col in ["mae_validation_mean", "roughness_mean", "nfev_mean"]:
        work[col] = work[col].fillna(np.inf)
    work = work.sort_values(
        ["rmse_validation_mean", "mae_validation_mean", "roughness_mean", "nfev_mean", "reg"],
        kind="mergesort",
    )
    value = float(work.iloc[0]["reg"])
    if not value.is_integer():
        raise ValueError("current workbook naming convention requires integer lambda_TS candidates")
    return int(value)


def build_regularisation_tradeoff_figure(
    reg_summary: pd.DataFrame,
    *,
    final_reg: int,
    output_dir: Path,
    evaluation_split: str,
) -> go.Figure:
    """Plot the validation-only regularisation trade-off used for selection."""
    split = str(evaluation_split).lower()
    if split != "validation":
        raise ValueError("regularisation selection figures must use the validation split")
    rmse_col = "rmse_validation_mean"

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.20,
        specs=[[{"secondary_y": True}, {"secondary_y": False}]],
    )
    x_labels = reg_summary["reg"].astype(str)

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=reg_summary[rmse_col],
            mode="lines+markers+text",
            name="Mean validation RMSE",
            text=x_labels,
            textposition="top center",
            textfont=dict(size=10),
            cliponaxis=False,
            line=dict(color="#111827", width=2),
            marker=dict(size=8),
            hovertemplate="lambda_TS=%{x}<br>Mean validation RMSE=%{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=reg_summary["nfev_mean"],
            mode="lines+markers",
            name="Mean function evaluations",
            line=dict(color="#2563eb", width=2, dash="dash"),
            marker=dict(size=8),
            hovertemplate="lambda_TS=%{x}<br>Mean function evaluations=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=reg_summary["roughness_mean"],
            mode="lines+markers+text",
            name="Roughness index",
            text=x_labels,
            textposition="top center",
            textfont=dict(size=10),
            cliponaxis=False,
            line=dict(color="#dc2626", width=2),
            marker=dict(size=8),
            hovertemplate="lambda_TS=%{x}<br>Roughness index=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    selected_label = str(int(final_reg))
    selected_positions = np.flatnonzero(x_labels.to_numpy() == selected_label)
    if len(selected_positions):
        selected_x = selected_label
        for col in [1, 2]:
            fig.add_vline(x=selected_x, line_width=1.5, line_dash="dot", line_color="#dc2626", row=1, col=col)

    fig.update_xaxes(type="category", title_text="lambda_TS candidate", row=1, col=1, showgrid=True)
    fig.update_xaxes(type="category", title_text="lambda_TS candidate", row=1, col=2, showgrid=True)
    fig.update_yaxes(title_text="Mean validation RMSE", row=1, col=1, secondary_y=False, tickformat=".5f", showgrid=True)
    fig.update_yaxes(title_text="Mean function evaluations", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="Roughness index", row=1, col=2, showgrid=True)

    style_figure(fig, height=520, title="Validation-only regularisation trade-off across the SVCJ sweep", legend_y=1.10)
    fig.update_layout(
        margin=dict(t=95, b=70, l=70, r=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    save_figure(fig, output_dir, "regularisation_tradeoff_validation")
    return fig


def load_locked_regularisation(record_path: Path) -> int:
    """Read the committed lambda_TS lock; refuse pending/unlocked records."""
    path = Path(record_path)
    text = path.read_text(encoding="utf-8")
    status_match = re.search(r"^\s*-\s*Status:\s*([A-Za-z_]+)\s*$", text, flags=re.MULTILINE)
    status = status_match.group(1).upper() if status_match else ""
    if status != "LOCKED":
        raise RuntimeError(
            f"regularisation selection is not locked in {path}; run the validation-only selection workflow first"
        )
    value_match = re.search(r"^\s*-\s*Selected\s+lambda_TS:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if not value_match:
        raise RuntimeError(f"locked regularisation record {path} has no integer Selected lambda_TS value")
    return int(value_match.group(1))



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
