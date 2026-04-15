from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .common import NotebookPaths


@dataclass
class FuturesData:
    perp: pd.DataFrame
    term: pd.DataFrame
    perp_price_col: str
    term_price_col: str


def resolve_futures_paths(paths: NotebookPaths) -> tuple[Path, Path]:
    perp_path = paths.data_dir / "perpetual_futures_prices.csv"
    term_path = paths.data_dir / "term_futures_marks.csv"
    if not perp_path.exists():
        raise FileNotFoundError(f"Missing perpetual futures file: {perp_path}")
    if not term_path.exists():
        raise FileNotFoundError(f"Missing term futures file: {term_path}")
    return perp_path, term_path


def load_futures_data(paths: NotebookPaths) -> FuturesData:
    perp_path, term_path = resolve_futures_paths(paths)
    perp = pd.read_csv(perp_path)
    term = pd.read_csv(term_path)

    if "obs_datetime" in perp.columns:
        perp["obs_datetime"] = pd.to_datetime(perp["obs_datetime"], utc=True, errors="coerce")
    elif "timestamp" in perp.columns:
        perp["obs_datetime"] = pd.to_datetime(perp["timestamp"], utc=True, errors="coerce")
    else:
        raise ValueError("Perp file must contain obs_datetime or timestamp")

    if "timestamp" in term.columns:
        term["timestamp"] = pd.to_datetime(term["timestamp"], utc=True, errors="coerce")
    elif "obs_datetime" in term.columns:
        term["timestamp"] = pd.to_datetime(term["obs_datetime"], utc=True, errors="coerce")
    else:
        raise ValueError("Term file must contain timestamp or obs_datetime")

    if "coin" not in perp.columns and "currency" in perp.columns:
        perp["coin"] = perp["currency"]
    if "currency" not in term.columns and "coin" in term.columns:
        term["currency"] = term["coin"]

    perp_price_col = "perp_futures_mark_price" if "perp_futures_mark_price" in perp.columns else "candle_price_used"
    term_price_col = "term_mark_price_final" if "term_mark_price_final" in term.columns else "term_candle_price_used"
    perp[perp_price_col] = pd.to_numeric(perp[perp_price_col], errors="coerce")
    term[term_price_col] = pd.to_numeric(term[term_price_col], errors="coerce")
    return FuturesData(perp=perp, term=term, perp_price_col=perp_price_col, term_price_col=term_price_col)


def suggest_term_instruments(term: pd.DataFrame) -> list[str]:
    term_non_syn = term.copy()
    if "futures_instrument_name" in term_non_syn.columns:
        term_non_syn = term_non_syn[~term_non_syn["futures_instrument_name"].astype(str).str.startswith("SYN.", na=False)].copy()
    rep = (
        term_non_syn.dropna(subset=["currency", "futures_instrument_name"])
        .groupby(["currency", "futures_instrument_name"])
        .size()
        .reset_index(name="n")
        .sort_values(["currency", "n"], ascending=[True, False])
    )

    suggested: list[str] = []
    for currency in sorted(rep["currency"].dropna().unique().tolist()):
        suggested.extend(rep.loc[rep["currency"] == currency, "futures_instrument_name"].head(3).tolist())
    return suggested


def build_perp_figure(data: FuturesData) -> go.Figure:
    fig = go.Figure()
    for coin, group in data.perp.dropna(subset=["obs_datetime"]).groupby("coin"):
        gg = group.sort_values("obs_datetime")
        fig.add_trace(
            go.Scatter(
                x=gg["obs_datetime"],
                y=gg[data.perp_price_col],
                mode="lines",
                name=str(coin),
                hovertemplate="Coin=%{fullData.name}<br>Time=%{x}<br>Price=%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Perpetual futures prices",
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
        height=550,
    )
    return fig


def build_term_figure(data: FuturesData, term_instruments: list[str]) -> go.Figure:
    term_plot = data.term.copy()
    if term_instruments:
        term_plot = term_plot[term_plot["futures_instrument_name"].isin(term_instruments)].copy()

    fig = go.Figure()
    for inst, group in term_plot.dropna(subset=["timestamp"]).groupby("futures_instrument_name"):
        gg = group.sort_values("timestamp")
        fig.add_trace(
            go.Scatter(
                x=gg["timestamp"],
                y=gg[data.term_price_col],
                mode="lines",
                name=str(inst),
                hovertemplate="Instrument=%{fullData.name}<br>Time=%{x}<br>Price=%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Selected term futures prices",
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
        height=650,
    )
    return fig


def build_diagnostics(data: FuturesData, term_instruments: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    perp_diag = (
        data.perp.groupby("coin", dropna=False)
        .agg(
            n_rows=("coin", "size"),
            first_time=("obs_datetime", "min"),
            last_time=("obs_datetime", "max"),
            n_nonnull_price=(data.perp_price_col, lambda series: series.notna().sum()),
        )
        .reset_index()
    )
    term_diag = (
        data.term[data.term["futures_instrument_name"].isin(term_instruments)]
        .groupby(["currency", "futures_instrument_name"], dropna=False)
        .agg(
            n_rows=("futures_instrument_name", "size"),
            first_time=("timestamp", "min"),
            last_time=("timestamp", "max"),
            n_nonnull_price=(data.term_price_col, lambda series: series.notna().sum()),
        )
        .reset_index()
        .sort_values(["currency", "futures_instrument_name"])
    )
    return perp_diag, term_diag
