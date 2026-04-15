from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


@dataclass(frozen=True)
class NotebookPaths:
    notebook_dir: Path
    project_root: Path
    data_dir: Path
    excel_dir: Path


def locate_notebook_paths(start: Path | None = None) -> NotebookPaths:
    start_path = (start or Path.cwd()).resolve()
    candidates = [start_path, *start_path.parents]

    for candidate in candidates:
        if (candidate / "notebooks").exists() and (candidate / "src").exists():
            project_root = candidate
            notebook_dir = candidate / "notebooks"
            break
        if candidate.name == "notebooks" and (candidate.parent / "src").exists():
            project_root = candidate.parent
            notebook_dir = candidate
            break
    else:
        raise FileNotFoundError(f"Could not locate project root from {start_path}")

    return NotebookPaths(
        notebook_dir=notebook_dir,
        project_root=project_root,
        data_dir=project_root / "data",
        excel_dir=project_root / "excel files",
    )


def configure_notebook_display(
    *,
    max_columns: int = 200,
    width: int = 200,
    template: str = "plotly_white",
    renderer: str | None = None,
) -> None:
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.width", width)
    pio.templates.default = template
    if renderer is not None:
        pio.renderers.default = renderer


def style_figure(
    fig: go.Figure,
    *,
    height: int = 700,
    width: int | None = None,
    title: str | None = None,
    legend_y: float = 1.10,
    hovermode: str = "x unified",
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.5} if title else None,
        height=height,
        width=width,
        hovermode=hovermode,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    return fig


def save_plotly_figure(
    fig: go.Figure,
    output_dir: Path,
    stem: str,
    *,
    include_png: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / f"{stem}.html", include_plotlyjs="cdn")
    if not include_png:
        return
    try:
        fig.write_image(output_dir / f"{stem}.png", scale=2)
    except Exception as exc:  # pragma: no cover - depends on optional kaleido
        print(f"[info] Static export skipped for {stem}: {exc}")


def export_tables_csv(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, table in tables.items():
        table.to_csv(output_dir / f"{stem}.csv", index=False)


def ensure_existing_path(path: Path, *, search_dirs: Iterable[Path] = ()) -> Path:
    if path.exists():
        return path.resolve()
    for directory in search_dirs:
        candidate = directory / path
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"File not found: {path}")
