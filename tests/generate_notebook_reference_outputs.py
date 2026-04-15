from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import sys

import nbformat
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


NOTEBOOKS = [
    "notebooks/calibration_analysis_complete_reg_0.ipynb",
    "notebooks/calibration_analysis_complete_reg_1.ipynb",
    "notebooks/calibration_analysis_complete_reg_3.ipynb",
    "notebooks/calibration_analysis_complete_reg_5.ipynb",
    "notebooks/calibration_analysis_complete_reg_10.ipynb",
    "notebooks/calibration_analysis_complete_reg_50.ipynb",
    "notebooks/calibration_analysis_complete_reg_100.ipynb",
    "notebooks/calibration_results_final.ipynb",
    "notebooks/futures_plotly_viewer.ipynb",
    "notebooks/hedging_analytics.ipynb",
    "notebooks/calibrate_all_to_excel.ipynb",
]

ARTIFACT_DIRS = [
    "notebooks/chapter3_outputs",
    "hedging_output/hedging_chapter4",
]

OUTPUT_PATH = ROOT / "tests" / "fixtures" / "notebook_reference_outputs.json"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)


def normalize_series(values: Any) -> tuple[int, str]:
    if values is None:
        return 0, stable_json_hash(None)
    if isinstance(values, dict) and {"dtype", "bdata"}.issubset(values):
        return -1, stable_json_hash(values)
    if not isinstance(values, list):
        return 1, stable_json_hash(values)
    return len(values), stable_json_hash(values)


def normalize_plotly_payload(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data", [])
    layout = payload.get("layout", {})
    traces = []
    for trace in data:
        x_len, x_hash = normalize_series(trace.get("x"))
        y_len, y_hash = normalize_series(trace.get("y"))
        z_len, z_hash = normalize_series(trace.get("z"))
        traces.append(
            {
                "type": trace.get("type"),
                "name": trace.get("name"),
                "mode": trace.get("mode"),
                "legendgroup": trace.get("legendgroup"),
                "marker_color": trace.get("marker", {}).get("color"),
                "line_color": trace.get("line", {}).get("color"),
                "x_len": x_len,
                "y_len": y_len,
                "z_len": z_len,
                "x_hash": x_hash,
                "y_hash": y_hash,
                "z_hash": z_hash,
            }
        )
    layout_summary = {
        "title": layout.get("title", {}).get("text"),
        "height": layout.get("height"),
        "width": layout.get("width"),
        "annotations_hash": stable_json_hash(layout.get("annotations")),
        "xaxes_hash": stable_json_hash({k: v.get("title") if isinstance(v, dict) else v for k, v in layout.items() if str(k).startswith("xaxis")}),
        "yaxes_hash": stable_json_hash({k: v.get("title") if isinstance(v, dict) else v for k, v in layout.items() if str(k).startswith("yaxis")}),
    }
    return {
        "trace_count": len(traces),
        "traces": traces,
        "layout": layout_summary,
    }


def normalize_output(output: dict[str, Any]) -> dict[str, Any]:
    data = output.get("data", {})
    summary = {"output_type": output.get("output_type")}
    if "application/vnd.plotly.v1+json" in data:
        summary["plotly"] = normalize_plotly_payload(data["application/vnd.plotly.v1+json"])
    if "text/html" in data:
        html_text = data["text/html"]
        if isinstance(html_text, list):
            html_text = "".join(html_text)
        summary["html_hash"] = sha256_text(str(html_text))
    if "text/plain" in data:
        plain_text = data["text/plain"]
        if isinstance(plain_text, list):
            plain_text = "".join(plain_text)
        summary["text_hash"] = sha256_text(str(plain_text))
    if output.get("output_type") == "stream":
        text = output.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        summary["stream_hash"] = sha256_text(str(text))
    if output.get("output_type") == "error":
        summary["ename"] = output.get("ename")
        summary["evalue"] = output.get("evalue")
    return summary


def snapshot_notebook(path: Path) -> dict[str, Any]:
    nb = nbformat.read(path, as_version=4)
    markdown_headers = []
    cells = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            text = "".join(cell.source)
            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            if first_line.startswith("#"):
                markdown_headers.append(first_line)
            cells.append({"index": index, "cell_type": "markdown", "source_hash": sha256_text(text)})
            continue
        outputs = [normalize_output(output) for output in cell.get("outputs", [])]
        cells.append(
            {
                "index": index,
                "cell_type": "code",
                "source_hash": sha256_text("".join(cell.source)),
                "output_count": len(outputs),
                "outputs": outputs,
            }
        )
    return {
        "metadata": {
            "kernelspec": nb.metadata.get("kernelspec", {}),
            "language_info": {
                "name": nb.metadata.get("language_info", {}).get("name"),
                "version": nb.metadata.get("language_info", {}).get("version"),
            },
        },
        "markdown_headers": markdown_headers,
        "cells": cells,
    }


def snapshot_artifact(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        return {
            "type": "csv",
            "shape": list(frame.shape),
            "columns": frame.columns.tolist(),
            "hash": stable_json_hash(frame.to_dict(orient="split")),
        }
    text = path.read_text(encoding="utf-8")
    return {
        "type": path.suffix.lower().lstrip("."),
        "size": len(text),
        "hash": sha256_text(text),
    }


def main() -> None:
    payload = {"notebooks": {}, "artifacts": {}}
    for rel_path in NOTEBOOKS:
        payload["notebooks"][rel_path] = snapshot_notebook(ROOT / rel_path)
    for rel_dir in ARTIFACT_DIRS:
        dir_path = ROOT / rel_dir
        payload["artifacts"][rel_dir] = {
            str(path.relative_to(ROOT)): snapshot_artifact(path)
            for path in sorted(dir_path.glob("*"))
            if path.is_file() and not path.name.startswith(".")
        }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
