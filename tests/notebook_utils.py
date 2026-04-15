from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from tests._path import ROOT
from tests.generate_notebook_reference_outputs import OUTPUT_PATH, snapshot_artifact, snapshot_notebook


def load_notebook_reference() -> dict:
    return json.loads(OUTPUT_PATH.read_text())


def execute_notebook_in_memory(
    notebook_path: Path,
    *,
    working_directory: Path,
    timeout: int = 600,
) -> nbformat.NotebookNode:
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook = copy.deepcopy(notebook)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(working_directory)}},
        allow_errors=False,
    )
    client.execute()
    return notebook


def execute_notebook_with_replacements(
    notebook_path: Path,
    replacements: dict[str, str],
    *,
    working_directory: Path,
    timeout: int = 600,
) -> nbformat.NotebookNode:
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook = copy.deepcopy(notebook)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = "".join(cell.source)
        for before, after in replacements.items():
            source = source.replace(before, after)
        cell.source = source
        cell.outputs = []
        cell.execution_count = None
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(working_directory)}},
        allow_errors=False,
    )
    client.execute()
    return notebook


def write_executed_notebook(notebook: nbformat.NotebookNode, path: Path) -> None:
    nbformat.write(notebook, path)


def snapshot_notebook_node(notebook: nbformat.NotebookNode, temp_path: Path) -> dict:
    nbformat.write(notebook, temp_path)
    return snapshot_notebook(temp_path)


def snapshot_artifact_tree(rel_dir: str) -> dict[str, dict]:
    dir_path = ROOT / rel_dir
    return {
        str(path.relative_to(ROOT)): snapshot_artifact(path)
        for path in sorted(dir_path.glob("*"))
        if path.is_file() and not path.name.startswith(".")
    }


def temporary_notebook_path(name: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="nb-tests-"))
    return tmpdir / name
