from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def remove_top_level_functions(path: Path, names: set[str]) -> None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    spans = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            spans.append((node.lineno - 1, node.end_lineno))
    missing = names - {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    }
    if missing:
        raise RuntimeError(f"missing expected functions in {path}: {sorted(missing)}")
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    path.write_text("".join(lines), encoding="utf-8")


def replace_top_level_function(path: Path, name: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name} in {path}, found {len(matches)}")
    node = matches[0]
    lines = text.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [replacement.rstrip() + "\n"]
    path.write_text("".join(lines), encoding="utf-8")


def clean_chapter3_final() -> None:
    path = ROOT / "notebooks" / "_lib" / "chapter3_final.py"
    remove_top_level_functions(
        path,
        {
            "find_calibration_files",
            "reg_from_path",
            "compute_regularisation_summary",
            "select_regularisation_candidate",
            "build_regularisation_tradeoff_figure",
            "load_locked_regularisation",
        },
    )
    text = path.read_text(encoding="utf-8")
    text = text.replace("import re\n", "")
    text = re.sub(
        r"\n# Floor used when normalising roughness by IQR — prevents division by zero for\n"
        r"# parameters that barely move across snapshots\.\n_EPS_SCALE = 1e-12\n",
        "\n",
        text,
    )
    path.write_text(text, encoding="utf-8")

    replacement = '''def load_baseline_workbook(
    baseline_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load parameter panels plus the canonical untouched 30% test holdout."""
    black_params = pd.read_excel(baseline_path, sheet_name="black_params")
    heston_params = pd.read_excel(baseline_path, sheet_name="heston_params")
    svcj_params = pd.read_excel(baseline_path, sheet_name="svcj_params")
    for frame in [black_params, heston_params, svcj_params]:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    test_data = pd.read_excel(baseline_path, sheet_name="test_data")
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
        raise ValueError(f"test_data missing required columns: {sorted(missing)}")
    test_data["snapshot_ts"] = pd.to_datetime(test_data["snapshot_ts"], utc=True)
    test_data["currency"] = test_data["currency"].astype(str).str.upper()
    test_data["evaluation_split"] = "test"
    test_data = add_common_unit_errors(test_data)
    return black_params, heston_params, svcj_params, test_data
'''
    replace_top_level_function(path, "load_baseline_workbook", replacement)


def clean_stage2_workflow() -> None:
    path = ROOT / "notebooks" / "_lib" / "stage2_workflow.py"
    text = path.read_text(encoding="utf-8")
    old = 'return chapter3.load_baseline_workbook(Path(workbook_path), evaluation_split="test")'
    new = 'return chapter3.load_baseline_workbook(Path(workbook_path))'
    if old not in text:
        raise RuntimeError("expected Stage 2 test-loader call was not found")
    path.write_text(text.replace(old, new), encoding="utf-8")


def remove_superseded_notebook_surface() -> None:
    notebooks = ROOT / "notebooks"
    allowed = {
        "00_check_data.ipynb",
        "01_run_calibration.ipynb",
        "02_pricing_analysis.ipynb",
        "03_run_hedging.ipynb",
        "04_final_analysis.ipynb",
        "_lib",
    }
    for child in notebooks.iterdir():
        if child.name in allowed:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    legacy_helper = notebooks / "_lib" / "chapter3_analysis.py"
    if legacy_helper.exists():
        legacy_helper.unlink()


def remove_remediation_automation() -> None:
    workflows = ROOT / ".github" / "workflows"
    allowed = {"collect_deribit_snapshot.yml", "update-futures-history.yml", "stage4-verification.yml", "stage5-cleanup.yml"}
    for path in workflows.glob("*.yml"):
        if path.name not in allowed:
            path.unlink()
    scripts = ROOT / "scripts"
    for path in scripts.glob("task13*.py"):
        path.unlink()


def remove_superseded_docs() -> None:
    path = ROOT / "docs" / "reproducibility" / "regularization_selection.md"
    if path.exists():
        path.unlink()
    superpowers = ROOT / "docs" / "superpowers"
    if superpowers.exists():
        shutil.rmtree(superpowers)


def promote_ci_and_remove_one_shot_files() -> None:
    workflows = ROOT / ".github" / "workflows"
    old = workflows / "stage4-verification.yml"
    new = workflows / "ci.yml"
    if old.exists():
        text = old.read_text(encoding="utf-8").replace("name: Stage 4 verification", "name: Canonical verification")
        new.write_text(text, encoding="utf-8")
        old.unlink()
    cleanup_workflow = workflows / "stage5-cleanup.yml"
    if cleanup_workflow.exists():
        cleanup_workflow.unlink()
    self_path = ROOT / "scripts" / "stage5_cleanup.py"
    if self_path.exists():
        self_path.unlink()
    scripts = ROOT / "scripts"
    if scripts.exists() and not any(scripts.iterdir()):
        scripts.rmdir()


def main() -> None:
    clean_chapter3_final()
    clean_stage2_workflow()
    remove_superseded_notebook_surface()
    remove_remediation_automation()
    remove_superseded_docs()
    promote_ci_and_remove_one_shot_files()


if __name__ == "__main__":
    main()
