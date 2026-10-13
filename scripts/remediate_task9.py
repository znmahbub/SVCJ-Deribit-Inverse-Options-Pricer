from __future__ import annotations

from pathlib import Path
import re
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, found {n}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# chapter3_final: validation-only regularisation selection + lock parser
# ---------------------------------------------------------------------------
p = ROOT / "notebooks/_lib/chapter3_final.py"
s = p.read_text(encoding="utf-8")
start = s.index("def compute_regularisation_summary(")
end = s.index("def load_baseline_workbook(")
new_block = r'''def compute_regularisation_summary(
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


'''
s = s[:start] + new_block + s[end:]
p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# chapter3_analysis: correct rho_j diagnostic semantics and holdout wording
# ---------------------------------------------------------------------------
p = ROOT / "notebooks/_lib/chapter3_analysis.py"
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    '    "SVCJ": {\n        "kappa": (1e-4, 50.0),\n        "theta": (1e-6, 5.0),\n        "sigma_v": (1e-4, 10.0),\n        "rho": (RHO_LB, RHO_UB),\n        "v0": (1e-6, 5.0),\n        "lam": (1e-6, 10.0),\n        "ell_y": (-5.0, 5.0),\n        "sigma_y": (1e-4, 5.0),\n        "ell_v": (1e-6, 10.0),\n        "rho_j": (RHO_LB, RHO_UB),\n    },',
    '    "SVCJ": {\n        "kappa": (1e-4, 50.0),\n        "theta": (1e-6, 5.0),\n        "sigma_v": (1e-4, 10.0),\n        "rho": (RHO_LB, RHO_UB),\n        "v0": (1e-6, 5.0),\n        "lam": (1e-6, 10.0),\n        "ell_y": (-5.0, 5.0),\n        "sigma_y": (1e-4, 5.0),\n        "ell_v": (1e-6, 10.0),\n        # rho_j is an unbounded jump-size loading; moment-domain diagnostics\n        # replace the obsolete [-1,1] boundary diagnostic.\n    },',
    "remove rho_j fake correlation bound",
)
# Add an explicit inference label near the public constants so any future HAC
# table/notebook uses the correct statistical interpretation.
anchor = 'FIGDIMS = dict(width=1200, height=1100)\n'
insert = (
    anchor
    + '\nHOLDOUT_INFERENCE_LABEL = (\n'
    + '    "snapshot-level cross-sectional holdout loss differential with HAC standard errors"\n'
    + ')\n'
)
s = replace_once(s, anchor, insert, "holdout inference label")
s = s.replace(
    'fig.update_layout(title=f"{currency} — {split.title()} error boxplots (snapshot-level)", showlegend=False, **FIGDIMS)',
    'split_label = "Training" if split == "train" else f"{split.title()} cross-sectional holdout"\n    fig.update_layout(title=f"{currency} — {split_label} error boxplots (snapshot-level)", showlegend=False, **FIGDIMS)',
)
s = s.replace(
    'fig.update_layout(title=f"{currency} — {split.title()} spread-relative diagnostics (per snapshot)", showlegend=False, width=1200, height=900)',
    'split_label = "Training" if split == "train" else f"{split.title()} cross-sectional holdout"\n    fig.update_layout(title=f"{currency} — {split_label} spread-relative diagnostics (per snapshot)", showlegend=False, width=1200, height=900)',
)
p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# Selection lock record (pending until corrected candidate workbooks are run)
# ---------------------------------------------------------------------------
record = ROOT / "docs/reproducibility/regularization_selection.md"
record.write_text(textwrap.dedent('''
# Regularization selection record

- Status: PENDING
- Selected lambda_TS: TBD
- Candidate grid: 0, 1, 3, 5, 10, 25, 50, 100, 250, 500, 1000
- Evaluation split: validation
- Primary selection statistic: mean SVCJ snapshot-level `rmse_validation`
- Deterministic tie-break order: `mae_validation`, parameter-path roughness, mean `nfev`, smaller `lambda_TS`
- Selection commit: TBD

## Protocol

This record is the gate between hyperparameter selection and final test-set reporting.
The candidate workbooks must be generated by the corrected pipeline using the frozen
sample, deterministic 60/20/20 split, and sequential temporal regularization. The
selection notebook reads only the `svcj_params` columns required for validation
selection and never reads `test_data`, `rmse_test`, or `mae_test`.

After the corrected candidate sweep is available:

1. Run `notebooks/regularization_selection.ipynb` against
   `excel files/remediation/calibration_results_reg_*.xlsx`.
2. Review the exported validation trade-off table and figure.
3. Confirm the deterministic candidate returned by
   `select_regularisation_candidate(...)`.
4. Replace `PENDING` by `LOCKED`, replace `TBD` with the selected integer
   `lambda_TS`, and record the commit SHA containing the candidate results.
5. Commit this file **before** running `notebooks/calibration_results_final.ipynb`.

`calibration_results_final.ipynb` intentionally raises while this record is pending.
That prevents the final untouched test set from participating in regularization
selection.

## Why no value is locked yet

Tasks 6–8 changed the objective diagnostics, restored the true chronological
regularization chain, and replaced the old 70/30 split with a 60/20/20
train/validation/test split. The old regularization workbooks are therefore not
eligible for a new selection decision. The numerical lock is deferred until the
corrected candidate workbooks are generated; using the old sweep here would
re-introduce the leakage this task is designed to remove.
''').lstrip(), encoding="utf-8")


# ---------------------------------------------------------------------------
# New validation-only selection notebook
# ---------------------------------------------------------------------------
selection_nb = new_notebook(
    cells=[
        new_markdown_cell(textwrap.dedent('''
        # Validation-only temporal-regularization selection

        This notebook selects `lambda_TS` **without opening the final test set**.
        It operates only on corrected candidate workbooks in `excel files/remediation/`
        and uses the validation metrics stored in `svcj_params` together with
        optimizer/stability/admissibility diagnostics.

        The notebook does not edit the lock record automatically. After reviewing
        the table and figure, update `docs/reproducibility/regularization_selection.md`
        and commit that lock before running the final Chapter 3 notebook.
        ''').strip()),
        new_code_cell(textwrap.dedent('''
        from __future__ import annotations

        from pathlib import Path
        import sys

        NOTEBOOK_ROOT = Path.cwd().resolve()
        if (NOTEBOOK_ROOT / "_lib").exists():
            NOTEBOOK_DIR = NOTEBOOK_ROOT
        elif (NOTEBOOK_ROOT / "notebooks" / "_lib").exists():
            NOTEBOOK_DIR = NOTEBOOK_ROOT / "notebooks"
        else:
            raise FileNotFoundError(f"Could not locate notebooks/_lib from {NOTEBOOK_ROOT}")

        if str(NOTEBOOK_DIR) not in sys.path:
            sys.path.insert(0, str(NOTEBOOK_DIR))

        from _lib import chapter3_final as chapter3
        from _lib import common

        PATHS = common.locate_notebook_paths(NOTEBOOK_DIR)
        REMEDIATION_EXCEL_DIR = PATHS.excel_dir / "remediation"
        OUTPUT_DIR = PATHS.notebook_dir / "chapter3_outputs" / "remediation" / "regularization_selection"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FILE_PATTERN = "calibration_results_reg_*.xlsx"
        ''').strip()),
        new_markdown_cell("## 1. Load corrected candidate workbooks"),
        new_code_cell(textwrap.dedent('''
        files = chapter3.find_calibration_files([REMEDIATION_EXCEL_DIR], FILE_PATTERN)
        [path.name for path in files]
        ''').strip()),
        new_markdown_cell("## 2. Summarize validation performance and diagnostics"),
        new_code_cell(textwrap.dedent('''
        reg_summary = chapter3.compute_regularisation_summary(
            files,
            OUTPUT_DIR,
            evaluation_split="validation",
        )
        reg_summary
        ''').strip()),
        new_markdown_cell("## 3. Apply the pre-specified validation selection rule"),
        new_code_cell(textwrap.dedent('''
        selected_reg = chapter3.select_regularisation_candidate(reg_summary)
        print(f"Validation-selected lambda_TS = {selected_reg}")
        ''').strip()),
        new_code_cell(textwrap.dedent('''
        fig = chapter3.build_regularisation_tradeoff_figure(
            reg_summary,
            final_reg=selected_reg,
            output_dir=OUTPUT_DIR,
            evaluation_split="validation",
        )
        fig.show()
        ''').strip()),
        new_markdown_cell(textwrap.dedent('''
        ## 4. Lock before final reporting

        Review the validation table/figure above. Then update and commit
        `docs/reproducibility/regularization_selection.md` with `Status: LOCKED`,
        the selected integer value, and the candidate-results commit SHA. Do not
        run the final Chapter 3 reporting notebook before that commit exists.
        ''').strip()),
    ],
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
nbformat.write(selection_nb, ROOT / "notebooks/regularization_selection.ipynb")


# ---------------------------------------------------------------------------
# Final Chapter 3 notebook: remove selection, require committed lock, test only
# ---------------------------------------------------------------------------
final_path = ROOT / "notebooks/calibration_results_final.ipynb"
nb = nbformat.read(final_path, as_version=4)

# Clear stale pre-remediation outputs before manipulating sources.
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []
        cell.metadata.pop("execution", None)

# Rewrite intro.
nb.cells[0].source = textwrap.dedent('''
# Chapter 3 final calibration results

This notebook performs **final reporting only**. Temporal regularization is
selected separately in `regularization_selection.ipynb` using validation data.
The committed lock in `docs/reproducibility/regularization_selection.md` determines
the single workbook loaded here; final metrics are computed on the untouched
within-snapshot cross-sectional **test holdout**.
''').strip()

# Bootstrap cell: lock first, remediation directory only.
for cell in nb.cells:
    if cell.cell_type == "code" and "FINAL_REG = 100" in cell.source:
        cell.source = textwrap.dedent('''
        from __future__ import annotations

        from pathlib import Path
        import sys
        import warnings

        import pandas as pd

        warnings.filterwarnings("ignore", category=FutureWarning)

        NOTEBOOK_ROOT = Path.cwd().resolve()
        if (NOTEBOOK_ROOT / "_lib").exists():
            NOTEBOOK_DIR = NOTEBOOK_ROOT
        elif (NOTEBOOK_ROOT / "notebooks" / "_lib").exists():
            NOTEBOOK_DIR = NOTEBOOK_ROOT / "notebooks"
        else:
            raise FileNotFoundError(f"Could not locate notebooks/_lib from {NOTEBOOK_ROOT}")

        if str(NOTEBOOK_DIR) not in sys.path:
            sys.path.insert(0, str(NOTEBOOK_DIR))

        from _lib import chapter3_final as chapter3
        from _lib import common

        common.configure_notebook_display(max_columns=200, width=200, template="plotly_white")

        PATHS = common.locate_notebook_paths(NOTEBOOK_DIR)
        SELECTION_RECORD = PATHS.project_root / "docs" / "reproducibility" / "regularization_selection.md"
        FINAL_REG = chapter3.load_locked_regularisation(SELECTION_RECORD)
        REMEDIATION_EXCEL_DIR = PATHS.excel_dir / "remediation"
        FILE_PATTERN = f"calibration_results_reg_{FINAL_REG}.xlsx"
        OUTPUT_DIR = PATHS.notebook_dir / "chapter3_outputs" / "remediation" / "final"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_LABELS = chapter3.MODEL_LABELS
        MODEL_COLORS = chapter3.MODEL_COLORS
        ''').strip()
        break
else:
    raise RuntimeError("could not find FINAL_REG bootstrap cell")

# Replace the old sweep discovery with single locked-workbook discovery.
for cell in nb.cells:
    if cell.cell_type == "code" and "files = find_calibration_files" in cell.source:
        cell.source = textwrap.dedent('''
        locked_files = chapter3.find_calibration_files([REMEDIATION_EXCEL_DIR], FILE_PATTERN)
        if len(locked_files) != 1:
            raise RuntimeError(f"Expected exactly one locked regularization workbook, found {locked_files}")
        baseline_path = locked_files[0]
        baseline_path
        ''').strip()
        break

# Remove obsolete sweep-summary/selection cells and markdown; remove candidate table.
filtered = []
for cell in nb.cells:
    source = "".join(cell.source)
    if cell.cell_type == "markdown" and "regularisation sweep" in source.lower():
        continue
    if cell.cell_type == "code" and (
        "compute_regularisation_summary" in source
        or "build_regularisation_tradeoff_figure" in source
        or "candidate_regs = reg_summary" in source
    ):
        continue
    filtered.append(cell)
nb.cells = filtered

# Make the baseline load explicitly test-only and reuse baseline_path.
for cell in nb.cells:
    if cell.cell_type == "code" and "baseline_path = next(path for path in files" in cell.source:
        cell.source = textwrap.dedent('''
        black_params, heston_params, svcj_params, test_data = chapter3.load_baseline_workbook(
            baseline_path,
            evaluation_split="test",
        )
        test_data.head()
        ''').strip()
        break
else:
    raise RuntimeError("could not find baseline workbook load cell")

# Remove aliases no longer needed and make parameter-path titles dynamic.
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    cell.source = cell.source.replace("find_calibration_files = chapter3.find_calibration_files\n", "")
    cell.source = cell.source.replace("reg_from_path = chapter3.reg_from_path\n", "")
    cell.source = cell.source.replace(
        'title="Selected SVCJ parameter paths for BTC at lambda_TS = 100",',
        'title=f"Selected SVCJ parameter paths for BTC at lambda_TS = {FINAL_REG}",',
    )
    cell.source = cell.source.replace(
        'title="Selected SVCJ parameter paths for ETH at lambda_TS = 100",',
        'title=f"Selected SVCJ parameter paths for ETH at lambda_TS = {FINAL_REG}",',
    )

nbformat.write(nb, final_path)
