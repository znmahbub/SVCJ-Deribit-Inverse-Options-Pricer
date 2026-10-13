from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd

from notebooks._lib import chapter3_final


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown_table(summary: pd.DataFrame) -> str:
    cols = [
        "reg",
        "rmse_validation_mean",
        "mae_validation_mean",
        "roughness_mean",
        "nfev_mean",
        "success_rate",
        "feller_violation_share",
        "svcj_martingale_violation_share",
        "svcj_contour_violation_share",
        "svcj_contour_margin_min",
    ]
    view = summary[cols].copy().sort_values("reg")
    lines = [
        "| lambda_TS | validation RMSE | validation MAE | roughness | mean nfev | success | Feller viol. | martingale viol. | contour viol. | min contour margin |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in view.iterrows():
        lines.append(
            "| {reg:g} | {rmse:.8f} | {mae:.8f} | {rough:.6f} | {nfev:.2f} | {success:.3%} | {feller:.3%} | {mart:.3%} | {contour:.3%} | {margin:.6g} |".format(
                reg=float(row["reg"]),
                rmse=float(row["rmse_validation_mean"]),
                mae=float(row["mae_validation_mean"]),
                rough=float(row["roughness_mean"]),
                nfev=float(row["nfev_mean"]),
                success=float(row["success_rate"]),
                feller=float(row["feller_violation_share"]),
                mart=float(row["svcj_martingale_violation_share"]),
                contour=float(row["svcj_contour_violation_share"]),
                margin=float(row["svcj_contour_margin_min"]),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    candidate_dir = args.candidate_dir.resolve()
    files = sorted(candidate_dir.glob("calibration_results_reg_*.xlsx"))
    if not files:
        raise RuntimeError(f"No candidate workbooks found in {candidate_dir}")

    expected = {0, 1, 3, 5, 10, 25, 50, 100, 250, 500, 1000}
    found = {chapter3_final.reg_from_path(path) for path in files}
    if found != expected:
        raise RuntimeError(f"Candidate grid mismatch: expected={sorted(expected)}, found={sorted(found)}")

    out_dir = root / "docs" / "reproducibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = chapter3_final.compute_regularisation_summary(
        files,
        out_dir,
        evaluation_split="validation",
    )
    selected = chapter3_final.select_regularisation_candidate(summary)
    summary_path = out_dir / "regularization_validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest_rows = []
    for path in files:
        manifest_rows.append(
            {
                "lambda_TS": chapter3_final.reg_from_path(path),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "workflow_run_id": str(args.workflow_run_id),
                "candidate_source_commit": str(args.source_sha),
            }
        )
    pd.DataFrame(manifest_rows).sort_values("lambda_TS").to_csv(
        out_dir / "regularization_candidate_manifest.csv", index=False
    )

    record = out_dir / "regularization_selection.md"
    record.write_text(
        "# Regularization selection record\n\n"
        "- Status: LOCKED\n"
        f"- Selected lambda_TS: {selected}\n"
        "- Candidate grid: 0, 1, 3, 5, 10, 25, 50, 100, 250, 500, 1000\n"
        "- Evaluation split: validation\n"
        "- Primary selection statistic: mean SVCJ snapshot-level `rmse_validation`\n"
        "- Deterministic tie-break order: `mae_validation`, parameter-path roughness, mean `nfev`, smaller `lambda_TS`\n"
        f"- Candidate source commit: `{args.source_sha}`\n"
        f"- GitHub Actions workflow run ID: `{args.workflow_run_id}`\n"
        "- Selection results commit: TBD\n\n"
        "## Validation sweep\n\n"
        + markdown_table(summary)
        + "\n\n"
        "## Selection rule\n\n"
        "The selected value is the candidate with the smallest mean snapshot-level SVCJ validation RMSE. "
        "Only exact primary-metric ties are resolved by validation MAE, then transformed-parameter path roughness, "
        "then mean optimizer evaluations, then the smaller lambda. The selection code does not read `test_data`, "
        "`rmse_test`, or `mae_test`.\n\n"
        "The candidate calibrations use the frozen 388-snapshot thesis window, deterministic 60/20/20 "
        "cross-sectional splits, and a fully chronological warm-start/regularization chain for both currencies. "
        "All lambda candidates, including lambda=0, were run sequentially in time to avoid changing the optimizer "
        "path across candidates.\n\n"
        "## Provenance\n\n"
        "The compact candidate workbooks are retained as workflow artifacts rather than committed to Git. "
        "`regularization_candidate_manifest.csv` records their SHA-256 hashes and sizes; "
        "`regularization_validation_summary.csv` contains the committed selection statistics.\n",
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"Selected lambda_TS={selected}")
    print(f"Wrote {record}")


if __name__ == "__main__":
    main()
