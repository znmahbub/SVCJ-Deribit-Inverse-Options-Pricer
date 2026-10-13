from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from notebooks._lib import chapter3_final


EXPECTED_GRID = {0, 1, 3, 5, 10, 25, 50, 100, 250, 500, 1000}
EXPECTED_CURRENCIES = {"BTC", "ETH"}
PART_RE = re.compile(r"^calibration_results_reg_(\d+)_(BTC|ETH)\.xlsx$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_candidate_parts(
    candidate_dir: Path,
    *,
    expected_grid: set[int] = EXPECTED_GRID,
) -> dict[int, dict[str, Path]]:
    """Discover the exact lambda x currency artifact grid produced by Task 9."""
    candidate_dir = Path(candidate_dir)
    parts: dict[int, dict[str, Path]] = {}
    for path in sorted(candidate_dir.glob("calibration_results_reg_*.xlsx")):
        match = PART_RE.fullmatch(path.name)
        if not match:
            raise RuntimeError(f"unexpected task 9 candidate filename: {path.name}")
        reg = int(match.group(1))
        currency = match.group(2)
        if reg not in expected_grid:
            raise RuntimeError(f"unexpected lambda_TS candidate {reg} in {path.name}")
        if currency in parts.setdefault(reg, {}):
            raise RuntimeError(f"duplicate candidate artifact for lambda_TS={reg}, currency={currency}")
        parts[reg][currency] = path

    actual_grid = set(parts)
    missing_regs = sorted(expected_grid - actual_grid)
    unexpected_regs = sorted(actual_grid - expected_grid)
    bad_currency_sets = {
        reg: sorted(EXPECTED_CURRENCIES - set(by_currency))
        for reg, by_currency in parts.items()
        if set(by_currency) != EXPECTED_CURRENCIES
    }
    if missing_regs or unexpected_regs or bad_currency_sets:
        raise RuntimeError(
            "candidate artifact grid mismatch: "
            f"missing_lambdas={missing_regs}, unexpected_lambdas={unexpected_regs}, "
            f"currency_mismatches={bad_currency_sets}"
        )
    return parts


def combine_candidate_parts(
    parts: dict[int, dict[str, Path]],
    combined_dir: Path,
) -> list[Path]:
    """Create one compact SVCJ parameter workbook per lambda for analysis."""
    combined_dir = Path(combined_dir)
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_files: list[Path] = []
    for reg in sorted(parts):
        frames = []
        for currency in sorted(EXPECTED_CURRENCIES):
            path = parts[reg][currency]
            frame = pd.read_excel(path, sheet_name="svcj_params")
            if len(frame) != 388:
                raise RuntimeError(
                    f"lambda_TS={reg}, currency={currency}: expected 388 rows, found {len(frame)}"
                )
            observed = set(frame["currency"].dropna().astype(str).str.upper().unique())
            if observed != {currency}:
                raise RuntimeError(
                    f"lambda_TS={reg}, artifact {path.name}: currency column is {sorted(observed)}, expected {[currency]}"
                )
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True)
        if len(combined) != 776:
            raise RuntimeError(f"lambda_TS={reg}: expected 776 combined rows, found {len(combined)}")
        out = combined_dir / f"calibration_results_reg_{reg}.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            combined.to_excel(writer, sheet_name="svcj_params", index=False)
        combined_files.append(out)
    return combined_files


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
    parts = discover_candidate_parts(candidate_dir)
    combined_files = combine_candidate_parts(parts, candidate_dir / "_combined")

    out_dir = root / "docs" / "reproducibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = chapter3_final.compute_regularisation_summary(
        combined_files,
        out_dir,
        evaluation_split="validation",
    )
    selected = chapter3_final.select_regularisation_candidate(summary)
    summary_path = out_dir / "regularization_validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest_rows = []
    for reg in sorted(parts):
        for currency in sorted(parts[reg]):
            path = parts[reg][currency]
            manifest_rows.append(
                {
                    "lambda_TS": reg,
                    "currency": currency,
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "workflow_run_id": str(args.workflow_run_id),
                    "candidate_source_commit": str(args.source_sha),
                }
            )
    pd.DataFrame(manifest_rows).sort_values(["lambda_TS", "currency"]).to_csv(
        out_dir / "regularization_candidate_manifest.csv", index=False
    )

    record = out_dir / "regularization_selection.md"
    record.write_text(
        "# Regularization selection record\n\n"
        "- Status: LOCKED\n"
        f"- Selected lambda_TS: {selected}\n"
        "- Candidate grid: 0, 1, 3, 5, 10, 25, 50, 100, 250, 500, 1000\n"
        "- Evaluation split: validation\n"
        "- Primary selection statistic: mean SVCJ snapshot-level `rmse_validation` across BTC and ETH\n"
        "- Deterministic tie-break order: `mae_validation`, parameter-path roughness, mean `nfev`, smaller `lambda_TS`\n"
        f"- Candidate source commit: `{args.source_sha}`\n"
        f"- GitHub Actions workflow run ID: `{args.workflow_run_id}`\n"
        "- Selection results commit: TBD\n\n"
        "## Validation sweep\n\n"
        + markdown_table(summary)
        + "\n\n"
        "## Selection rule\n\n"
        "The selected value is the candidate with the smallest mean snapshot-level SVCJ validation RMSE across "
        "the two currency paths. Only exact primary-metric ties are resolved by validation MAE, then "
        "transformed-parameter path roughness, then mean optimizer evaluations, then the smaller lambda. "
        "The selection code does not read `test_data`, `rmse_test`, or `mae_test`.\n\n"
        "The candidate calibrations use the reconciled 388-snapshot submitted-thesis sample, deterministic "
        "60/20/20 cross-sectional splits, and strictly chronological warm-start/regularization chains within "
        "each currency. BTC and ETH are run independently for computational parallelism and are concatenated "
        "only after calibration for the validation summary. All lambda candidates, including lambda=0, use "
        "the same chronological initialization logic. The optimized Task 9 runner omits Heston and omits "
        "full train/test repricing because neither enters SVCJ regularization selection; Black is calibrated "
        "only until the first successful SVCJ warm start, exactly where it can affect SVCJ initialization.\n\n"
        "## Provenance\n\n"
        "The 22 compact candidate workbooks are retained as workflow artifacts rather than committed to Git. "
        "`regularization_candidate_manifest.csv` records their SHA-256 hashes and sizes; "
        "`regularization_validation_summary.csv` contains the committed selection statistics.\n",
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"Selected lambda_TS={selected}")
    print(f"Wrote {record}")


if __name__ == "__main__":
    main()
