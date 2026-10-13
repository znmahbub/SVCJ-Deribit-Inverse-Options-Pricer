from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


GRID = (0, 1, 3, 5, 10, 25, 50, 100, 250, 500, 1000)
ANALYSIS_START = "2025-12-30T17:31:15Z"
ANALYSIS_END = "2026-04-08T05:03:27Z"
EXPECTED_PARAM_ROWS = 776


def one_match(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match for {pattern!r} under {root}, found {matches}")
    return matches[0]


def main() -> None:
    download_root = Path(os.environ.get("CANDIDATE_ROOT", ROOT / "_task13_selection" / "candidates"))
    remediation_dir = ROOT / "excel files" / "remediation"
    remediation_dir.mkdir(parents=True, exist_ok=True)

    qcs: list[dict] = []
    for reg in GRID:
        artifact_dir = download_root / f"candidate-reg-{reg}"
        csv_path = one_match(artifact_dir, f"**/svcj_params_reg_{reg}.csv")
        qc_path = one_match(artifact_dir, f"**/candidate_qc_reg_{reg}.json")
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if int(qc["lambda_TS"]) != reg:
            raise RuntimeError(f"Candidate artifact mismatch: expected lambda {reg}, got {qc['lambda_TS']}")
        if qc["analysis_start"] != ANALYSIS_START or qc["analysis_end"] != ANALYSIS_END:
            raise RuntimeError(f"lambda={reg}: frozen analysis window mismatch in candidate QC")
        if int(qc["svcj_param_rows"]) != EXPECTED_PARAM_ROWS:
            raise RuntimeError(f"lambda={reg}: expected {EXPECTED_PARAM_ROWS} parameter rows")

        params = pd.read_csv(csv_path)
        if len(params) != EXPECTED_PARAM_ROWS:
            raise RuntimeError(f"lambda={reg}: compact parameter export has {len(params)} rows")
        forbidden = [col for col in params.columns if "test" in col.lower()]
        if forbidden:
            raise RuntimeError(f"lambda={reg}: selection artifact leaks final-test columns: {forbidden}")
        required_metrics = {"rmse_validation", "mae_validation"}
        if not required_metrics.issubset(params.columns):
            raise RuntimeError(f"lambda={reg}: compact export is missing validation metrics")

        xlsx = remediation_dir / f"calibration_results_reg_{reg}.xlsx"
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            params.to_excel(writer, sheet_name="svcj_params", index=False)
        qcs.append(qc)

    qc_table = pd.DataFrame(qcs).sort_values("lambda_TS").reset_index(drop=True)
    out_dir = ROOT / "notebooks" / "chapter3_outputs" / "remediation" / "regularization_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_table.to_csv(out_dir / "candidate_qc.csv", index=False)
    print(qc_table[["lambda_TS", "svcj_success_rate", "validation_rmse_mean", "validation_mae_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
