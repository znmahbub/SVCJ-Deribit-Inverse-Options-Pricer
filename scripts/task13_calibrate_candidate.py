"""Temporary Task 13 runner for one (lambda_TS, currency) calibration chain.

This script is orchestration-only and is removed after the remediation rerun.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.batch_runner import BatchConfig, run_currency_to_excel

ANALYSIS_START = "2025-12-30T17:31:15Z"
ANALYSIS_END = "2026-04-08T05:03:27Z"
EXPECTED_SNAPSHOTS = 388


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg", type=float, required=True)
    parser.add_argument("--currency", choices=("BTC", "ETH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = BatchConfig(
        project_root=root,
        data_dir=root / "data",
        output_xlsx=output,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        resume=False,
        save_every_n_files=20,
        l2_prev_strength=float(args.reg),
        n_workers=max(1, min(6, (os.cpu_count() or 4) - 1)),
        verbose=True,
    )
    run_currency_to_excel(args.currency, cfg=cfg)

    sheets = pd.read_excel(output, sheet_name=None)
    for sheet in ("black_params", "heston_params", "svcj_params"):
        frame = sheets.get(sheet, pd.DataFrame())
        if "currency" not in frame.columns:
            raise RuntimeError(f"{sheet} has no currency column")
        count = int((frame["currency"].astype(str).str.upper() == args.currency).sum())
        if count != EXPECTED_SNAPSHOTS:
            raise RuntimeError(
                f"{args.currency} lambda={args.reg:g}: expected {EXPECTED_SNAPSHOTS} {sheet} rows, got {count}"
            )

    svcj = sheets["svcj_params"]
    svcj_ccy = svcj.loc[svcj["currency"].astype(str).str.upper() == args.currency]
    success = pd.to_numeric(svcj_ccy.get("success"), errors="coerce")
    n_success = int((success == 1).sum())
    print(
        f"TASK13_CANDIDATE_OK reg={args.reg:g} currency={args.currency} "
        f"snapshots={EXPECTED_SNAPSHOTS} svcj_success={n_success}/{EXPECTED_SNAPSHOTS} output={output}"
    )


if __name__ == "__main__":
    main()
