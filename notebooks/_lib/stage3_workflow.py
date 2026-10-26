from __future__ import annotations

"""Canonical Stage 3 hedging/reporting orchestration.

The maintained notebooks remain thin runners. This module owns the fixed
PWLS/WLS hedging configuration, test-only hedging execution, reusable hedging
report assembly, and the final packaging step that copies already-generated
pricing/hedging artifacts without recomputing models.
"""

from pathlib import Path
from shutil import copy2
from typing import Mapping

import pandas as pd

from src.hedging_analysis import HedgingConfig, run_hedging_analysis
from src import hedging_analytics as analytics

from .stage2_workflow import (
    CANONICAL_PRICING_OUTPUT_SUBDIR,
    canonical_calibration_paths,
)


CANONICAL_HEDGING_REPORT_SUBDIR = Path("hedging_outputs") / "canonical"
CANONICAL_FINAL_OUTPUT_SUBDIR = Path("final_outputs") / "canonical"


def canonical_hedging_paths(project_root: Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    output_dir = root / "excel files" / "canonical"
    return {
        "wls": output_dir / "hedging_wls.xlsx",
        "pwls": output_dir / "hedging_pwls.xlsx",
    }


def build_hedging_configs(
    project_root: Path,
    *,
    run_wls: bool = False,
    run_pwls: bool = True,
    start_dt=None,
    end_dt=None,
    rebalance_every_n_snapshots: int = 1,
    currencies: tuple[str, ...] | None = None,
    verify_prices: bool = True,
) -> dict[str, HedgingConfig]:
    """Build canonical test-only hedging configs.

    PWLS is the default final specification. WLS is optional and exists only
    for direct benchmark comparison. ``None`` date bounds mean all available
    snapshots; bounded runs use the same inclusive UTC window in the hedging
    engine and forward-snapshot lookup.
    """
    root = Path(project_root).resolve()
    calibration = canonical_calibration_paths(root)
    outputs = canonical_hedging_paths(root)

    requested: list[str] = []
    if run_wls:
        requested.append("wls")
    if run_pwls:
        requested.append("pwls")
    if not requested:
        raise ValueError("At least one of run_wls or run_pwls must be True")

    configs: dict[str, HedgingConfig] = {}
    for key in requested:
        configs[key] = HedgingConfig(
            calibration_xlsx=calibration[key],
            data_dir=root / "data",
            output_xlsx=outputs[key],
            analysis_start=start_dt,
            analysis_end=end_dt,
            verify_prices=bool(verify_prices),
            rebalance_every_n_snapshots=int(rebalance_every_n_snapshots),
            currencies=currencies,
            evaluation_splits=("test",),
            exclude_expiry_crossing=True,
        )
    return configs


def run_hedging_suite(configs: Mapping[str, HedgingConfig]) -> dict[str, Path]:
    """Run requested hedging specs and retain only persisted workbook paths."""
    output_paths: dict[str, Path] = {}
    for key, cfg in configs.items():
        if not cfg.calibration_xlsx.exists():
            raise FileNotFoundError(
                f"Calibration workbook not found for {key}: {cfg.calibration_xlsx}. "
                "Run 01_run_calibration.ipynb first."
            )
        cfg.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
        run_hedging_analysis(cfg)
        output_paths[key] = cfg.output_xlsx
    return output_paths


def _equal_count_tables(
    interval_long: pd.DataFrame,
    *,
    value_col: str,
    bucket_col: str,
    value_kind: str,
    q: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, summary, ranges = analytics.build_equal_count_bucket_summary(
        interval_long,
        value_col=value_col,
        bucket_col=bucket_col,
        q=q,
        group_cols=("currency",),
    )
    label_col = f"{bucket_col}_label"
    summary = analytics.attach_bucket_range_labels(
        summary,
        ranges,
        bucket_col=bucket_col,
        output_col=label_col,
        value_kind=value_kind,
    )
    return summary, ranges


def build_hedging_report(
    hedging_workbook: Path,
    output_dir: Path,
    *,
    q: int = 5,
    make_figures: bool = True,
) -> dict[str, object]:
    """Build reusable tables/figures from one persisted hedging workbook."""
    hedging_workbook = Path(hedging_workbook)
    if not hedging_workbook.exists():
        raise FileNotFoundError(f"Hedging workbook not found: {hedging_workbook}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workbook = analytics.load_output_workbook(hedging_workbook)
    if "hedge_interval_panel" not in workbook:
        raise ValueError(f"{hedging_workbook} is missing hedge_interval_panel")

    panel = analytics.ensure_analysis_columns(workbook["hedge_interval_panel"])
    interval_long = analytics.make_interval_long(panel)
    test_long = interval_long.loc[
        interval_long["split"].astype(str).str.lower() == "test"
    ].copy()

    sample = analytics.build_sample_construction(panel)
    headline = analytics.pooled_summary(
        test_long,
        ["currency", "split", "model", "hedge_type"],
    )
    by_timestamp = analytics.pooled_summary(
        test_long,
        ["snapshot_ts", "currency", "split", "model", "hedge_type"],
    )
    maturity, maturity_ranges = _equal_count_tables(
        test_long,
        value_col="time_to_maturity_days",
        bucket_col="maturity_equal_bucket",
        value_kind="maturity_days",
        q=q,
    )
    moneyness, moneyness_ranges = _equal_count_tables(
        test_long,
        value_col="moneyness_ratio",
        bucket_col="moneyness_equal_bucket",
        value_kind="moneyness",
        q=q,
    )
    basis, basis_ranges = _equal_count_tables(
        test_long,
        value_col="basis_abs",
        bucket_col="basis_equal_bucket",
        value_kind="basis_abs",
        q=q,
    )

    representative_model = analytics.choose_representative_model(headline)
    representative = analytics.representative_panel(
        test_long,
        model=representative_model,
        split="test",
    )
    representative_series = analytics.unhedged_vs_hedged_panel(representative)
    sensitivity = analytics.build_sensitivity_table(representative)

    tables = {
        "sample_construction": sample,
        "headline_test": headline,
        "by_timestamp_test": by_timestamp,
        "maturity_test": maturity,
        "maturity_ranges": maturity_ranges,
        "moneyness_test": moneyness,
        "moneyness_ranges": moneyness_ranges,
        "basis_test": basis,
        "basis_ranges": basis_ranges,
        "sensitivity_table": sensitivity,
    }
    analytics.export_tables_csv(tables, output_dir)

    figures = {}
    if make_figures:
        figures = {
            "pooled_rmse_reduction": analytics.figure_pooled_metric(
                headline, metric="rmse_reduction", split="test"
            ),
            "pooled_mae_reduction": analytics.figure_pooled_metric(
                headline, metric="mae_reduction", split="test"
            ),
            "snapshot_rmse_reduction": analytics.figure_snapshot_metric(
                by_timestamp,
                metric="rmse_reduction",
                representative_model=representative_model,
                hedge_type="net_delta",
            ),
            "ecdf_absolute_pnl": analytics.figure_ecdf_abs_pnl(representative_series),
            "tail_comparison": analytics.figure_tail_comparison(representative_series),
            "maturity_rmse_reduction": analytics.figure_equal_count_maturity_metric(
                representative,
                metric="rmse_reduction",
                q=q,
            ),
            "moneyness_rmse_reduction": analytics.figure_equal_count_moneyness_metric(
                representative,
                metric="rmse_reduction",
                q=q,
            ),
            "basis_rmse_reduction": analytics.figure_equal_count_basis_metric(
                representative,
                metric="rmse_reduction",
                q=q,
            ),
        }
        analytics.export_figures_html(figures, output_dir)

    return {
        "tables": tables,
        "figures": figures,
        "representative_model": representative_model,
    }


def run_hedging_reporting(
    hedging_workbooks: Mapping[str, Path],
    output_root: Path,
    *,
    q: int = 5,
    make_figures: bool = True,
) -> dict[str, dict[str, object]]:
    """Build reports for each persisted hedging specification."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, object]] = {}
    for key, workbook in hedging_workbooks.items():
        reports[key] = build_hedging_report(
            Path(workbook),
            output_root / key,
            q=q,
            make_figures=make_figures,
        )
    return reports


def _copy_artifacts(
    source_root: Path,
    destination_root: Path,
    *,
    suffix: str,
    artifact_type: str,
    manifest_rows: list[dict[str, str]],
) -> None:
    if not source_root.exists():
        return
    for source in sorted(source_root.rglob(f"*{suffix}")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)
        manifest_rows.append(
            {
                "artifact_type": artifact_type,
                "source": str(source),
                "destination": str(destination),
            }
        )


def assemble_final_package(
    pricing_output_root: Path,
    hedging_output_root: Path,
    final_output_root: Path,
) -> pd.DataFrame:
    """Collect existing pricing/hedging CSV and HTML artifacts only.

    This function deliberately performs no calibration, pricing, or hedging
    recomputation. It creates a stable final directory tree and manifest from
    outputs generated by notebooks 02 and 03.
    """
    pricing_output_root = Path(pricing_output_root)
    hedging_output_root = Path(hedging_output_root)
    final_output_root = Path(final_output_root)

    tables_root = final_output_root / "tables"
    figures_root = final_output_root / "figures"
    tables_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    _copy_artifacts(
        pricing_output_root,
        tables_root / "pricing",
        suffix=".csv",
        artifact_type="pricing_table",
        manifest_rows=manifest_rows,
    )
    _copy_artifacts(
        hedging_output_root,
        tables_root / "hedging",
        suffix=".csv",
        artifact_type="hedging_table",
        manifest_rows=manifest_rows,
    )
    _copy_artifacts(
        pricing_output_root,
        figures_root / "pricing",
        suffix=".html",
        artifact_type="pricing_figure",
        manifest_rows=manifest_rows,
    )
    _copy_artifacts(
        hedging_output_root,
        figures_root / "hedging",
        suffix=".html",
        artifact_type="hedging_figure",
        manifest_rows=manifest_rows,
    )

    manifest = pd.DataFrame(
        manifest_rows,
        columns=["artifact_type", "source", "destination"],
    )
    final_output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(final_output_root / "manifest.csv", index=False)
    return manifest


__all__ = [
    "CANONICAL_HEDGING_REPORT_SUBDIR",
    "CANONICAL_FINAL_OUTPUT_SUBDIR",
    "CANONICAL_PRICING_OUTPUT_SUBDIR",
    "canonical_hedging_paths",
    "build_hedging_configs",
    "run_hedging_suite",
    "build_hedging_report",
    "run_hedging_reporting",
    "assemble_final_package",
]
