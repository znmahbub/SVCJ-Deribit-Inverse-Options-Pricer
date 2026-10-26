from __future__ import annotations

"""Shared orchestration helpers for the canonical Stage 2 notebooks.

The notebooks in ``00_check_data.ipynb``, ``01_run_calibration.ipynb`` and
``02_pricing_analysis.ipynb`` are intentionally thin. This module owns the
reusable configuration, data checks, calibration-suite orchestration and
pricing-report assembly used by those notebooks.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.batch_runner import (
    BatchConfig,
    PWLS_LAMBDA_TS,
    WLS_LAMBDA_TS,
    run_all_snapshots_to_excel,
)
from src.calibration import WeightConfig
from src.inverse_fft_pricer import FFTParams
from src.snapshot_job import timestamp_from_filename

from . import chapter3_final as chapter3


REQUIRED_SNAPSHOT_COLUMNS = {
    "currency",
    "option_type",
    "strike",
    "time_to_maturity",
    "bid_price",
    "ask_price",
    "futures_price",
    "vega",
    "open_interest",
    "expiry_datetime",
}

CANONICAL_EXCEL_SUBDIR = "canonical"
CANONICAL_PRICING_OUTPUT_SUBDIR = Path("pricing_outputs") / "canonical"


@dataclass(frozen=True)
class CalibrationSpec:
    key: str
    label: str
    lambda_ts: float
    workbook_name: str


def canonical_calibration_specs() -> dict[str, CalibrationSpec]:
    """Return the two fixed paper specifications in execution order."""
    return {
        "wls": CalibrationSpec(
            key="wls",
            label="WLS",
            lambda_ts=float(WLS_LAMBDA_TS),
            workbook_name="calibration_wls.xlsx",
        ),
        "pwls": CalibrationSpec(
            key="pwls",
            label="PWLS",
            lambda_ts=float(PWLS_LAMBDA_TS),
            workbook_name="calibration_pwls.xlsx",
        ),
    }


def canonical_calibration_paths(project_root: Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    output_dir = root / "excel files" / CANONICAL_EXCEL_SUBDIR
    return {
        key: output_dir / spec.workbook_name
        for key, spec in canonical_calibration_specs().items()
    }


def build_calibration_configs(
    project_root: Path,
    *,
    start_dt=None,
    end_dt=None,
    resume: bool = True,
    n_workers: int | None = None,
    verbose: bool = True,
) -> dict[str, BatchConfig]:
    """Build canonical full-dataset WLS/PWLS batch configurations.

    ``start_dt`` and ``end_dt`` default to ``None`` deliberately: the canonical
    runner uses every snapshot present in ``data/`` unless the user explicitly
    requests a bounded robustness run. The scientific split is always 70/30
    train/test and the only temporal-penalty values are 0 (WLS) and 1 (PWLS).
    """
    root = Path(project_root).resolve()
    workers = int(n_workers) if n_workers is not None else max(1, int((os.cpu_count() or 4) - 2))
    paths = canonical_calibration_paths(root)

    common = dict(
        project_root=root,
        data_dir=root / "data",
        analysis_start=start_dt,
        analysis_end=end_dt,
        resume=bool(resume),
        save_every_n_files=50,
        smoke_test_max_files_per_currency=None,
        filter_rules=dict(
            require_bid_ask=True,
            min_time_to_maturity=1 / 365,
            max_time_to_maturity=None,
            min_open_interest=1.0,
            min_vega=0.0,
            max_rel_spread=0.50,
            moneyness_range=(0.5, 2.0),
            drop_synthetic_underlyings=False,
        ),
        min_options_after_filter=50,
        weight_config=WeightConfig(
            use_spread=True,
            use_vega=False,
            use_open_interest=False,
            spread_power=1.0,
            vega_power=0.5,
            oi_power=0.5,
            eps_spread=1e-6,
            eps_other=1e-12,
            cap=1e6,
        ),
        fft_base=FFTParams(
            N=2**12,
            eta=0.10,
            alpha=1.5,
            b=-10.0,
            use_simpson=True,
        ),
        train_frac=0.70,
        test_frac=0.30,
        global_random_seed=123,
        max_nfev=dict(black=250, heston=250, svcj=250),
        runtime_top_expiries_by_oi=None,
        runtime_max_options=None,
        n_workers=workers,
        limit_internal_threads=True,
        internal_num_threads=1,
        currencies=("BTC", "ETH"),
        verbose=bool(verbose),
    )

    return {
        key: BatchConfig(
            **common,
            output_xlsx=paths[key],
            l2_prev_strength=spec.lambda_ts,
        )
        for key, spec in canonical_calibration_specs().items()
    }


def run_calibration_suite(configs: Mapping[str, BatchConfig]) -> dict[str, Path]:
    """Run canonical WLS then PWLS and return only their workbook paths.

    ``run_all_snapshots_to_excel`` returns the full in-memory workbook payload.
    The canonical full-sample runner deliberately discards that payload after
    each specification is persisted so WLS data are not retained while PWLS
    is running.
    """
    required = tuple(canonical_calibration_specs())
    missing = [key for key in required if key not in configs]
    if missing:
        raise ValueError(f"missing canonical calibration configs: {missing}")

    output_paths: dict[str, Path] = {}
    for key in required:
        cfg = configs[key]
        cfg.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
        run_all_snapshots_to_excel(cfg)
        output_paths[key] = cfg.output_xlsx
    return output_paths


def inspect_snapshot_data(
    data_dir: Path,
    *,
    currencies: tuple[str, ...] = ("BTC", "ETH"),
) -> pd.DataFrame:
    """Build a light-weight inventory of all option snapshot CSVs.

    Only the header and ``currency`` column are read. This is intended as a
    pre-flight data check, not as a second cleaning implementation.
    """
    root = Path(data_dir)
    files = sorted(root.glob("deribit_options_snapshot_*.csv"), key=timestamp_from_filename)
    rows: list[dict[str, object]] = []

    for path in files:
        row: dict[str, object] = {
            "snapshot_filename": path.name,
            "snapshot_ts": timestamp_from_filename(path),
            "n_rows": 0,
            "schema_ok": False,
            "missing_columns": "",
            "read_error": "",
        }
        for currency in currencies:
            row[f"n_{currency.upper()}"] = 0

        try:
            header = pd.read_csv(path, nrows=0)
            missing = sorted(REQUIRED_SNAPSHOT_COLUMNS - set(header.columns))
            row["missing_columns"] = ",".join(missing)
            row["schema_ok"] = not missing

            if "currency" in header.columns:
                currency_col = pd.read_csv(path, usecols=["currency"])["currency"].astype(str).str.strip().str.upper()
                row["n_rows"] = int(len(currency_col))
                counts = currency_col.value_counts()
                for currency in currencies:
                    row[f"n_{currency.upper()}"] = int(counts.get(currency.upper(), 0))
        except Exception as exc:
            row["read_error"] = repr(exc)
            row["schema_ok"] = False

        rows.append(row)

    columns = [
        "snapshot_filename",
        "snapshot_ts",
        "n_rows",
        "schema_ok",
        "missing_columns",
        "read_error",
        *[f"n_{currency.upper()}" for currency in currencies],
    ]
    return pd.DataFrame(rows, columns=columns)


def summarize_snapshot_data(
    inventory: pd.DataFrame,
    *,
    currencies: tuple[str, ...] = ("BTC", "ETH"),
) -> pd.DataFrame:
    """Return a compact one-row coverage summary for the data-check notebook."""
    if inventory.empty:
        first = last = pd.NaT
    else:
        timestamps = pd.to_datetime(inventory["snapshot_ts"], utc=True, errors="coerce")
        first = timestamps.min()
        last = timestamps.max()

    row: dict[str, object] = {
        "n_snapshots": int(len(inventory)),
        "first_snapshot": first,
        "last_snapshot": last,
        "n_schema_failures": int((~inventory["schema_ok"].fillna(False)).sum()) if len(inventory) else 0,
        "n_duplicate_timestamps": int(pd.to_datetime(inventory.get("snapshot_ts"), utc=True, errors="coerce").duplicated().sum()) if len(inventory) else 0,
    }
    for currency in currencies:
        col = f"n_{currency.upper()}"
        values = pd.to_numeric(inventory[col], errors="coerce") if col in inventory else pd.Series(dtype=float)
        row[f"n_missing_{currency.upper()}"] = int((values.fillna(0) <= 0).sum()) if len(inventory) else 0
        row[f"n_rows_{currency.upper()}"] = int(values.fillna(0).sum()) if len(values) else 0
    return pd.DataFrame([row])


def assert_snapshot_data_ready(
    inventory: pd.DataFrame,
    *,
    currencies: tuple[str, ...] = ("BTC", "ETH"),
) -> None:
    """Fail fast when the canonical full-dataset run is not data-ready."""
    if inventory.empty:
        raise RuntimeError("no option snapshot CSVs were found")

    bad_schema = inventory.loc[~inventory["schema_ok"].fillna(False)]
    if len(bad_schema):
        details = bad_schema[["snapshot_filename", "missing_columns", "read_error"]].head(5).to_dict("records")
        raise RuntimeError(f"snapshot schema/read failures detected: {details}")

    ts = pd.to_datetime(inventory["snapshot_ts"], utc=True, errors="coerce")
    if ts.isna().any():
        raise RuntimeError("one or more snapshot timestamps could not be parsed")
    if ts.duplicated().any():
        duplicated = inventory.loc[ts.duplicated(keep=False), "snapshot_filename"].tolist()
        raise RuntimeError(f"duplicate snapshot timestamps detected: {duplicated[:10]}")

    for currency in currencies:
        col = f"n_{currency.upper()}"
        if col not in inventory.columns:
            raise RuntimeError(f"snapshot inventory is missing currency-count column {col}")
        missing = inventory.loc[pd.to_numeric(inventory[col], errors="coerce").fillna(0) <= 0, "snapshot_filename"]
        if len(missing):
            raise RuntimeError(f"{currency.upper()} is absent from {len(missing)} snapshots; first: {missing.iloc[0]}")


def load_pricing_inputs(
    workbook_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load parameter panels plus the untouched 30% test holdout only."""
    return chapter3.load_baseline_workbook(Path(workbook_path))


def _snapshot_win_rates(snapshot_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for currency, frame in snapshot_metrics.groupby("currency"):
        rows.append(
            {
                "currency": currency,
                "n_snapshots": int(len(frame)),
                "svcj_beats_heston_rmse": float((frame["rmse_svcj"] < frame["rmse_heston"]).mean()),
                "svcj_beats_heston_mae": float((frame["mae_svcj"] < frame["mae_heston"]).mean()),
                "svcj_beats_heston_within_spread": float((frame["within_svcj"] > frame["within_heston"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def run_single_pricing_analysis(
    workbook_path: Path,
    output_dir: Path,
    *,
    n_buckets: int = 5,
    make_figures: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build the pricing tables/figures for one fixed calibration specification."""
    workbook_path = Path(workbook_path)
    if not workbook_path.exists():
        raise FileNotFoundError(f"calibration workbook not found: {workbook_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    black_params, heston_params, svcj_params, test_data = load_pricing_inputs(workbook_path)
    overall = chapter3.compute_overall_baseline_metrics(test_data, output_dir)
    snapshot_metrics = chapter3.compute_snapshot_metrics(test_data, output_dir)
    bucket_metrics = chapter3.compute_bucket_metrics(test_data, n_buckets=n_buckets, output_dir=output_dir)
    parameter_summary = chapter3.compute_parameter_summary(svcj_params, output_dir)
    win_rates = _snapshot_win_rates(snapshot_metrics)
    win_rates.to_csv(output_dir / "snapshot_model_win_rates.csv", index=False)

    if make_figures:
        chapter3.plot_snapshot_rmse_mae_panel(snapshot_metrics, output_dir)
        chapter3.plot_snapshot_within_spread(snapshot_metrics, output_dir)
        chapter3.plot_bucket_panel(
            bucket_metrics,
            dimension="time_to_maturity",
            stem="bucket_diagnostics_maturity",
            title="",
            output_dir=output_dir,
        )
        chapter3.plot_bucket_panel(
            bucket_metrics,
            dimension="moneyness",
            stem="bucket_diagnostics_moneyness",
            title="",
            output_dir=output_dir,
        )
        for currency in ("BTC", "ETH"):
            if (svcj_params["currency"].astype(str).str.upper() == currency).any():
                chapter3.plot_parameter_paths(
                    svcj_params,
                    currency=currency,
                    stem=f"svcj_parameter_paths_{currency.lower()}",
                    title=f"SVCJ parameter paths — {currency}",
                    output_dir=output_dir,
                )

    return {
        "black_params": black_params,
        "heston_params": heston_params,
        "svcj_params": svcj_params,
        "test_data": test_data,
        "overall": overall,
        "snapshot_metrics": snapshot_metrics,
        "bucket_metrics": bucket_metrics,
        "parameter_summary": parameter_summary,
        "win_rates": win_rates,
    }


def run_pricing_analysis(
    workbook_paths: Mapping[str, Path],
    output_root: Path,
    *,
    n_buckets: int = 5,
    make_figures: bool = True,
) -> dict[str, object]:
    """Run complete test-holdout pricing reporting for WLS and PWLS."""
    required = tuple(canonical_calibration_specs())
    missing = [key for key in required if key not in workbook_paths]
    if missing:
        raise ValueError(f"missing canonical calibration workbooks: {missing}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, pd.DataFrame]] = {}
    overall_tables: list[pd.DataFrame] = []
    win_tables: list[pd.DataFrame] = []

    for key in required:
        run_dir = output_root / key
        report = run_single_pricing_analysis(
            Path(workbook_paths[key]),
            run_dir,
            n_buckets=n_buckets,
            make_figures=make_figures,
        )
        runs[key] = report
        overall_tables.append(report["overall"].assign(specification=key))
        win_tables.append(report["win_rates"].assign(specification=key))

    overall_comparison = pd.concat(overall_tables, ignore_index=True, sort=False)
    win_rate_comparison = pd.concat(win_tables, ignore_index=True, sort=False)
    overall_comparison.to_csv(output_root / "overall_metrics_by_specification.csv", index=False)
    win_rate_comparison.to_csv(output_root / "snapshot_win_rates_by_specification.csv", index=False)

    return {
        "runs": runs,
        "overall_comparison": overall_comparison,
        "win_rate_comparison": win_rate_comparison,
    }


__all__ = [
    "CalibrationSpec",
    "REQUIRED_SNAPSHOT_COLUMNS",
    "CANONICAL_EXCEL_SUBDIR",
    "CANONICAL_PRICING_OUTPUT_SUBDIR",
    "canonical_calibration_specs",
    "canonical_calibration_paths",
    "build_calibration_configs",
    "run_calibration_suite",
    "inspect_snapshot_data",
    "summarize_snapshot_data",
    "assert_snapshot_data_ready",
    "load_pricing_inputs",
    "run_single_pricing_analysis",
    "run_pricing_analysis",
]
