from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests._path import ROOT
from tests.notebook_utils import execute_notebook_in_memory, execute_notebook_with_replacements


NOTEBOOK_DIR = ROOT / "notebooks"
EXCEL_DIR = ROOT / "excel files"


def code_output_count(notebook) -> int:
    return sum(len(cell.get("outputs", [])) for cell in notebook.cells if cell.cell_type == "code")


def write_snapshot_fixture(data_dir: Path) -> None:
    rows = []
    for currency, forward in [("BTC", 100.0), ("ETH", 10.0)]:
        for option_type, strike in [("call", forward), ("put", 0.9 * forward)]:
            rows.append(
                {
                    "currency": currency,
                    "option_type": option_type,
                    "strike": strike,
                    "time_to_maturity": 30 / 365,
                    "bid_price": 0.01,
                    "ask_price": 0.02,
                    "futures_price": forward,
                    "vega": 1.0,
                    "open_interest": 10.0,
                    "expiry_datetime": "2026-02-01T00:00:00Z",
                }
            )
    frame = pd.DataFrame(rows)
    data_dir.mkdir(parents=True, exist_ok=True)
    for stamp in ["20260101T000000Z", "20260101T010000Z"]:
        frame.to_csv(data_dir / f"deribit_options_snapshot_{stamp}.csv", index=False)


def write_pricing_workbook(path: Path, *, price_shift: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    param_rows = []
    option_rows = []
    # Excel stores datetimes without timezone metadata. The production loader
    # restores UTC explicitly, so the smoke fixture mirrors the persisted form.
    for currency, forward in [("BTC", 100.0), ("ETH", 10.0)]:
        for i, ts in enumerate(pd.date_range("2026-01-01", periods=3, freq="h")):
            param_rows.append(
                {
                    "timestamp": ts,
                    "currency": currency,
                    "success": True,
                    "kappa": 2.0 + 0.05 * i,
                    "theta": 0.04,
                    "sigma_v": 0.30,
                    "rho": -0.5,
                    "v0": 0.04,
                    "lam": 0.2,
                    "ell_y": -0.05,
                    "sigma_y": 0.20,
                    "ell_v": 0.10,
                    "rho_j": -1.20 + 0.10 * i,
                }
            )
            for j in range(10):
                moneyness = 0.75 + 0.055 * j
                mid = 0.02 + 0.001 * j
                option_rows.append(
                    {
                        "snapshot_ts": ts,
                        "currency": currency,
                        "time_to_maturity": 0.03 + 0.01 * j,
                        "moneyness": moneyness,
                        "F0": forward,
                        "bid_price": mid - 0.002,
                        "ask_price": mid + 0.002,
                        "mid_price_clean": mid,
                        "price_black": mid + 0.0015 + price_shift,
                        "price_heston": mid + 0.0010 + price_shift,
                        "price_svcj": mid + 0.0005 + price_shift,
                    }
                )

    svcj = pd.DataFrame(param_rows)
    base_params = svcj[["timestamp", "currency", "success"]].copy()
    black = base_params.assign(sigma=0.60)
    heston = base_params.assign(kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.5, v0=0.04)
    test_data = pd.DataFrame(option_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        black.to_excel(writer, sheet_name="black_params", index=False)
        heston.to_excel(writer, sheet_name="heston_params", index=False)
        svcj.to_excel(writer, sheet_name="svcj_params", index=False)
        test_data.to_excel(writer, sheet_name="test_data", index=False)


class NotebookSmokeTests(unittest.TestCase):
    def _execute_or_skip(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except PermissionError as exc:
            self.skipTest(f"Notebook execution is not permitted in this environment: {exc}")

    def test_stage2_data_check_notebook_executes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            write_snapshot_fixture(data_dir)
            notebook = self._execute_or_skip(
                execute_notebook_with_replacements,
                NOTEBOOK_DIR / "00_check_data.ipynb",
                {
                    "inventory = inspect_snapshot_data(PATHS.data_dir)": f'inventory = inspect_snapshot_data(Path(r"{data_dir}"))',
                },
                working_directory=NOTEBOOK_DIR,
                timeout=600,
            )
        self.assertGreater(code_output_count(notebook), 2)

    def test_stage2_calibration_notebook_config_smoke_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data"
            write_snapshot_fixture(data_dir)
            notebook = self._execute_or_skip(
                execute_notebook_with_replacements,
                NOTEBOOK_DIR / "01_run_calibration.ipynb",
                {
                    "inventory = inspect_snapshot_data(PATHS.data_dir)": f'inventory = inspect_snapshot_data(Path(r"{data_dir}"))',
                    "calibration_outputs = run_calibration_suite(configs)": "calibration_outputs = {key: cfg.output_xlsx for key, cfg in configs.items()}",
                },
                working_directory=NOTEBOOK_DIR,
                timeout=600,
            )
        self.assertGreater(code_output_count(notebook), 2)

    def test_stage2_pricing_analysis_notebook_executes_on_fixture_workbooks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wls = root / "calibration_wls.xlsx"
            pwls = root / "calibration_pwls.xlsx"
            output_root = root / "pricing_outputs"
            write_pricing_workbook(wls, price_shift=0.0)
            write_pricing_workbook(pwls, price_shift=-0.0002)
            replacement = "WORKBOOKS = {\"wls\": Path(r\"%s\"), \"pwls\": Path(r\"%s\")}" % (wls, pwls)
            notebook = self._execute_or_skip(
                execute_notebook_with_replacements,
                NOTEBOOK_DIR / "02_pricing_analysis.ipynb",
                {
                    "WORKBOOKS = canonical_calibration_paths(ROOT)": replacement,
                    "OUTPUT_ROOT = NOTEBOOK_DIR / CANONICAL_PRICING_OUTPUT_SUBDIR": f'OUTPUT_ROOT = Path(r"{output_root}")',
                },
                working_directory=NOTEBOOK_DIR,
                timeout=1200,
            )
            self.assertTrue((output_root / "overall_metrics_by_specification.csv").exists())
            self.assertTrue((output_root / "snapshot_win_rates_by_specification.csv").exists())
        self.assertGreater(code_output_count(notebook), 4)

    def test_futures_viewer_notebook_executes(self) -> None:
        notebook = self._execute_or_skip(
            execute_notebook_in_memory,
            NOTEBOOK_DIR / "futures_plotly_viewer.ipynb",
            working_directory=NOTEBOOK_DIR,
            timeout=600,
        )
        self.assertGreater(code_output_count(notebook), 4)

    def test_hedging_notebook_executes_saved_output_mode(self) -> None:
        required = EXCEL_DIR / "hedging_results.xlsx"
        if not required.exists():
            self.skipTest(
                "Saved-output hedging smoke requires the external workbook "
                f"{required.name}, which is intentionally not versioned."
            )
        notebook = self._execute_or_skip(
            execute_notebook_with_replacements,
            NOTEBOOK_DIR / "hedging_analytics.ipynb",
            {"RUN_ENGINE = None": "RUN_ENGINE = False"},
            working_directory=NOTEBOOK_DIR,
            timeout=1200,
        )
        self.assertGreater(code_output_count(notebook), 10)


if __name__ == "__main__":
    unittest.main()
