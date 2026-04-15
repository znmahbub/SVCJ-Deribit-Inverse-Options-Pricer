from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOKS = [0, 1, 3, 5, 10, 50, 100]


def bootstrap_cell(reg_val: int) -> str:
    return f"""from pathlib import Path
import sys

from IPython.display import display

NOTEBOOK_ROOT = Path.cwd().resolve()
if (NOTEBOOK_ROOT / "_lib").exists():
    NOTEBOOK_DIR = NOTEBOOK_ROOT
elif (NOTEBOOK_ROOT / "notebooks" / "_lib").exists():
    NOTEBOOK_DIR = NOTEBOOK_ROOT / "notebooks"
else:
    raise FileNotFoundError(f"Could not locate notebooks/_lib from {{NOTEBOOK_ROOT}}")

if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from _lib import chapter3_analysis as analysis
from _lib.common import ensure_existing_path, locate_notebook_paths

RNG = analysis.initialize_notebook_defaults()
PATHS = locate_notebook_paths(NOTEBOOK_DIR)
DATA_PATH = "calibration_results_reg_{reg_val}.xlsx"
MODEL_SPECS = analysis.MODEL_SPECS
COLORS = analysis.COLORS
FIGDIMS = analysis.FIGDIMS
DATA_FILE = ensure_existing_path(Path(DATA_PATH), search_dirs=[PATHS.excel_dir, PATHS.project_root, PATHS.notebook_dir])
"""


def load_state_cell() -> str:
    return """state = analysis.build_analysis_state(DATA_FILE, rng=RNG)

black_params = state.black_params
heston_params = state.heston_params
svcj_params = state.svcj_params
train_data = state.train_data
test_data = state.test_data

print("Loaded from:", DATA_FILE)
print(" - black_params:", black_params.shape)
print(" - heston_params:", heston_params.shape)
print(" - svcj_params:", svcj_params.shape)
print(" - train_data:", train_data.shape)
print(" - test_data :", test_data.shape)

display(black_params.head(3))
"""


def code_cells(reg_val: int) -> list[str]:
    return [
        bootstrap_cell(reg_val),
        load_state_cell(),
        """results_long = state.results_long
results_ok = state.results_ok

display(results_long.head(6))
print("Currencies:", results_long["currency"].unique())
""",
        """opt_metrics = state.opt_metrics

display(opt_metrics.head(6))
print("Option-derived snapshot metrics:", opt_metrics.shape)
""",
        """def bootstrap_mean_ci(values, n_boot=3000, alpha=0.05, rng=RNG):
    return analysis.bootstrap_mean_ci(values, n_boot=n_boot, alpha=alpha, rng=rng)


def summarize_snapshot_series(values, n_boot=3000):
    return analysis.summarize_snapshot_series(values, n_boot=n_boot, rng=RNG)
""",
        """add_line = analysis.add_line
add_box = analysis.add_box
add_subplot_legend = analysis.add_subplot_legend
plot_error_timeseries = analysis.plot_error_timeseries
plot_error_boxplots = analysis.plot_error_boxplots
""",
        """def error_summary_table(results_long_df, currency, split="test", n_boot=3000):
    return analysis.error_summary_table(results_long_df, currency, split=split, n_boot=n_boot, rng=RNG)
""",
        """convergence_table = analysis.convergence_table

display(convergence_table(results_long))
""",
        """spread_metric_timeseries = analysis.spread_metric_timeseries


def spread_metric_summary_table(opt_metrics_df, currency, split="test", n_boot=3000):
    return analysis.spread_metric_summary_table(opt_metrics_df, currency, split=split, n_boot=n_boot, rng=RNG)
""",
        """MONEY_BINS = analysis.MONEY_BINS
MONEY_LABELS = analysis.MONEY_LABELS
T_BINS = analysis.T_BINS
T_LABELS = analysis.T_LABELS
bucket_mae_by_snapshot = analysis.bucket_mae_by_snapshot
bucket_all = state.bucket_all

display(bucket_all.head(6))
""",
        """def bucket_summary_table(bucket_df, currency, bucket_type, n_boot=2000):
    return analysis.bucket_summary_table(bucket_df, currency, bucket_type, n_boot=n_boot, rng=RNG)


plot_bucket_bars = analysis.plot_bucket_bars
""",
        """RHO_LB = analysis.RHO_LB
RHO_UB = analysis.RHO_UB
BOUNDS = analysis.BOUNDS
EPS = analysis.EPS
add_feller_ratio = analysis.add_feller_ratio
near_bound_rates = analysis.near_bound_rates
results_ok = add_feller_ratio(results_ok)
""",
        """plot_param_timeseries = analysis.plot_param_timeseries
plot_param_boxplots = analysis.plot_param_boxplots
""",
        """def run_currency_report(currency, n_boot=3000):
    return analysis.run_currency_report(state, currency, n_boot=n_boot)


RUN_REPORTS = True
N_BOOT = 3000

if RUN_REPORTS:
    run_currency_report("BTC", n_boot=N_BOOT)
    run_currency_report("ETH", n_boot=N_BOOT)
else:
    print("RUN_REPORTS=False. Set RUN_REPORTS=True to generate the full BTC/ETH report outputs.")
""",
        """EXPORT = False

OUT_TABLES = Path("tables")
OUT_FIGS = Path("figs")


def _safe_write_image(fig, path_png):
    try:
        fig.write_image(path_png, scale=2)
        return True
    except Exception as exc:
        print(f"[warn] Could not write PNG (needs kaleido): {path_png}\\n  {exc}")
        return False


if EXPORT:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    conv = convergence_table(results_long)
    conv.to_csv(OUT_TABLES / "convergence_table.csv", index=False)

    for currency in results_long["currency"].unique():
        for split in ["train", "test"]:
            tbl = error_summary_table(results_long, currency, split=split)
            tbl.to_csv(OUT_TABLES / f"{currency.lower()}_{split}_error_summary.csv", index=False)

            tbl_sp = spread_metric_summary_table(opt_metrics, currency, split=split)
            tbl_sp.to_csv(OUT_TABLES / f"{currency.lower()}_{split}_spread_summary.csv", index=False)

            fig_ts = plot_error_timeseries(results_long, currency, split=split)
            fig_ts.write_html(OUT_FIGS / f"{currency.lower()}_{split}_errors_timeseries.html")
            _safe_write_image(fig_ts, OUT_FIGS / f"{currency.lower()}_{split}_errors_timeseries.png")

            fig_box = plot_error_boxplots(results_long, currency, split=split)
            fig_box.write_html(OUT_FIGS / f"{currency.lower()}_{split}_errors_boxplots.html")
            _safe_write_image(fig_box, OUT_FIGS / f"{currency.lower()}_{split}_errors_boxplots.png")

            fig_sp = spread_metric_timeseries(opt_metrics, currency, split=split)
            fig_sp.write_html(OUT_FIGS / f"{currency.lower()}_{split}_spread_timeseries.html")
            _safe_write_image(fig_sp, OUT_FIGS / f"{currency.lower()}_{split}_spread_timeseries.png")

        b1 = bucket_summary_table(bucket_all, currency, "moneyness")
        b2 = bucket_summary_table(bucket_all, currency, "maturity")
        b1.to_csv(OUT_TABLES / f"{currency.lower()}_test_bucket_moneyness.csv", index=False)
        b2.to_csv(OUT_TABLES / f"{currency.lower()}_test_bucket_maturity.csv", index=False)

        fig_bm = plot_bucket_bars(bucket_all, currency, "moneyness")
        fig_bt = plot_bucket_bars(bucket_all, currency, "maturity")
        fig_bm.write_html(OUT_FIGS / f"{currency.lower()}_test_bucket_moneyness.html")
        fig_bt.write_html(OUT_FIGS / f"{currency.lower()}_test_bucket_maturity.html")
        _safe_write_image(fig_bm, OUT_FIGS / f"{currency.lower()}_test_bucket_moneyness.png")
        _safe_write_image(fig_bt, OUT_FIGS / f"{currency.lower()}_test_bucket_maturity.png")

        nb_hes = near_bound_rates(results_long[results_long["currency"] == currency], "Heston")
        nb_svcj = near_bound_rates(results_long[results_long["currency"] == currency], "SVCJ")
        nb_hes.to_csv(OUT_TABLES / f"{currency.lower()}_heston_near_bound_rates.csv", index=False)
        nb_svcj.to_csv(OUT_TABLES / f"{currency.lower()}_svcj_near_bound_rates.csv", index=False)

    print("Export complete.")
else:
    print("EXPORT=False (no files written). Set EXPORT=True to save tables/figures.")
""",
    ]


def sync_notebook(reg_val: int) -> None:
    notebook_path = NOTEBOOK_DIR / f"calibration_analysis_complete_reg_{reg_val}.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)
    code_sources = code_cells(reg_val)
    code_cells_in_nb = [cell for cell in nb.cells if cell.cell_type == "code"]
    if len(code_cells_in_nb) != len(code_sources):
        raise ValueError(f"{notebook_path} has {len(code_cells_in_nb)} code cells, expected {len(code_sources)}")

    for cell, source in zip(code_cells_in_nb, code_sources):
        cell["source"] = source
        cell["outputs"] = []
        cell["execution_count"] = None

    nbformat.write(nbformat.validator.normalize(nb)[1], notebook_path)
    print(f"Synced {notebook_path.name}")


def main() -> None:
    for reg_val in NOTEBOOKS:
        sync_notebook(reg_val)


if __name__ == "__main__":
    main()
