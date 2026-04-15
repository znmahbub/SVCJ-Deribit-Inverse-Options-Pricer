from __future__ import annotations

from pathlib import Path


def collect_named_objects(namespace: dict[str, object], names: dict[str, str]) -> dict[str, object]:
    return {label: namespace[var_name] for label, var_name in names.items() if var_name in namespace}


def export_hedging_notebook_outputs(
    namespace: dict[str, object],
    *,
    export_tables: bool,
    export_figures_html: bool,
    export_dir: Path,
    analytics_module,
) -> None:
    if export_tables or export_figures_html:
        export_dir.mkdir(parents=True, exist_ok=True)

    figure_names = [
        "fig_pooled_rmse",
        "fig_pooled_mae",
        "fig_snapshot_rmse",
        "fig_ecdf",
        "fig_tail",
        "fig_maturity_rmse",
        "fig_basis_maturity",
        "fig_moneyness_bucket",
        "fig_basis_bucket",
    ]
    figures = {name: namespace[name] for name in figure_names if name in namespace}

    table_names = {
        "sample_construction": "sample_table",
        "headline_test": "headline_test",
        "maturity_test": "maturity_test",
        "maturity_ranges": "maturity_ranges",
        "moneyness_test": "moneyness_test",
        "moneyness_ranges": "moneyness_ranges",
        "basis_test": "basis_test",
        "basis_ranges": "basis_ranges",
        "sensitivity_table": "sensitivity_table",
    }
    tables = collect_named_objects(namespace, table_names)

    if export_tables and tables:
        analytics_module.export_tables_csv(tables, export_dir)
        print("Exported CSV tables to:", export_dir)
    elif export_tables:
        print("EXPORT_TABLES=True, but no tables are currently defined in memory.")

    if export_figures_html and figures:
        analytics_module.export_figures_html(figures, export_dir)
        print("Exported HTML figures to:", export_dir)
    elif export_figures_html:
        print("EXPORT_FIGURES_HTML=True, but no figures are currently defined in memory.")
