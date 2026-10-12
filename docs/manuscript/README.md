# Manuscript source status

The editable thesis source supplied for the remediation pass is:

`Pricing and Hedging Deribit Inverse Options under Affine Stochastic Volatility Models with Jumps.tex`

Source integrity:

- Size: `204101` bytes
- Lines: `4085`
- SHA-256: `72a10d9253eef54779caf3c53c29fac7267a647f84d39ab88a37ca91bb4186de`
- Expected Git blob SHA-1 for the exact bytes: `a934ab14b440ce27bbd0f130bce8f3ac96513752`

The source is a standalone LaTeX document (no `\\input`/`\\include` subfiles), but a faithful full build also expects external assets.

## Missing build inputs on this branch

The supplied source references `references.bib`; that file is not currently present in the repository. The source contains an `\\IfFileExists{references.bib}` fallback, so LaTeX can proceed without it, but citations/bibliography are then unresolved and the result is not a faithful manuscript build.

It also references the following image files under `images/`, none of which are currently present under `docs/manuscript/`:

- `filtered_vs_raw.png`
- `exploratory_distributions.png`
- `regularisation_tradeoff.png`
- `ts_rmse_mae_panel.png`
- `ts_within_spread_panel.png`
- `bucket_diagnostics_maturity.png`
- `bucket_diagnostics_moneyness.png`
- `param_paths_btc.png`
- `param_paths_eth.png`
- `hedge_pooled_rmse.png`
- `hedge_pooled_mae.png`
- `hedge_snapshot_btc.png`
- `hedge_snapshot_eth.png`
- `hedge_maturity_rmse.png`
- `hedge_moneyness_rmse.png`
- `hedge_basis_rmse.png`

A local compile attempt was started with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The first pass generated fonts and reported undefined citations before the execution time limit was reached. It did not establish a faithful baseline build because the bibliography and figure assets above are not available alongside the source.

## Rule for Task 2

Do **not** reconstruct `references.bib` or the missing figures from the PDF. Add the exact original assets when available, then compile and compare the resulting PDF to the committed baseline PDF before making manuscript edits.
