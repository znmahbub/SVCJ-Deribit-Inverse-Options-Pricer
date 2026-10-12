# Remediation baseline

This file freezes the pre-remediation empirical state so that corrected reruns can be compared against the submitted thesis rather than against a moving live-data directory.

## Repository baseline

- Remediation branch: `paper-code-remediation`
- Branch point / pre-remediation code commit: `d93dbcb1b0a801af5487abe31e136d3d3b06c613`
- Baseline thesis PDF: `docs/Pricing and Hedging Deribit Inverse Options under Affine Stochastic Volatility Models with Jumps.pdf`
- Baseline thesis PDF Git blob: `501089ccb930a78c1e21abbb15dfa497947ff9eb`
- Original uploaded LaTeX source SHA-256: `72a10d9253eef54779caf3c53c29fac7267a647f84d39ab88a37ca91bb4186de`
- Calibration/hedging workbooks are intentionally stored outside Git; `excel files/link.txt` points to the shared Drive folder used by the thesis.

## Frozen thesis sample

The submitted thesis states that the option panel covers **30 December 2025 through 8 April 2026** and contains **388 snapshots**, collected roughly four times per day. The local snapshot-index file pins the exact observed endpoints used by the repository:

- First snapshot file: `deribit_options_snapshot_20251230T173115Z.csv`
- Frozen analysis start: `2025-12-30T17:31:15Z`
- Last snapshot file: `deribit_options_snapshot_20260408T100119Z.csv`
- Frozen analysis end: `2026-04-08T10:01:19Z`
- Expected unique option-surface snapshots: `388`
- Expected snapshot/currency rows when both BTC and ETH are present: `776`

For remediation runs, pass these exact inclusive bounds through `BatchConfig.analysis_start` and `BatchConfig.analysis_end`. This prevents snapshots collected after the thesis endpoint from silently entering a rerun.

### Immutable data anchors at the branch point

| Input | Git object |
|---|---|
| `data/` tree | `7c1e2c9bb692a82e66ff9f8b17d8f298b592b729` |
| `data/snapshot_currency_combos.csv` | `8285645206dbd0cccc19cc17a31f6b817b3c1d68` |
| `data/perpetual_futures_prices.csv` | `2e1efb874574cb9fd9d52c94e1bb475733c8845c` |
| `data/term_futures_marks.csv` | `082940cd35c643a3d71b8fcf8a6d922ea137bedc` |

`docs/reproducibility/thesis_sample_manifest.csv` is a compact immutable manifest. Exact snapshot membership is defined by the unique `snapshot_file` values in `data/snapshot_currency_combos.csv` at the Git blob above, restricted to the inclusive start/end bounds. This avoids duplicating hundreds of filenames while still fixing the exact bytes that define membership.

## Submitted-thesis benchmark specification

The submitted Chapter 3 baseline uses temporal regularization `lambda_TS = 100`. These values are recorded only as **pre-remediation comparison targets**; they are not presumed correct after the code/methodology fixes.

### Pricing: Table 3.3.1

| Asset | Model | RMSE | MAE | Within spread |
|---|---|---:|---:|---:|
| BTC | Black | 0.005330 | 0.003565 | 25.7% |
| BTC | Heston | 0.002892 | 0.001482 | 45.2% |
| BTC | SVCJ | 0.002546 | 0.001004 | 61.9% |
| ETH | Black | 0.006151 | 0.003929 | 32.1% |
| ETH | Heston | 0.004485 | 0.002112 | 49.1% |
| ETH | SVCJ | 0.004227 | 0.001626 | 63.1% |
| Pooled (as reported) | Black | 0.005592 | 0.003772 | 28.7% |
| Pooled (as reported) | Heston | 0.003247 | 0.001771 | 47.0% |
| Pooled (as reported) | SVCJ | 0.002911 | 0.001304 | 62.4% |

The pooled coin-denominated error rows are retained here only for old/new reconciliation. The remediation plan explicitly replaces economically ambiguous cross-asset pooled coin errors with common-unit/dimensionless metrics.

### Hedging: Table 4.3.1 RMSE reductions

| Currency | Model | Regular delta | Net delta |
|---|---|---:|---:|
| BTC | Black | 52.8% | 65.2% |
| BTC | Heston | 52.3% | 64.9% |
| BTC | SVCJ | 52.2% | 64.8% |
| ETH | Black | 48.1% | 62.6% |
| ETH | Heston | 47.9% | 62.5% |
| ETH | SVCJ | 47.9% | 62.5% |

## Reproduction rule

Do not update the benchmark numbers in this file. Corrected results belong in `docs/reproducibility/remediation_results.md` and must be compared against this frozen baseline. Live collection can continue on `main`; research reruns on the remediation branch must use the frozen analysis window explicitly.
