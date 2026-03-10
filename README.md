# Deribit Inverse Options Pricer (Black, Heston, SVCJ)

This repository contains a full pricing and calibration workflow for **Deribit inverse options** (coin-settled options on crypto futures) under three models:

- **Black (Black-76)**
- **Heston**
- **SVCJ**

The core implementation combines a Carr-Madan FFT engine, data filtering for liquid quotes, weighted least-squares calibration, and Excel-based result persistence for multi-snapshot batch runs.

## What the pipeline does

For each `deribit_options_snapshot_*.csv` file, the code:

1. Loads and validates option snapshot data.
2. Filters to liquid quotes (`bid/ask`, spread, maturity, OI, vega, moneyness).
3. Builds per-expiry FFT pricing plans (one call-grid per expiry bucket).
4. Splits filtered data into deterministic train/test subsets.
5. Calibrates Black, Heston, and SVCJ on the train split (optionally warm-started).
6. Reprices the full filtered dataset with calibrated parameters.
7. Computes train/test metrics and appends everything to an Excel workbook.

## Repository layout

```text
.
├── src/
│   ├── inverse_fft_pricer.py       # Characteristic functions + Carr-Madan FFT pricer
│   ├── calibration.py              # Filtering, weighting, vectorized pricing, calibration
│   ├── snapshot_job.py             # Process one snapshot into an Excel-ready payload
│   ├── batch_runner.py             # Multi-snapshot / multi-currency batch orchestration
│   ├── results_store.py            # Workbook schema, append helpers, atomic flush
│   ├── collect_deribit_snapshot.py # Deribit API snapshot collector
│   └── __init__.py
├── docs/
│   ├── code_documentation.md       # Detailed module/function documentation
│   └── MASTERS_DIPLOM.pdf          # Theory and derivations
├── pricing_examples.ipynb
├── calibration_example.ipynb
├── calibrate_all_to_excel.ipynb
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data collection

Collect fresh Deribit snapshots (options + perpetual futures):

```bash
python -m src.collect_deribit_snapshot --currency BOTH --outdir data
```

Useful flags:

- `--currency BTC|ETH|BOTH`
- `--max-instruments <n>` for quick test snapshots
- `--sleep <seconds>` between ticker requests

## Running calibrations

### Single snapshot (programmatic)

Use `process_snapshot_to_payload(...)` from `src.snapshot_job` when you want one-file processing and full control over config objects.

### Batch run to Excel (programmatic)

Use `BatchConfig` + `run_all_snapshots_to_excel(...)` from `src.batch_runner`.
The typical pattern is demonstrated in `calibrate_all_to_excel.ipynb`.

## Output workbook schema

`calibration_results.xlsx` contains five sheets:

- `black_params`
- `heston_params`
- `svcj_params`
- `train_data`
- `test_data`

The batch runner is resume-safe: it checks the latest processed timestamp per currency and continues from there when `resume=True`.

## Pricing conventions

- Inputs are futures/forward prices `F0` and strikes `K` in USD per coin.
- Model call prices are produced in USD on FFT grids, then converted to coin units.
- Put prices are derived from inverse put-call parity in coin terms.

## Notes

- The implementation supports warm starts from previously calibrated parameters.
- Soft penalties are applied for Heston Feller violations and SVCJ moment-stability violations.
- FFT grids are cached and can be cleared via `clear_fft_cache()`.

For deeper implementation details, see `docs/code_documentation.md`.
