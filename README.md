# Deribit Inverse Options Pricing & Calibration (FFT)

A research codebase (thesis project) for **pricing and calibrating European *inverse* options** (coin-settled options) using the **Carr–Madan (1999) FFT** approach.

The project contains:

- An FFT pricer for inverse calls/puts under **Black-76**, **Heston**, and **SVCJ (stochastic volatility with correlated jumps)**.
- A lightweight data collector that pulls **Deribit** option surface snapshots into CSV.
- Calibration utilities to clean the snapshot data and fit model parameters via **weighted least squares**.
- Example notebooks demonstrating pricing and calibration workflows.

> **Inverse option pricing note:** prices are produced in **coin units** (e.g., BTC) by converting USD-denominated FFT call prices using the current futures price and put–call parity.

---

## Repository layout

```
thesis/
  src/
    inverse_fft_pricer.py        # Carr–Madan FFT engine + model characteristic functions
    calibration.py               # Cleaning + weighted-LS calibration (Black/Heston/SVCJ)
    collect_deribit_snapshot.py  # CLI to fetch Deribit option snapshots to CSV
  data/
    *.csv                        # Example snapshot(s)
  docs/
    *.pdf                        # Thesis / background documentation
  pricing_examples.ipynb
  calibration_example.ipynb
```

---

## Installation

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Make the `src/` modules importable

From the `thesis/` directory, either:

```bash
export PYTHONPATH="$PWD/src"
```

or (in notebooks) add `thesis/src` to `sys.path`.

---

## Quickstart

### Price an inverse option (FFT)

```python
from src.inverse_fft_pricer import price_inverse_option

# Example inputs (coin = BTC style)
price_coin = price_inverse_option(
    option_type="call",   # "call" or "put"
    K=40000.0,            # strike
    T=30/365,             # maturity in years
    F0=39500.0,           # current futures price
    r=0.0,                # rate (often ~0 for crypto short maturities)
    model="heston",       # "black", "heston", or "svcj"
    params={              # model parameters depend on chosen model
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.6,
        "rho": -0.5,
        "v0": 0.04,
    },
)
print(price_coin)
```

For additional worked examples, see:

- `pricing_examples.ipynb`

---

## Collect a Deribit snapshot (CSV)

The collector hits Deribit public endpoints and writes a timestamped CSV to `data/`:

```bash
python src/collect_deribit_snapshot.py --currency BTC --outdir data
# or both BTC + ETH:
python src/collect_deribit_snapshot.py --currency BOTH --outdir data
```

Helpful flags:

- `--max-instruments N` for quick tests (0 = no cap)
- `--sleep SECONDS` to be extra rate-limit friendly

---

## Calibrate a model to snapshot data

Calibration utilities live in `src/calibration.py` and are designed to work with the FFT pricer.

High-level workflow:

1. Load a snapshot CSV as a `pandas.DataFrame`
2. Filter to liquid options (`filter_liquid_options`)
3. Choose a model (`black`, `heston`, `svcj`)
4. Fit parameters via weighted least squares (`calibrate_model`)

See the full walkthrough in:

- `calibration_example.ipynb`

---

## Reproducibility notes

- Results depend on the **exact snapshot timestamp** (market data) and on the chosen filtering/weighting scheme.
- The included CSV in `data/` is an example snapshot; for fresh results, run the snapshot collector again.

---

## Citation

If you use this code in academic work, please cite the accompanying thesis / documentation in `docs/`.

---

## License

Add a license that matches your intended distribution (e.g., MIT, BSD-3, or an academic/research license).
