# Carr–Madan FFT numerical validation

This note records the Stage 4 numerical check of the canonical pricing grid. It is a numerical implementation check, not a calibration exercise and not a historical-data tuning exercise.

## Canonical configuration

The maintained pricing workflow uses:

- `N = 4096`
- `eta = 0.10`
- `alpha = 1.5`
- `use_simpson = True`
- the log-strike grid is centred on the strikes being priced.

## Independent benchmarks

Black prices are compared with an independent Black–76 implementation using SciPy's normal CDF. The deterministic benchmark includes short, medium and long maturities and ATM/ITM/OTM strikes.

For `F0 = 10,000`, `sigma = 0.50`:

| maturity | strike | Black–76 USD | canonical FFT USD | absolute error USD |
|---:|---:|---:|---:|---:|
| 0.01 | 10,000 | 199.450364 | 199.450364 | < 1e-8 |
| 0.25 | 8,000 | 2,226.559013 | 2,226.836727 | 0.277714 |
| 0.25 | 10,000 | 994.764497 | 994.764497 | < 1e-8 |
| 0.25 | 12,000 | 370.588309 | 370.748685 | 0.160376 |
| 1.00 | 10,000 | 1,974.126514 | 1,974.126514 | < 1e-8 |

The nonzero ITM/OTM errors are dominated by interpolation from the FFT log-strike grid; ATM points chosen at the grid centre agree to numerical precision.

The representative Heston benchmark uses

`kappa=2.0, theta=0.04, sigma_v=0.30, rho=-0.50, v0=0.04`, `T=0.50`, `F0=10,000`.

The reference is the project's independent Gil–Pelaez semi-analytical integration. Tightening its Gauss–Legendre order from 256 to 512 changes the benchmark by less than `1e-6` USD on the test strikes.

| strike | semi-analytic USD | canonical FFT USD | absolute error USD | relative error |
|---:|---:|---:|---:|---:|
| 8,000 | 2,053.577764 | 2,053.757113 | 0.179349 | 0.0087% |
| 10,000 | 546.617411 | 546.617411 | < 1e-8 | < 1e-8% |
| 12,000 | 42.544542 | 42.718526 | 0.173984 | 0.4089% |

## Nearby-grid and quadrature diagnostic

The same Heston case was checked at nearby grids. Representative maximum absolute errors were approximately:

| configuration | max absolute error USD |
|---|---:|
| `N=2048, eta=0.10`, current Simpson weights | 0.549 |
| `N=4096, eta=0.10`, current Simpson weights | 0.179 |
| `N=8192, eta=0.10`, current Simpson weights | 0.073 |
| `N=4096, eta=0.05`, current Simpson weights | 0.549 |
| `N=4096, eta=0.20`, current Simpson weights | 0.073 |

At `N=4096, eta=0.10`, replacing the current endpoint convention by the common Carr–Madan half-line Simpson sequence, or by trapezoidal weights, changes these benchmark prices only at numerical-noise scale. The final high-frequency contribution is negligible for the tested admissible characteristic functions.

The convergence pattern follows the log-strike spacing `2*pi/(N*eta)`, which is consistent with strike-grid interpolation being the visible residual error.

## Stage 4 decision

No Carr–Madan production code change is warranted. The canonical configuration is accurate on independent Black and Heston benchmarks, the errors decrease as the strike grid is refined, and the questioned final Simpson weight does not produce a measurable pricing bias at canonical settings.

Regression coverage is provided by `tests/test_fft_numerical_validation.py`. The FFT implementation should only be changed in the future if an independent benchmark exposes a material numerical error outside these tolerances.
