# Paper and Code Remediation Design

**Date:** 2026-10-13  
**Branch:** `paper-code-remediation`  
**Baseline commit:** `d93dbcb1b0a801af5487abe31e136d3d3b06c613`

## Purpose

Bring the thesis manuscript and the repository into exact theoretical, empirical, and computational agreement before any journal-paper extraction. The remediation is complete only when the equations in the manuscript describe the code that generated the reported results, the evaluation design is free of train/test leakage, the sample is reproducible, and all headline tables/figures can be regenerated from tested code.

## Accepted issue baseline

This design treats the preceding paper/code audit as the accepted scope. The primary issues are:

1. The manuscript does not document the cross-sectional holdout construction.
2. The manuscript conflates the cash/forward pricing measure under which the futures price is a martingale with the measure associated with using the futures price as numeraire; Appendix D consequently mislabels the shifted characteristic-function probability.
3. The manuscript hedge-notional formula does not match the implementation, and the current code does not use an identical term/perpetual scaling for the regular-delta and net-delta benchmark rules.
4. Temporal regularization is implemented in transformed parameter space but written as raw-parameter regularization in the manuscript.
5. Per-currency worker chunking breaks the sequential previous-snapshot regularization chain at chunk boundaries.
6. The regularization strength is selected using the same test sample later reported as out-of-sample evidence.
7. Reported aggregation conventions and cross-asset raw coin metrics need to be made explicit and economically coherent.
8. The baseline residual scale is inverse spread, so the squared objective is inverse-spread-squared; the manuscript should say this exactly.
9. `rho_j` is a conditional jump-size loading coefficient in the implemented SVCJ specification, not a bounded correlation coefficient; the current `tanh` transform can create artificial boundary behavior.
10. SVCJ calibration checks the first-moment jump condition but does not explicitly enforce Carr-Madan contour admissibility for `alpha > 0`.
11. The Feller condition is softly penalized, not hard-imposed.
12. The perpetual-futures history is a candle-based price proxy in the current backfill path, despite `mark_price` naming.
13. The manuscript says stale quotes are removed even though no quote-age filter is implemented.
15. Diebold-Mariano-style inference should be described as inference on snapshot-level cross-sectional holdout-loss differentials, not conventional time-series forecast evaluation.
16. Structural interpretation of weakly identified ETH parameters should be softened.
17. The literature review and presentation should be strengthened for a research-paper-quality version.
18. Existing correct choices should be preserved: the manuscript's moneyness buckets are raw `K/F` buckets rather than incorrect pooled ITM/OTM labels, and the hedging chapter already states that transaction costs/funding are excluded from the clean hedging diagnostic.

## Design decisions

### 1. Reproducibility before model changes

Freeze the thesis analysis window and create a manifest of all inputs used for the remediation run. The live collector may continue adding files to `data/`, but calibration/hedging jobs used for the paper must accept an explicit start/end window and must not silently ingest later snapshots.

Old workbooks and the current thesis PDF are baseline artifacts. Corrected outputs are generated into remediation-specific paths until all checks pass; only then are final canonical outputs replaced.

### 2. Separate pricing measure from numeraire change

Use explicit notation throughout:

- `Q`: the pricing measure used by `cf_black`, `cf_heston`, and `cf_svcj`, under which the dated futures price `F_t` is a martingale.
- `Q^F`: the measure induced by the futures-price numeraire when performing the inverse-option numeraire change.

The regular call price is computed under `Q`. The regular call delta is the shifted-characteristic-function quantity

`P1 = Q^F(F_T > K)`,

with characteristic function `phi_Q(u-i) / phi_Q(-i)`. Under Black this must reduce exactly to `N(d1)`, while `Q(F_T>K)` reduces to `N(d2)`.

### 3. Centralize Greeks and hedge mapping

Move regular-delta/net-delta identities and term-futures-to-perpetual hedge-notional mapping into small tested modules rather than leaving the formulas embedded in `src/hedging_analysis.py`.

For an inverse option whose coin sensitivity is approximated by `Delta_h / F_term`, and an inverse perpetual with local coin P&L `N_USD * dH / H^2`, the same basis mapping must be used for both hedge rules. Under the baseline local assumption `dF_term = dH`, the notional is

`N_USD = (H^2 / F_term) * Delta_h`

for a short one-option position, with `Delta_h` equal to either regular delta (benchmark) or net delta (contract-consistent rule). Thus the benchmark comparison differs only in the delta definition, not in an additional basis-scaling approximation.

### 4. Keep transformed-space regularization, but make it exact and sequential

The current optimizer regularizes the transformed optimization vector. Keep that approach and document it explicitly because it is numerically sensible. When `l2_prev_strength > 0`, each currency must be processed strictly in chronological order so observation `t` is anchored to the successful estimate from `t-1` (or the latest prior successful estimate under resume). Parallelism may be used across independent regularization sweeps or currencies outside the sequential per-currency chain, but not across consecutive snapshots within the same chain.

### 5. Replace train/test with train/validation/test

Use a deterministic within-snapshot cross-sectional split with default fractions 60/20/20. Persist all three splits. Calibration fits only the train rows. Regularization strength and other tuning decisions use validation rows only. Once the regularization value is locked and recorded, final pricing and hedging claims use the untouched test rows.

The manuscript will call this a **within-snapshot cross-sectional holdout**, not a time-series forecasting exercise.

### 6. Preserve the current spread-standardized baseline objective

Keep the baseline residual

`r_i = (P_model_i - P_market_i) / (spread_i + eps)`.

Therefore the least-squares objective uses inverse-spread-squared contributions. Rename/document this as spread-standardized residual scaling rather than ambiguous WLS language. Vega/open-interest scaling remains optional robustness functionality.

### 7. Correct the SVCJ jump-loading domain and FFT admissibility

Treat `rho_j` as an unbounded conditional jump-size loading coefficient in the implemented specification, not a correlation constrained by `tanh`. Remove the `tanh` transform for this coordinate and retain explicit moment/admissibility checks.

At minimum, candidate parameters must satisfy the martingale first-moment condition and the positive exponential-moment condition required by the actual Carr-Madan contour. The pricing code must reject a candidate that passes `1 - ell_v*rho_j > 0` but fails the `alpha+1` contour requirement. Characteristic-function evaluation must also reject any quadrature point for which the joint-jump transform leaves its domain.

### 8. Separate asset-level raw metrics from cross-asset pooled metrics

BTC and ETH raw coin RMSE/MAE remain valid asset-specific diagnostics. Do not report a raw BTC+ETH pooled coin RMSE/MAE as if the coin units were commensurable. Cross-asset pooled reporting should use common-unit or dimensionless metrics such as USD-converted error, spread-standardized error, and within-spread share.

The current `notebooks/_lib/chapter3_final.py` already computes an all-row pooled RMSE mechanically correctly; the remediation must still remove/rename the economically misleading pooled raw-coin metric and regenerate the manuscript table from one explicit aggregation convention.

### 9. Make data provenance truthful

The candle-backed perpetual field will be described and named as a perpetual price proxy unless true historical Deribit mark prices are sourced. Loader compatibility with existing legacy `perp_futures_mark_price` files may be retained, but new outputs must expose the proxy/source semantics explicitly.

The manuscript will stop claiming that a quote-age stale-quote filter exists unless one is actually implemented.

### 10. Code first, manuscript second, final rerun last

The editable manuscript source is not currently present on the branch; `docs/` contains the PDF and code documentation only. Before manuscript edits begin, add the actual LaTeX source and bibliography to the repository. Do not manually change headline numbers in prose/tables before corrected workbooks and exported tables exist.

## Completion criteria

The remediation is complete when:

- the full unit/regression suite passes;
- closed-form Black identities validate the measure/delta implementation;
- regular and net hedge rules differ only in the chosen delta under the same basis mapping;
- regularized calibration is deterministic with respect to worker count and resume boundaries;
- train/validation/test splits are deterministic and persisted;
- regularization is chosen without reading test metrics;
- SVCJ rejects Carr-Madan-inadmissible parameter vectors and no longer artificially bounds `rho_j` to `[-1,1]`;
- all final calibration/hedging workbooks are regenerated from the frozen sample;
- every number and equation in the manuscript can be traced to current code/output;
- the final PDF compiles from committed editable source and contains the corrected theory, empirical design, data provenance, inference wording, limitations, and figures.