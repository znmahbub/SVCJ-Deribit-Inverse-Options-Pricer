# Paper and Code Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the theoretical, empirical, computational, and manuscript inconsistencies identified in the thesis audit, then regenerate the thesis results from a frozen reproducible sample before any journal-paper extraction.

**Architecture:** The remediation proceeds from foundations outward: freeze the data/sample, centralize and test pricing-measure/Greek identities, centralize and test hedge mapping, correct SVCJ parameter/admissibility logic, make temporal regularization strictly sequential, replace the two-way split with train/validation/test, then rebuild reporting, rerun calibration/hedging, and only then rewrite and recompile the manuscript. Generated workbooks/tables remain downstream artifacts; no headline number is edited manually.

**Tech Stack:** Python 3, NumPy, pandas, SciPy `least_squares`, OpenPyXL, Plotly/Jupyter reporting helpers already in the repo, `unittest` regression suite, LaTeX for the manuscript.

**Spec:** `docs/superpowers/specs/2026-10-13-paper-code-remediation-design.md`

## Global Constraints

- Work only on branch `paper-code-remediation`, created from baseline commit `d93dbcb1b0a801af5487abe31e136d3d3b06c613`.
- Preserve the current thesis PDF and existing calibration/hedging workbooks as baseline artifacts until corrected outputs pass all validation gates.
- Freeze the thesis sample window; live snapshots added after the thesis endpoint must never enter a remediation run implicitly.
- Use one deterministic split assignment per snapshot and persist the assignment; all models must see exactly the same train/validation/test rows.
- Calibration uses train rows only. Hyperparameter/model-selection decisions use validation rows only. Test rows are read only after the chosen specification is locked.
- For `l2_prev_strength > 0`, per-currency snapshot processing must be chronological and sequential; no worker boundary may reset the regularization anchor.
- Keep transformed-space temporal regularization and document its transform explicitly.
- Keep the current baseline spread-standardized residual definition unless a separately documented robustness experiment changes it.
- Do not constrain SVCJ `rho_j` with `tanh`; treat it as the conditional jump-loading coefficient in the implemented specification.
- Reject SVCJ parameter vectors that fail the exponential-moment domain required by the actual Carr-Madan contour.
- Use the same term-futures/perpetual mapping for regular-delta and net-delta hedge rules; the comparison must differ only by the chosen delta.
- Do not report a raw BTC+ETH pooled coin RMSE/MAE as a common-unit statistic.
- Retain the current clean-hedging interpretation: transaction costs and funding are not silently introduced into the baseline P&L experiment.
- Retain raw `K/F` moneyness buckets; do not replace them with pooled call/put ITM/OTM labels.
- All manuscript numbers must come from exported tables produced by current code, not hand calculations.
- The editable manuscript source must be committed before manuscript changes begin; the current branch contains only the PDF, not the LaTeX source.

## Review Focus

1. **Resume and worker-boundary dependence:** a regularized run resumed halfway through the sample must produce the same parameter path as an uninterrupted run, within optimizer tolerance. Task 7 owns this test.
2. **Small expiry/type strata in the three-way split:** strata with 1–4 rows must still receive a deterministic valid assignment without emptying the overall train set. Task 8 owns this test.
3. **SVCJ positive jump-loading cases:** parameters can satisfy `1 - ell_v*rho_j > 0` but fail the `alpha+1` Carr-Madan moment requirement; those candidates must be rejected before producing misleading prices. Task 5 owns this test.
4. **Nonzero term/perpetual basis:** when `H != F_term`, regular-delta and net-delta benchmark notionals must use the same scaling formula so the hedge-rule comparison is not confounded by basis scaling. Task 4 owns this test.
5. **Cross-asset metric units:** BTC/ETH coin errors may be reported separately but a pooled table must use USD/common-unit or dimensionless errors; no helper may silently average coin units. Task 10 owns this test.

---

## Issue-to-task map

| Audit issue | Owning task(s) |
|---|---|
| Missing train/test description and ambiguous “out-of-sample” wording | 8, 13 |
| Pricing-measure / futures-numeraire inconsistency | 3, 13 |
| Reversed/mislabeled measure change in Appendix D | 3, 13 |
| Paper hedge notional differs from code | 4, 13 |
| Regular and net hedge rules use inconsistent basis scaling | 4, 11 |
| Paper writes raw-parameter L2 but code regularizes transformed vector | 6, 13 |
| Worker chunks break temporal chain | 7 |
| `lambda_TS` chosen on test data | 8, 9 |
| Pooled metric inconsistency / pooled coin units | 10, 13 |
| Spread weighting language is ambiguous | 6, 13 |
| `rho_j` incorrectly bounded/interpreted as correlation | 5, 12, 13 |
| Carr-Madan moment admissibility too weak | 5 |
| Feller described as imposed instead of penalized | 6, 13 |
| Perpetual “mark” is a candle proxy | 11, 13 |
| “Stale quote” claim unsupported | 13 |
| DM terminology overstates forecast interpretation | 9, 13 |
| Weakly identified ETH parameters over-interpreted | 12, 13 |
| Literature review too narrow | 14 |
| Figure/appendix presentation cleanup | 14 |
| Preserve valid moneyness-bucket and clean-P&L choices | 13, 14 |

---

### Task 1: Freeze the remediation baseline and thesis sample

**Files:**
- Create: `docs/reproducibility/remediation_baseline.md`
- Create: `docs/reproducibility/thesis_sample_manifest.csv`
- Create: `tests/test_analysis_window.py`
- Modify: `src/batch_runner.py`
- Modify: `src/hedging_analysis.py`
- Modify: `README.md`
- Modify: `docs/code_documentation.md`

**Interfaces:**
- Consumes: current snapshot filenames and thesis sample dates from the baseline workbooks/PDF.
- Produces: `BatchConfig.analysis_start: pd.Timestamp | None`, `BatchConfig.analysis_end: pd.Timestamp | None`, matching fields on `HedgingConfig`, and deterministic filtering of snapshot files to the frozen window.

- [ ] **Step 1: Record the baseline provenance.** In `docs/reproducibility/remediation_baseline.md`, record the branch base SHA, current PDF filename/date, current regularization workbooks used by the thesis, current headline pricing/hedging numbers, and the exact thesis sample start/end recovered from the 388-snapshot workbooks.

- [ ] **Step 2: Write failing tests for analysis-window filtering.** Add tests asserting that files before `analysis_start` and after `analysis_end` are excluded, boundary timestamps are included, and `None` preserves current behavior.

- [ ] **Step 3: Run the targeted test and verify failure.**

```bash
python -m unittest tests.test_analysis_window -v
```

Expected: FAIL because the config/window filter does not yet exist.

- [ ] **Step 4: Add explicit analysis-window fields to `BatchConfig` and `HedgingConfig`.** Normalize values to UTC once and apply them in snapshot discovery before resume/smoke-test truncation.

- [ ] **Step 5: Generate `thesis_sample_manifest.csv`.** Include at least `snapshot_filename`, `snapshot_ts`, `currency_available`, and a stable file hash for every snapshot used in the thesis analysis window. Include the perpetual-history and term-futures-history input filenames/hashes in the baseline markdown.

- [ ] **Step 6: Run tests.**

```bash
python -m unittest tests.test_analysis_window -v
python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 7: Update README/code documentation.** State explicitly that the live data directory can contain newer snapshots and paper runs must set the frozen analysis window.

- [ ] **Step 8: Commit.**

```bash
git add src/batch_runner.py src/hedging_analysis.py tests/test_analysis_window.py docs/reproducibility README.md docs/code_documentation.md
git commit -m "fix: freeze reproducible thesis analysis window"
```

---

### Task 2: Add the editable manuscript source to version control

**Files:**
- Create: `docs/manuscript/main.tex`
- Create: `docs/manuscript/references.bib` or the exact bibliography file used by the thesis
- Create: any required `docs/manuscript/*.sty`, figure, or subfile inputs
- Modify: `README.md`

**Interfaces:**
- Consumes: the exact LaTeX/Overleaf source that produced the current 89-page PDF.
- Produces: a reproducible local manuscript build with no content edits yet.

- [ ] **Step 1: Obtain the exact editable source.** Do not reconstruct LaTeX from the PDF. If the source is not available in the repo, the implementation pauses here until the user supplies/exports it.

- [ ] **Step 2: Add the source under `docs/manuscript/` without changing wording or equations.** Preserve bibliography keys and figure filenames.

- [ ] **Step 3: Compile the unmodified source.** Use the project’s actual engine (`latexmk`, `pdflatex`/`biber`, etc.) and record the command in README.

- [ ] **Step 4: Compare the compiled PDF to the committed baseline.** Page count and substantive content must match; small metadata/font-layout differences are acceptable only if documented.

- [ ] **Step 5: Commit.**

```bash
git add docs/manuscript README.md
git commit -m "docs: add reproducible thesis source"
```

---

### Task 3: Correct pricing-measure, numeraire-change, and regular-delta identities

**Files:**
- Create: `src/greeks.py`
- Create: `tests/test_measure_delta_identities.py`
- Modify: `src/inverse_fft_pricer.py`
- Modify: `src/hedging_analysis.py`
- Modify: `docs/code_documentation.md`

**Interfaces:**
- Produces:
  - `black_regular_delta(F, K, T, sigma, option_type) -> np.ndarray`
  - `call_p1_from_logprice_cf(K, F0, cf, *, quad_u_max, quad_n) -> np.ndarray`
  - `regular_delta_from_call_p1(p1, option_type) -> np.ndarray`
  - `net_delta(regular_delta, inverse_price) -> np.ndarray`
- `cf_black`, `cf_heston`, `cf_svcj` remain characteristic functions under the measure `Q` for which `F_t` is a martingale.

- [ ] **Step 1: Write the Black closed-form identity tests.** Assert `phi_Q(-i)=F0`, direct ITM probability equals `N(d2)`, shifted-CF `P1` equals `N(d1)`, call regular delta equals `N(d1)`, put regular delta equals `N(d1)-1`, and `net_delta = regular_delta - price_coin`.

- [ ] **Step 2: Add Heston/SVCJ martingale and shifted-CF sanity tests.** Assert `phi(-i) ≈ F0` for stable reference parameters and finite `P1` values in `[0,1]` for calls.

- [ ] **Step 3: Run tests to confirm the new module is missing.**

```bash
python -m unittest tests.test_measure_delta_identities -v
```

Expected: FAIL on import/functions.

- [ ] **Step 4: Move the current Black and semi-analytic `P1` delta logic from `src/hedging_analysis.py` into `src/greeks.py`.** The shifted characteristic function must use `phi_Q(u-i)/phi_Q(-i)`; do not relabel `phi_Q` as the `Q^F` characteristic function.

- [ ] **Step 5: Update pricer docstrings/comments.** Replace ambiguous statements such as “under the futures measure/futures price numeraire” when they refer to the measure under which `F` is a martingale. Use `Q` for that pricing measure and reserve `Q^F` for the numeraire-changed measure.

- [ ] **Step 6: Make `hedging_analysis.compute_prices_and_deltas(...)` import the centralized Greek functions.** Remove duplicate private implementations after equivalence tests pass.

- [ ] **Step 7: Run targeted and full tests.**

```bash
python -m unittest tests.test_measure_delta_identities tests.test_sanity_pricing tests.test_hedging_invariants -v
python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 8: Commit.**

```bash
git add src/greeks.py src/inverse_fft_pricer.py src/hedging_analysis.py tests/test_measure_delta_identities.py docs/code_documentation.md
git commit -m "fix: separate pricing measure and numeraire delta logic"
```

---

### Task 4: Make the perpetual hedge mapping theoretically and computationally identical across hedge rules

**Files:**
- Create: `src/hedge_mapping.py`
- Create: `tests/test_hedge_mapping.py`
- Modify: `src/hedging_analysis.py`
- Modify: `tests/test_hedging_invariants.py`
- Modify: `docs/code_documentation.md`

**Interfaces:**
- Produces:
  - `inverse_perp_pnl(notional_usd, H_t, H_t1) -> np.ndarray`
  - `hedge_notional_usd(delta_used, F_term, H_perp, *, option_position=-1.0, dF_dH=1.0) -> np.ndarray`
- Baseline short-one-option mapping with `dF/dH = 1` must reduce to `N_USD = H_perp**2 / F_term * delta_used` for both regular and net delta.

- [ ] **Step 1: Write failing unit tests.** Test `H=F` reduction to `H*delta`, nonzero basis `H!=F`, sign behavior for long/short option positions, and exact equality of scaling between regular/net rules when fed the same synthetic delta.

- [ ] **Step 2: Add a regression test exposing the current confound.** Construct `F_term != H_perp` and assert that the two hedge-rule formulas differ only because `delta_used` differs; current code should fail this because regular uses `H*delta` while net uses `H^2/F*delta`.

- [ ] **Step 3: Implement `src/hedge_mapping.py`.** Keep the local basis assumption explicit through `dF_dH`; default to 1.0 to reproduce the intended baseline.

- [ ] **Step 4: Refactor `build_hedge_interval_panel(...)`.** Use the centralized notional function for both hedge rules and centralized inverse-perpetual P&L.

- [ ] **Step 5: Add output columns recording `F_term`, `H_perp`, `basis_ratio`, and the notional scale used.** These make paper diagnostics auditable.

- [ ] **Step 6: Run tests.**

```bash
python -m unittest tests.test_hedge_mapping tests.test_hedging_invariants -v
python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add src/hedge_mapping.py src/hedging_analysis.py tests/test_hedge_mapping.py tests/test_hedging_invariants.py docs/code_documentation.md
git commit -m "fix: unify inverse perpetual hedge notional mapping"
```

---

### Task 5: Correct SVCJ jump-loading parameterization and Carr-Madan admissibility

**Files:**
- Create: `tests/test_svcj_admissibility.py`
- Modify: `src/calibration.py`
- Modify: `src/inverse_fft_pricer.py`
- Modify: `tests/reference_inputs.py`
- Modify: `tests/test_sanity_pricing.py`
- Modify: `docs/code_documentation.md`

**Interfaces:**
- Produces:
  - direct, unbounded optimization coordinate for `rho_j` rather than `atanh/tanh` mapping;
  - `svcj_martingale_moment_ok(ell_v, rho_j, eps) -> bool`;
  - `svcj_carr_madan_admissible(..., alpha, T, ...) -> bool` or an equivalent internal validator used by calibration/pricing.

- [ ] **Step 1: Write transform round-trip tests.** Assert SVCJ pack/unpack preserves values with `rho_j=-2.0`, `rho_j=0`, and `rho_j=+1.25`; current `tanh` mapping must fail these cases.

- [ ] **Step 2: Write the contour counterexample test.** Use parameters satisfying `1 - ell_v*rho_j > 0` but violating the positive exponential moment at `alpha+1` (for example a positive `rho_j`/`ell_v` combination where `(alpha+1)*ell_v*rho_j >= 1` at `s=0`). Assert that calibration/pricing rejects the candidate before returning a finite Carr-Madan price.

- [ ] **Step 3: Write a stable negative-loading case.** Assert that materially negative `rho_j` values are not clipped to `-1` and still produce finite characteristic-function values for the reference case.

- [ ] **Step 4: Replace the `rho_j` transform.** `_pack_svcj` stores it directly; `_unpack_svcj` returns it directly. `_default_bounds('svcj')` uses infinite or sufficiently nonbinding numeric bounds for this coordinate rather than `[-atanh(.999...), +atanh(.999...)]`.

- [ ] **Step 5: Add explicit moment-domain checks.** Keep the martingale condition and add the Carr-Madan shifted-contour condition using the actual `FFTParams.alpha`. In `cf_svcj`, check the real domain of the jump transform across quadrature points, not only whether the complex denominator has small absolute value.

- [ ] **Step 6: Make calibration return a controlled penalty for inadmissible candidates.** Do not allow runtime warnings/NaNs to be the only rejection path.

- [ ] **Step 7: Run pricing/admissibility tests.**

```bash
python -m unittest tests.test_svcj_admissibility tests.test_sanity_pricing tests.test_regression_reference_outputs -v
```

Expected: pricing reference fixtures may fail because the parameter transform/domain has changed; regenerate fixtures only after manually checking the numerical differences are intended.

- [ ] **Step 8: Regenerate frozen pricing fixtures after review.**

```bash
python tests/generate_reference_outputs.py
python -m unittest tests.test_regression_reference_outputs -v
```

- [ ] **Step 9: Commit.**

```bash
git add src/calibration.py src/inverse_fft_pricer.py tests docs/code_documentation.md
git commit -m "fix: correct SVCJ jump loading and FFT admissibility"
```

---

### Task 6: Make the calibration objective and soft constraints explicit

**Files:**
- Create: `tests/test_calibration_objective.py`
- Modify: `src/calibration.py`
- Modify: `src/snapshot_job.py`
- Modify: `docs/code_documentation.md`

**Interfaces:**
- Existing `WeightConfig` remains public, but its documentation must distinguish residual scaling from objective weights.
- Calibration output metadata must record whether Feller and SVCJ moment penalties were active/violated.

- [ ] **Step 1: Write tests for baseline residual scaling.** For known spreads and price errors, assert `r_i = error_i/(spread_i+eps)` and therefore `sum(r_i^2)` is inverse-spread-squared.

- [ ] **Step 2: Write tests for transformed-space regularization.** With a fixed `previous_params`, assert the appended regularization residual equals `sqrt(lambda)*(x-x_prev)` in packed coordinates, not raw economic parameters.

- [ ] **Step 3: Write a Feller soft-penalty test.** Confirm a Feller-violating vector receives a finite positive penalty instead of being hard-rejected solely for the violation.

- [ ] **Step 4: Refactor private helpers only as needed to make these quantities testable.** Prefer small pure functions such as `_residual_scales(...)`, `_regularization_residual(...)`, and `_constraint_residuals(...)`; do not change the baseline objective numerically.

- [ ] **Step 5: Add diagnostics to `CalibrationResult`/parameter rows if needed.** Persist Feller gap, jump-moment margin, and whether regularization was active so the paper's parameter-stability discussion can use generated fields instead of recomputing ambiguous conditions.

- [ ] **Step 6: Run tests.**

```bash
python -m unittest tests.test_calibration_objective -v
python -m unittest discover -s tests -v
```

- [ ] **Step 7: Commit.**

```bash
git add src/calibration.py src/snapshot_job.py tests/test_calibration_objective.py docs/code_documentation.md
git commit -m "refactor: make calibration objective semantics explicit"
```

---

### Task 7: Restore a true sequential temporal-regularization chain

**Files:**
- Create: `tests/test_batch_regularization_sequence.py`
- Modify: `src/batch_runner.py`
- Modify: `docs/code_documentation.md`
- Modify: `notebooks/calibrate_all_to_excel.ipynb` only if its displayed parallelism claims/config need updating

**Interfaces:**
- `run_currency_to_excel(...)` must process one currency sequentially whenever `cfg.l2_prev_strength > 0`.
- `n_workers` may remain available for `l2_prev_strength == 0`; the function must not silently break the regularization chain.

- [ ] **Step 1: Write a deterministic fake-calibrator test.** Patch `process_snapshot_to_payload` so each output encodes its received `warm_start`; assert snapshot `t+1` receives `t`'s successful parameters even when config asks for multiple workers.

- [ ] **Step 2: Write a resume-equivalence test.** Run a small synthetic sequence uninterrupted and then as two runs with resume; assert identical warm-start chain and committed parameter rows.

- [ ] **Step 3: Run tests and confirm current chunked execution fails.**

```bash
python -m unittest tests.test_batch_regularization_sequence -v
```

- [ ] **Step 4: Implement the minimal correctness fix.** When `l2_prev_strength > 0`, force a single chronological worker for that currency (or a dedicated sequential loop). Do not attempt complex dependency-aware parallelism.

- [ ] **Step 5: Preserve parallel behavior for unregularized runs.** Existing ordered commit/resume tests must continue to pass.

- [ ] **Step 6: Run tests.**

```bash
python -m unittest tests.test_batch_regularization_sequence tests.test_results_store_io -v
python -m unittest discover -s tests -v
```

- [ ] **Step 7: Commit.**

```bash
git add src/batch_runner.py tests/test_batch_regularization_sequence.py docs/code_documentation.md notebooks/calibrate_all_to_excel.ipynb
git commit -m "fix: keep temporal regularization sequential"
```

---

### Task 8: Replace the two-way holdout with deterministic train/validation/test splits

**Files:**
- Create: `src/splitting.py`
- Create: `tests/test_snapshot_splitting.py`
- Modify: `src/snapshot_job.py`
- Modify: `src/results_store.py`
- Modify: `src/batch_runner.py`
- Modify: `src/hedging_analysis.py`
- Modify: `notebooks/_lib/chapter3_analysis.py`
- Modify: `notebooks/_lib/chapter3_final.py`
- Modify: `calibration_example.ipynb`
- Modify: `notebooks/calibrate_all_to_excel.ipynb`
- Modify: `docs/code_documentation.md`

**Interfaces:**
- Produces `SplitConfig(train_frac=0.60, validation_frac=0.20, test_frac=0.20, random_seed=...)`.
- Produces `assign_cross_sectional_split(df, cfg) -> pd.Series` with values `train`, `validation`, `test`.
- Adds canonical workbook sheet `validation_data` and keeps `train_data`/`test_data`.

- [ ] **Step 1: Write deterministic split tests.** Same snapshot+seed must produce identical assignments; different models must never resplit; total fractions must sum to one; every row appears exactly once.

- [ ] **Step 2: Add stratification/edge-case tests.** Use `(expiry_datetime, option_type)` strata where feasible, and deterministic fallback for tiny strata. Assert the overall train set is nonempty and no row is duplicated/dropped for strata of sizes 1–4.

- [ ] **Step 3: Implement `src/splitting.py`.** Keep split logic independent of pricing/calibration so it can be tested without FFT calls.

- [ ] **Step 4: Update `process_snapshot_to_payload(...)`.** Fit all models on train only; price the full filtered snapshot once per model; persist train, validation, and test DataFrames plus separate train/validation/test metrics in parameter rows.

- [ ] **Step 5: Extend workbook schema.** Add `VALIDATION_SHEET`, update initialization/load/append/flush semantics, and retain tolerant loading of legacy workbooks for comparison only.

- [ ] **Step 6: Update batch runner and hedging loader.** Batch writes all three sheets. Hedging defaults to the untouched `test_data` rows for headline evaluation; train/validation may be optionally loaded only for diagnostics.

- [ ] **Step 7: Update notebook helpers and examples.** Replace “train/test” assumptions and make the reporting layer explicit about which split it consumes.

- [ ] **Step 8: Run tests.**

```bash
python -m unittest tests.test_snapshot_splitting tests.test_results_store_io tests.test_hedging_invariants -v
python -m unittest discover -s tests -v
```

- [ ] **Step 9: Commit.**

```bash
git add src/splitting.py src/snapshot_job.py src/results_store.py src/batch_runner.py src/hedging_analysis.py notebooks calibration_example.ipynb tests docs/code_documentation.md
git commit -m "feat: add train validation test calibration split"
```

---

### Task 9: Remove regularization-selection leakage and formalize holdout inference

**Files:**
- Create: `notebooks/regularization_selection.ipynb`
- Create: `docs/reproducibility/regularization_selection.md`
- Create: `tests/test_regularization_selection.py`
- Modify: `notebooks/_lib/chapter3_final.py`
- Modify: `notebooks/calibration_results_final.ipynb`
- Modify: `notebooks/_lib/chapter3_analysis.py`

**Interfaces:**
- `compute_regularisation_summary(...)` must take an explicit evaluation split and selection workflow must use `validation` only.
- Final chapter-3 test reporting must accept a locked `final_reg` value and must not choose it from test metrics.

- [ ] **Step 1: Write a leakage test.** Build synthetic workbooks where validation prefers one lambda and test prefers another; assert selection helper returns the validation-preferred candidate and never reads test columns.

- [ ] **Step 2: Refactor regularization summary.** Report validation RMSE/MAE, nfev, roughness, Feller/jump-margin diagnostics. Remove “test RMSE” from the selection plot.

- [ ] **Step 3: Create the selection notebook.** It may inspect only train/validation/parameter sheets and exports the trade-off table/figure used to choose `lambda_TS`.

- [ ] **Step 4: Lock the selected value in `docs/reproducibility/regularization_selection.md`.** Record candidate grid, validation metrics, roughness, chosen value, and commit SHA **before** running the final test-reporting notebook.

- [ ] **Step 5: Update final reporting.** `calibration_results_final.ipynb` loads only the already-locked regularization workbook and computes untouched test metrics.

- [ ] **Step 6: Rename inference in helper comments/tables.** Use “snapshot-level cross-sectional holdout loss differential with HAC standard errors” rather than implying a conventional temporal forecast DM design.

- [ ] **Step 7: Run tests and notebook structure checks.**

```bash
python -m unittest tests.test_regularization_selection tests.test_notebook_structure tests.test_notebook_integrity -v
```

- [ ] **Step 8: Commit.**

```bash
git add notebooks docs/reproducibility/regularization_selection.md tests/test_regularization_selection.py
git commit -m "fix: select regularization on validation data only"
```

---

### Task 10: Make pricing metrics aggregation and units economically coherent

**Files:**
- Create: `tests/test_chapter3_metrics.py`
- Modify: `notebooks/_lib/chapter3_final.py`
- Modify: `notebooks/_lib/chapter3_analysis.py`
- Modify: `notebooks/calibration_results_final.ipynb`

**Interfaces:**
- Asset-level outputs keep `rmse_coin`, `mae_coin`, and within-spread share.
- Cross-asset pooled outputs use explicit common-unit/dimensionless columns, e.g. `error_usd = F0 * error_coin`, `error_spread_units = error_coin / spread_coin`, plus within-spread share.
- No helper exposes a column named simply `pooled_rmse` without a unit suffix.

- [ ] **Step 1: Write aggregation tests.** Verify option-weighted RMSE is `sqrt(mean(e_i^2))`, snapshot-mean RMSE is a separately named statistic, and the two are not accidentally substituted for one another.

- [ ] **Step 2: Write unit tests for pooled BTC/ETH data.** Assert pooled raw coin RMSE is absent/rejected; pooled USD and spread-unit metrics match hand calculations on a tiny fixture.

- [ ] **Step 3: Refactor `compute_overall_baseline_metrics(...)`.** Emit BTC/ETH coin metrics separately and pooled common-unit/dimensionless metrics. The existing all-row pooled calculation should be treated as a mechanical baseline, not retained under an economically ambiguous label.

- [ ] **Step 4: Update exported table schemas and plots.** Every axis/table header must include units or a clear normalization.

- [ ] **Step 5: Run tests.**

```bash
python -m unittest tests.test_chapter3_metrics -v
```

- [ ] **Step 6: Commit.**

```bash
git add notebooks/_lib notebooks/calibration_results_final.ipynb tests/test_chapter3_metrics.py
git commit -m "fix: make pricing metric aggregation unit consistent"
```

---

### Task 11: Correct perpetual-price provenance and hedging input schema

**Files:**
- Create: `tests/test_perp_price_provenance.py`
- Modify: `src/backfill_futures_marks.py`
- Modify: `src/hedging_analysis.py`
- Modify: `notebooks/_lib/futures_viewer.py`
- Modify: `README.md`
- Modify: `docs/code_documentation.md`

**Interfaces:**
- New backfill outputs must expose a neutral canonical price field such as `perp_futures_price_proxy` plus `price_source`/`price_kind` metadata.
- `load_perp_history(...)` accepts the new field first and legacy `perp_futures_mark_price` as a backward-compatible fallback.

- [ ] **Step 1: Write loader tests for new and legacy schemas.** New schema must preserve source metadata; legacy files must still load with a warning/note identifying them as legacy.

- [ ] **Step 2: Rename new backfill output semantics.** Do not label nearest TradingView one-minute candle open/close as a Deribit historical mark price.

- [ ] **Step 3: Preserve candle alignment diagnostics.** Keep timestamp gap, open/close choice, resolution, and source columns so hedging timing can be audited.

- [ ] **Step 4: Update hedging workbook notes.** Record the exact proxy file and price-kind used.

- [ ] **Step 5: Run tests.**

```bash
python -m unittest tests.test_perp_price_provenance -v
python -m unittest discover -s tests -v
```

- [ ] **Step 6: Commit.**

```bash
git add src/backfill_futures_marks.py src/hedging_analysis.py notebooks/_lib/futures_viewer.py tests/test_perp_price_provenance.py README.md docs/code_documentation.md
git commit -m "fix: make perpetual hedge price provenance explicit"
```

---

### Task 12: Add SVCJ identification diagnostics

**Files:**
- Create: `notebooks/_lib/robustness.py`
- Create: `notebooks/robustness_checks.ipynb`
- Create: `tests/test_robustness_helpers.py`

**Interfaces:**
- Produces parameter-path diagnostics for Feller gaps, Brownian-`rho` boundary clustering, jump-parameter near-bound behavior, SVCJ moment-domain margins, the empirical distribution of unbounded `rho_j`, and parameter correlations.

- [ ] **Step 1: Write identification-helper tests.** Verify currency-specific summaries, Feller diagnostics, moment margins, and that `rho_j` is treated as an unbounded loading rather than a bounded correlation.

- [ ] **Step 2: Implement identification helpers.** Keep the module diagnostic-only; it must not alter calibration, pricing, or hedging conventions.

- [ ] **Step 3: Add the identification notebook.** Export summary tables and currency-specific parameter-correlation matrices for the corrected calibration paths.

- [ ] **Step 4: Run tests.**

```bash
python -m unittest tests.test_robustness_helpers -v
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add notebooks/_lib/robustness.py notebooks/robustness_checks.ipynb tests/test_robustness_helpers.py
git commit -m "feat: add SVCJ identification diagnostics"
```

---

### Task 13: Regenerate calibration and hedging results from a clean slate

**Files:**
- Generate: `excel files/remediation/calibration_results_reg_*.xlsx`
- Generate: `notebooks/chapter3_outputs/remediation/*`
- Generate: `hedging_output/remediation/*`
- Create: `docs/reproducibility/remediation_results.md`
- Update frozen semantic fixtures only after review

**Interfaces:**
- Consumes: frozen analysis manifest, corrected code, 60/20/20 split, sequential regularization, validation-only lambda selection.
- Produces: the only result artifacts allowed to feed the manuscript rewrite.

- [ ] **Step 1: Run the full test suite before any expensive rerun.**

```bash
python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 2: Run all candidate regularization workbooks into the remediation output directory.** Use the frozen sample window and `resume=False`. Run different lambda values as independent jobs if parallel compute is desired; each per-currency time chain remains sequential.

- [ ] **Step 3: Run `regularization_selection.ipynb` using validation only.** Record and commit the selected lambda in `docs/reproducibility/regularization_selection.md` before opening final test summaries.

- [ ] **Step 4: Generate final untouched-test calibration outputs.** Export asset-level coin metrics, pooled common-unit metrics, maturity/moneyness buckets, HAC loss-differential tests, parameter diagnostics, and figures.

- [ ] **Step 5: Run the corrected hedging engine on test rows only.** Generate regular-delta, net-delta, tail-risk, basis-bucket, maturity/moneyness, and robustness summaries using identical hedge scaling.

- [ ] **Step 6: Compare old versus corrected results.** `remediation_results.md` must tabulate every thesis headline statistic side by side and attribute material changes to one of: split redesign, sequential regularization, `rho_j` domain/admissibility, hedge mapping, metric definition, or data-provenance change.

- [ ] **Step 7: Investigate any unexpected discontinuity before manuscript edits.** In particular, inspect parameter-path jumps, failed calibrations, near-singular SVCJ moment margins, and hedging outliers.

- [ ] **Step 8: Refresh notebook regression fixtures only after the new outputs are approved.**

```bash
python tests/generate_notebook_reference_outputs.py
python -m unittest tests.test_notebook_regression_outputs -v
```

- [ ] **Step 9: Commit reproducibility notes and approved generated summary artifacts.** Avoid committing unnecessary giant intermediate files if the repository policy does not require them; preserve enough outputs to reproduce every manuscript table/figure.

```bash
git add docs/reproducibility notebooks/chapter3_outputs tests/fixtures
# add selected workbooks/hedging exports only if they are intentionally versioned
git commit -m "results: regenerate corrected thesis evidence"
```

---

### Task 14: Rewrite the manuscript to match the corrected code and results

**Files:**
- Modify: `docs/manuscript/main.tex`
- Modify: `docs/manuscript/references.bib`
- Modify/add manuscript figure/table inputs generated in Task 13

**Interfaces:**
- Consumes: only approved outputs from Task 13 plus the corrected theory/code from Tasks 3–12.
- Produces: a manuscript whose equations, terminology, methods, tables, and conclusions are traceable to current code.

- [ ] **Step 1: Correct the theory notation in Chapters 1–2 and Appendix D.** Distinguish `Q` (pricing measure under which `F` is a martingale) from `Q^F` (futures-numeraire measure). Define `P2=Q(F_T>K)` and `P1=Q^F(F_T>K)` with shifted CF `phi_Q(u-i)/phi_Q(-i)`. Ensure Black gives `N(d2)` and `N(d1)` respectively.

- [ ] **Step 2: Rewrite inverse-pricing and delta derivations.** Keep `V_coin=V_USD/F_t` under the appropriate numeraire-change statement and derive `dV_coin/dF=(Delta_reg-V_coin)/F` without measure-label ambiguity.

- [ ] **Step 3: Rewrite the perpetual hedge mapping.** State the local basis assumption explicitly and give the actual implemented notional `H^2/F_term * Delta_used` for both regular and net hedge rules; explain that only `Delta_used` differs between the benchmark and contract-consistent hedge.

- [ ] **Step 4: Rewrite calibration methodology.** Document the 60/20/20 within-snapshot cross-sectional split, deterministic assignment, train-only fitting, validation-only regularization selection, untouched test evaluation, spread-standardized residuals, transformed-space temporal regularization, and soft—not hard—Feller penalty.

- [ ] **Step 5: Rewrite SVCJ parameter discussion.** Describe `rho_j` as a jump-size loading/dependence coefficient, not a correlation bounded by `[-1,1]`; report new admissibility diagnostics and soften structural interpretation of weakly identified parameters.

- [ ] **Step 6: Correct data provenance.** Describe the perpetual hedge input as the exact proxy actually used; remove the unsupported statement that quote-age stale quotes are filtered unless a true staleness filter has been added.

- [ ] **Step 7: Correct evaluation language.** Replace generic “out-of-sample forecasting” language with “within-snapshot cross-sectional holdout” where appropriate. Describe the HAC loss-differential test accurately.

- [ ] **Step 8: Replace all Chapter 3 and 4 headline tables/figures with Task 13 exports.** Never transcribe numbers by hand when an exported CSV/LaTeX table can be included/generated.


- [ ] **Step 10: Preserve already-correct caveats.** Keep the statement that transaction costs/funding are excluded from clean hedge diagnostics; keep raw `K/F` moneyness buckets.

- [ ] **Step 11: Fix appendix numbering and notation.** Appendix D equations must no longer be labeled `(C.x)`; standardize `F_t`, `H_t`, `Q`, `Q^F`, `Delta_reg`, and `Delta_net` across the text.

- [ ] **Step 12: Compile and inspect the PDF.** Resolve all undefined references/citations, overfull equations/tables, and stale figures.

- [ ] **Step 13: Commit.**

```bash
git add docs/manuscript
git commit -m "docs: align thesis manuscript with corrected implementation"
```

---

### Task 15: Expand literature positioning and tighten interpretation/presentation

**Files:**
- Modify: `docs/manuscript/main.tex`
- Modify: `docs/manuscript/references.bib`

**Interfaces:**
- Consumes: verified bibliographic sources only; no invented references.
- Produces: a research-paper-quality literature review and presentation while preserving the corrected empirical claims.

- [ ] **Step 1: Refresh the inverse/quanto literature.** Verify current publication status of Lucic–Sepp and related inverse/perpetual-option work; update bibliographic metadata rather than retaining stale “forthcoming” labels if publication has occurred.

- [ ] **Step 2: Expand crypto option-surface calibration literature.** Add directly relevant Black/Heston/jump/rough-volatility/nonparametric surface papers and explain exactly how this thesis differs.

- [ ] **Step 3: Expand crypto hedging and dynamic-calibration literature.** Position the premium-adjusted delta result, basis-risk result, and proposed state-space extension against real prior work.

- [ ] **Step 4: Rewrite the contribution statement conservatively.** Avoid “first” claims unless the literature search establishes them. Separate theoretical adaptation, empirical pricing evidence, and hedging evidence.

- [ ] **Step 5: Soften weak-identification language.** Replace statements such as “ETH relies on…” with calibration-representation language unless identification diagnostics support a structural claim.

- [ ] **Step 6: Simplify figures.** Main-text figures should have readable labels and 2–4 panels where possible; move dense parameter-path panels and secondary robustness plots to appendices.

- [ ] **Step 7: Compile and inspect again.** Ensure figure/table references and page flow remain coherent.

- [ ] **Step 8: Commit.**

```bash
git add docs/manuscript
git commit -m "docs: strengthen thesis literature and presentation"
```

---

### Task 16: Final end-to-end verification and canonical artifact replacement

**Files:**
- Modify: `README.md`
- Modify: `docs/code_documentation.md`
- Replace/update: canonical thesis PDF under `docs/`
- Replace/update: canonical result artifacts only after comparison approval
- Create: `docs/reproducibility/final_verification.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: final branch ready for review/merge, with one documented command sequence from raw/frozen inputs to manuscript outputs.

- [ ] **Step 1: Run the full Python test suite from a clean environment.**

```bash
python -m unittest discover -s tests -v
```

Expected: all required tests PASS; environment-dependent notebook execution tests may skip only for documented kernel/permission reasons.

- [ ] **Step 2: Run notebook integrity/regression/smoke checks explicitly.**

```bash
python -m unittest tests.test_notebook_integrity tests.test_notebook_structure tests.test_notebook_regression_outputs tests.test_notebook_smoke -v
```

- [ ] **Step 3: Run one clean miniature end-to-end calibration/hedging smoke pipeline.** Use a small frozen subset and assert train/validation/test sheets, sequential warm starts, finite prices/deltas, and hedging outputs are created.

- [ ] **Step 4: Recompile the manuscript from committed source.** Record compiler versions/commands and the final PDF checksum in `final_verification.md`.

- [ ] **Step 5: Audit manuscript-code traceability.** For every headline table/figure, record the producing notebook/helper/export filename. For every central equation, record the implementing function/test that validates it.

- [ ] **Step 6: Audit the old-to-new result delta.** Confirm all changed thesis numbers are explained in `remediation_results.md` and no old table/figure remains accidentally embedded.

- [ ] **Step 7: Update README and code documentation.** Document the three-way split, validation selection protocol, sequential regularization requirement, corrected hedge mapping, price-proxy semantics, and exact paper reproduction workflow.

- [ ] **Step 8: Replace the canonical PDF/result artifacts only now.** Preserve the old PDF under a clearly named baseline/archive path if desired; do not silently overwrite without provenance.

- [ ] **Step 9: Check branch diff against `main`.** Ensure no unrelated live-data snapshots, caches, or generated junk were added as part of remediation.

```bash
git status
git diff --stat main...HEAD
```

- [ ] **Step 10: Final commit.**

```bash
git add README.md docs tests/fixtures
# add only intentionally versioned final artifacts
git commit -m "chore: finalize reproducible thesis remediation"
```

---

## Execution order and stop conditions

Implement strictly in task order. The following are hard stop conditions:

1. **Task 2:** if the exact editable manuscript source is unavailable, continue code tasks but do not attempt manuscript reconstruction from PDF.
2. **Task 3:** if the Black shifted-CF delta does not reproduce `N(d1)` to numerical tolerance, stop before any hedging rerun.
3. **Task 5:** if SVCJ contour admissibility cannot be made stable for the thesis parameter region, stop and redesign the SVCJ numerical integration before recalibration.
4. **Task 7:** if sequential and resumed regularized runs disagree materially, do not generate any regularization sweep.
5. **Task 9:** do not inspect/quote final test metrics until the regularization choice is recorded.
6. **Task 12:** review SVCJ identification diagnostics before interpreting structural parameter paths.
7. **Task 13:** if corrected results materially overturn a thesis conclusion, rewrite the conclusion to match the evidence rather than trying to preserve the old claim.

## Expected corrected empirical narrative

The plan does **not** assume that the old numerical values survive. It tests whether the qualitative thesis conclusions remain true after removing methodological ambiguity and implementation mismatches. The expected—but not guaranteed—outcome is:

- SVCJ continues to outperform Black/Heston in cross-sectional pricing, with the size of the advantage re-estimated on the untouched test sample;
- net delta continues to outperform regular delta when both are mapped through the same perpetual hedge-notional formula;
- cross-model hedging differences remain small relative to the net-delta effect, with basis risk still explaining part of the residual;
- ETH jump-parameter paths become less artificially boundary-constrained after removing the `tanh` bound, though weak identification may remain.

If any of these fail after the corrected rerun, the manuscript must report the corrected result.

## Final deliverable checklist

- [ ] Frozen input manifest and baseline provenance
- [ ] Editable manuscript source in repo
- [ ] Correct measure/numeraire and Greek implementation + tests
- [ ] Correct common hedge mapping + tests
- [ ] Unbounded `rho_j` + Carr-Madan admissibility + tests
- [ ] Explicit residual/regularization/soft-constraint semantics
- [ ] Sequential regularized calibration + resume-equivalence tests
- [ ] Deterministic 60/20/20 train/validation/test split
- [ ] Validation-only regularization selection record
- [ ] Unit-consistent pricing metrics
- [ ] Honest perpetual-price proxy schema/provenance
- [ ] SVCJ identification diagnostics
- [ ] Full corrected calibration and hedging rerun
- [ ] Manuscript rewritten from generated outputs
- [ ] Literature/presentation pass
- [ ] Full regression/notebook/manuscript verification
