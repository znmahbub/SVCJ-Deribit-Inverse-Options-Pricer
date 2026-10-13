from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, repl: str, label: str) -> str:
    out, count = re.subn(pattern, repl, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    return out


# ---------------------------------------------------------------------------
# New independent split module
# ---------------------------------------------------------------------------
(ROOT / "src/splitting.py").write_text(textwrap.dedent('''
"""Deterministic within-snapshot train/validation/test assignment."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.60
    validation_frac: float = 0.20
    test_frac: float = 0.20
    random_seed: int = 123

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.train_frac, self.validation_frac, self.test_frac], dtype=float
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("split fractions must be finite and non-negative")
        if float(self.train_frac) <= 0.0:
            raise ValueError("train_frac must be strictly positive")
        if not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("train/validation/test fractions must sum to one")


def _allocation_counts(n: int, cfg: SplitConfig) -> dict[str, int]:
    """Largest-remainder allocation with a nonempty train set when n>0."""
    n = int(n)
    if n <= 0:
        return {name: 0 for name in _SPLITS}
    fractions = np.asarray(
        [cfg.train_frac, cfg.validation_frac, cfg.test_frac], dtype=float
    )
    raw = n * fractions
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    # Stable tie-break order: validation, test, train after fractional size.
    tie_priority = np.asarray([2, 0, 1])
    order = sorted(range(3), key=lambda i: (-float(raw[i] - counts[i]), int(tie_priority[i])))
    for i in order[:remainder]:
        counts[i] += 1

    if counts[0] == 0:
        donor = int(np.argmax(counts[1:])) + 1
        if counts[donor] > 0:
            counts[donor] -= 1
            counts[0] += 1

    # When there are enough observations, every positive-fraction split should
    # be represented inside the stratum.
    positive = [i for i, frac in enumerate(fractions) if frac > 0.0]
    if n >= len(positive):
        for i in positive:
            if counts[i] == 0:
                donors = [j for j in positive if counts[j] > 1]
                if donors:
                    donor = max(donors, key=lambda j: counts[j])
                    counts[donor] -= 1
                    counts[i] += 1

    return {name: int(counts[i]) for i, name in enumerate(_SPLITS)}


def _assign_positions(positions: np.ndarray, cfg: SplitConfig, rng: np.random.Generator, out: np.ndarray) -> None:
    positions = np.asarray(positions, dtype=int)
    if positions.size == 0:
        return
    perm = positions[rng.permutation(len(positions))]
    counts = _allocation_counts(len(perm), cfg)
    start = 0
    for name in _SPLITS:
        end = start + counts[name]
        out[perm[start:end]] = name
        start = end


def assign_cross_sectional_split(df: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    """Assign each row exactly once to train, validation, or test.

    Strata are ``(expiry_datetime, option_type)`` when both fields exist.
    Strata of at least three rows are assigned independently so all feasible
    splits are represented. Rows belonging to tiny strata (sizes one or two)
    are pooled into one deterministic fallback allocation; this avoids forcing
    every singleton into the same holdout while preserving all observations.
    """
    n = len(df)
    if n == 0:
        return pd.Series([], index=df.index, dtype="object", name="split")

    rng = np.random.default_rng(int(cfg.random_seed))
    out = np.empty(n, dtype=object)
    out[:] = None

    if {"expiry_datetime", "option_type"}.issubset(df.columns):
        groups = df.groupby(["expiry_datetime", "option_type"], sort=True, dropna=False).indices
        tiny: list[int] = []
        for _, positions in groups.items():
            pos = np.asarray(positions, dtype=int)
            if len(pos) >= 3:
                _assign_positions(pos, cfg, rng, out)
            else:
                tiny.extend(pos.tolist())
        _assign_positions(np.asarray(tiny, dtype=int), cfg, rng, out)
    else:
        _assign_positions(np.arange(n, dtype=int), cfg, rng, out)

    if any(value is None for value in out):
        raise RuntimeError("internal split assignment left unassigned rows")
    result = pd.Series(out, index=df.index, dtype="object", name="split")
    if not bool((result == "train").any()):
        # Defensive fallback for pathological future fraction logic.
        result.iloc[0] = "train"
    return result


__all__ = ["SplitConfig", "assign_cross_sectional_split"]
''').lstrip(), encoding="utf-8")


# ---------------------------------------------------------------------------
# snapshot_job.py
# ---------------------------------------------------------------------------
p = ROOT / "src/snapshot_job.py"
s = p.read_text(encoding="utf-8")
s = s.replace("3. Splits deterministically into train/test.\n4. Calibrates Black, Heston, and SVCJ on the train split (warm-started).", "3. Assigns a deterministic train/validation/test cross-sectional split.\n4. Calibrates Black, Heston, and SVCJ on the train split only (warm-started).")
s = replace_once(
    s,
    "from .inverse_fft_pricer import FFTParams, center_fft_params_on_strikes\n",
    "from .inverse_fft_pricer import FFTParams, center_fft_params_on_strikes\nfrom .splitting import SplitConfig, assign_cross_sectional_split\n",
    "snapshot splitting import",
)
s = replace_once(
    s,
    '''    train_frac: float,
    random_seed: int,
''',
    '''    train_frac: float,
    validation_frac: float,
    test_frac: float,
    random_seed: int,
''',
    "snapshot split fractions signature",
)
s = s.replace("      - ``train_df`` / ``test_df``: priced option rows (with price_* columns)", "      - ``train_df`` / ``validation_df`` / ``test_df``: priced option rows")
s = replace_once(
    s,
    '''        rmse_train: float,
        mae_train: float,
        rmse_test: float,
        mae_test: float,
        n_total: int,
        n_train: int,
        n_test: int,
''',
    '''        rmse_train: float,
        mae_train: float,
        rmse_validation: float,
        mae_validation: float,
        rmse_test: float,
        mae_test: float,
        n_total: int,
        n_train: int,
        n_validation: int,
        n_test: int,
''',
    "parameter row validation metrics signature",
)
s = replace_once(
    s,
    '''            rmse_train=float(rmse_train) if rmse_train is not None else float("nan"),
            mae_train=float(mae_train) if mae_train is not None else float("nan"),
            rmse_test=float(rmse_test) if rmse_test is not None else float("nan"),
            mae_test=float(mae_test) if mae_test is not None else float("nan"),
            n_options_total=int(n_total),
            n_train=int(n_train),
            n_test=int(n_test),
''',
    '''            rmse_train=float(rmse_train) if rmse_train is not None else float("nan"),
            mae_train=float(mae_train) if mae_train is not None else float("nan"),
            rmse_validation=float(rmse_validation) if rmse_validation is not None else float("nan"),
            mae_validation=float(mae_validation) if mae_validation is not None else float("nan"),
            rmse_test=float(rmse_test) if rmse_test is not None else float("nan"),
            mae_test=float(mae_test) if mae_test is not None else float("nan"),
            n_options_total=int(n_total),
            n_train=int(n_train),
            n_validation=int(n_validation),
            n_test=int(n_test),
''',
    "parameter row validation metrics body",
)
s = replace_once(
    s,
    '''                rmse_train=np.nan, mae_train=np.nan,
                rmse_test=np.nan,  mae_test=np.nan,
                n_total=int(n_total), n_train=0, n_test=0,
''',
    '''                rmse_train=np.nan, mae_train=np.nan,
                rmse_validation=np.nan, mae_validation=np.nan,
                rmse_test=np.nan,  mae_test=np.nan,
                n_total=int(n_total), n_train=0, n_validation=0, n_test=0,
''',
    "failure validation metrics",
)
s = replace_once(
    s,
    '''            "train_df": empty_tt,
            "test_df": empty_tt,
''',
    '''            "train_df": empty_tt,
            "validation_df": empty_tt,
            "test_df": empty_tt,
''',
    "failure validation frame",
)

# Replace old shuffled two-way split.
pattern = r'''    # Deterministic split \(shuffled\)\n    df_filt = df_filt\.sample\(frac=1\.0, random_state=random_seed\)\.reset_index\(drop=True\)\n    n_total_rows = int\(len\(df_filt\)\)\n    n_train = int\(np\.floor\(train_frac \* n_total_rows\)\)\n    # Keep calibration numerically meaningful even if train_frac is extreme\.\n    if n_total_rows > 1:\n        n_train = min\(max\(n_train, 1\), n_total_rows - 1\)\n    else:\n        n_train = n_total_rows\n    train = df_filt\.iloc\[:n_train\]\.copy\(\)\n    test = df_filt\.iloc\[n_train:\]\.copy\(\)\n'''
replacement = '''    # Deterministic within-snapshot three-way cross-sectional split.  The split
    # is assigned once and then reused by every model; model choice never resplits.
    df_filt = df_filt.reset_index(drop=True).copy()
    split_cfg = SplitConfig(
        train_frac=float(train_frac),
        validation_frac=float(validation_frac),
        test_frac=float(test_frac),
        random_seed=int(random_seed),
    )
    df_filt["split"] = assign_cross_sectional_split(df_filt, split_cfg).to_numpy()
    n_total_rows = int(len(df_filt))
    train = df_filt.loc[df_filt["split"] == "train"].copy()
    validation = df_filt.loc[df_filt["split"] == "validation"].copy()
    test = df_filt.loc[df_filt["split"] == "test"].copy()
    n_train = int(len(train))
    n_validation = int(len(validation))
'''
s = regex_once(s, pattern, replacement, "three-way split block")

s = s.replace("# Reprice full snapshot once per model (then split train/test for output & errors)", "# Reprice full snapshot once per model, then reuse the frozen three-way split.")
s = replace_once(
    s,
    '''    train_out = df_out.iloc[:n_train].copy()
    test_out = df_out.iloc[n_train:].copy()

    # Errors (unweighted)
    y_train = train_out["mid_price_clean"].to_numpy(dtype=float) if len(train_out) else np.array([])
    y_test = test_out["mid_price_clean"].to_numpy(dtype=float) if len(test_out) else np.array([])
''',
    '''    train_out = df_out.loc[df_out["split"] == "train"].copy()
    validation_out = df_out.loc[df_out["split"] == "validation"].copy()
    test_out = df_out.loc[df_out["split"] == "test"].copy()

    # Errors (unweighted) are reported separately for all three splits.
    y_train = train_out["mid_price_clean"].to_numpy(dtype=float) if len(train_out) else np.array([])
    y_validation = validation_out["mid_price_clean"].to_numpy(dtype=float) if len(validation_out) else np.array([])
    y_test = test_out["mid_price_clean"].to_numpy(dtype=float) if len(test_out) else np.array([])
''',
    "output split frames",
)
s = replace_once(
    s,
    '''        p_train = train_out[col].to_numpy(dtype=float) if len(train_out) else np.array([])
        p_test = test_out[col].to_numpy(dtype=float) if len(test_out) else np.array([])
        e_tr = compute_errors(y_train, p_train) if len(train_out) else {"rmse": np.nan, "mae": np.nan}
        e_te = compute_errors(y_test, p_test) if len(test_out) else {"rmse": np.nan, "mae": np.nan}
        errs[model] = dict(
            rmse_train=float(e_tr["rmse"]),
            mae_train=float(e_tr["mae"]),
            rmse_test=float(e_te["rmse"]),
            mae_test=float(e_te["mae"]),
        )

    n_total = int(len(df_out))
    n_test = int(len(test_out))
''',
    '''        p_train = train_out[col].to_numpy(dtype=float) if len(train_out) else np.array([])
        p_validation = validation_out[col].to_numpy(dtype=float) if len(validation_out) else np.array([])
        p_test = test_out[col].to_numpy(dtype=float) if len(test_out) else np.array([])
        e_tr = compute_errors(y_train, p_train) if len(train_out) else {"rmse": np.nan, "mae": np.nan}
        e_va = compute_errors(y_validation, p_validation) if len(validation_out) else {"rmse": np.nan, "mae": np.nan}
        e_te = compute_errors(y_test, p_test) if len(test_out) else {"rmse": np.nan, "mae": np.nan}
        errs[model] = dict(
            rmse_train=float(e_tr["rmse"]),
            mae_train=float(e_tr["mae"]),
            rmse_validation=float(e_va["rmse"]),
            mae_validation=float(e_va["mae"]),
            rmse_test=float(e_te["rmse"]),
            mae_test=float(e_te["mae"]),
        )

    n_total = int(len(df_out))
    n_validation = int(len(validation_out))
    n_test = int(len(test_out))
''',
    "validation errors",
)
s = replace_once(
    s,
    '''            rmse_train=model_errs["rmse_train"],
            mae_train=model_errs["mae_train"],
            rmse_test=model_errs["rmse_test"],
            mae_test=model_errs["mae_test"],
            n_total=n_total, n_train=n_train, n_test=n_test,
''',
    '''            rmse_train=model_errs["rmse_train"],
            mae_train=model_errs["mae_train"],
            rmse_validation=model_errs["rmse_validation"],
            mae_validation=model_errs["mae_validation"],
            rmse_test=model_errs["rmse_test"],
            mae_test=model_errs["mae_test"],
            n_total=n_total, n_train=n_train, n_validation=n_validation, n_test=n_test,
''',
    "common validation fields",
)
s = replace_once(
    s,
    '''        "train_df": train_out,
        "test_df": test_out,
''',
    '''        "train_df": train_out,
        "validation_df": validation_out,
        "test_df": test_out,
''',
    "return validation frame",
)
p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# results_store.py
# ---------------------------------------------------------------------------
p = ROOT / "src/results_store.py"
s = p.read_text(encoding="utf-8")
s = s.replace("appending new rows (params + train/test option rows)", "appending new rows (params + train/validation/test option rows)")
s = replace_once(s, 'TRAIN_SHEET = "train_data"\nTEST_SHEET = "test_data"\n', 'TRAIN_SHEET = "train_data"\nVALIDATION_SHEET = "validation_data"\nTEST_SHEET = "test_data"\n', "validation sheet constant")
s = replace_once(
    s,
    '''    "rmse_train",
    "mae_train",
    "rmse_test",
    "mae_test",
    "n_options_total",
    "n_train",
    "n_test",
''',
    '''    "rmse_train",
    "mae_train",
    "rmse_validation",
    "mae_validation",
    "rmse_test",
    "mae_test",
    "n_options_total",
    "n_train",
    "n_validation",
    "n_test",
''',
    "parameter validation schema",
)
s = s.replace("# Preferred (front) columns for train/test sheets", "# Preferred (front) columns for train/validation/test sheets")
s = replace_once(s, '    "random_seed",\n    "price_black",', '    "random_seed",\n    "split",\n    "price_black",', "front split column")
s = replace_once(
    s,
    '''    train_sheet: str = TRAIN_SHEET
    test_sheet: str = TEST_SHEET
''',
    '''    train_sheet: str = TRAIN_SHEET
    validation_sheet: str = VALIDATION_SHEET
    test_sheet: str = TEST_SHEET
''',
    "WorkbookSchema validation",
)
s = replace_once(
    s,
    '''        TRAIN_SHEET: pd.DataFrame(),
        TEST_SHEET: pd.DataFrame(),
''',
    '''        TRAIN_SHEET: pd.DataFrame(),
        VALIDATION_SHEET: pd.DataFrame(),
        TEST_SHEET: pd.DataFrame(),
''',
    "empty workbook validation",
)
# Replace train/test loading with all option sheets, tolerant of legacy absence.
pattern = r'''    # train/test: keep as-is; we'll align columns on append\n    if TRAIN_SHEET in sheets:\n.*?    if TEST_SHEET in sheets:\n        wb\[TEST_SHEET\] = sheets\[TEST_SHEET\]\.copy\(\)\n        if "snapshot_ts" in wb\[TEST_SHEET\]\.columns:\n            wb\[TEST_SHEET\]\["snapshot_ts"\] = normalize_timestamp_series\(wb\[TEST_SHEET\]\["snapshot_ts"\]\)\n'''
replacement = '''    # Option splits: legacy workbooks may not contain validation_data.  Such
    # workbooks remain loadable for comparison; new writes always include it.
    for sheet_name in (TRAIN_SHEET, VALIDATION_SHEET, TEST_SHEET):
        if sheet_name in sheets:
            wb[sheet_name] = sheets[sheet_name].copy()
            if "snapshot_ts" in wb[sheet_name].columns:
                wb[sheet_name]["snapshot_ts"] = normalize_timestamp_series(wb[sheet_name]["snapshot_ts"])
'''
s = regex_once(s, pattern, replacement, "tolerant validation load")
s = s.replace("# order train/test columns", "# order option-split columns")
s = replace_once(
    s,
    '''    wb_to_write[TRAIN_SHEET] = order_train_test_columns(wb_to_write.get(TRAIN_SHEET, pd.DataFrame()))
    wb_to_write[TEST_SHEET] = order_train_test_columns(wb_to_write.get(TEST_SHEET, pd.DataFrame()))

    # sort train/test rows
    for sh in [TRAIN_SHEET, TEST_SHEET]:
''',
    '''    for sh in [TRAIN_SHEET, VALIDATION_SHEET, TEST_SHEET]:
        wb_to_write[sh] = order_train_test_columns(wb_to_write.get(sh, pd.DataFrame()))

    # sort option-split rows
    for sh in [TRAIN_SHEET, VALIDATION_SHEET, TEST_SHEET]:
''',
    "flush validation sheet",
)
s = replace_once(s, '    "TRAIN_SHEET",\n    "TEST_SHEET",', '    "TRAIN_SHEET",\n    "VALIDATION_SHEET",\n    "TEST_SHEET",', "results store exports")
p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# batch_runner.py
# ---------------------------------------------------------------------------
p = ROOT / "src/batch_runner.py"
s = p.read_text(encoding="utf-8")
s = replace_once(s, '    TEST_SHEET,\n    TRAIN_SHEET,\n', '    TEST_SHEET,\n    TRAIN_SHEET,\n    VALIDATION_SHEET,\n', "batch validation import")
s = replace_once(s, 'from .snapshot_job import process_snapshot_to_payload, timestamp_from_filename\n', 'from .snapshot_job import process_snapshot_to_payload, timestamp_from_filename\nfrom .splitting import SplitConfig\n', "batch SplitConfig import")
s = replace_once(
    s,
    '''    train_frac: float = 0.70
    global_random_seed: int = 123
''',
    '''    train_frac: float = 0.60
    validation_frac: float = 0.20
    test_frac: float = 0.20
    global_random_seed: int = 123
''',
    "BatchConfig three-way fractions",
)
s = replace_once(
    s,
    '''        self.save_every_n_files = int(self.save_every_n_files)
        self.n_workers = int(self.n_workers)
''',
    '''        self.save_every_n_files = int(self.save_every_n_files)
        self.n_workers = int(self.n_workers)
        # Validate the configured fractions once at construction time.
        SplitConfig(
            train_frac=float(self.train_frac),
            validation_frac=float(self.validation_frac),
            test_frac=float(self.test_frac),
            random_seed=int(self.global_random_seed),
        )
''',
    "BatchConfig split validation",
)
s = replace_once(
    s,
    '''            rmse_train=float("nan"),
            mae_train=float("nan"),
            rmse_test=float("nan"),
            mae_test=float("nan"),
            n_options_total=0,
            n_train=0,
            n_test=0,
''',
    '''            rmse_train=float("nan"),
            mae_train=float("nan"),
            rmse_validation=float("nan"),
            mae_validation=float("nan"),
            rmse_test=float("nan"),
            mae_test=float("nan"),
            n_options_total=0,
            n_train=0,
            n_validation=0,
            n_test=0,
''',
    "batch exception validation metrics",
)
s = replace_once(
    s,
    '''            "train_df": pd.DataFrame(),
            "test_df": pd.DataFrame(),
''',
    '''            "train_df": pd.DataFrame(),
            "validation_df": pd.DataFrame(),
            "test_df": pd.DataFrame(),
''',
    "batch exception validation frame",
)
s = replace_once(
    s,
    '''                            train_frac=cfg.train_frac,
                            random_seed=seed,
''',
    '''                            train_frac=cfg.train_frac,
                            validation_frac=cfg.validation_frac,
                            test_frac=cfg.test_frac,
                            random_seed=seed,
''',
    "batch pass split fractions",
)
s = replace_once(
    s,
    '''            append_df(workbook, TRAIN_SHEET, pld.get("train_df", pd.DataFrame()))
            append_df(workbook, TEST_SHEET, pld.get("test_df", pd.DataFrame()))
''',
    '''            append_df(workbook, TRAIN_SHEET, pld.get("train_df", pd.DataFrame()))
            append_df(workbook, VALIDATION_SHEET, pld.get("validation_df", pd.DataFrame()))
            append_df(workbook, TEST_SHEET, pld.get("test_df", pd.DataFrame()))
''',
    "batch append validation",
)
p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# hedging_analysis.py: default headline analysis to untouched test only.
# ---------------------------------------------------------------------------
p = ROOT / "src/hedging_analysis.py"
s = p.read_text(encoding="utf-8")
s = replace_once(s, '    TEST_SHEET,\n    TRAIN_SHEET,\n', '    TEST_SHEET,\n    TRAIN_SHEET,\n    VALIDATION_SHEET,\n', "hedging validation import")
s = replace_once(
    s,
    '''    currencies: tuple[str, ...] | None = None
    exclude_expiry_crossing: bool = True
''',
    '''    currencies: tuple[str, ...] | None = None
    evaluation_splits: tuple[str, ...] = ("test",)
    exclude_expiry_crossing: bool = True
''',
    "hedging evaluation splits config",
)
s = replace_once(
    s,
    '''        if int(self.rebalance_every_n_snapshots) < 1:
            raise ValueError("rebalance_every_n_snapshots must be >= 1")
''',
    '''        if int(self.rebalance_every_n_snapshots) < 1:
            raise ValueError("rebalance_every_n_snapshots must be >= 1")
        allowed = {"train", "validation", "test"}
        requested = tuple(str(x).lower() for x in self.evaluation_splits)
        if not requested or not set(requested).issubset(allowed):
            raise ValueError("evaluation_splits must contain train/validation/test only")
        object.__setattr__(self, "evaluation_splits", requested)
''',
    "hedging split config validation",
)
s = replace_once(
    s,
    '''def load_calibration_workbook(path: Path) -> dict[str, pd.DataFrame]:
    needed = [*PARAM_SHEETS.values(), TRAIN_SHEET, TEST_SHEET]
    return {sheet: pd.read_excel(path, sheet_name=sheet, engine="openpyxl") for sheet in needed}


def combine_option_sheets(workbook: dict[str, pd.DataFrame]) -> pd.DataFrame:
    train = workbook[TRAIN_SHEET].copy()
    test = workbook[TEST_SHEET].copy()
    train["split"] = "train"
    test["split"] = "test"
    df = pd.concat([train, test], ignore_index=True, sort=False)
''',
    '''def load_calibration_workbook(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    needed = [*PARAM_SHEETS.values(), TRAIN_SHEET, VALIDATION_SHEET, TEST_SHEET]
    out: dict[str, pd.DataFrame] = {}
    for sheet in needed:
        if sheet in sheets:
            out[sheet] = sheets[sheet].copy()
        elif sheet == VALIDATION_SHEET:
            out[sheet] = pd.DataFrame()
        else:
            raise ValueError(f"Calibration workbook missing required sheet: {sheet}")
    return out


def combine_option_sheets(
    workbook: dict[str, pd.DataFrame],
    *,
    splits: tuple[str, ...] = ("test",),
) -> pd.DataFrame:
    """Combine explicitly requested calibration splits; default is untouched test."""
    sheet_for = {"train": TRAIN_SHEET, "validation": VALIDATION_SHEET, "test": TEST_SHEET}
    frames: list[pd.DataFrame] = []
    for split in tuple(str(x).lower() for x in splits):
        if split not in sheet_for:
            raise ValueError(f"Unknown split: {split}")
        frame = workbook.get(sheet_for[split], pd.DataFrame()).copy()
        if frame.empty:
            continue
        frame["split"] = split
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
''',
    "hedging split combiner",
)

# replace prepare_output_option_sheets function
pattern = r'''def prepare_output_option_sheets\(options: pd\.DataFrame, workbook: dict\[str, pd\.DataFrame\]\) -> tuple\[pd\.DataFrame, pd\.DataFrame\]:\n.*?    return train_out, test_out\n'''
replacement = '''def prepare_output_option_sheets(
    options: pd.DataFrame,
    workbook: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched = options.copy()

    def one(split: str, sheet: str) -> pd.DataFrame:
        base = workbook.get(sheet, pd.DataFrame()).copy()
        base_cols = list(base.columns)
        selected = enriched.loc[enriched.get("split", pd.Series(index=enriched.index, dtype=object)) == split].copy()
        if selected.empty:
            for col in DELTA_OUTPUT_COLS:
                if col not in base.columns:
                    base[col] = np.nan
            return base
        for col in DELTA_OUTPUT_COLS:
            if col not in selected.columns:
                selected[col] = np.nan
        return selected[[c for c in base_cols if c in selected.columns] + DELTA_OUTPUT_COLS].copy()

    return (
        one("train", TRAIN_SHEET),
        one("validation", VALIDATION_SHEET),
        one("test", TEST_SHEET),
    )
'''
s = regex_once(s, pattern, replacement, "hedging output three splits")

s = replace_once(
    s,
    '''def write_hedge_workbook(output_path: Path, workbook: dict[str, pd.DataFrame], train_data: pd.DataFrame, test_data: pd.DataFrame, price_verification: pd.DataFrame, interval_panel: pd.DataFrame, summary_tables: dict[str, pd.DataFrame], notes: pd.DataFrame) -> None:
''',
    '''def write_hedge_workbook(output_path: Path, workbook: dict[str, pd.DataFrame], train_data: pd.DataFrame, validation_data: pd.DataFrame, test_data: pd.DataFrame, price_verification: pd.DataFrame, interval_panel: pd.DataFrame, summary_tables: dict[str, pd.DataFrame], notes: pd.DataFrame) -> None:
''',
    "hedge workbook validation signature",
)
s = replace_once(
    s,
    '''        (TRAIN_SHEET, train_data),
        (TEST_SHEET, test_data),
''',
    '''        (TRAIN_SHEET, train_data),
        (VALIDATION_SHEET, validation_data),
        (TEST_SHEET, test_data),
''',
    "hedge workbook validation sheet",
)
s = replace_once(
    s,
    '        options = filter_options_for_run(combine_option_sheets(workbook), cfg)\n',
    '        options = filter_options_for_run(combine_option_sheets(workbook, splits=cfg.evaluation_splits), cfg)\n',
    "hedging default evaluation split",
)
s = replace_once(
    s,
    '''        train_data, test_data = prepare_output_option_sheets(options, workbook)
        notes = make_notes(cfg, interval_panel, perp, funding, used_perp_file, used_funding_file)
''',
    '''        train_data, validation_data, test_data = prepare_output_option_sheets(options, workbook)
        notes = make_notes(cfg, interval_panel, perp, funding, used_perp_file, used_funding_file)
''',
    "prepare validation output",
)
s = replace_once(
    s,
    '        write_hedge_workbook(cfg.output_xlsx, workbook, train_data, test_data, price_verification, interval_panel, summary_tables, notes)\n',
    '        write_hedge_workbook(cfg.output_xlsx, workbook, train_data, validation_data, test_data, price_verification, interval_panel, summary_tables, notes)\n',
    "write validation output",
)
s = replace_once(
    s,
    '''            TRAIN_SHEET: train_data,
            TEST_SHEET: test_data,
''',
    '''            TRAIN_SHEET: train_data,
            VALIDATION_SHEET: validation_data,
            TEST_SHEET: test_data,
''',
    "return validation output",
)
s = replace_once(
    s,
    '        ("currencies", None if cfg.currencies is None else ",".join(cfg.currencies)),\n',
    '        ("currencies", None if cfg.currencies is None else ",".join(cfg.currencies)),\n        ("evaluation_splits", ",".join(cfg.evaluation_splits)),\n',
    "hedge notes split",
)
p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# Notebook analysis helpers
# ---------------------------------------------------------------------------
p = ROOT / "notebooks/_lib/chapter3_analysis.py"
s = p.read_text(encoding="utf-8")
s = replace_once(s, '    train_data: pd.DataFrame\n    test_data: pd.DataFrame\n', '    train_data: pd.DataFrame\n    validation_data: pd.DataFrame\n    test_data: pd.DataFrame\n', "AnalysisState validation")
pattern = r'''def load_workbook\(path: Path\) -> tuple\[pd\.DataFrame, pd\.DataFrame, pd\.DataFrame, pd\.DataFrame, pd\.DataFrame\]:\n.*?    return black_params, heston_params, svcj_params, train_data, test_data\n'''
replacement = '''def load_workbook(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    black_params = pd.read_excel(path, sheet_name="black_params")
    heston_params = pd.read_excel(path, sheet_name="heston_params")
    svcj_params = pd.read_excel(path, sheet_name="svcj_params")
    train_data = pd.read_excel(path, sheet_name="train_data")
    with pd.ExcelFile(path, engine="openpyxl") as xls:
        validation_data = pd.read_excel(xls, sheet_name="validation_data") if "validation_data" in xls.sheet_names else pd.DataFrame(columns=train_data.columns)
        test_data = pd.read_excel(xls, sheet_name="test_data")

    for frame in (black_params, heston_params, svcj_params):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    for frame in (train_data, validation_data, test_data):
        if "snapshot_ts" in frame.columns:
            frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True)
        if "expiry_datetime" in frame.columns:
            frame["expiry_datetime"] = pd.to_datetime(frame["expiry_datetime"], utc=True)

    return black_params, heston_params, svcj_params, train_data, validation_data, test_data
'''
s = regex_once(s, pattern, replacement, "chapter3 load validation")
s = replace_once(
    s,
    '''        "rmse_train",
        "mae_train",
        "rmse_test",
        "mae_test",
        "n_options_total",
        "n_train",
        "n_test",
        "random_seed",
    ]
    keep = base_cols + param_cols
    out = df[keep].copy()
''',
    '''        "rmse_train",
        "mae_train",
        "rmse_validation",
        "mae_validation",
        "rmse_test",
        "mae_test",
        "n_options_total",
        "n_train",
        "n_validation",
        "n_test",
        "random_seed",
    ]
    work = df.copy()
    for col in base_cols:
        if col not in work.columns:
            work[col] = np.nan
    keep = base_cols + param_cols
    out = work[keep].copy()
''',
    "chapter3 parameter validation fields",
)
s = replace_once(
    s,
    '''    black_params, heston_params, svcj_params, train_data, test_data = load_workbook(data_path)
''',
    '''    black_params, heston_params, svcj_params, train_data, validation_data, test_data = load_workbook(data_path)
''',
    "build state validation load",
)
s = replace_once(
    s,
    '''            compute_snapshot_metrics_from_quotes(train_data, "train"),
            compute_snapshot_metrics_from_quotes(test_data, "test"),
''',
    '''            compute_snapshot_metrics_from_quotes(train_data, "train"),
            compute_snapshot_metrics_from_quotes(validation_data, "validation"),
            compute_snapshot_metrics_from_quotes(test_data, "test"),
''',
    "chapter3 validation metrics",
)
s = replace_once(s, '        train_data=train_data,\n        test_data=test_data,\n', '        train_data=train_data,\n        validation_data=validation_data,\n        test_data=test_data,\n', "AnalysisState return validation")
s = s.replace('for split in ["train", "test"]:', 'for split in ["train", "validation", "test"]:')
p.write_text(s, encoding="utf-8")

p = ROOT / "notebooks/_lib/chapter3_final.py"
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    'def load_baseline_workbook(baseline_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:\n',
    'def load_baseline_workbook(baseline_path: Path, *, evaluation_split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:\n',
    "final explicit split signature",
)
s = replace_once(
    s,
    '''    test_data = pd.read_excel(baseline_path, sheet_name="test_data", usecols=test_cols)
    test_data["snapshot_ts"] = pd.to_datetime(test_data["snapshot_ts"], utc=True)
''',
    '''    evaluation_split = str(evaluation_split).lower()
    if evaluation_split not in {"train", "validation", "test"}:
        raise ValueError("evaluation_split must be train, validation, or test")
    # Final headline reporting should leave this at the default untouched test split.
    test_data = pd.read_excel(baseline_path, sheet_name=f"{evaluation_split}_data", usecols=test_cols)
    test_data["snapshot_ts"] = pd.to_datetime(test_data["snapshot_ts"], utc=True)
    test_data["evaluation_split"] = evaluation_split
''',
    "final explicit evaluation split",
)
p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------------------
# Notebooks: update sources only, clear stale outputs.
# ---------------------------------------------------------------------------
p = ROOT / "notebooks/calibrate_all_to_excel.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
for cell in nb.get("cells", []):
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown":
        source = source.replace("- `train_data`, `test_data`: the option rows used, each with model price columns", "- `train_data`, `validation_data`, `test_data`: deterministic 60/20/20 option rows, each with model price columns")
        cell["source"] = source.splitlines(keepends=True)
    elif cell.get("cell_type") == "code":
        source = source.replace("    train_frac=0.70,\n", "    train_frac=0.60,\n    validation_frac=0.20,\n    test_frac=0.20,\n")
        cell["source"] = source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

p = ROOT / "calibration_example.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
for cell in nb.get("cells", []):
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown":
        source = source.replace("3. Splitting into train/test", "3. Deterministic 60/20/20 train/validation/test splitting")
        source = source.replace("5. Repricing train/test and plotting **true vs model** prices", "5. Repricing train/validation/test and plotting **true vs model** prices")
        cell["source"] = source.splitlines(keepends=True)
    elif cell.get("cell_type") == "code":
        if "from src.inverse_fft_pricer import FFTParams" in source and "src.splitting" not in source:
            source = source.replace("from src.inverse_fft_pricer import FFTParams\n", "from src.inverse_fft_pricer import FFTParams\nfrom src.splitting import SplitConfig, assign_cross_sectional_split\n")
        if "# --- Train/test split (random) ---" in source:
            source = '''# --- Deterministic train/validation/test split ---\nsplit_cfg = SplitConfig(train_frac=0.60, validation_frac=0.20, test_frac=0.20, random_seed=42)\nsplit = assign_cross_sectional_split(df, split_cfg)\ntrain = df.loc[split == "train"].reset_index(drop=True)\nvalidation = df.loc[split == "validation"].reset_index(drop=True)\ntest = df.loc[split == "test"].reset_index(drop=True)\n\ntrain.shape, validation.shape, test.shape\n'''
        if "# --- Reprice TRAIN and TEST sets with calibrated params ---" in source:
            source = source.replace("# --- Reprice TRAIN and TEST sets with calibrated params ---", "# --- Reprice TRAIN, VALIDATION, and TEST sets with calibrated params ---")
            source = source.replace("train_p = train.copy()\ntest_p  = test.copy()", "train_p = train.copy()\nvalidation_p = validation.copy()\ntest_p  = test.copy()")
            source = source.replace("    train_p = add_preds(train_p, fit, m)\n    test_p  = add_preds(test_p, fit, m)", "    train_p = add_preds(train_p, fit, m)\n    validation_p = add_preds(validation_p, fit, m)\n    test_p  = add_preds(test_p, fit, m)")
        if "# --- Fit diagnostics on train/test ---" in source:
            source = source.replace("# --- Fit diagnostics on train/test ---", "# --- Fit diagnostics on train/validation/test ---")
            source = source.replace("    te = metrics(test_p,  f\"model_price_{m}\")\n    print(f\"{m:6s}  train RMSE={tr[0]:.6f}, MAE={tr[1]:.6f}   |   test RMSE={te[0]:.6f}, MAE={te[1]:.6f}\")", "    va = metrics(validation_p, f\"model_price_{m}\")\n    te = metrics(test_p,  f\"model_price_{m}\")\n    print(f\"{m:6s}  train RMSE={tr[0]:.6f}, MAE={tr[1]:.6f}   |   validation RMSE={va[0]:.6f}, MAE={va[1]:.6f}   |   test RMSE={te[0]:.6f}, MAE={te[1]:.6f}\")")
        cell["source"] = source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------
p = ROOT / "docs/code_documentation.md"
s = p.read_text(encoding="utf-8")
if "### Deterministic train/validation/test holdout" not in s:
    s += textwrap.dedent('''

### Deterministic train/validation/test holdout

Each filtered snapshot is assigned once to a deterministic 60/20/20
cross-sectional split through `SplitConfig` and
`assign_cross_sectional_split(...)`. Assignment is stratified by
`(expiry_datetime, option_type)` when the stratum is large enough; rows from
one- and two-observation strata use a deterministic pooled fallback. Every row
appears in exactly one split and the train set is always nonempty.

All models calibrate on `train` only and then price the full frozen snapshot.
Workbooks now contain `train_data`, `validation_data`, and `test_data`, together
with separate parameter-row RMSE/MAE and counts for each split. Legacy two-way
workbooks remain loadable for historical comparison with an empty validation
sheet. The hedging pipeline defaults to `test_data` only for headline evidence;
other splits must be requested explicitly through `evaluation_splits` for
diagnostics.
''')
p.write_text(s, encoding="utf-8")
