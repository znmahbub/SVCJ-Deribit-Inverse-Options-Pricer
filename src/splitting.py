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
