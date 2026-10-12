"""Utilities for freezing analyses to an explicit UTC snapshot window.

The live ``data/`` directory is continuously extended by the collection
workflows, so thesis/research reruns must never rely on "all files currently
present" as the empirical sample definition.  This module provides the small,
shared normalization/filtering primitives used by batch jobs.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re
from typing import Callable, Iterable, TypeVar

import pandas as pd


TimestampLike = str | date | datetime | pd.Timestamp
T = TypeVar("T")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_date_only(value: TimestampLike) -> bool:
    if isinstance(value, str):
        return bool(_DATE_ONLY_RE.fullmatch(value.strip()))
    return isinstance(value, date) and not isinstance(value, (datetime, pd.Timestamp))


def normalize_utc_boundary(
    value: TimestampLike | None,
    *,
    is_end: bool = False,
) -> pd.Timestamp | None:
    """Normalize an analysis-window boundary to an aware UTC timestamp.

    Date-only end values are interpreted as inclusive calendar dates.  Thus
    ``analysis_end="2026-04-08"`` becomes the final nanosecond of 8 April UTC,
    rather than midnight at the start of the day.
    """

    if value is None:
        return None

    date_only = _is_date_only(value)
    ts = pd.to_datetime(value, utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid analysis-window boundary: {value!r}")
    ts = pd.Timestamp(ts)

    if is_end and date_only:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def normalize_analysis_window(
    start: TimestampLike | None,
    end: TimestampLike | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return validated inclusive UTC start/end boundaries."""

    start_ts = normalize_utc_boundary(start, is_end=False)
    end_ts = normalize_utc_boundary(end, is_end=True)
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(
            f"analysis_start ({start_ts}) must be <= analysis_end ({end_ts})"
        )
    return start_ts, end_ts


def timestamp_in_window(
    timestamp: TimestampLike,
    *,
    start: TimestampLike | None = None,
    end: TimestampLike | None = None,
) -> bool:
    """Return whether ``timestamp`` lies in the inclusive analysis window."""

    start_ts, end_ts = normalize_analysis_window(start, end)
    ts = pd.Timestamp(pd.to_datetime(timestamp, utc=True))
    if start_ts is not None and ts < start_ts:
        return False
    if end_ts is not None and ts > end_ts:
        return False
    return True


def filter_items_by_timestamp(
    items: Iterable[T],
    timestamp_fn: Callable[[T], TimestampLike],
    *,
    start: TimestampLike | None = None,
    end: TimestampLike | None = None,
) -> list[T]:
    """Filter arbitrary items by an extracted timestamp, preserving order."""

    start_ts, end_ts = normalize_analysis_window(start, end)
    out: list[T] = []
    for item in items:
        ts = pd.Timestamp(pd.to_datetime(timestamp_fn(item), utc=True))
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        out.append(item)
    return out


def filter_paths_by_timestamp(
    paths: Iterable[Path],
    timestamp_fn: Callable[[Path], TimestampLike],
    *,
    start: TimestampLike | None = None,
    end: TimestampLike | None = None,
) -> list[Path]:
    """Convenience wrapper for snapshot ``Path`` objects."""

    return filter_items_by_timestamp(
        paths,
        timestamp_fn,
        start=start,
        end=end,
    )


__all__ = [
    "TimestampLike",
    "normalize_utc_boundary",
    "normalize_analysis_window",
    "timestamp_in_window",
    "filter_items_by_timestamp",
    "filter_paths_by_timestamp",
]
