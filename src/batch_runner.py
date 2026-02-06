"""src.batch_runner

Batch calibration pipeline.

This module orchestrates:

- enumerating snapshot CSV files
- optional resume-from-Excel
- multithreaded processing with warm-start
- ordered (timestamp) committing to the workbook
- periodic, atomic flushing to disk

The implementation is intentionally conservative about correctness:

- Worker threads may finish out-of-order, but the main thread *commits* in
  chronological order (by snapshot timestamp). This keeps resume logic safe.
- Skip/failure cases still produce *parameter rows* ("Option A"), so resume
  does not retry the same snapshot forever.
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .calibration import WeightConfig
from .inverse_fft_pricer import FFTParams
from .results_store import (
    PARAM_SHEET_BLACK,
    PARAM_SHEET_HESTON,
    PARAM_SHEET_SVCJ,
    TEST_SHEET,
    TRAIN_SHEET,
    append_df,
    flush_workbook_atomic,
    get_latest_processed_timestamp,
    init_empty_workbook,
    latest_successful_params_before,
    load_existing_workbook,
)
from .snapshot_job import process_snapshot_to_payload, timestamp_from_filename


try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _default_n_workers() -> int:
    return max(1, int((os.cpu_count() or 4) - 2))


@dataclass
class BatchConfig:
    """Configuration for :func:`run_all_snapshots_to_excel`.

    The defaults are chosen to match the behaviour of the original
    ``calibrate_all_to_excel.ipynb`` notebook.
    """

    # Paths
    project_root: Path
    data_dir: Optional[Path] = None
    output_xlsx: Optional[Path] = None

    # Resume / saving
    resume: bool = True
    save_every_n_files: int = 3
    smoke_test_max_files_per_currency: Optional[int] = None

    # Filtering
    filter_rules: Dict[str, Any] = field(
        default_factory=lambda: dict(
            require_bid_ask=True,
            min_time_to_maturity=1 / 365,
            max_time_to_maturity=None,
            min_open_interest=1.0,
            min_vega=0.0,
            max_rel_spread=0.50,
            moneyness_range=(0.5, 2.0),
            drop_synthetic_underlyings=False,
        )
    )
    min_options_after_filter: int = 50

    # Calibration objective weights
    weight_config: WeightConfig = WeightConfig(
        use_spread=True,
        use_vega=False,
        use_open_interest=False,
        spread_power=1.0,
        vega_power=0.5,
        oi_power=0.5,
        eps_spread=1e-6,
        eps_other=1e-12,
        cap=1e6,
    )

    # FFT base (per-expiry b is computed per snapshot; so dynamic_b=False in pricing/calibration)
    fft_base: FFTParams = FFTParams(N=2 ** 12, eta=0.10, alpha=1.5, b=-10.0, use_simpson=True)

    # Calibration knobs
    train_frac: float = 0.70
    global_random_seed: int = 123
    max_nfev: Dict[str, int] = field(default_factory=lambda: dict(black=200, heston=200, svcj=200))

    # Runtime throttles
    runtime_top_expiries_by_oi: Optional[int] = None
    runtime_max_options: Optional[int] = None

    # Parallelism
    n_workers: int = field(default_factory=_default_n_workers)
    limit_internal_threads: bool = True
    internal_num_threads: int = 1

    # Currencies
    currencies: tuple[str, ...] = ("BTC", "ETH")

    # Logging
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = Path(self.project_root) / "data"
        if self.output_xlsx is None:
            self.output_xlsx = Path(self.project_root) / "calibration_results.xlsx"
        self.project_root = Path(self.project_root)
        self.data_dir = Path(self.data_dir)
        self.output_xlsx = Path(self.output_xlsx)
        self.save_every_n_files = int(self.save_every_n_files)
        self.n_workers = int(self.n_workers)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def list_snapshot_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("deribit_options_snapshot_*.csv"))
    files = [f for f in files if not f.name.startswith("._")]  # macOS metadata
    return files


def _make_chunks(seq: list[Path], k: int) -> list[tuple[int, list[Path]]]:
    """Contiguous chunks to preserve time ordering within each worker."""

    n = len(seq)
    chunks: list[tuple[int, list[Path]]] = []
    for w in range(k):
        start = (w * n) // k
        end = ((w + 1) * n) // k
        if start < end:
            chunks.append((start, seq[start:end]))
    return chunks


# ---------------------------------------------------------------------------
# Warm start helpers
# ---------------------------------------------------------------------------


_REQUIRED_PARAM_COLS: dict[str, list[str]] = {
    "black": ["sigma"],
    "heston": ["kappa", "theta", "sigma_v", "rho", "v0"],
    "svcj": ["kappa", "theta", "sigma_v", "rho", "v0", "lam", "ell_y", "sigma_y", "ell_v", "rho_j"],
}


def _chunk_warm_start_from_workbook(
    wb: dict[str, pd.DataFrame], *, currency: str, ts0: pd.Timestamp
) -> dict[str, dict[str, float]]:
    """Build warm-start params for the first file in a worker chunk."""

    warm: dict[str, dict[str, float]] = {}

    b = latest_successful_params_before(
        wb.get(PARAM_SHEET_BLACK, pd.DataFrame()),
        currency=currency,
        ts0=ts0,
        required_cols=_REQUIRED_PARAM_COLS["black"],
    )
    if b:
        warm["black"] = b

    h = latest_successful_params_before(
        wb.get(PARAM_SHEET_HESTON, pd.DataFrame()),
        currency=currency,
        ts0=ts0,
        required_cols=_REQUIRED_PARAM_COLS["heston"],
    )
    if h:
        warm["heston"] = h

    s = latest_successful_params_before(
        wb.get(PARAM_SHEET_SVCJ, pd.DataFrame()),
        currency=currency,
        ts0=ts0,
        required_cols=_REQUIRED_PARAM_COLS["svcj"],
    )
    if s:
        warm["svcj"] = s

    return warm


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_currency_to_excel(
    currency: str,
    *,
    cfg: BatchConfig,
    workbook: Optional[dict[str, pd.DataFrame]] = None,
) -> dict[str, pd.DataFrame]:
    """Run the pipeline for one currency and return the updated workbook."""

    currency = str(currency).upper()
    if workbook is None:
        workbook = load_existing_workbook(cfg.output_xlsx) if cfg.resume else init_empty_workbook()

    all_files = list_snapshot_files(cfg.data_dir)
    if not all_files:
        raise RuntimeError(f"No snapshot files found in {cfg.data_dir}")

    all_files_sorted = sorted(all_files, key=timestamp_from_filename)
    file_index_map = {p: i for i, p in enumerate(all_files_sorted)}
    currency_index = cfg.currencies.index(currency) if currency in cfg.currencies else 0

    # ---- Resume logic
    pending = all_files_sorted
    if cfg.resume:
        last_ts = get_latest_processed_timestamp(workbook, currency)
        if last_ts is not None:
            pending = [p for p in all_files_sorted if timestamp_from_filename(p) > last_ts]

    if cfg.smoke_test_max_files_per_currency is not None:
        pending = pending[: int(cfg.smoke_test_max_files_per_currency)]

    if not pending:
        if cfg.verbose:
            print(f"[{currency}] Nothing to do (resume found up-to-date workbook).")
        return workbook

    n_workers = max(1, min(int(cfg.n_workers), len(pending)))

    if cfg.verbose:
        print(f"[{currency}] Processing {len(pending)} snapshots using {n_workers} workers")

    out_q: queue.Queue = queue.Queue()

    chunks = _make_chunks(pending, n_workers)

    def _exception_payload(path: Path, *, seed: int, message: str, warm: Optional[dict[str, dict[str, float]]]) -> dict[str, Any]:
        ts = timestamp_from_filename(path)
        ts_iso = pd.to_datetime(ts, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")

        base = dict(
            timestamp=ts_iso,
            currency=currency,
            success=False,
            message=str(message),
            nfev=0,
            rmse_fit=float("nan"),
            mae_fit=float("nan"),
            rmse_train=float("nan"),
            mae_train=float("nan"),
            rmse_test=float("nan"),
            mae_test=float("nan"),
            n_options_total=0,
            n_train=0,
            n_test=0,
            random_seed=int(seed),
        )

        black_row = pd.DataFrame([dict(base, sigma=float("nan"))])
        heston_row = pd.DataFrame([dict(base, kappa=float("nan"), theta=float("nan"), sigma_v=float("nan"), rho=float("nan"), v0=float("nan"))])
        svcj_row = pd.DataFrame(
            [
                dict(
                    base,
                    kappa=float("nan"),
                    theta=float("nan"),
                    sigma_v=float("nan"),
                    rho=float("nan"),
                    v0=float("nan"),
                    lam=float("nan"),
                    ell_y=float("nan"),
                    sigma_y=float("nan"),
                    ell_v=float("nan"),
                    rho_j=float("nan"),
                )
            ]
        )

        return {
            "timestamp_iso": ts_iso,
            "timestamp": ts,
            "currency": currency,
            "param_rows": {"black": black_row, "heston": heston_row, "svcj": svcj_row},
            "train_df": pd.DataFrame(),
            "test_df": pd.DataFrame(),
            "warm_next": warm or {},
        }

    def _run_chunk(start_idx: int, chunk_files: list[Path]):
        """Run a contiguous chunk. Always emits a DONE message."""

        try:
            warm: Optional[dict[str, dict[str, float]]] = None
            if chunk_files:
                ts0 = timestamp_from_filename(chunk_files[0])
                warm = _chunk_warm_start_from_workbook(workbook, currency=currency, ts0=ts0)

            ctx = contextlib.nullcontext()
            if cfg.limit_internal_threads and (threadpool_limits is not None):
                ctx = threadpool_limits(limits=int(cfg.internal_num_threads))

            with ctx:
                for j, path in enumerate(chunk_files):
                    i = start_idx + j
                    seed = int(cfg.global_random_seed + 10_000 * currency_index + file_index_map[path])
                    try:
                        payload = process_snapshot_to_payload(
                            path,
                            currency=currency,
                            filter_rules=cfg.filter_rules,
                            weight_config=cfg.weight_config,
                            fft_base=cfg.fft_base,
                            max_nfev=cfg.max_nfev,
                            train_frac=cfg.train_frac,
                            random_seed=seed,
                            runtime_top_expiries_by_oi=cfg.runtime_top_expiries_by_oi,
                            runtime_max_options=cfg.runtime_max_options,
                            min_options_after_filter=cfg.min_options_after_filter,
                            warm_start=warm,
                            verbose=cfg.verbose,
                        )
                    except Exception as e:  # defensive: never drop an index
                        payload = _exception_payload(path, seed=seed, message=f"Unhandled exception: {repr(e)}", warm=warm)

                    warm = payload.get("warm_next") or warm
                    out_q.put((i, payload))
        finally:
            out_q.put(("DONE", start_idx))

    # Launch workers
    threads: list[threading.Thread] = []
    for start_idx, chunk_files in chunks:
        t = threading.Thread(target=_run_chunk, args=(start_idx, chunk_files), daemon=True)
        t.start()
        threads.append(t)

    # Ordered commit loop
    pending_payloads: dict[int, dict[str, Any]] = {}
    next_commit = 0
    committed_since_flush = 0
    done_workers = 0

    total = len(pending)
    while done_workers < len(threads):
        msg = out_q.get()
        if msg[0] == "DONE":
            done_workers += 1
            continue

        idx, payload = msg
        pending_payloads[int(idx)] = payload

        while next_commit in pending_payloads:
            pld = pending_payloads.pop(next_commit)

            # Append to workbook
            param_rows = pld.get("param_rows", {})
            if "black" in param_rows:
                append_df(workbook, PARAM_SHEET_BLACK, param_rows["black"])
            if "heston" in param_rows:
                append_df(workbook, PARAM_SHEET_HESTON, param_rows["heston"])
            if "svcj" in param_rows:
                append_df(workbook, PARAM_SHEET_SVCJ, param_rows["svcj"])

            append_df(workbook, TRAIN_SHEET, pld.get("train_df", pd.DataFrame()))
            append_df(workbook, TEST_SHEET, pld.get("test_df", pd.DataFrame()))

            next_commit += 1
            committed_since_flush += 1

            if cfg.verbose:
                ts_iso = pld.get("timestamp_iso", "?")
                print(f"[{currency}] committed {next_commit}/{total} | {ts_iso}")

            if committed_since_flush >= int(cfg.save_every_n_files):
                flush_workbook_atomic(workbook, cfg.output_xlsx)
                committed_since_flush = 0

    # Ensure worker threads finished
    for t in threads:
        t.join(timeout=0.1)

    # Final flush
    flush_workbook_atomic(workbook, cfg.output_xlsx)

    return workbook


def run_all_snapshots_to_excel(cfg: BatchConfig) -> dict[str, pd.DataFrame]:
    """Run all currencies sequentially (BTC then ETH by default)."""

    workbook = load_existing_workbook(cfg.output_xlsx) if cfg.resume else None
    for ccy in cfg.currencies:
        workbook = run_currency_to_excel(ccy, cfg=cfg, workbook=workbook)
    return workbook


__all__ = [
    "BatchConfig",
    "list_snapshot_files",
    "run_currency_to_excel",
    "run_all_snapshots_to_excel",
]
