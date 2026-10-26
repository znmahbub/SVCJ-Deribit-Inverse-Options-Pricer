"""src.batch_runner

Batch calibration pipeline.

This module orchestrates:

- enumerating snapshot CSV files
- optional resume-from-Excel
- chronological warm-start processing for temporally regularized runs
- optional multithreaded processing for unregularized runs
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
import multiprocessing as mp
import os
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
from tqdm.auto import tqdm

from .analysis_window import (
    TimestampLike,
    filter_paths_by_timestamp,
    normalize_analysis_window,
)
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
from .splitting import SplitConfig


try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


WLS_LAMBDA_TS = 0.0
PWLS_LAMBDA_TS = 1.0

_WORKBOOK_SHEETS = (
    PARAM_SHEET_BLACK,
    PARAM_SHEET_HESTON,
    PARAM_SHEET_SVCJ,
    TRAIN_SHEET,
    TEST_SHEET,
)

_PROGRESS_EVENT = tuple[str, str, int]
_PROCESS_PROGRESS_QUEUE = None


def _default_n_workers() -> int:
    return max(1, int((os.cpu_count() or 4) - 2))


@dataclass
class BatchConfig:
    """Configuration for :func:`run_all_snapshots_to_excel`.

    The defaults are chosen to match the behaviour of the original
    ``calibrate_all_to_excel.ipynb`` notebook.

    ``analysis_start`` and ``analysis_end`` freeze a paper/research run to an
    explicit inclusive UTC window before resume logic or smoke-test truncation
    is applied.  Date-only end values include the whole UTC calendar day.
    """

    # Paths
    project_root: Path
    data_dir: Optional[Path] = None
    output_xlsx: Optional[Path] = None

    # Frozen empirical sample (inclusive UTC boundaries)
    analysis_start: TimestampLike | None = None
    analysis_end: TimestampLike | None = None

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
    test_frac: float = 0.30
    global_random_seed: int = 123
    max_nfev: Dict[str, int] = field(default_factory=lambda: dict(black=200, heston=200, svcj=200))
    l2_prev_strength: float = 0.0

    # Runtime throttles
    runtime_top_expiries_by_oi: Optional[int] = None
    runtime_max_options: Optional[int] = None

    # Parallelism
    n_workers: int = field(default_factory=_default_n_workers)
    limit_internal_threads: bool = True
    internal_num_threads: int = 1

    # Currencies
    currencies: tuple[str, ...] = ("BTC", "ETH")

    # Progress / logging
    show_progress: bool = False
    progress_label: Optional[str] = None
    progress_position_offset: int = 0
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.data_dir is None:
            self.data_dir = Path(self.project_root) / "data"
        if self.output_xlsx is None:
            self.output_xlsx = Path(self.project_root) / "calibration_results.xlsx"
        self.project_root = Path(self.project_root)
        self.data_dir = Path(self.data_dir)
        self.output_xlsx = Path(self.output_xlsx)
        self.analysis_start, self.analysis_end = normalize_analysis_window(
            self.analysis_start,
            self.analysis_end,
        )
        self.save_every_n_files = int(self.save_every_n_files)
        self.n_workers = int(self.n_workers)
        self.progress_position_offset = int(self.progress_position_offset)
        # Validate the configured fractions once at construction time.
        SplitConfig(
            train_frac=float(self.train_frac),
            test_frac=float(self.test_frac),
            random_seed=int(self.global_random_seed),
        )


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


def _currency_checkpoint_dir(output_xlsx: Path, currency: str) -> Path:
    """Return the persistent Parquet checkpoint directory for one currency."""
    output_xlsx = Path(output_xlsx)
    return output_xlsx.parent / ".checkpoints" / output_xlsx.stem / str(currency).upper()


def _currency_only_workbook(
    workbook: dict[str, pd.DataFrame],
    currency: str,
) -> dict[str, pd.DataFrame]:
    """Copy only one currency from a canonical workbook payload."""
    currency = str(currency).upper()
    out = init_empty_workbook()
    for sheet in _WORKBOOK_SHEETS:
        frame = workbook.get(sheet, pd.DataFrame())
        if frame is None or frame.empty:
            out[sheet] = pd.DataFrame(columns=getattr(frame, "columns", None))
            continue
        if "currency" in frame.columns:
            mask = frame["currency"].astype(str).str.strip().str.upper() == currency
            out[sheet] = frame.loc[mask].copy().reset_index(drop=True)
        else:
            out[sheet] = frame.copy().reset_index(drop=True)
    return out


def _checkpoint_is_complete(checkpoint_dir: Path) -> bool:
    checkpoint_dir = Path(checkpoint_dir)
    return all((checkpoint_dir / f"{sheet}.parquet").exists() for sheet in _WORKBOOK_SHEETS)


def flush_currency_checkpoint(
    workbook: dict[str, pd.DataFrame],
    checkpoint_dir: Path,
) -> None:
    """Persist one currency workbook as lightweight per-sheet Parquet files."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for sheet in _WORKBOOK_SHEETS:
        target = checkpoint_dir / f"{sheet}.parquet"
        tmp = checkpoint_dir / f".{sheet}.parquet.tmp"
        frame = workbook.get(sheet, pd.DataFrame())
        frame.to_parquet(tmp, index=False, engine="pyarrow")
        os.replace(tmp, target)


def load_currency_checkpoint(checkpoint_dir: Path) -> dict[str, pd.DataFrame]:
    """Load a complete per-currency Parquet checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    if not _checkpoint_is_complete(checkpoint_dir):
        return init_empty_workbook()
    out = init_empty_workbook()
    for sheet in _WORKBOOK_SHEETS:
        out[sheet] = pd.read_parquet(checkpoint_dir / f"{sheet}.parquet", engine="pyarrow")
    return out


def _load_currency_resume_workbook(cfg: BatchConfig, currency: str) -> dict[str, pd.DataFrame]:
    """Prefer a currency checkpoint; otherwise seed resume state from final Excel."""
    if not cfg.resume:
        return init_empty_workbook()
    checkpoint_dir = _currency_checkpoint_dir(cfg.output_xlsx, currency)
    if _checkpoint_is_complete(checkpoint_dir):
        return load_currency_checkpoint(checkpoint_dir)
    if cfg.output_xlsx.exists():
        return _currency_only_workbook(load_existing_workbook(cfg.output_xlsx), currency)
    return init_empty_workbook()


def _merge_currency_workbooks(
    workbooks: dict[str, dict[str, pd.DataFrame]],
    currencies: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Merge independent per-currency workbooks in canonical currency order."""
    merged = init_empty_workbook()
    for sheet in _WORKBOOK_SHEETS:
        frames = [
            workbooks[str(currency).upper()].get(sheet, pd.DataFrame())
            for currency in currencies
            if str(currency).upper() in workbooks
        ]
        nonempty = [frame for frame in frames if frame is not None and len(frame)]
        if nonempty:
            merged[sheet] = pd.concat(nonempty, ignore_index=True, sort=False)
        elif frames:
            merged[sheet] = frames[0].iloc[0:0].copy()
    return merged


def _currency_process_count(cfg: BatchConfig, available_cpus: int | None = None) -> int:
    """Number of independent currency processes for a temporally regularized run."""
    if float(cfg.l2_prev_strength) <= 0.0:
        return 1
    cpus = max(1, int(available_cpus if available_cpus is not None else (os.cpu_count() or 1)))
    return max(1, min(len(cfg.currencies), cpus))


class _ProgressDisplay:
    """Own the fixed per-currency tqdm bars for one calibration specification."""

    def __init__(self, cfg: BatchConfig, currencies: tuple[str, ...]):
        self.enabled = bool(cfg.show_progress)
        self._bars: dict[str, Any] = {}
        if not self.enabled:
            return
        label = str(cfg.progress_label or cfg.output_xlsx.stem).strip()
        for index, currency in enumerate(currencies):
            currency = str(currency).upper()
            self._bars[currency] = tqdm(
                total=0,
                desc=f"{label} {currency}",
                position=int(cfg.progress_position_offset) + index,
                leave=True,
                unit="snapshot",
                dynamic_ncols=True,
            )

    def handle(self, event: _PROGRESS_EVENT) -> None:
        if not self.enabled:
            return
        kind, currency, value = event
        bar = self._bars.get(str(currency).upper())
        if bar is None:
            return
        if kind == "total":
            bar.reset(total=max(0, int(value)))
        elif kind == "advance":
            bar.update(int(value))

    def close(self) -> None:
        for bar in self._bars.values():
            bar.close()


def _init_currency_progress_queue(progress_queue) -> None:
    """Install the parent-owned progress queue in a spawned currency worker."""
    global _PROCESS_PROGRESS_QUEUE
    _PROCESS_PROGRESS_QUEUE = progress_queue


def _listen_for_progress(progress_queue, display: _ProgressDisplay) -> None:
    while True:
        event = progress_queue.get()
        if event is None:
            return
        display.handle(event)


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
    progress_callback: Optional[Callable[[_PROGRESS_EVENT], None]] = None,
) -> dict[str, pd.DataFrame]:
    """Run one currency and checkpoint it to Parquet; final Excel is written later."""
    currency = str(currency).upper()
    if workbook is None:
        workbook = _load_currency_resume_workbook(cfg, currency)

    all_files = list_snapshot_files(cfg.data_dir)
    if not all_files:
        raise RuntimeError(f"No snapshot files found in {cfg.data_dir}")

    all_files_sorted = sorted(all_files, key=timestamp_from_filename)
    all_files_sorted = filter_paths_by_timestamp(
        all_files_sorted,
        timestamp_from_filename,
        start=cfg.analysis_start,
        end=cfg.analysis_end,
    )
    if not all_files_sorted:
        raise RuntimeError(
            "No snapshot files fall inside the configured analysis window "
            f"[{cfg.analysis_start}, {cfg.analysis_end}]"
        )

    file_index_map = {p: i for i, p in enumerate(all_files_sorted)}
    currency_index = cfg.currencies.index(currency) if currency in cfg.currencies else 0
    checkpoint_dir = _currency_checkpoint_dir(cfg.output_xlsx, currency)

    pending = all_files_sorted
    if cfg.resume:
        last_ts = get_latest_processed_timestamp(workbook, currency)
        if last_ts is not None:
            pending = [p for p in all_files_sorted if timestamp_from_filename(p) > last_ts]

    if cfg.smoke_test_max_files_per_currency is not None:
        pending = pending[: int(cfg.smoke_test_max_files_per_currency)]

    if progress_callback is not None:
        progress_callback(("total", currency, len(pending)))

    log_verbose = bool(cfg.verbose and not cfg.show_progress)

    if not pending:
        flush_currency_checkpoint(_currency_only_workbook(workbook, currency), checkpoint_dir)
        if log_verbose:
            print(f"[{currency}] Nothing to do (resume found up-to-date workbook).")
        return workbook

    requested_workers = max(1, min(int(cfg.n_workers), len(pending)))
    n_workers = 1 if float(cfg.l2_prev_strength) > 0.0 else requested_workers

    if log_verbose:
        suffix = (
            " (forced sequential by temporal regularization)"
            if n_workers == 1 and requested_workers > 1 and float(cfg.l2_prev_strength) > 0.0
            else ""
        )
        print(f"[{currency}] Processing {len(pending)} snapshots using {n_workers} workers{suffix}")

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
        svcj_row = pd.DataFrame([
            dict(base, kappa=float("nan"), theta=float("nan"), sigma_v=float("nan"), rho=float("nan"), v0=float("nan"), lam=float("nan"), ell_y=float("nan"), sigma_y=float("nan"), ell_v=float("nan"), rho_j=float("nan"))
        ])
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
                            test_frac=cfg.test_frac,
                            random_seed=seed,
                            runtime_top_expiries_by_oi=cfg.runtime_top_expiries_by_oi,
                            runtime_max_options=cfg.runtime_max_options,
                            min_options_after_filter=cfg.min_options_after_filter,
                            warm_start=warm,
                            l2_prev_strength=cfg.l2_prev_strength,
                            verbose=log_verbose,
                        )
                    except Exception as e:
                        payload = _exception_payload(path, seed=seed, message=f"Unhandled exception: {repr(e)}", warm=warm)
                    warm = payload.get("warm_next") or warm
                    out_q.put((i, payload))
        finally:
            out_q.put(("DONE", start_idx))

    threads: list[threading.Thread] = []
    for start_idx, chunk_files in chunks:
        t = threading.Thread(target=_run_chunk, args=(start_idx, chunk_files), daemon=True)
        t.start()
        threads.append(t)

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
            if progress_callback is not None:
                progress_callback(("advance", currency, 1))
            if log_verbose:
                ts_iso = pld.get("timestamp_iso", "?")
                print(f"[{currency}] committed {next_commit}/{total} | {ts_iso}")
            if committed_since_flush >= int(cfg.save_every_n_files):
                flush_currency_checkpoint(_currency_only_workbook(workbook, currency), checkpoint_dir)
                committed_since_flush = 0

    for t in threads:
        t.join(timeout=0.1)

    flush_currency_checkpoint(_currency_only_workbook(workbook, currency), checkpoint_dir)
    return workbook


def _run_currency_checkpoint_worker(currency: str, cfg: BatchConfig) -> str:
    """Process one independent currency chain inside a child process."""
    workbook = _load_currency_resume_workbook(cfg, currency)
    progress_callback = None
    if _PROCESS_PROGRESS_QUEUE is not None:
        progress_callback = _PROCESS_PROGRESS_QUEUE.put
    run_currency_to_excel(
        currency,
        cfg=cfg,
        workbook=workbook,
        progress_callback=progress_callback,
    )
    return str(currency).upper()


def run_all_snapshots_to_excel(cfg: BatchConfig) -> dict[str, pd.DataFrame]:
    """Run all currencies, merge their checkpoints, and write final Excel once."""
    currencies = tuple(str(currency).upper() for currency in cfg.currencies)
    currency_workbooks: dict[str, dict[str, pd.DataFrame]] = {}
    process_count = _currency_process_count(cfg)
    display = _ProgressDisplay(cfg, currencies)

    try:
        if process_count > 1:
            ctx = mp.get_context("spawn")
            progress_queue = ctx.Queue() if display.enabled else None
            listener = None
            if progress_queue is not None:
                listener = threading.Thread(
                    target=_listen_for_progress,
                    args=(progress_queue, display),
                    daemon=True,
                )
                listener.start()

            try:
                executor_kwargs: dict[str, Any] = {
                    "max_workers": process_count,
                    "mp_context": ctx,
                }
                if progress_queue is not None:
                    executor_kwargs.update(
                        initializer=_init_currency_progress_queue,
                        initargs=(progress_queue,),
                    )
                with ProcessPoolExecutor(**executor_kwargs) as executor:
                    futures = {
                        executor.submit(_run_currency_checkpoint_worker, currency, cfg): currency
                        for currency in currencies
                    }
                    for future in as_completed(futures):
                        currency = futures[future]
                        future.result()
                        currency_workbooks[currency] = load_currency_checkpoint(
                            _currency_checkpoint_dir(cfg.output_xlsx, currency)
                        )
            finally:
                if progress_queue is not None:
                    progress_queue.put(None)
                    if listener is not None:
                        listener.join()
                    progress_queue.close()
                    progress_queue.join_thread()
        else:
            callback = display.handle if display.enabled else None
            for currency in currencies:
                workbook = _load_currency_resume_workbook(cfg, currency)
                workbook = run_currency_to_excel(
                    currency,
                    cfg=cfg,
                    workbook=workbook,
                    progress_callback=callback,
                )
                currency_workbooks[currency] = _currency_only_workbook(workbook, currency)

        merged = _merge_currency_workbooks(currency_workbooks, currencies)
        cfg.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
        flush_workbook_atomic(merged, cfg.output_xlsx)
        return merged
    finally:
        display.close()


__all__ = [
    "WLS_LAMBDA_TS",
    "PWLS_LAMBDA_TS",
    "BatchConfig",
    "list_snapshot_files",
    "flush_currency_checkpoint",
    "load_currency_checkpoint",
    "run_currency_to_excel",
    "run_all_snapshots_to_excel",
]
