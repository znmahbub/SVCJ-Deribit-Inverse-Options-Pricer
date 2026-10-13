from __future__ import annotations

import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


p = ROOT / "src/batch_runner.py"
s = p.read_text(encoding="utf-8")
s = replace_once(
    s,
    '''- multithreaded processing with warm-start
- ordered (timestamp) committing to the workbook
''',
    '''- chronological warm-start processing for temporally regularized runs
- optional multithreaded processing for unregularized runs
- ordered (timestamp) committing to the workbook
''',
    "batch runner overview",
)
s = replace_once(
    s,
    '''    n_workers = max(1, min(int(cfg.n_workers), len(pending)))

    if cfg.verbose:
        print(f"[{currency}] Processing {len(pending)} snapshots using {n_workers} workers")
''',
    '''    requested_workers = max(1, min(int(cfg.n_workers), len(pending)))
    # Temporal L2 regularization creates a true dependency chain: snapshot t+1
    # must receive the successful parameters from t. Chunk-level warm starts are
    # not sufficient because independent chunks break that chain.
    n_workers = 1 if float(cfg.l2_prev_strength) > 0.0 else requested_workers

    if cfg.verbose:
        suffix = " (forced sequential by temporal regularization)" if n_workers == 1 and requested_workers > 1 and float(cfg.l2_prev_strength) > 0.0 else ""
        print(f"[{currency}] Processing {len(pending)} snapshots using {n_workers} workers{suffix}")
''',
    "force sequential regularization",
)
p.write_text(s, encoding="utf-8")

# Update the execution notebook without changing cell order/IDs.
p = ROOT / "notebooks/calibrate_all_to_excel.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
for cell in nb.get("cells", []):
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and "Key features:" in source:
        source = source.replace(
            "- **Warm-start**: each worker chunk warm-starts from the most recent successful params prior to the chunk start.\n- **Threading**: worker threads may finish out-of-order, but commits are written in chronological order.\n",
            "- **Sequential temporal regularization**: when `l2_prev_strength > 0`, snapshots for each currency are processed chronologically with one worker so snapshot `t+1` receives snapshot `t`'s successful parameters.\n- **Threading for unregularized runs**: `n_workers` is used only when `l2_prev_strength == 0`; commits remain chronological.\n",
        )
        cell["source"] = source.splitlines(keepends=True)
    if cell.get("cell_type") == "code" and "l2_prev_strength=1000.0" in source and "n_workers=" in source:
        source = source.replace(
            "    n_workers=int(os.cpu_count() - 2),\n",
            "    # Ignored for this regularized run: temporal chaining forces one worker per currency.\n    n_workers=max(1, int((os.cpu_count() or 4) - 2)),\n",
        )
        cell["source"] = source.splitlines(keepends=True)
p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

p = ROOT / "docs/code_documentation.md"
s = p.read_text(encoding="utf-8")
if "### Sequential temporal regularization" not in s:
    s += textwrap.dedent('''

### Sequential temporal regularization

A positive `BatchConfig.l2_prev_strength` creates a chronological dependency
within each currency. `run_currency_to_excel(...)` therefore forces one worker
for that currency whenever temporal regularization is active, regardless of the
requested `n_workers`. After a successful snapshot, its calibrated parameters
become the warm start and transformed-space L2 anchor for the next snapshot.
Resume reconstructs the anchor from the latest successful parameter row before
the first pending timestamp. Parallel chunk processing remains available only
when `l2_prev_strength == 0`.
''')
p.write_text(s, encoding="utf-8")
