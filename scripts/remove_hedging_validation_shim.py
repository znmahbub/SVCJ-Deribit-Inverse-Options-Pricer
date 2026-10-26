from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src" / "hedging_analysis.py"

text = PATH.read_text(encoding="utf-8")

replacements = {
    '''    TEST_SHEET,\n    TRAIN_SHEET,\n    VALIDATION_SHEET,\n)''': '''    TEST_SHEET,\n    TRAIN_SHEET,\n)''',
    '''    needed = [*PARAM_SHEETS.values(), TRAIN_SHEET, VALIDATION_SHEET, TEST_SHEET]\n    out: dict[str, pd.DataFrame] = {}\n    for sheet in needed:\n        if sheet in sheets:\n            out[sheet] = sheets[sheet].copy()\n        elif sheet == VALIDATION_SHEET:\n            out[sheet] = pd.DataFrame()\n        else:\n            raise ValueError(f"Calibration workbook missing required sheet: {sheet}")''': '''    needed = [*PARAM_SHEETS.values(), TRAIN_SHEET, TEST_SHEET]\n    out: dict[str, pd.DataFrame] = {}\n    for sheet in needed:\n        if sheet not in sheets:\n            raise ValueError(f"Calibration workbook missing required sheet: {sheet}")\n        out[sheet] = sheets[sheet].copy()''',
    '''    sheet_for = {"train": TRAIN_SHEET, "validation": VALIDATION_SHEET, "test": TEST_SHEET}''': '''    sheet_for = {"train": TRAIN_SHEET, "test": TEST_SHEET}''',
}

for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match, got {count}: {old[:80]!r}")
    text = text.replace(old, new)

PATH.write_text(text, encoding="utf-8")
Path(__file__).unlink()
