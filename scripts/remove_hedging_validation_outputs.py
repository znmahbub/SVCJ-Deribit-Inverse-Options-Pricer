from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "src" / "hedging_analysis.py"
text = PATH.read_text(encoding="utf-8")

replacements = {
    ''') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:\n    enriched = options.copy()''': ''') -> tuple[pd.DataFrame, pd.DataFrame]:\n    enriched = options.copy()''',
    '''    return (\n        one("train", TRAIN_SHEET),\n        one("validation", VALIDATION_SHEET),\n        one("test", TEST_SHEET),\n    )''': '''    return (\n        one("train", TRAIN_SHEET),\n        one("test", TEST_SHEET),\n    )''',
    '''def write_hedge_workbook(output_path: Path, workbook: dict[str, pd.DataFrame], train_data: pd.DataFrame, validation_data: pd.DataFrame, test_data: pd.DataFrame, price_verification: pd.DataFrame, interval_panel: pd.DataFrame, summary_tables: dict[str, pd.DataFrame], notes: pd.DataFrame) -> None:''': '''def write_hedge_workbook(output_path: Path, workbook: dict[str, pd.DataFrame], train_data: pd.DataFrame, test_data: pd.DataFrame, price_verification: pd.DataFrame, interval_panel: pd.DataFrame, summary_tables: dict[str, pd.DataFrame], notes: pd.DataFrame) -> None:''',
    '''        (TRAIN_SHEET, train_data),\n        (VALIDATION_SHEET, validation_data),\n        (TEST_SHEET, test_data),''': '''        (TRAIN_SHEET, train_data),\n        (TEST_SHEET, test_data),''',
    '''        train_data, validation_data, test_data = prepare_output_option_sheets(options, workbook)''': '''        train_data, test_data = prepare_output_option_sheets(options, workbook)''',
    '''        write_hedge_workbook(cfg.output_xlsx, workbook, train_data, validation_data, test_data, price_verification, interval_panel, summary_tables, notes)''': '''        write_hedge_workbook(cfg.output_xlsx, workbook, train_data, test_data, price_verification, interval_panel, summary_tables, notes)''',
    '''            TRAIN_SHEET: train_data,\n            VALIDATION_SHEET: validation_data,\n            TEST_SHEET: test_data,''': '''            TRAIN_SHEET: train_data,\n            TEST_SHEET: test_data,''',
}

for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match, got {count}: {old[:100]!r}")
    text = text.replace(old, new)

PATH.write_text(text, encoding="utf-8")
Path(__file__).unlink()
