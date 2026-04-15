from __future__ import annotations

import unittest
from pathlib import Path

import nbformat

from tests._path import ROOT
from tests.notebook_utils import load_notebook_reference


class NotebookIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = load_notebook_reference()

    def test_notebooks_parse_and_preserve_metadata(self) -> None:
        for rel_path, reference_snapshot in self.reference["notebooks"].items():
            with self.subTest(notebook=rel_path):
                nb = nbformat.read(ROOT / rel_path, as_version=4)
                self.assertEqual(nb.metadata.get("kernelspec", {}), reference_snapshot["metadata"]["kernelspec"])
                self.assertEqual(
                    nb.metadata.get("language_info", {}).get("name"),
                    reference_snapshot["metadata"]["language_info"]["name"],
                )

    def test_markdown_headers_are_preserved(self) -> None:
        for rel_path, reference_snapshot in self.reference["notebooks"].items():
            with self.subTest(notebook=rel_path):
                nb = nbformat.read(ROOT / rel_path, as_version=4)
                headers = []
                for cell in nb.cells:
                    if cell.cell_type != "markdown":
                        continue
                    text = "".join(cell.source)
                    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
                    if first_line.startswith("#"):
                        headers.append(first_line)
                self.assertEqual(headers, reference_snapshot["markdown_headers"])

    def test_cell_type_order_is_stable(self) -> None:
        for rel_path, reference_snapshot in self.reference["notebooks"].items():
            with self.subTest(notebook=rel_path):
                nb = nbformat.read(ROOT / rel_path, as_version=4)
                current_types = [cell.cell_type for cell in nb.cells]
                reference_types = [cell["cell_type"] for cell in reference_snapshot["cells"]]
                self.assertEqual(current_types, reference_types)


if __name__ == "__main__":
    unittest.main()
