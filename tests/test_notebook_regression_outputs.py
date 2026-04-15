from __future__ import annotations

import unittest

from tests._path import ROOT
from tests.generate_notebook_reference_outputs import snapshot_notebook
from tests.notebook_utils import load_notebook_reference, snapshot_artifact_tree


class NotebookRegressionOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = load_notebook_reference()

    def test_committed_notebook_outputs_match_reference(self) -> None:
        for rel_path, reference_snapshot in self.reference["notebooks"].items():
            with self.subTest(notebook=rel_path):
                current_snapshot = snapshot_notebook(ROOT / rel_path)
                self.assertEqual(current_snapshot["metadata"], reference_snapshot["metadata"])
                self.assertEqual(current_snapshot["markdown_headers"], reference_snapshot["markdown_headers"])
                self.assertEqual(self._plotly_signature(current_snapshot), self._plotly_signature(reference_snapshot))
                self.assertEqual(self._table_output_count(current_snapshot), self._table_output_count(reference_snapshot))

    def test_exported_artifacts_match_reference(self) -> None:
        for rel_dir, reference_artifacts in self.reference["artifacts"].items():
            with self.subTest(artifact_dir=rel_dir):
                current_artifacts = snapshot_artifact_tree(rel_dir)
                self.assertEqual(set(current_artifacts), set(reference_artifacts))
                for path, reference_artifact in reference_artifacts.items():
                    current_artifact = current_artifacts[path]
                    self.assertEqual(current_artifact["type"], reference_artifact["type"])
                    if reference_artifact["type"] == "csv":
                        self.assertEqual(current_artifact, reference_artifact)
                    else:
                        self.assertGreater(current_artifact["size"], 0)

    @staticmethod
    def _plotly_signature(snapshot: dict) -> list[tuple]:
        signature = []
        for cell in snapshot["cells"]:
            if cell["cell_type"] != "code":
                continue
            for output in cell["outputs"]:
                if "plotly" in output:
                    plotly = output["plotly"]
                    signature.append(
                        (
                            plotly["layout"]["title"],
                            plotly["trace_count"],
                            tuple(trace["name"] for trace in plotly["traces"]),
                            tuple(trace["type"] for trace in plotly["traces"]),
                        )
                    )
        return signature

    @staticmethod
    def _table_output_count(snapshot: dict) -> int:
        count = 0
        for cell in snapshot["cells"]:
            if cell["cell_type"] != "code":
                continue
            count += sum(
                1
                for output in cell["outputs"]
                if "html_hash" in output and "text_hash" in output and "plotly" not in output
            )
        return count


if __name__ == "__main__":
    unittest.main()
