import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "promptshield_deberta_colab.ipynb"
)


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            line
            for cell in cls.notebook["cells"]
            for line in cell.get("source", [])
        )

    def test_notebook_is_valid_and_has_required_stages(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertIn("DATASET_REVISION", self.source)
        self.assertIn("MODEL_REVISION", self.source)
        self.assertIn("PREFERRED_DEVICE", self.source)
        self.assertIn("train_ds", self.source)
        self.assertIn("validation_ds", self.source)
        self.assertIn("train_and_validate", self.source)
        self.assertIn("files.download", self.source)

    def test_notebook_never_references_forbidden_split_or_prints_prompts(self):
        self.assertNotIn("test.parquet", self.source)
        self.assertIn('"raw_prompt_text_printed": False', self.source)
        self.assertNotIn("print(train_ds['text']", self.source)
        self.assertNotIn("print(validation_ds['text']", self.source)

    def test_notebook_has_explicit_text_only_boundary(self):
        self.assertIn("text", self.source)
        self.assertIn("label", self.source)
        self.assertIn("category", self.source)
        self.assertIn("group_id", self.source)
        self.assertIn("EXPECTED_COLUMNS", self.source)


if __name__ == "__main__":
    unittest.main()
