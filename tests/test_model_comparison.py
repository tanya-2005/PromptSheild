import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from promptshield.model_comparison import (
    APPROVED_VALIDATION_PATH,
    compare_models,
    render_markdown,
)


class FakePipeline:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, texts):
        return SimpleNamespace(
            tolist=lambda: list(self.predictions[:len(texts)])
        )


class ModelComparisonTests(unittest.TestCase):
    def test_compares_all_models_on_same_synthetic_rows(self):
        labels = [0, 1, 0, 1]
        texts = ["a", "b", "c", "d"]
        fake_service = SimpleNamespace(
            threshold=0.90,
            device="cpu",
            _torch=SimpleNamespace(__version__="synthetic"),
            _tokenizer=SimpleNamespace(model_max_length=128),
            _model=SimpleNamespace(config=SimpleNamespace(
                architectures=["DebertaV2ForSequenceClassification"],
                model_type="deberta-v2",
                num_labels=2,
            )),
        )
        fake_service.load = lambda: None
        fake_service.predict_logits_batch = lambda texts, **kwargs: [
            [3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0],
        ]
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            logistic_path = directory / "baseline.joblib"
            svm_path = directory / "linear-svm.joblib"
            logistic_path.write_bytes(b"fixture")
            svm_path.write_bytes(b"fixture")
            checkpoint = directory / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            artifacts = {
                str(logistic_path): {"pipeline": FakePipeline([0, 1, 1, 1]), "experiment": "lr"},
                str(svm_path): {"pipeline": FakePipeline([0, 1, 0, 0]), "experiment": "svm"},
            }
            with patch(
                "promptshield.model_comparison.load_labeled_parquet",
                return_value=(texts, labels),
            ), patch(
                "promptshield.model_comparison.joblib.load",
                side_effect=lambda path: artifacts[str(path)],
            ), patch(
                "promptshield.model_comparison.DebertaInference",
                return_value=fake_service,
            ):
                report = compare_models(
                    APPROVED_VALIDATION_PATH,
                    logistic_path,
                    svm_path,
                    checkpoint,
                )
        self.assertEqual(report["validation_rows"], 4)
        self.assertEqual(set(report["models"]), {"logistic_regression", "linear_svm", "deberta"})
        self.assertEqual(report["models"]["logistic_regression"]["metrics"]["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(report["models"]["linear_svm"]["metrics"]["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(report["models"]["deberta"]["metrics"]["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertFalse(report["prompt_text_included"])

    def test_rejects_noncanonical_validation_path_before_loading(self):
        with patch("promptshield.model_comparison.load_labeled_parquet") as loader:
            with self.assertRaisesRegex(ValueError, "repository-approved"):
                compare_models(
                    APPROVED_VALIDATION_PATH.parent / "backup" / "validation.parquet"
                )
            loader.assert_not_called()

    def test_markdown_does_not_claim_calibrated_scores(self):
        report = {
            "validation_rows": 4,
            "models": {
                "logistic_regression": {
                    "metrics": {
                        "accuracy": 1.0, "macro_f1": 1.0,
                        "false_positive_rate_benign": 0.0,
                        "false_negative_rate_malicious": 0.0,
                    },
                    "latency": {"mean_ms_per_text": 1.0},
                },
            },
            "decision_rule_notes": ["Scores are not calibrated probabilities."],
            "limitations": [],
        }
        markdown = render_markdown(report)
        self.assertIn("not calibrated probabilities", markdown)
        self.assertNotIn("secret prompt", markdown)


if __name__ == "__main__":
    unittest.main()
