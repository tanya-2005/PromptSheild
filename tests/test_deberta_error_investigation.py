import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from promptshield.deberta_error_investigation import (
    APPROVED_VALIDATION_PATH,
    _confidence_summary,
    investigate_validation_errors,
    render_markdown,
)


class DebertaErrorInvestigationTests(unittest.TestCase):
    def test_confidence_summary_handles_empty_and_even_values(self):
        self.assertEqual(_confidence_summary([])["sample_count"], 0)
        summary = _confidence_summary([0.1, 0.9, 0.8, 0.2])
        self.assertEqual(summary["sample_count"], 4)
        self.assertAlmostEqual(summary["median"], 0.5)

    def test_investigation_reports_confusion_confidence_and_sources(self):
        service = SimpleNamespace(
            threshold=0.90,
            device="cpu",
            _torch=SimpleNamespace(__version__="synthetic"),
            _tokenizer=SimpleNamespace(model_max_length=128),
            _model=SimpleNamespace(config=SimpleNamespace(
                architectures=["DebertaV2ForSequenceClassification"],
                model_type="deberta-v2",
                num_labels=2,
                id2label={"0": "LABEL_0", "1": "LABEL_1"},
                label2id={"LABEL_0": 0, "LABEL_1": 1},
            )),
        )
        service.load = lambda: None
        service.predict_logits_batch = lambda texts, **kwargs: [
            [0.0, 3.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [2.0, 0.0],
        ]
        records = (
            ["a", "b", "c", "d"],
            [1, 0, 0, 1],
            [
                {"category": "attack", "source": "wildguard_judgecomp", "severity": "high", "augmented": False},
                {"category": "benign", "source": "wildguard_judgecomp", "severity": "low", "augmented": False},
                {"category": "benign", "source": "harmbench", "severity": "low", "augmented": True},
                {"category": "benign", "source": "harmbench", "severity": "high", "augmented": None},
            ],
        )
        with tempfile.TemporaryDirectory() as directory_name:
            checkpoint = Path(directory_name)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            with patch(
                "promptshield.deberta_error_investigation.load_validation_records",
                return_value=records,
            ), patch(
                "promptshield.deberta_error_investigation.DebertaInference",
                return_value=service,
            ):
                report = investigate_validation_errors(
                    APPROVED_VALIDATION_PATH, checkpoint
                )
        groups = report["confidence_by_confusion_group"]
        self.assertEqual(groups["true_positive"]["sample_count"], 1)
        self.assertEqual(groups["true_negative"]["sample_count"], 1)
        self.assertEqual(groups["false_positive"]["sample_count"], 1)
        self.assertEqual(groups["false_negative"]["sample_count"], 1)
        self.assertEqual(
            report["requested_source_comparison"]["harmbench"]["sample_count"],
            2,
        )
        self.assertIn(
            "benign",
            report["metadata_label_consistency"]["category"][
                "potential_mixed_label_values"
            ],
        )

    def test_noncanonical_path_is_rejected(self):
        with patch(
            "promptshield.deberta_error_investigation.load_validation_records"
        ) as loader:
            with self.assertRaisesRegex(ValueError, "repository-approved"):
                investigate_validation_errors(
                    APPROVED_VALIDATION_PATH.parent / "backup" / "validation.parquet",
                    Path("checkpoint"),
                )
            loader.assert_not_called()

    def test_markdown_contains_no_prompt_text(self):
        report = {
            "validation_rows": 1,
            "model_configuration": {"threshold": 0.90},
            "confidence_by_confusion_group": {
                name: {
                    "sample_count": 0,
                    "confidence": {
                        "mean": None, "median": None,
                        "minimum": None, "maximum": None,
                    },
                }
                for name in ("true_positive", "true_negative", "false_positive", "false_negative")
            },
            "requested_source_comparison": {},
            "metadata_label_consistency": {},
            "limitations": [],
        }
        self.assertNotIn("secret prompt", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
