import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from promptshield.deberta_error_analysis import (
    APPROVED_VALIDATION_PATH,
    analyze_validation_errors,
    main,
    render_markdown,
    write_reports,
)


class DebertaErrorAnalysisTests(unittest.TestCase):
    def _service(self):
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
            [2.0, 0.0],
            [0.0, 3.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ]
        return service

    def test_counts_and_metadata_breakdown_use_threshold_predictions(self):
        records = (
            ["safe", "attack", "ambiguous", "safe"],
            [0, 1, 0, 1],
            [
                {"category": "normal", "source": "a", "severity": None, "augmented": False},
                {"category": "injection", "source": "a", "severity": "high", "augmented": True},
                {"category": "normal", "source": "b", "severity": "low", "augmented": False},
                {"category": "injection", "source": "b", "severity": "high", "augmented": None},
            ],
        )
        with tempfile.TemporaryDirectory() as directory_name:
            checkpoint = Path(directory_name)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            with patch(
                "promptshield.deberta_error_analysis.load_validation_records",
                return_value=records,
            ), patch(
                "promptshield.deberta_error_analysis.DebertaInference",
                return_value=self._service(),
            ):
                report = analyze_validation_errors(
                    APPROVED_VALIDATION_PATH, checkpoint
                )
        self.assertEqual(report["validation_rows"], 4)
        self.assertEqual(report["counts"]["false_positive"], 0)
        self.assertEqual(report["counts"]["false_negative"], 1)
        self.assertEqual(report["counts"]["true_negative"], 2)
        self.assertEqual(report["counts"]["true_positive"], 1)
        self.assertEqual(
            report["metadata_breakdown"]["severity"]["<missing>"]["sample_count"],
            1,
        )
        self.assertFalse(report["prompt_text_included"])

    def test_noncanonical_path_is_rejected_before_loader(self):
        with patch(
            "promptshield.deberta_error_analysis.load_validation_records"
        ) as loader:
            with self.assertRaisesRegex(ValueError, "repository-approved"):
                analyze_validation_errors(
                    APPROVED_VALIDATION_PATH.parent / "backup" / "validation.parquet",
                    Path("checkpoint"),
                )
            loader.assert_not_called()

    def test_reports_do_not_include_prompt_text(self):
        report = {
            "validation_rows": 1,
            "timestamp_utc": "synthetic",
            "model_configuration": {"threshold": 0.90},
            "counts": {
                "true_negative": 1, "false_positive": 0,
                "false_negative": 0, "true_positive": 0,
            },
            "metadata_breakdown": {"category": {}, "source": {}, "severity": {}, "augmented": {}},
            "limitations": [],
        }
        markdown = render_markdown(report)
        self.assertNotIn("safe secret prompt", markdown)
        with tempfile.TemporaryDirectory() as directory_name:
            paths = write_reports(report, Path(directory_name))
            self.assertTrue(all(path.is_file() for path in paths))

    def test_cli_failure_is_nonzero_and_prompt_free(self):
        with tempfile.TemporaryDirectory() as directory_name:
            reports_dir = Path(directory_name)
            with patch(
                "sys.argv",
                [
                    "deberta_error_analysis",
                    "--validation", str(APPROVED_VALIDATION_PATH),
                    "--checkpoint", "checkpoint",
                    "--reports-dir", str(reports_dir),
                ],
            ), patch(
                "promptshield.deberta_error_analysis.load_validation_records",
                side_effect=RuntimeError("secret prompt text"),
            ):
                self.assertEqual(main(), 1)
            report_text = next(
                path.read_text(encoding="utf-8")
                for path in reports_dir.iterdir()
                if path.suffix == ".json"
            )
            self.assertIn('"evaluation_status": "failed"', report_text)
            self.assertNotIn("secret prompt text", report_text)


if __name__ == "__main__":
    unittest.main()
