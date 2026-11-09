import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from promptshield.deberta_decision_audit import (
    APPROVED_VALIDATION_PATH,
    audit_validation,
    compare_decision_rules,
    hash_checkpoint_files,
    render_markdown,
    main,
    validate_audit_inputs,
    write_failure_report,
    write_reports,
)


class DecisionAuditTests(unittest.TestCase):
    def test_compare_rules_reports_metrics_and_disagreements(self):
        result = compare_decision_rules(
            [0, 1, 1],
            [[2.0, 0.0], [0.0, 3.0], [2.0, 2.1]],
        )
        self.assertEqual(result["argmax"]["predictions"], [0, 1, 1])
        self.assertEqual(result["threshold"]["predictions"], [0, 1, 0])
        self.assertEqual(result["decision_disagreement"], {"count": 1, "rate": 1 / 3})
        self.assertEqual(
            result["threshold"]["metrics"]["confusion_matrix"],
            [[1, 0], [1, 1]],
        )

    def test_invalid_audit_inputs_fail_before_metrics(self):
        with self.assertRaisesRegex(ValueError, "expected 2"):
            validate_audit_inputs([0], [[0.0]])
        with self.assertRaisesRegex(ValueError, "invalid values"):
            validate_audit_inputs([0], [[0.0, float("nan")]])
        with self.assertRaisesRegex(ValueError, "labels"):
            validate_audit_inputs([2], [[0.0, 1.0]])

    def test_hashes_are_relative_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "nested").mkdir()
            (directory / "nested" / "weights.bin").write_bytes(b"weights")
            first = hash_checkpoint_files(directory)
            second = hash_checkpoint_files(directory)
            self.assertEqual(first, second)
            self.assertEqual(
                [item["path"] for item in first],
                ["config.json", "nested/weights.bin"],
            )

    def test_synthetic_mocked_audit_path_does_not_load_parquet(self):
        fake_service = SimpleNamespace(
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
        fake_service.load = lambda: None
        fake_service.predict_logits_batch = lambda texts, **kwargs: [
            [2.0, 0.0],
            [0.0, 3.0],
        ]
        with tempfile.TemporaryDirectory() as directory_name:
            checkpoint = Path(directory_name)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            with patch(
                "promptshield.deberta_decision_audit.load_labeled_parquet",
                return_value=(["synthetic benign", "synthetic malicious"], [0, 1]),
            ), patch(
                "promptshield.deberta_decision_audit.DebertaInference",
                return_value=fake_service,
            ):
                from promptshield.deberta_decision_audit import audit_validation
                report = audit_validation(
                    APPROVED_VALIDATION_PATH, checkpoint
                )
        self.assertEqual(report["validation_rows"], 2)
        self.assertFalse(report["prompt_text_included"])
        self.assertEqual(report["comparison"]["decision_disagreement"]["count"], 0)

    def test_only_repository_approved_validation_path_is_accepted(self):
        with patch(
            "promptshield.deberta_decision_audit.load_labeled_parquet",
            return_value=(["synthetic"], [0]),
        ), patch(
            "promptshield.deberta_decision_audit.DebertaInference",
        ):
            with self.assertRaisesRegex(ValueError, "repository-approved"):
                audit_validation(
                    APPROVED_VALIDATION_PATH.parent.parent
                    / "backup"
                    / "validation.parquet",
                    Path("checkpoint"),
                )
            with self.assertRaisesRegex(ValueError, "repository-approved"):
                audit_validation(
                    APPROVED_VALIDATION_PATH.parent.parent
                    / "train_data"
                    / "validation.parquet",
                    Path("checkpoint"),
                )

    def test_approved_path_normalizes_relative_and_mixed_separators(self):
        fake_service = SimpleNamespace(
            threshold=0.90,
            device="cpu",
            _torch=SimpleNamespace(__version__="synthetic"),
            _tokenizer=SimpleNamespace(model_max_length=128),
            _model=SimpleNamespace(config=SimpleNamespace(num_labels=2)),
        )
        fake_service.load = lambda: None
        fake_service.predict_logits_batch = lambda texts, **kwargs: [[2.0, 0.0]]
        relative = Path(
            os.path.relpath(APPROVED_VALIDATION_PATH, Path.cwd())
        )
        mixed = Path(str(relative).replace("\\", "/"))
        normalized = (
            Path(".") / "data" / "raw" / ".." / "raw" / "validation.parquet"
        )
        with tempfile.TemporaryDirectory() as directory_name:
            checkpoint = Path(directory_name)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            with patch(
                "promptshield.deberta_decision_audit.load_labeled_parquet",
                return_value=(["synthetic"], [0]),
            ), patch(
                "promptshield.deberta_decision_audit.DebertaInference",
                return_value=fake_service,
            ):
                for path in (relative, mixed, normalized):
                    report = audit_validation(path, checkpoint)
                    self.assertEqual(report["validation_rows"], 1)

    def test_failure_report_is_prompt_free_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory_name:
            reports_dir = Path(directory_name)
            existing_json = reports_dir / "deberta-decision-audit-existing.json"
            existing_markdown = reports_dir / "deberta-decision-audit-existing.md"
            existing_json.write_text("keep-json", encoding="utf-8")
            existing_markdown.write_text("keep-markdown", encoding="utf-8")
            json_path, markdown_path = write_failure_report(
                reports_dir, RuntimeError("model loading failed")
            )
            self.assertEqual(existing_json.read_text(encoding="utf-8"), "keep-json")
            self.assertEqual(
                existing_markdown.read_text(encoding="utf-8"),
                "keep-markdown",
            )
            json_text = json_path.read_text(encoding="utf-8")
            markdown_text = markdown_path.read_text(encoding="utf-8")
            self.assertIn('"evaluation_status": "failed"', json_text)
            self.assertIn("audit operation failed", json_text)
            self.assertNotIn("prompt", json_text.lower().replace(
                "prompt_text_included", ""
            ))
            self.assertIn("Evaluation status: `failed`", markdown_text)

    def test_main_records_model_failure_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory_name:
            reports_dir = Path(directory_name)
            with patch(
                "sys.argv",
                [
                    "deberta_decision_audit",
                    "--validation",
                    str(APPROVED_VALIDATION_PATH),
                    "--checkpoint",
                    "synthetic-checkpoint",
                    "--reports-dir",
                    str(reports_dir),
                ],
            ), patch(
                "promptshield.deberta_decision_audit.DebertaInference",
                side_effect=RuntimeError("synthetic model loading failed"),
            ), patch(
                "promptshield.deberta_decision_audit.load_labeled_parquet",
                return_value=(["synthetic"], [0]),
            ):
                self.assertEqual(main(), 1)
            reports = sorted(reports_dir.iterdir())
            self.assertEqual(len(reports), 2)
            report_text = next(
                path.read_text(encoding="utf-8")
                for path in reports
                if path.suffix == ".json"
            )
            self.assertIn('"evaluation_status": "failed"', report_text)
            self.assertIn('"test_split_loaded": false', report_text)

    def _assert_main_failure_category(self, category, *, inference=False):
        with tempfile.TemporaryDirectory() as directory_name:
            reports_dir = Path(directory_name)
            service = SimpleNamespace(load=lambda: None)
            if inference:
                service.predict_logits_batch = lambda *args, **kwargs: (
                    (_ for _ in ()).throw(RuntimeError("prompt leaked"))
                )
            else:
                service.predict_logits_batch = lambda *args, **kwargs: [
                    [2.0, 0.0]
                ]
            with patch(
                "sys.argv",
                [
                    "deberta_decision_audit",
                    "--validation",
                    str(APPROVED_VALIDATION_PATH),
                    "--checkpoint",
                    "synthetic-checkpoint",
                    "--reports-dir",
                    str(reports_dir),
                ],
            ), patch(
                "promptshield.deberta_decision_audit.DebertaInference",
                return_value=service,
            ), patch(
                "promptshield.deberta_decision_audit.load_labeled_parquet",
                return_value=(["synthetic"], [0]),
            ), patch(
                "promptshield.deberta_decision_audit.compare_decision_rules",
                side_effect=ValueError("raw prompt must not be reported"),
            ) as compare:
                self.assertEqual(main(), 1)
                if inference:
                    compare.assert_not_called()
            report_text = next(
                path.read_text(encoding="utf-8")
                for path in reports_dir.iterdir()
                if path.suffix == ".json"
            )
            self.assertIn(f'"error": "{category}"', report_text)
            self.assertNotIn("prompt leaked", report_text)
            self.assertNotIn("raw prompt", report_text)

    def test_main_records_inference_failure_without_sensitive_details(self):
        self._assert_main_failure_category(
            "inference failure", inference=True
        )

    def test_main_records_metric_failure_without_sensitive_details(self):
        self._assert_main_failure_category("metric calculation failure")

    def test_reports_are_unique_and_markdown_has_no_prompt_text(self):
        report = {
            "validation_rows": 2,
            "timestamp_utc": "synthetic",
            "model_configuration": {"threshold": 0.9},
            "comparison": {
                "decision_disagreement": {"count": 0, "rate": 0.0},
                "argmax": {"metrics": {
                    "accuracy": 1.0, "macro_f1": 1.0,
                    "false_positive_rate_benign": 0.0,
                    "false_negative_rate_malicious": 0.0,
                    "confusion_matrix": [[1, 0], [0, 1]],
                }},
                "threshold": {"metrics": {
                    "accuracy": 1.0, "macro_f1": 1.0,
                    "false_positive_rate_benign": 0.0,
                    "false_negative_rate_malicious": 0.0,
                    "confusion_matrix": [[1, 0], [0, 1]],
                }},
            },
        }
        markdown = render_markdown(report)
        self.assertNotIn("synthetic benign", markdown)
        with tempfile.TemporaryDirectory() as directory_name:
            paths = write_reports(report, Path(directory_name))
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
