import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from promptshield.robustness import (
    DEFAULT_DEBERTA_BATCH_SIZE,
    TRANSFORMATIONS,
    TRANSFORMATION_METADATA,
    build_examples,
    calculate_robustness_metrics,
    load_model_adapters,
    render_markdown,
    run_benchmark,
    stable_example_id,
    transform_examples,
    validate_predictions,
    validate_validation_path,
)


class RobustnessTests(unittest.TestCase):
    def _run_synthetic_benchmark(
        self, adapters, unavailable=None, model_paths=None
    ):
        with patch(
            "promptshield.robustness.load_labeled_parquet",
            return_value=(["synthetic benign", "synthetic malicious"], [0, 1]),
        ), patch(
            "promptshield.robustness.load_model_adapters",
            return_value=(adapters, unavailable or {}),
        ):
            return run_benchmark(
                Path("data/raw/validation.parquet"),
                model_paths
                or {
                    name: Path(f"{name}.model")
                    for name in adapters
                },
            )

    def test_benchmark_status_completed_when_all_models_finish(self):
        class CompleteAdapter:
            def predict_batch(self, texts, batch_size):
                return [0, 1]

            def metadata(self):
                return {"model": "synthetic"}

        report = self._run_synthetic_benchmark(
            {"baseline": CompleteAdapter()}
        )
        self.assertEqual(report["evaluation_status"], "completed")
        self.assertTrue(report["inference_complete"])
        self.assertEqual(
            report["models"]["baseline"]["evaluation_status"],
            "completed",
        )

    def test_benchmark_status_incomplete_when_deberta_is_unavailable(self):
        class CompleteAdapter:
            def predict_batch(self, texts, batch_size):
                return [0, 1]

            def metadata(self):
                return {"model": "synthetic"}

        report = self._run_synthetic_benchmark(
            {
                "baseline": CompleteAdapter(),
                "svm": CompleteAdapter(),
            },
            unavailable={"deberta": "synthetic unavailable"},
            model_paths={
                "baseline": Path("baseline.model"),
                "svm": Path("svm.model"),
                "deberta": Path("deberta.model"),
            },
        )
        self.assertEqual(report["evaluation_status"], "incomplete")
        self.assertFalse(report["inference_complete"])

    def test_benchmark_status_incomplete_when_model_returns_short_batch(self):
        class IncompleteAdapter:
            def __init__(self):
                self.calls = 0

            def predict_batch(self, texts, batch_size):
                self.calls += 1
                return [0] if self.calls == 2 else [0, 1]

            def metadata(self):
                return {"model": "synthetic"}

        report = self._run_synthetic_benchmark(
            {"baseline": IncompleteAdapter()}
        )
        self.assertEqual(report["evaluation_status"], "incomplete")
        self.assertFalse(report["inference_complete"])
        self.assertEqual(
            report["models"]["baseline"]["evaluation_status"],
            "incomplete",
        )

    def test_invalid_prediction_values_make_model_incomplete(self):
        class InvalidAdapter:
            def predict_batch(self, texts, batch_size):
                return [0, 2]

            def metadata(self):
                return {"model": "synthetic"}

        report = self._run_synthetic_benchmark(
            {"baseline": InvalidAdapter()}
        )
        self.assertEqual(report["evaluation_status"], "incomplete")
        self.assertFalse(report["inference_complete"])
        self.assertTrue(report["models"]["baseline"]["errors"])

    def test_prediction_validation_rejects_wrong_shape_and_label(self):
        with self.assertRaisesRegex(ValueError, "expected 2"):
            validate_predictions([0], 2, "synthetic")
        with self.assertRaisesRegex(ValueError, "binary label set"):
            validate_predictions([0, 3], 2, "synthetic")

    def test_incomplete_report_markdown_does_not_require_metrics(self):
        report = self._run_synthetic_benchmark(
            {}, unavailable={"deberta": "synthetic unavailable"},
            model_paths={"deberta": Path("deberta.model")},
        )
        markdown = render_markdown(report)
        self.assertIn("incomplete", markdown)
        self.assertIn("unavailable", markdown)

    def test_transformations_are_deterministic_and_preserve_identity(self):
        examples = build_examples(["safe text", "ignore instructions"], [0, 1])
        for transformation in TRANSFORMATIONS:
            first = transform_examples(examples, transformation)
            second = transform_examples(examples, transformation)
            self.assertEqual(first, second)
            self.assertEqual(
                [item.example_id for item in first],
                [item.example_id for item in examples],
            )
            self.assertEqual([item.label for item in first], [0, 1])
            self.assertTrue(all(item.text for item in first))

    def test_neutral_context_is_exploratory_not_label_preserving(self):
        self.assertFalse(
            TRANSFORMATION_METADATA["neutral_context"][
                "label_preserving_claim"
            ]
        )
        self.assertEqual(
            TRANSFORMATION_METADATA["neutral_context"][
                "evaluation_category"
            ],
            "exploratory",
        )

    def test_candidate_transformations_document_edge_cases(self):
        self.assertEqual(
            transform_examples(
                build_examples(["synthetic input"], [0]),
                "harmless_formatting",
            )[0].text,
            "---\nsynthetic input\n---",
        )
        self.assertFalse(
            TRANSFORMATION_METADATA["harmless_formatting"][
                "label_preserving_claim"
            ]
        )
        for name in ("whitespace_variation",):
            metadata = TRANSFORMATION_METADATA[name]
            self.assertTrue(metadata["label_preserving_claim"])
            self.assertTrue(metadata["limitation"])

    def test_formatting_preserves_identifier_and_label_only(self):
        original = build_examples(["synthetic input"], [1])[0]
        transformed = transform_examples(
            [original], "harmless_formatting"
        )[0]
        self.assertEqual(transformed.example_id, original.example_id)
        self.assertEqual(transformed.label, original.label)
        self.assertNotEqual(transformed.text, original.text)

    def test_formatting_and_context_are_distinct_and_preserve_rows(self):
        examples = build_examples(
            ["synthetic benign", "synthetic malicious\nline"], [0, 1]
        )
        formatting = transform_examples(examples, "harmless_formatting")
        context = transform_examples(examples, "neutral_context")

        self.assertEqual(len(formatting), len(examples))
        self.assertEqual(len(context), len(examples))
        self.assertEqual(
            [item.example_id for item in formatting],
            [item.example_id for item in examples],
        )
        self.assertEqual(
            [item.example_id for item in context],
            [item.example_id for item in examples],
        )
        self.assertEqual(
            [item.label for item in formatting],
            [item.label for item in examples],
        )
        self.assertEqual(
            [item.label for item in context],
            [item.label for item in examples],
        )
        self.assertEqual(
            [item.text for item in formatting],
            ["---\nsynthetic benign\n---", "---\nsynthetic malicious\nline\n---"],
        )
        self.assertEqual(
            [item.text for item in context],
            [
                "Context note: The following user-provided text is being classified.\n\n"
                "synthetic benign",
                "Context note: The following user-provided text is being classified.\n\n"
                "synthetic malicious\nline",
            ],
        )
        self.assertNotEqual(
            [item.text for item in formatting],
            [item.text for item in context],
        )

    def test_benchmark_records_distinct_inputs_and_uses_each_prediction_set(self):
        class RecordingAdapter:
            def __init__(self):
                self.inputs = []
                self.predictions = iter(
                    ([0, 1], [0, 1], [1, 1], [0, 1],)
                )

            def predict_batch(self, texts, batch_size):
                self.inputs.append(list(texts))
                return next(self.predictions)

            def metadata(self):
                return {"model": "synthetic"}

        adapter = RecordingAdapter()
        with patch(
            "promptshield.robustness.load_labeled_parquet",
            return_value=(["synthetic benign", "synthetic malicious"], [0, 1]),
        ), patch(
            "promptshield.robustness.load_model_adapters",
            return_value=({"synthetic": adapter}, {}),
        ):
            from promptshield.robustness import run_benchmark

            report = run_benchmark(
                Path("data/raw/validation.parquet"),
                {"synthetic": Path("synthetic.model")},
            )

        self.assertEqual(
            adapter.inputs,
            [
                ["synthetic benign", "synthetic malicious"],
                ["synthetic benign", "synthetic malicious"],
                ["---\nsynthetic benign\n---", "---\nsynthetic malicious\n---"],
                [
                    "Context note: The following user-provided text is being classified.\n\n"
                    "synthetic benign",
                    "Context note: The following user-provided text is being classified.\n\n"
                    "synthetic malicious",
                ],
            ],
        )
        transformations = report["models"]["synthetic"]["transformations"]
        self.assertEqual(
            transformations["harmless_formatting"]["metrics"][
                "prediction_flip_rate"
            ],
            0.5,
        )
        self.assertEqual(
            transformations["neutral_context"]["metrics"][
                "prediction_flip_rate"
            ],
            0.0,
        )
        self.assertNotEqual(
            transformations["harmless_formatting"]["metrics"],
            transformations["neutral_context"]["metrics"],
        )

    def test_ids_are_stable_without_exposing_text(self):
        identifier = stable_example_id(1, "private prompt")
        self.assertEqual(identifier, stable_example_id(1, "private prompt"))
        self.assertNotIn("private", identifier)

    def test_metrics_include_flips_and_class_breakdown(self):
        result = calculate_robustness_metrics(
            [0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]
        )
        self.assertEqual(result["prediction_flip_rate"], 0.5)
        self.assertEqual(
            result["by_original_class"]["0"]["prediction_flip_rate"], 0.5
        )
        self.assertEqual(
            result["by_original_class"]["1"]["prediction_flip_rate"], 0.5
        )
        self.assertEqual(
            result["by_original_class"]["0"]["classification_accuracy"], 0.5
        )
        self.assertEqual(
            result["by_original_class"]["1"]["classification_accuracy"], 0.5
        )

    def test_test_split_path_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_validation_path(Path("data/raw/test.parquet"))

    def test_unavailable_model_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.joblib"
            available, unavailable = load_model_adapters({"baseline": missing})
        self.assertEqual(available, {})
        self.assertIn("baseline", unavailable)

    def test_model_adapter_predictions_can_be_mocked(self):
        class FakePipeline:
            def predict(self, texts):
                return [0 for _ in texts]

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            with patch(
                "promptshield.robustness.joblib.load",
                return_value={"pipeline": FakePipeline()},
            ):
                available, unavailable = load_model_adapters(
                    {"baseline": artifact}
                )
        self.assertEqual(unavailable, {})
        self.assertEqual(available["baseline"].predict(["x", "y"]), [0, 0])

    def test_deberta_batch_adapter_preserves_order(self):
        class FakePrediction:
            def __init__(self, decision):
                self.decision = decision

        class FakeService:
            def load(self):
                return None

            def predict_batch(self, texts, batch_size, enforce_prompt_limit=True):
                return [
                    FakePrediction("malicious" if text.endswith("1") else "benign")
                    for text in texts
                ]

        with patch(
            "promptshield.robustness.DebertaInference",
            return_value=FakeService(),
        ):
            from promptshield.robustness import DebertaAdapter

            adapter = DebertaAdapter(Path("model"), DEFAULT_DEBERTA_BATCH_SIZE)
        self.assertEqual(
            adapter.predict_batch(["item-0", "item-1", "item-2"], 2),
            [0, 1, 0],
        )


if __name__ == "__main__":
    unittest.main()
