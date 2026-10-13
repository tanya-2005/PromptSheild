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
    stable_example_id,
    transform_examples,
    validate_validation_path,
)


class RobustnessTests(unittest.TestCase):
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
