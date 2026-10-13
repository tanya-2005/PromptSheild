import unittest

from promptshield.transformer_experiment import (
    MODEL_ID,
    TransformerConfig,
    require_transformer_dependencies,
    validate_config,
)


class TransformerExperimentTests(unittest.TestCase):
    def test_conservative_config_is_explicit(self):
        config = TransformerConfig(model_revision="pinned-revision")
        self.assertEqual(config.model_id, MODEL_ID)
        self.assertEqual(config.max_length, 128)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.epochs, 1)

    def test_model_revision_is_required(self):
        with self.assertRaisesRegex(ValueError, "model_revision"):
            validate_config(TransformerConfig())

    def test_invalid_resource_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_config(
                TransformerConfig(model_revision="rev", batch_size=0)
            )

    def test_device_setting_is_explicit(self):
        validate_config(TransformerConfig(model_revision="rev", device="cuda"))
        with self.assertRaisesRegex(ValueError, "device"):
            validate_config(
                TransformerConfig(model_revision="rev", device="tpu")
            )

    def test_missing_optional_dependencies_are_reported(self):
        try:
            require_transformer_dependencies()
        except RuntimeError as error:
            self.assertIn("requires torch and transformers", str(error))


if __name__ == "__main__":
    unittest.main()
