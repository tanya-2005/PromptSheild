import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from promptshield.api import PromptRequest, create_app
from promptshield.inference import DEFAULT_THRESHOLD, Prediction


class FakeService:
    def __init__(self, model_dir, threshold, device):
        self.loaded = True
        self.device = "cpu"
        self.load_error = None
        self.threshold = threshold

    def load(self):
        return None

    def predict(self, prompt):
        return Prediction(0.95, "malicious", self.threshold)


class FailingService(FakeService):
    def predict(self, prompt):
        raise RuntimeError("synthetic inference failure")


class FailingLoadService(FakeService):
    def __init__(self, model_dir, threshold, device):
        super().__init__(model_dir, threshold, device)
        self.loaded = False

    def load(self):
        raise FileNotFoundError("synthetic missing model directory")


def route_endpoint(app, path, method):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in route.methods:
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


class ApiTests(unittest.TestCase):
    def test_health_and_prediction_endpoints(self):
        with patch("promptshield.api.DebertaInference", FakeService):
            app = create_app(Path("models/deberta-v3-small"))
        health = route_endpoint(app, "/health", "GET")()
        prediction = route_endpoint(app, "/predict", "POST")(
            PromptRequest(prompt="test prompt")
        )
        self.assertTrue(health["loaded"])
        self.assertEqual(health["device"], "cpu")
        self.assertEqual(prediction["decision"], "malicious")
        self.assertEqual(prediction["malicious_probability"], 0.95)
        self.assertIn("experimental threshold", prediction["threshold_status"])

    def test_predict_reports_unavailable_model(self):
        with patch("promptshield.api.DebertaInference", FakeService):
            app = create_app(Path("models/deberta-v3-small"))
        app.state.inference.loaded = False
        with self.assertRaises(HTTPException) as context:
            route_endpoint(app, "/predict", "POST")(
                PromptRequest(prompt="test prompt")
            )
        self.assertEqual(context.exception.status_code, 503)

    def test_predict_reports_runtime_inference_failure_without_details(self):
        with patch("promptshield.api.DebertaInference", FailingService):
            app = create_app(Path("models/deberta-v3-small"))
        with self.assertRaises(HTTPException) as context:
            route_endpoint(app, "/predict", "POST")(
                PromptRequest(prompt="test prompt")
            )
        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(
            context.exception.detail,
            "model is unavailable for inference",
        )
        self.assertEqual(app.state.inference.load_error, "RuntimeError")

    def test_request_model_rejects_empty_and_oversized_prompts(self):
        with self.assertRaises(ValueError):
            PromptRequest(prompt=" ")
        with self.assertRaises(ValueError):
            PromptRequest(prompt="x" * 4097)

    def test_request_model_rejects_non_string_prompt(self):
        with self.assertRaises(ValueError):
            PromptRequest(prompt={"raw": "prompt"})

    def test_prediction_response_contract_is_exact_and_threshold_echoed(self):
        with patch("promptshield.api.DebertaInference", FakeService):
            app = create_app(Path("models/deberta-v3-small"))
        prediction = route_endpoint(app, "/predict", "POST")(
            PromptRequest(prompt="test prompt")
        )
        self.assertEqual(
            set(prediction),
            {
                "malicious_probability",
                "decision",
                "threshold",
                "threshold_status",
            },
        )
        self.assertEqual(prediction["threshold"], DEFAULT_THRESHOLD)

    def test_request_model_requires_prompt_field(self):
        with self.assertRaises(ValidationError):
            PromptRequest()

    def test_startup_load_failure_is_reported_not_raised_and_blocks_predict(self):
        with patch("promptshield.api.DebertaInference", FailingLoadService):
            app = create_app(Path("models/deberta-v3-small"))

        async def run_lifespan():
            async with app.router.lifespan_context(app):
                pass

        asyncio.run(run_lifespan())
        health = route_endpoint(app, "/health", "GET")()
        self.assertFalse(health["loaded"])
        self.assertIn("synthetic missing model directory", health["load_error"])
        with self.assertRaises(HTTPException) as context:
            route_endpoint(app, "/predict", "POST")(
                PromptRequest(prompt="test prompt")
            )
        self.assertEqual(context.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
