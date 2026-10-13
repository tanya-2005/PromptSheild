import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from promptshield.api import PromptRequest, create_app
from promptshield.inference import Prediction


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

    def test_request_model_rejects_empty_and_oversized_prompts(self):
        with self.assertRaises(ValueError):
            PromptRequest(prompt=" ")
        with self.assertRaises(ValueError):
            PromptRequest(prompt="x" * 4097)


if __name__ == "__main__":
    unittest.main()
