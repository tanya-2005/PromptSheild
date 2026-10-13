"""Local-only inference service for the saved DeBERTa classifier."""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "microsoft/deberta-v3-small"
MAX_LENGTH = 128
MAX_PROMPT_LENGTH = 4096
DEFAULT_THRESHOLD = 0.90
LABEL_MAPPING = {0: "benign", 1: "malicious"}


@dataclass(frozen=True)
class Prediction:
    malicious_probability: float
    decision: str
    threshold: float


class DebertaInference:
    """Load and run the saved classifier without network access."""

    def __init__(
        self,
        model_dir: Path,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "auto",
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.requested_device = device
        self.loaded = False
        self.device = "unloaded"
        self.load_error: str | None = None
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._inference_lock = threading.Lock()

    def load(self) -> None:
        """Load local files only and put the model in evaluation mode."""
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"model directory not found: {self.model_dir}")

        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = (
            "cuda"
            if self.requested_device == "cuda"
            or (self.requested_device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        if getattr(getattr(model, "config", None), "num_labels", 2) != 2:
            raise ValueError("saved model must have exactly two classifier labels")
        model.to(device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self.device = device
        self.loaded = True
        self.load_error = None

    def predict(self, prompt: str) -> Prediction:
        if not self.loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("model is not loaded")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"prompt must be at most {MAX_PROMPT_LENGTH} characters"
            )

        encoded = self._tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(self.device) for key, value in encoded.items()
        }
        with self._inference_lock:
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=1)
        malicious_probability = float(probabilities[0, 1].item())
        decision = (
            LABEL_MAPPING[1]
            if malicious_probability >= self.threshold
            else LABEL_MAPPING[0]
        )
        return Prediction(
            malicious_probability=malicious_probability,
            decision=decision,
            threshold=self.threshold,
        )

    def predict_batch(
        self,
        prompts: list[str],
        batch_size: int = 8,
        enforce_prompt_limit: bool = True,
    ) -> list[Prediction]:
        """Run local-only inference on prompts in conservative mini-batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not self.loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("model is not loaded")
        for prompt in prompts:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("each prompt must be a non-empty string")
            if enforce_prompt_limit and len(prompt) > MAX_PROMPT_LENGTH:
                raise ValueError(
                    f"prompt must be at most {MAX_PROMPT_LENGTH} characters"
                )

        predictions: list[Prediction] = []
        with self._inference_lock:
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device) for key, value in encoded.items()
                }
                with self._torch.inference_mode():
                    logits = self._model(**encoded).logits
                    probabilities = self._torch.softmax(logits, dim=1)
                for probability in probabilities[:, 1]:
                    malicious_probability = float(probability.item())
                    predictions.append(
                        Prediction(
                            malicious_probability=malicious_probability,
                            decision=(
                                LABEL_MAPPING[1]
                                if malicious_probability >= self.threshold
                                else LABEL_MAPPING[0]
                            ),
                            threshold=self.threshold,
                        )
                    )
        return predictions
