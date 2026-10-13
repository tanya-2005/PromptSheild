"""Conservative local DeBERTa-v3-small classification experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from promptshield.baseline import (
    calculate_metrics,
    load_labeled_parquet,
)

MODEL_ID = "microsoft/deberta-v3-small"
FEATURE_COLUMN = "text"
TARGET_COLUMN = "label"
SEED = 42


@dataclass(frozen=True)
class TransformerConfig:
    model_id: str = MODEL_ID
    model_revision: str = ""
    max_length: int = 128
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    cpu_threads: int = 2
    seed: int = SEED
    device: str = "auto"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_transformer_dependencies() -> tuple[Any, Any]:
    """Load optional dependencies or explain the installation blocker."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Transformer experiment requires torch and transformers. "
            "Install them only after confirming sufficient RAM and a "
            "compatible Windows/Python environment."
        ) from error
    return torch, (AutoTokenizer, AutoModelForSequenceClassification)


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def validate_config(config: TransformerConfig) -> None:
    if not config.model_revision:
        raise ValueError(
            "model_revision must be a pinned model revision, not an implicit "
            "moving branch."
        )
    if config.max_length < 1 or config.batch_size < 1 or config.epochs < 1:
        raise ValueError("max_length, batch_size, and epochs must be positive")
    if config.learning_rate <= 0 or config.cpu_threads < 1:
        raise ValueError("learning_rate and cpu_threads must be positive")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")


def _make_batches(
    torch: Any,
    tokenizer: Any,
    texts: list[str],
    labels: list[int],
    config: TransformerConfig,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    return [
        {
            key: value[start:start + config.batch_size]
            for key, value in encoded.items()
        }
        for start in range(0, len(texts), config.batch_size)
    ]


def _predict(
    torch: Any,
    model: Any,
    batches: list[dict[str, Any]],
    device: Any,
) -> list[int]:
    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            logits = model(**inputs).logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
    return predictions


def train_and_validate(
    train_path: Path,
    validation_path: Path,
    artifact_path: Path,
    report_path: Path,
    config: TransformerConfig,
) -> dict[str, Any]:
    """Fine-tune on train and evaluate on validation only."""
    validate_config(config)
    torch, (auto_tokenizer, auto_model) = require_transformer_dependencies()
    torch.set_num_threads(config.cpu_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but CUDA is unavailable")
    device = (
        torch.device("cuda")
        if config.device == "cuda"
        or (config.device == "auto" and torch.cuda.is_available())
        else torch.device("cpu")
    )
    set_seed(torch, config.seed)
    train_texts, train_labels = load_labeled_parquet(train_path)
    validation_texts, validation_labels = load_labeled_parquet(validation_path)
    tokenizer = auto_tokenizer.from_pretrained(
        config.model_id, revision=config.model_revision
    )
    model = auto_model.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        num_labels=2,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_batches = _make_batches(
        torch, tokenizer, train_texts, train_labels, config
    )
    validation_batches = _make_batches(
        torch, tokenizer, validation_texts, validation_labels, config
    )
    started = time.perf_counter()
    model.train()
    for _ in range(config.epochs):
        for batch in train_batches:
            optimizer.zero_grad()
            outputs = model(
                **{key: value.to(device) for key, value in batch.items()}
            )
            outputs.loss.backward()
            optimizer.step()
    training_seconds = time.perf_counter() - started
    warmup = validation_batches[:1]
    _predict(torch, model, warmup, device)
    started = time.perf_counter()
    predictions = _predict(torch, model, validation_batches, device)
    latency_seconds = time.perf_counter() - started
    report = {
        "experiment": "deberta_v3_small_local_cpu",
        "feature_column": FEATURE_COLUMN,
        "target_column": TARGET_COLUMN,
        "excluded_feature_columns": [
            "category", "source", "severity", "group_id", "augmented", "tags"
        ],
        "config": asdict(config),
        "train": {
            "path": train_path.as_posix(),
            "sha256": _sha256(train_path),
            "rows": len(train_texts),
        },
        "validation": {
            "path": validation_path.as_posix(),
            "sha256": _sha256(validation_path),
            "rows": len(validation_texts),
        },
        "metrics": calculate_metrics(validation_labels, predictions),
        "latency": {
            "method": (
                "One warm-up validation batch followed by one full validation "
                "pass using time.perf_counter; excludes model loading."
            ),
            "warmup_batches": len(warmup),
            "measured_examples": len(validation_texts),
            "mean_ms_per_text": latency_seconds / len(validation_texts) * 1000,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "training_seconds": training_seconds,
            "device": str(device),
        },
        "test_split_loaded": False,
        "limitations": [
            "Validation metrics are not final held-out test metrics.",
            "This classifier is not a production security guarantee.",
            "The base model and dataset may have domain and source overlap.",
        ],
    }
    artifact_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact_path)
    tokenizer.save_pretrained(artifact_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()
    report = train_and_validate(
        args.train,
        args.validation,
        args.artifact,
        args.report,
        TransformerConfig(
            model_revision=args.model_revision,
            device=args.device,
        ),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
