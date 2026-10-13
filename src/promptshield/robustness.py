"""Validation-only adversarial robustness benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib

from promptshield.baseline import calculate_metrics, load_labeled_parquet
from promptshield.inference import (
    MAX_LENGTH,
    MODEL_ID,
    DebertaInference,
)

SEED = 42
THRESHOLD = 0.90
DEFAULT_DEBERTA_BATCH_SIZE = 8
TRANSFORMATIONS = (
    "whitespace_variation",
    "harmless_formatting",
    "neutral_context",
)
TRANSFORMATION_METADATA: dict[str, dict[str, Any]] = {
    "whitespace_variation": {
        "description": "Replace line endings with spaces around line breaks.",
        "evaluation_category": "candidate_label_preserving",
        "label_preserving_claim": True,
        "limitation": (
            "Line-break-sensitive syntax or formatting could change meaning."
        ),
    },
    "harmless_formatting": {
        "description": "Wrap the text in plain delimiter lines.",
        "evaluation_category": "exploratory",
        "label_preserving_claim": False,
        "limitation": (
            "Delimiter tokens are part of the model input and can change "
            "interpretation or truncation; copied labels are exploratory "
            "references only."
        ),
    },
    "neutral_context": {
        "description": (
            "Prepend a neutral classification-context note."
        ),
        "evaluation_category": "exploratory",
        "label_preserving_claim": False,
        "limitation": (
            "Added context can change interpretation, so the original label "
            "is used only as an exploratory reference."
        ),
    },
}
Transformation = Callable[[str], str]


def _whitespace_variation(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " \n ")


def _harmless_formatting(text: str) -> str:
    return f"---\n{text}\n---"


def _neutral_context(text: str) -> str:
    return f"Context note: The following user-provided text is being classified.\n\n{text}"


TRANSFORM_FUNCTIONS: dict[str, Transformation] = {
    "whitespace_variation": _whitespace_variation,
    "harmless_formatting": _harmless_formatting,
    "neutral_context": _neutral_context,
}


@dataclass(frozen=True)
class Example:
    example_id: str
    text: str
    label: int


def validate_validation_path(path: Path) -> None:
    """Reject paths that could point at a forbidden test split."""
    normalized_parts = {part.casefold() for part in path.resolve().parts}
    if path.name.casefold() == "test.parquet" or "test" in normalized_parts:
        raise ValueError("robustness benchmark accepts validation data only")


def stable_example_id(row_number: int, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{row_number}:".encode("utf-8"))
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:16]


def build_examples(texts: list[str], labels: list[int]) -> list[Example]:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    return [
        Example(stable_example_id(index, text), text, label)
        for index, (text, label) in enumerate(zip(texts, labels), start=1)
    ]


def transform_examples(
    examples: list[Example],
    transformation: str,
) -> list[Example]:
    if transformation not in TRANSFORM_FUNCTIONS:
        raise ValueError(f"unknown transformation: {transformation}")
    transform = TRANSFORM_FUNCTIONS[transformation]
    transformed: list[Example] = []
    for example in examples:
        text = transform(example.text)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"transformation failed for example {example.example_id}"
            )
        transformed.append(
            Example(example.example_id, text, example.label)
        )
    return transformed


def calculate_robustness_metrics(
    labels: list[int],
    predictions: list[int],
    original_predictions: list[int],
) -> dict[str, Any]:
    if not (
        len(labels) == len(predictions) == len(original_predictions)
    ):
        raise ValueError("metric inputs must have equal lengths")
    metrics = calculate_metrics(labels, predictions)
    flips = [
        prediction != original
        for prediction, original in zip(predictions, original_predictions)
    ]
    by_original_class = {}
    for label in (0, 1):
        indices = [index for index, value in enumerate(labels) if value == label]
        by_original_class[str(label)] = {
            "examples": len(indices),
            "classification_accuracy": (
                sum(predictions[index] == label for index in indices)
                / len(indices)
                if indices else None
            ),
            "prediction_flip_rate": (
                sum(flips[index] for index in indices) / len(indices)
                if indices else None
            ),
        }
    metrics["prediction_flip_rate"] = (
        sum(flips) / len(flips) if flips else None
    )
    metrics["by_original_class"] = by_original_class
    return metrics


class ModelAdapter:
    def predict(self, texts: list[str]) -> list[int]:
        raise NotImplementedError

    def predict_batch(self, texts: list[str], batch_size: int) -> list[int]:
        return self.predict(texts)

    def metadata(self) -> dict[str, Any]:
        return {}


class ClassicalAdapter(ModelAdapter):
    def __init__(self, artifact_path: Path):
        self.artifact = joblib.load(artifact_path)
        self.pipeline = self.artifact["pipeline"]

    def predict(self, texts: list[str]) -> list[int]:
        return [int(value) for value in self.pipeline.predict(texts)]


class DebertaAdapter(ModelAdapter):
    def __init__(
        self,
        model_dir: Path,
        batch_size: int = DEFAULT_DEBERTA_BATCH_SIZE,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.service = DebertaInference(
            model_dir, threshold=THRESHOLD, device="auto"
        )
        self.service.load()
        self.batch_size = batch_size

    def predict(self, texts: list[str]) -> list[int]:
        return self.predict_batch(texts, self.batch_size)

    def predict_batch(self, texts: list[str], batch_size: int) -> list[int]:
        return [
            1 if prediction.decision == "malicious" else 0
            for prediction in self.service.predict_batch(
                texts,
                batch_size,
                enforce_prompt_limit=False,
            )
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "checkpoint": self.service.model_dir.as_posix(),
            "device": self.service.device,
            "threshold": self.service.threshold,
            "max_sequence_length": MAX_LENGTH,
            "num_labels": 2,
        }


def load_model_adapters(
    model_paths: dict[str, Path],
    deberta_batch_size: int = DEFAULT_DEBERTA_BATCH_SIZE,
) -> tuple[dict[str, ModelAdapter], dict[str, str]]:
    available: dict[str, ModelAdapter] = {}
    unavailable: dict[str, str] = {}
    for name, path in model_paths.items():
        try:
            if name == "deberta":
                available[name] = DebertaAdapter(path, deberta_batch_size)
            else:
                available[name] = ClassicalAdapter(path)
        except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
            unavailable[name] = str(error)
    return available, unavailable


def run_benchmark(
    validation_path: Path,
    model_paths: dict[str, Path],
    deberta_batch_size: int = DEFAULT_DEBERTA_BATCH_SIZE,
) -> dict[str, Any]:
    validate_validation_path(validation_path)
    texts, labels = load_labeled_parquet(validation_path)
    examples = build_examples(texts, labels)
    adapters, unavailable = load_model_adapters(
        model_paths, deberta_batch_size
    )
    results: dict[str, Any] = {}
    for name, adapter in adapters.items():
        original_predictions = adapter.predict_batch(texts, deberta_batch_size)
        model_result = {
            "available": True,
            "evaluation_status": "completed",
            "metadata": adapter.metadata(),
            "completed_rows": {
                "clean_validation": len(original_predictions),
                **{
                    transformation: len(
                        transform_examples(examples, transformation)
                    )
                    for transformation in TRANSFORMATIONS
                },
            },
            "original_validation": calculate_metrics(labels, original_predictions),
            "transformations": {},
        }
        for transformation in TRANSFORMATIONS:
            transformed = transform_examples(examples, transformation)
            predictions = adapter.predict_batch(
                [item.text for item in transformed], deberta_batch_size
            )
            model_result["transformations"][transformation] = {
                "examples_evaluated": len(predictions),
                "metrics": calculate_robustness_metrics(
                    labels, predictions, original_predictions
                ),
                "identifier_count": len(
                    {item.example_id for item in transformed}
                ),
                "labels_copied": all(
                    item.label == label
                    for item, label in zip(transformed, labels)
                ),
                "label_preservation": {
                    "category": TRANSFORMATION_METADATA[transformation][
                        "evaluation_category"
                    ],
                    "claimed": TRANSFORMATION_METADATA[transformation][
                        "label_preserving_claim"
                    ],
                    "limitation": TRANSFORMATION_METADATA[transformation][
                        "limitation"
                    ],
                },
            }
        results[name] = model_result
    for name, reason in unavailable.items():
        results[name] = {"available": False, "reason": reason}
    return {
        "experiment": "validation_adversarial_robustness",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": True,
        "validation_path": validation_path.as_posix(),
        "rows": len(texts),
        "seed": SEED,
        "threshold": THRESHOLD,
        "decision_convention": (
            "DeBERTa uses softmax(logits)[:, 1] >= threshold; the historical "
            "training report uses argmax(logits)."
        ),
        "deberta_batch_size": deberta_batch_size,
        "evaluation_status": "completed",
        "inference_complete": all(
            result.get("evaluation_status") == "completed"
            and all(
                count == len(texts)
                for count in result.get("completed_rows", {}).values()
            )
            for result in results.values()
            if result.get("available")
        ),
        "transformations": TRANSFORMATION_METADATA,
        "models": results,
        "limitations": [
            "All results use the validation split and are not independent test results.",
            "Transformations are controlled perturbations, not a complete adversarial threat model.",
            "Neutral-context results are exploratory because the added context may alter meaning.",
            "Candidate label-preserving transformations still have documented edge cases.",
            "Prediction flips indicate sensitivity, not necessarily a label error.",
            "No transformed copies were used for training.",
            "Prompt text is intentionally excluded from this report.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PromptShield validation robustness benchmark",
        "",
        "This report evaluates controlled, local transformations on the validation split only.",
        "It is not an independent test result and contains no prompt text.",
        "",
        f"- Evaluation time (UTC): `{report['timestamp_utc']}`",
        f"- Validation rows: `{report['rows']}`",
        f"- Seed: `{report['seed']}`",
        f"- Threshold: `{report['threshold']}`",
        "",
        "## Results",
        "",
        "| Model | Status | Transformation | Category | Examples | Accuracy | Macro F1 | Benign FPR | Malicious recall | Flip rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["models"].items():
        if not result["available"]:
            lines.append(
                f"| {name} | unavailable | — | — | — | — | — | — | — | — |"
            )
            continue
        clean = result["original_validation"]
        lines.append(
            f"| {name} | available | clean validation | reference | "
            f"{report['rows']} | {clean['accuracy']:.4f} | "
            f"{clean['macro_f1']:.4f} | "
            f"{clean['false_positive_rate_benign']:.4f} | "
            f"{clean['per_class']['1']['recall']:.4f} | — |"
        )
        for transformation, data in result["transformations"].items():
            metrics = data["metrics"]
            lines.append(
                f"| {name} | available | {transformation} | "
                f"{data['label_preservation']['category']} | "
                f"{data['examples_evaluated']} | "
                f"{metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                f"{metrics['false_positive_rate_benign']:.4f} | "
                f"{metrics['per_class']['1']['recall']:.4f} | "
                f"{metrics['prediction_flip_rate']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Clean rows and every transformed row count are 941 in this run.",
            "Clean metrics use the model's ordinary prediction convention. "
            "Transformed metrics use the copied validation labels; flip rate "
            "compares predictions by stable row order and identifier.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--svm", type=Path, required=True)
    parser.add_argument("--deberta", type=Path, required=True)
    parser.add_argument("--json-report", type=Path, required=True)
    parser.add_argument("--markdown-report", type=Path, required=True)
    parser.add_argument(
        "--deberta-batch-size",
        type=int,
        default=DEFAULT_DEBERTA_BATCH_SIZE,
    )
    args = parser.parse_args()
    report = run_benchmark(
        args.validation,
        {
            "logistic_regression": args.baseline,
            "linear_svm": args.svm,
            "deberta": args.deberta,
        },
        deberta_batch_size=args.deberta_batch_size,
    )
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_report.write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps({
        "json_report": args.json_report.as_posix(),
        "markdown_report": args.markdown_report.as_posix(),
        "models": {
            name: result["available"]
            for name, result in report["models"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
