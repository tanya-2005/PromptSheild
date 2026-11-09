"""Validation-only comparison of PromptShield's three classifiers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from promptshield.baseline import calculate_metrics, load_labeled_parquet
from promptshield.deberta_decision_audit import (
    APPROVED_VALIDATION_PATH,
    DEFAULT_BATCH_SIZE,
    _model_configuration,
)
from promptshield.inference import DEFAULT_THRESHOLD, DebertaInference

DEFAULT_LOGISTIC_ARTIFACT = Path("models/baseline.joblib")
DEFAULT_SVM_ARTIFACT = Path("models/linear-svm.joblib")
DEFAULT_DEBERTA_CHECKPOINT = Path("models/deberta-v3-small")


def _validate_validation_path(validation_path: Path) -> None:
    if validation_path.resolve(strict=False) != APPROVED_VALIDATION_PATH:
        raise ValueError(
            "comparison accepts only the repository-approved "
            "data/raw/validation.parquet path"
        )


def _validate_predictions(
    name: str,
    labels: list[int],
    predictions: list[int],
) -> None:
    if len(predictions) != len(labels):
        raise ValueError(
            f"{name} returned {len(predictions)} predictions; "
            f"expected {len(labels)}"
        )
    if any(prediction not in (0, 1) for prediction in predictions):
        raise ValueError(f"{name} returned a non-binary prediction")


def _timed_predict(
    pipeline: Any,
    texts: list[str],
    name: str,
) -> tuple[list[int], dict[str, Any]]:
    started = time.perf_counter()
    predictions = pipeline.predict(texts)
    elapsed = time.perf_counter() - started
    values = [int(value) for value in predictions.tolist()]
    return values, {
        "model": name,
        "measured_examples": len(texts),
        "elapsed_seconds": elapsed,
        "mean_ms_per_text": (
            elapsed / len(texts) * 1000 if texts else None
        ),
        "method": "One full validation-batch prediction using perf_counter.",
    }


def _load_classical_artifact(path: Path, name: str) -> tuple[Any, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} artifact not found")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"{name} artifact does not contain a pipeline")
    return artifact["pipeline"], {
        "artifact_path": path.as_posix(),
        "experiment": artifact.get("experiment"),
        "decision_rule": "pipeline.predict(texts) -> numeric class label",
        "score_interpretation": (
            "Logistic Regression probabilities are model scores; "
            "no calibration claim is made."
            if name == "logistic_regression"
            else "LinearSVC decision_function scores are not calibrated probabilities."
        ),
    }


def compare_models(
    validation_path: Path,
    logistic_artifact: Path = DEFAULT_LOGISTIC_ARTIFACT,
    svm_artifact: Path = DEFAULT_SVM_ARTIFACT,
    deberta_checkpoint: Path = DEFAULT_DEBERTA_CHECKPOINT,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Evaluate all three saved models on one validation split only."""
    _validate_validation_path(validation_path)
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    texts, labels = load_labeled_parquet(validation_path)
    logistic, logistic_info = _load_classical_artifact(
        logistic_artifact, "logistic_regression"
    )
    svm, svm_info = _load_classical_artifact(svm_artifact, "linear_svm")

    model_reports: dict[str, Any] = {}
    logistic_predictions, logistic_latency = _timed_predict(
        logistic, texts, "logistic_regression"
    )
    _validate_predictions("logistic_regression", labels, logistic_predictions)
    model_reports["logistic_regression"] = {
        "metrics": calculate_metrics(labels, logistic_predictions),
        "latency": logistic_latency,
        "configuration": logistic_info,
    }

    svm_predictions, svm_latency = _timed_predict(
        svm, texts, "linear_svm"
    )
    _validate_predictions("linear_svm", labels, svm_predictions)
    model_reports["linear_svm"] = {
        "metrics": calculate_metrics(labels, svm_predictions),
        "latency": svm_latency,
        "configuration": svm_info,
    }

    service = DebertaInference(
        deberta_checkpoint, threshold=threshold, device="auto"
    )
    service.load()
    started = time.perf_counter()
    logits = service.predict_logits_batch(
        texts, batch_size=batch_size, enforce_prompt_limit=False
    )
    elapsed = time.perf_counter() - started
    if len(logits) != len(labels):
        raise ValueError(
            f"deberta returned {len(logits)} predictions; expected {len(labels)}"
        )
    from promptshield.deberta_decision_audit import compare_decision_rules

    comparison = compare_decision_rules(labels, logits, threshold)
    deberta_predictions = comparison["threshold"]["predictions"]
    _validate_predictions("deberta", labels, deberta_predictions)
    model_reports["deberta"] = {
        "metrics": comparison["threshold"]["metrics"],
        "latency": {
            "model": "deberta",
            "measured_examples": len(texts),
            "elapsed_seconds": elapsed,
            "mean_ms_per_text": elapsed / len(texts) * 1000,
            "method": "One full validation-batch raw-logit inference using perf_counter.",
        },
        "configuration": {
            "checkpoint_directory": deberta_checkpoint.as_posix(),
            "decision_rule": "softmax(logits)[:, 1] >= threshold",
            "threshold": threshold,
            "model_configuration": _model_configuration(service),
            "score_interpretation": (
                "Class-1 softmax scores are model outputs; calibration has "
                "not been established."
            ),
        },
    }

    return {
        "experiment": "promptshield_validation_model_comparison",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "completed",
        "validation_only": True,
        "validation_path": validation_path.as_posix(),
        "validation_rows": len(labels),
        "label_distribution": {
            "0": labels.count(0),
            "1": labels.count(1),
        },
        "models": model_reports,
        "decision_rule_notes": [
            "Logistic Regression and Linear SVM use their serialized pipeline.predict numeric class decisions.",
            "DeBERTa uses class-1 softmax output >= 0.90, matching the current inference adapter.",
            "LinearSVC decision_function values are not probabilities.",
            "No threshold was tuned and no model was retrained.",
        ],
        "test_split_loaded": False,
        "prompt_text_included": False,
        "limitations": [
            "All metrics are validation-only and do not establish held-out test performance.",
            "Latency depends on hardware, runtime, batch behavior, and process state; these measurements are not a production capacity benchmark.",
            "The three models have different score semantics and preprocessing/runtime costs, so score magnitudes are not directly comparable.",
            "Checkpoint and artifact identity do not prove historical training provenance.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PromptShield validation model comparison",
        "",
        "Validation-only aggregate comparison; no prompt text is included.",
        "",
        f"- Validation rows: `{report['validation_rows']}`",
        "",
        "## Metrics",
        "",
        "| Model | Accuracy | Macro F1 | Benign FPR | Malicious FNR | Mean ms/text |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, model in report["models"].items():
        metrics = model["metrics"]
        fpr = metrics["false_positive_rate_benign"]
        fnr = metrics["false_negative_rate_malicious"]
        latency = model["latency"]["mean_ms_per_text"]
        lines.append(
            f"| {name} | {metrics['accuracy']:.6f} | "
            f"{metrics['macro_f1']:.6f} | "
            f"{fpr if fpr is not None else 'n/a'} | "
            f"{fnr if fnr is not None else 'n/a'} | "
            f"{latency if latency is not None else 'n/a'} |"
        )
    lines.extend(["", "## Decision rules", ""])
    lines.extend(f"- {note}" for note in report["decision_rule_notes"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"validation-model-comparison-{stamp}.json"
    markdown_path = reports_dir / f"validation-model-comparison-{stamp}.md"
    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write(render_markdown(report))
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--logistic-artifact", type=Path, default=DEFAULT_LOGISTIC_ARTIFACT)
    parser.add_argument("--svm-artifact", type=Path, default=DEFAULT_SVM_ARTIFACT)
    parser.add_argument("--deberta-checkpoint", type=Path, default=DEFAULT_DEBERTA_CHECKPOINT)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    try:
        report = compare_models(
            args.validation,
            args.logistic_artifact,
            args.svm_artifact,
            args.deberta_checkpoint,
            batch_size=args.batch_size,
        )
        json_path, markdown_path = write_reports(report, args.reports_dir)
        print(json.dumps({
            "json_report": json_path.as_posix(),
            "markdown_report": markdown_path.as_posix(),
            "validation_rows": report["validation_rows"],
        }, indent=2))
        return 0
    except Exception as error:
        print(
            f"model comparison failed: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
