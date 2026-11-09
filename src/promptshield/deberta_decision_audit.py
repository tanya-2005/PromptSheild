"""Validation-only audit comparing DeBERTa decision conventions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from promptshield.baseline import calculate_metrics, load_labeled_parquet
from promptshield.inference import (
    DEFAULT_THRESHOLD,
    LABEL_MAPPING,
    MAX_LENGTH,
    MODEL_ID,
    DebertaInference,
)

REQUIRED_LOGIT_WIDTH = 2
DEFAULT_BATCH_SIZE = 8
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
APPROVED_VALIDATION_PATH = (
    REPOSITORY_ROOT / "data" / "raw" / "validation.parquet"
)


class AuditFailure(RuntimeError):
    """Raised by the library API with a report-safe failure category."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def hash_checkpoint_files(checkpoint_dir: Path) -> list[dict[str, str]]:
    """Hash every regular file below a local checkpoint directory."""
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    files: list[dict[str, str]] = []
    for path in sorted(
        (item for item in checkpoint_dir.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(checkpoint_dir).as_posix(),
    ):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append({
            "path": path.relative_to(checkpoint_dir).as_posix(),
            "sha256": digest.hexdigest(),
        })
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {checkpoint_dir}")
    return files


def validate_audit_inputs(
    labels: list[int],
    logits: list[list[float]],
) -> None:
    """Reject incomplete or malformed audit inputs before metric calculation."""
    if not labels:
        raise ValueError("validation split contains no rows")
    if any(
        isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1)
        for label in labels
    ):
        raise ValueError("validation labels must be integers in {0, 1}")
    if len(logits) != len(labels):
        raise ValueError(
            f"model returned {len(logits)} logits; expected {len(labels)}"
        )
    for row_number, row in enumerate(logits, start=1):
        if len(row) != REQUIRED_LOGIT_WIDTH:
            raise ValueError(
                f"logit row {row_number} has {len(row)} values; expected 2"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in row
        ):
            raise ValueError(f"logit row {row_number} contains invalid values")


def compare_decision_rules(
    labels: list[int],
    logits: list[list[float]],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Calculate argmax and threshold metrics from validated two-class logits."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    validate_audit_inputs(labels, logits)
    argmax_predictions: list[int] = []
    threshold_predictions: list[int] = []
    class_1_probabilities: list[float] = []
    for class_0, class_1 in logits:
        argmax_predictions.append(0 if class_0 >= class_1 else 1)
        difference = float(class_1) - float(class_0)
        if difference >= 0:
            probability = 1.0 / (1.0 + math.exp(-difference))
        else:
            exponent = math.exp(difference)
            probability = exponent / (1.0 + exponent)
        class_1_probabilities.append(probability)
        threshold_predictions.append(1 if probability >= threshold else 0)

    disagreements = [
        left != right
        for left, right in zip(argmax_predictions, threshold_predictions)
    ]
    return {
        "argmax": {
            "decision_convention": "argmax(logits)",
            "metrics": calculate_metrics(labels, argmax_predictions),
            "predictions": argmax_predictions,
        },
        "threshold": {
            "decision_convention": "softmax(logits)[:, 1] >= threshold",
            "threshold": threshold,
            "metrics": calculate_metrics(labels, threshold_predictions),
            "predictions": threshold_predictions,
        },
        "class_1_probability_summary": {
            "minimum": min(class_1_probabilities),
            "maximum": max(class_1_probabilities),
            "mean": sum(class_1_probabilities) / len(class_1_probabilities),
        },
        "decision_disagreement": {
            "count": sum(disagreements),
            "rate": sum(disagreements) / len(disagreements),
        },
    }


def _dependency_version(name: str, fallback: Any = None) -> Any:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _model_configuration(service: DebertaInference) -> dict[str, Any]:
    config = service._model.config
    return {
        "model_id": MODEL_ID,
        "architecture": getattr(config, "architectures", None),
        "model_type": getattr(config, "model_type", None),
        "num_labels": getattr(config, "num_labels", None),
        "id2label": getattr(config, "id2label", None),
        "label2id": getattr(config, "label2id", None),
        "label_mapping_expected_by_project": LABEL_MAPPING,
        "tokenizer_class": type(service._tokenizer).__name__,
        "tokenizer_declared_model_max_length": getattr(
            service._tokenizer, "model_max_length", None
        ),
        "effective_max_sequence_length": MAX_LENGTH,
        "threshold": service.threshold,
    }


def audit_validation(
    validation_path: Path,
    checkpoint_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Run the audit against validation.parquet and a local checkpoint only.

    This library function raises on failure. Persisted failure reports are
    guaranteed only when the CLI entry point, :func:`main`, is used.
    """
    resolved_validation = validation_path.resolve(strict=False)
    if resolved_validation != APPROVED_VALIDATION_PATH:
        raise ValueError(
            "audit accepts only the repository-approved "
            "data/raw/validation.parquet path"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    try:
        texts, labels = load_labeled_parquet(validation_path)
    except Exception as error:
        raise AuditFailure("invalid validation input") from error
    try:
        service = DebertaInference(
            checkpoint_dir, threshold=threshold, device="auto"
        )
        service.load()
    except FileNotFoundError as error:
        raise AuditFailure("missing checkpoint") from error
    except Exception as error:
        raise AuditFailure("model-loading failure") from error
    try:
        logits = service.predict_logits_batch(
            texts, batch_size=batch_size, enforce_prompt_limit=False
        )
    except Exception as error:
        raise AuditFailure("inference failure") from error
    try:
        validate_audit_inputs(labels, logits)
    except Exception as error:
        raise AuditFailure("invalid logits or labels") from error
    try:
        comparison = compare_decision_rules(labels, logits, threshold)
    except Exception as error:
        raise AuditFailure("metric calculation failure") from error
    return {
        "experiment": "deberta_validation_decision_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": True,
        "validation_path": validation_path.as_posix(),
        "validation_rows": len(labels),
        "label_distribution": {
            "0": labels.count(0),
            "1": labels.count(1),
        },
        "checkpoint": {
            "directory": checkpoint_dir.as_posix(),
            "files": hash_checkpoint_files(checkpoint_dir),
            "hashes_establish_artifact_identity_only_from_this_audit_onward": True,
        },
        "model_configuration": _model_configuration(service),
        "runtime": {
            "python": platform.python_version(),
            "torch": getattr(service._torch, "__version__", None),
            "transformers": _dependency_version("transformers"),
            "sentencepiece": _dependency_version("sentencepiece"),
            "device": service.device,
            "batch_size": batch_size,
        },
        "comparison": comparison,
        "test_split_loaded": False,
        "prompt_text_included": False,
        "limitations": [
            "This audit uses validation data only and does not establish held-out test performance.",
            "Artifact hashes identify files captured by this audit; they do not prove historical training provenance.",
            "The project mapping is recorded as expected metadata, not inferred from confidence values.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    lines = [
        "# PromptShield DeBERTa validation decision audit",
        "",
        "This report contains validation metadata and aggregate metrics only; it does not contain prompt text.",
        "",
        f"- Validation rows: `{report['validation_rows']}`",
        f"- Evaluation time (UTC): `{report['timestamp_utc']}`",
        f"- Threshold: `{report['model_configuration']['threshold']}`",
        f"- Decision disagreements: `{comparison['decision_disagreement']['count']}` "
        f"({comparison['decision_disagreement']['rate']:.6f})",
        "",
        "## Decision-rule metrics",
        "",
        "| Rule | Accuracy | Macro F1 | Benign FPR | Malicious FNR | Confusion matrix |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in ("argmax", "threshold"):
        metrics = comparison[name]["metrics"]
        lines.append(
            f"| {name} | {metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} | "
            f"{metrics['false_positive_rate_benign']:.6f} | "
            f"{metrics['false_negative_rate_malicious']:.6f} | "
            f"`{metrics['confusion_matrix']}` |"
        )
    lines.extend([
        "",
        "Per-class precision, recall, and F1 are available in the JSON report under each rule's `metrics.per_class` field.",
        "",
        "## Provenance limitation",
        "",
        "The checkpoint file hashes establish artifact identity from the time of this audit onward. "
        "They do not, by themselves, prove which historical training run produced the checkpoint.",
        "",
    ])
    return "\n".join(lines)


def write_reports(
    report: dict[str, Any],
    reports_dir: Path,
) -> tuple[Path, Path]:
    """Write uniquely timestamped reports without overwriting existing files."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = reports_dir / f"deberta-decision-audit-{stamp}.json"
    markdown_path = reports_dir / f"deberta-decision-audit-{stamp}.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for path, contents in (
        (json_path, json.dumps(report, indent=2) + "\n"),
        (markdown_path, render_markdown(report)),
    ):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(contents)
    return json_path, markdown_path


def render_failure_markdown(report: dict[str, Any]) -> str:
    """Render a failure report without exposing inputs or partial metrics."""
    return "\n".join([
        "# PromptShield DeBERTa validation decision audit",
        "",
        "The validation-only audit failed before complete metrics were calculated.",
        "",
        f"- Evaluation time (UTC): `{report['timestamp_utc']}`",
        "- Evaluation status: `failed`",
        "- Validation only: `true`",
        "- Test split loaded: `false`",
        "- Prompt text included: `false`",
        f"- Error: `{report['error']}`",
        "",
    ])


def _safe_failure_description(error: Exception) -> str:
    if isinstance(error, AuditFailure):
        return error.category
    if isinstance(error, FileNotFoundError):
        return "missing checkpoint"
    if isinstance(error, ValueError):
        return "invalid validation input"
    if isinstance(error, RuntimeError):
        return "audit operation failed"
    return "unexpected audit failure"


def write_failure_report(
    reports_dir: Path,
    error: Exception,
) -> tuple[Path, Path]:
    """Write a unique failure report, surfacing report-write failures clearly."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = reports_dir / f"deberta-decision-audit-{stamp}.json"
    markdown_path = reports_dir / f"deberta-decision-audit-{stamp}.md"
    report = {
        "experiment": "deberta_validation_decision_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "failed",
        "validation_only": True,
        "error": _safe_failure_description(error),
        "test_split_loaded": False,
        "prompt_text_included": False,
        "metrics": None,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    try:
        with json_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
        with markdown_path.open("x", encoding="utf-8") as stream:
            stream.write(render_failure_markdown(report))
    except Exception as report_error:
        raise RuntimeError(
            "audit failed; failure report could not be written: "
            f"{type(report_error).__name__}: {report_error}"
        ) from report_error
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    try:
        report = audit_validation(
            args.validation, args.checkpoint, batch_size=args.batch_size
        )
        json_path, markdown_path = write_reports(report, args.reports_dir)
        print(json.dumps({
            "json_report": json_path.as_posix(),
            "markdown_report": markdown_path.as_posix(),
            "validation_rows": report["validation_rows"],
            "test_split_loaded": report["test_split_loaded"],
        }, indent=2))
        return 0
    except Exception as error:
        print(
            f"audit failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        try:
            json_path, markdown_path = write_failure_report(
                args.reports_dir, error
            )
        except RuntimeError as report_error:
            print(str(report_error))
            return 1
        print(json.dumps({
            "json_report": json_path.as_posix(),
            "markdown_report": markdown_path.as_posix(),
            "evaluation_status": "failed",
        }, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
