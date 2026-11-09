"""Validation-only error analysis for the local DeBERTa classifier."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

from promptshield.baseline import calculate_metrics
from promptshield.deberta_decision_audit import (
    APPROVED_VALIDATION_PATH,
    DEFAULT_BATCH_SIZE,
    _model_configuration,
    compare_decision_rules,
    hash_checkpoint_files,
)
from promptshield.inference import DEFAULT_THRESHOLD, DebertaInference

METADATA_FIELDS = ("category", "source", "severity", "augmented")
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


class ErrorAnalysisFailure(RuntimeError):
    """Raised with a report-safe category for CLI failure reporting."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _canonical_validation_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved != APPROVED_VALIDATION_PATH:
        raise ValueError(
            "error analysis accepts only the repository-approved "
            "data/raw/validation.parquet path"
        )
    return resolved


def load_validation_records(
    validation_path: Path,
) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    """Load only the approved validation split and its optional metadata."""
    _canonical_validation_path(validation_path)
    table = parquet.read_table(validation_path)
    columns = set(table.column_names)
    missing = {TEXT_COLUMN, LABEL_COLUMN} - columns
    if missing:
        raise ValueError("validation data is missing required columns")

    selected = [TEXT_COLUMN, LABEL_COLUMN]
    selected.extend(field for field in METADATA_FIELDS if field in columns)
    rows = table.select(selected).to_pylist()
    if not rows:
        raise ValueError("validation data contains no rows")

    texts: list[str] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        text = row.get(TEXT_COLUMN)
        label = row.get(LABEL_COLUMN)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("validation data contains an invalid text value")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("validation data contains an invalid label value")
        texts.append(text)
        labels.append(int(label))
        metadata.append({
            field: row.get(field) for field in METADATA_FIELDS
        })
    return texts, labels, metadata


def _safe_metadata_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "<missing>"
        return str(value)
    return "<unserializable>"


def _counts(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    true_negative = sum(
        actual == 0 and predicted == 0
        for actual, predicted in zip(labels, predictions)
    )
    false_positive = sum(
        actual == 0 and predicted == 1
        for actual, predicted in zip(labels, predictions)
    )
    false_negative = sum(
        actual == 1 and predicted == 0
        for actual, predicted in zip(labels, predictions)
    )
    true_positive = sum(
        actual == 1 and predicted == 1
        for actual, predicted in zip(labels, predictions)
    )
    benign_total = true_negative + false_positive
    malicious_total = false_negative + true_positive
    return {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "rates": {
            "false_positive_rate": {
                "numerator": false_positive,
                "denominator": benign_total,
                "value": (
                    false_positive / benign_total if benign_total else None
                ),
            },
            "false_negative_rate": {
                "numerator": false_negative,
                "denominator": malicious_total,
                "value": (
                    false_negative / malicious_total
                    if malicious_total else None
                ),
            },
        },
    }


def _metadata_breakdown(
    labels: list[int],
    predictions: list[int],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    for field in METADATA_FIELDS:
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(metadata):
            groups.setdefault(
                _safe_metadata_value(row.get(field)), []
            ).append(index)
        field_result: dict[str, Any] = {}
        for value, indexes in sorted(groups.items()):
            group_labels = [labels[index] for index in indexes]
            group_predictions = [predictions[index] for index in indexes]
            counts = _counts(group_labels, group_predictions)
            field_result[value] = {
                "sample_count": len(indexes),
                "actual_label_counts": {
                    "0": group_labels.count(0),
                    "1": group_labels.count(1),
                },
                "predicted_label_counts": {
                    "0": group_predictions.count(0),
                    "1": group_predictions.count(1),
                },
                "counts": counts,
                "metrics": calculate_metrics(
                    group_labels, group_predictions
                ),
            }
        breakdown[field] = field_result
    return breakdown


def analyze_validation_errors(
    validation_path: Path,
    checkpoint_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Analyze threshold decisions on the approved validation split only."""
    _canonical_validation_path(validation_path)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    texts, labels, metadata = load_validation_records(validation_path)
    service = DebertaInference(
        checkpoint_dir, threshold=threshold, device="auto"
    )
    service.load()
    logits = service.predict_logits_batch(
        texts, batch_size=batch_size, enforce_prompt_limit=False
    )
    comparison = compare_decision_rules(labels, logits, threshold)
    predictions = comparison["threshold"]["predictions"]
    if len(predictions) != len(labels):
        raise ValueError("model returned an incomplete prediction batch")

    return {
        "experiment": "deberta_validation_error_analysis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "completed",
        "validation_only": True,
        "validation_path": validation_path.as_posix(),
        "validation_rows": len(labels),
        "label_distribution": {
            "0": labels.count(0),
            "1": labels.count(1),
        },
        "model_configuration": _model_configuration(service),
        "checkpoint": {
            "directory": checkpoint_dir.as_posix(),
            "files": hash_checkpoint_files(checkpoint_dir),
        },
        "metrics": comparison["threshold"]["metrics"],
        "counts": _counts(labels, predictions),
        "metadata_fields_available": [
            field for field in METADATA_FIELDS
            if any(row.get(field) is not None for row in metadata)
        ],
        "metadata_breakdown": _metadata_breakdown(
            labels, predictions, metadata
        ),
        "test_split_loaded": False,
        "prompt_text_included": False,
        "limitations": [
            "This analysis uses validation data only and does not estimate held-out test performance.",
            "Metadata group rates can be unstable when sample_count is small.",
            "The threshold was not tuned or selected from this analysis.",
            "Checkpoint hashes identify the artifact captured during this run; they do not prove historical training provenance.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# PromptShield DeBERTa validation error analysis",
        "",
        "Aggregate validation analysis only; no prompt text is included.",
        "",
        f"- Validation rows: `{report['validation_rows']}`",
        f"- Threshold: `{report['model_configuration']['threshold']}`",
        f"- True negatives: `{counts['true_negative']}`",
        f"- False positives: `{counts['false_positive']}`",
        f"- False negatives: `{counts['false_negative']}`",
        f"- True positives: `{counts['true_positive']}`",
        "",
        "## Metadata breakdowns",
        "",
    ]
    for field, groups in report["metadata_breakdown"].items():
        lines.append(f"### `{field}`")
        lines.extend([
            "",
            "| Value | Samples | FP | FN | TP | TN | FPR | FNR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for value, details in groups.items():
            group_counts = details["counts"]
            rates = group_counts["rates"]
            fpr = rates["false_positive_rate"]["value"]
            fnr = rates["false_negative_rate"]["value"]
            lines.append(
                f"| {value} | {details['sample_count']} | "
                f"{group_counts['false_positive']} | "
                f"{group_counts['false_negative']} | "
                f"{group_counts['true_positive']} | "
                f"{group_counts['true_negative']} | "
                f"{fpr if fpr is not None else 'n/a'} | "
                f"{fnr if fnr is not None else 'n/a'} |"
            )
        lines.append("")
    lines.extend([
        "## Limitations",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ])
    return "\n".join(lines)


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    """Write uniquely named JSON and Markdown reports without overwriting."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"deberta-error-analysis-{stamp}.json"
    markdown_path = reports_dir / f"deberta-error-analysis-{stamp}.md"
    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write(render_markdown(report))
    return json_path, markdown_path


def _failure_category(error: Exception) -> str:
    if isinstance(error, ErrorAnalysisFailure):
        return error.category
    if isinstance(error, FileNotFoundError):
        return "missing checkpoint or validation input"
    if isinstance(error, ValueError):
        return "invalid validation input or predictions"
    return "error analysis failed"


def write_failure_report(
    reports_dir: Path,
    error: Exception,
) -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "deberta_validation_error_analysis",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "failed",
        "validation_only": True,
        "error": _failure_category(error),
        "test_split_loaded": False,
        "prompt_text_included": False,
        "metrics": None,
    }
    json_path = reports_dir / f"deberta-error-analysis-{stamp}.json"
    markdown_path = reports_dir / f"deberta-error-analysis-{stamp}.md"
    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write(
            "# PromptShield DeBERTa validation error analysis\n\n"
            "The validation-only analysis failed before complete metrics "
            "were calculated.\n\n"
            f"- Evaluation status: `failed`\n"
            f"- Error category: `{report['error']}`\n"
            "- Prompt text included: `false`\n"
        )
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    try:
        report = analyze_validation_errors(
            args.validation, args.checkpoint, batch_size=args.batch_size
        )
        json_path, markdown_path = write_reports(report, args.reports_dir)
        print(json.dumps({
            "json_report": json_path.as_posix(),
            "markdown_report": markdown_path.as_posix(),
            "validation_rows": report["validation_rows"],
        }, indent=2))
        return 0
    except Exception as error:
        print(f"error analysis failed: {type(error).__name__}", file=sys.stderr)
        try:
            json_path, markdown_path = write_failure_report(
                args.reports_dir, error
            )
        except Exception:
            print("error analysis failure report could not be written",
                  file=sys.stderr)
            return 1
        print(json.dumps({
            "json_report": json_path.as_posix(),
            "markdown_report": markdown_path.as_posix(),
            "evaluation_status": "failed",
        }, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
