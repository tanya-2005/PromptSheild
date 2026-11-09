"""Privacy-safe aggregate investigation of DeBERTa validation errors."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from promptshield.deberta_decision_audit import (
    APPROVED_VALIDATION_PATH,
    DEFAULT_BATCH_SIZE,
    _model_configuration,
    compare_decision_rules,
    hash_checkpoint_files,
)
from promptshield.deberta_error_analysis import (
    METADATA_FIELDS,
    load_validation_records,
)
from promptshield.inference import DEFAULT_THRESHOLD, DebertaInference

REQUESTED_SOURCES = (
    "wildguard_judgecomp",
    "harmbench",
    "harmbench_benign",
    "neuralchemy_v1",
)
CONFUSION_GROUPS = (
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
)


def _class_1_probabilities(logits: list[list[float]]) -> list[float]:
    probabilities: list[float] = []
    for class_0, class_1 in logits:
        difference = float(class_1) - float(class_0)
        if difference >= 0:
            probabilities.append(1.0 / (1.0 + math.exp(-difference)))
        else:
            exponent = math.exp(difference)
            probabilities.append(exponent / (1.0 + exponent))
    return probabilities


def _confidence_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
            "below_threshold_count": 0,
            "at_or_above_threshold_count": 0,
        }
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return {
        "sample_count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
        "median": median,
    }


def _safe_metadata_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "<missing>"
        return str(value)[:200]
    return "<unserializable>"


def _group_statistics(
    indexes: list[int],
    labels: list[int],
    predictions: list[int],
    probabilities: list[float],
) -> dict[str, Any]:
    actual = [labels[index] for index in indexes]
    predicted = [predictions[index] for index in indexes]
    confidence = [probabilities[index] for index in indexes]
    counts = {
        "sample_count": len(indexes),
        "actual_label_counts": {
            "0": actual.count(0),
            "1": actual.count(1),
        },
        "predicted_label_counts": {
            "0": predicted.count(0),
            "1": predicted.count(1),
        },
        "true_positive": sum(a == 1 and p == 1 for a, p in zip(actual, predicted)),
        "true_negative": sum(a == 0 and p == 0 for a, p in zip(actual, predicted)),
        "false_positive": sum(a == 0 and p == 1 for a, p in zip(actual, predicted)),
        "false_negative": sum(a == 1 and p == 0 for a, p in zip(actual, predicted)),
    }
    benign_count = actual.count(0)
    malicious_count = actual.count(1)
    return {
        "sample_count": len(indexes),
        "actual_label_counts": counts["actual_label_counts"],
        "predicted_label_counts": counts["predicted_label_counts"],
        "counts": {
            key: counts[key]
            for key in CONFUSION_GROUPS
        },
        "rates": {
            "false_positive_rate": {
                "numerator": counts["false_positive"],
                "denominator": benign_count,
                "value": (
                    counts["false_positive"] / benign_count
                    if benign_count else None
                ),
            },
            "false_negative_rate": {
                "numerator": counts["false_negative"],
                "denominator": malicious_count,
                "value": (
                    counts["false_negative"] / malicious_count
                    if malicious_count else None
                ),
            },
        },
        "confidence": _confidence_summary(confidence),
    }


def _breakdown(
    field: str,
    labels: list[int],
    predictions: list[int],
    probabilities: list[float],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        groups.setdefault(_safe_metadata_value(row.get(field)), []).append(index)
    return {
        value: _group_statistics(indexes, labels, predictions, probabilities)
        for value, indexes in sorted(groups.items())
    }


def _metadata_label_consistency(
    labels: list[int],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("category", "source"):
        groups: dict[str, dict[str, int]] = {}
        for label, row in zip(labels, metadata):
            value = _safe_metadata_value(row.get(field))
            counts = groups.setdefault(value, {"0": 0, "1": 0})
            counts[str(label)] += 1
        mixed = sorted(
            value for value, counts in groups.items()
            if counts["0"] and counts["1"]
        )
        result[field] = {
            "label_counts_by_value": groups,
            "potential_mixed_label_values": mixed,
            "interpretation": (
                "Mixed labels indicate metadata/label combinations worth "
                "reviewing; metadata is not treated as ground truth."
            ),
        }
    return result


def investigate_validation_errors(
    validation_path: Path,
    checkpoint_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Generate aggregate confidence and metadata findings for validation only."""
    resolved = validation_path.resolve(strict=False)
    if resolved != APPROVED_VALIDATION_PATH:
        raise ValueError(
            "investigation accepts only the repository-approved "
            "data/raw/validation.parquet path"
        )
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
    probabilities = _class_1_probabilities(logits)
    if len(predictions) != len(labels) or len(probabilities) != len(labels):
        raise ValueError("model returned an incomplete prediction batch")

    groups: dict[str, list[int]] = {name: [] for name in CONFUSION_GROUPS}
    for index, (label, prediction) in enumerate(zip(labels, predictions)):
        if label == 1 and prediction == 1:
            groups["true_positive"].append(index)
        elif label == 0 and prediction == 0:
            groups["true_negative"].append(index)
        elif label == 0 and prediction == 1:
            groups["false_positive"].append(index)
        else:
            groups["false_negative"].append(index)

    return {
        "experiment": "deberta_validation_error_investigation",
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
        "confidence_by_confusion_group": {
            name: _group_statistics(
                indexes, labels, predictions, probabilities
            )
            for name, indexes in groups.items()
        },
        "requested_source_comparison": {
            source: _breakdown(
                "source", labels, predictions, probabilities,
                metadata,
            ).get(source, _group_statistics(
                [], labels, predictions, probabilities
            ))
            for source in REQUESTED_SOURCES
        },
        "metadata_breakdowns": {
            field: _breakdown(
                field, labels, predictions, probabilities, metadata
            )
            for field in METADATA_FIELDS
        },
        "metadata_label_consistency": _metadata_label_consistency(
            labels, metadata
        ),
        "test_split_loaded": False,
        "prompt_text_included": False,
        "limitations": [
            "This investigation uses validation data only and does not establish held-out test performance.",
            "Confidence summaries describe model scores; they are not calibrated probabilities.",
            "Metadata is descriptive and is not treated as ground truth for labels.",
            "Small groups can produce unstable rates and confidence summaries.",
            "The threshold was not tuned or changed for this investigation.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PromptShield DeBERTa validation error investigation",
        "",
        "Aggregate confidence and metadata analysis only; no prompt text is included.",
        "",
        f"- Validation rows: `{report['validation_rows']}`",
        f"- Threshold: `{report['model_configuration']['threshold']}`",
        "",
        "## Confidence by confusion group",
        "",
        "| Group | Samples | Mean class-1 score | Median | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, details in report["confidence_by_confusion_group"].items():
        confidence = details["confidence"]
        lines.append(
            f"| {name} | {details['sample_count']} | "
            f"{confidence['mean'] if confidence['mean'] is not None else 'n/a'} | "
            f"{confidence['median'] if confidence['median'] is not None else 'n/a'} | "
            f"{confidence['minimum'] if confidence['minimum'] is not None else 'n/a'} | "
            f"{confidence['maximum'] if confidence['maximum'] is not None else 'n/a'} |"
        )
    lines.extend(["", "## Requested source comparison", ""])
    lines.extend([
        "| Source | Samples | FP | FN | FPR | FNR | Mean score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for source, details in report["requested_source_comparison"].items():
        counts = details["counts"]
        rates = details["rates"]
        mean = details["confidence"]["mean"]
        lines.append(
            f"| {source} | {details['sample_count']} | "
            f"{counts['false_positive']} | {counts['false_negative']} | "
            f"{rates['false_positive_rate']['value'] if rates['false_positive_rate']['value'] is not None else 'n/a'} | "
            f"{rates['false_negative_rate']['value'] if rates['false_negative_rate']['value'] is not None else 'n/a'} | "
            f"{mean if mean is not None else 'n/a'} |"
        )
    lines.extend(["", "## Metadata/label consistency", ""])
    for field, details in report["metadata_label_consistency"].items():
        lines.append(
            f"- `{field}` values with both numeric labels: "
            f"{details['potential_mixed_label_values'] or 'none'}"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"deberta-error-investigation-{stamp}.json"
    markdown_path = reports_dir / f"deberta-error-investigation-{stamp}.md"
    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write(render_markdown(report))
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    try:
        report = investigate_validation_errors(
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
        print(
            f"error investigation failed: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
