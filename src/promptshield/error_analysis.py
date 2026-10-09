"""Analyze validation errors from a saved PromptShield baseline artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import pyarrow.parquet as parquet

DISPLAY_TEXT_LIMIT = 300
ERROR_COLUMNS = (
    "text",
    "label",
    "category",
    "source",
    "severity",
    "group_id",
    "augmented",
    "tags",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_by_metadata(
    errors: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, int]]:
    counts: Counter[tuple[str, str]] = Counter(
        (str(error["metadata"].get(field)), error["error_type"])
        for error in errors
    )
    values = sorted({value for value, _ in counts})
    return {
        value: {
            "false_positive": counts[(value, "false_positive")],
            "false_negative": counts[(value, "false_negative")],
            "total": counts[(value, "false_positive")]
            + counts[(value, "false_negative")],
        }
        for value in values
    }


def _observations(errors: list[dict[str, Any]]) -> list[dict[str, str]]:
    observations: list[dict[str, str]] = []
    false_positives = [e for e in errors if e["error_type"] == "false_positive"]
    false_negatives = [e for e in errors if e["error_type"] == "false_negative"]
    if false_positives:
        observations.append({
            "type": "observation",
            "text": (
                f"{len(false_positives)} benign validation examples were "
                "predicted as malicious."
            ),
        })
    if false_negatives:
        observations.append({
            "type": "observation",
            "text": (
                f"{len(false_negatives)} malicious validation examples were "
                "predicted as benign."
            ),
        })
    category_counts = Counter(
        e["metadata"].get("category")
        for e in false_positives
        if e["metadata"].get("category")
    )
    if category_counts:
        category, count = category_counts.most_common(1)[0]
        observations.append({
            "type": "observation",
            "text": (
                f"The most common false-positive category in this validation "
                f"sample was {category!r} ({count} examples)."
            ),
        })
    observations.append({
        "type": "hypothesis",
        "text": (
            "Repeated metadata categories or lexical patterns may indicate "
            "dataset-specific ambiguity, but this report cannot establish "
            "causation or real-world generalization."
        ),
    })
    return observations


def analyze_validation_errors(
    artifact_path: Path,
    validation_path: Path,
    validation_report_path: Path,
) -> dict[str, Any]:
    """Create a complete, validation-only error analysis."""
    artifact = joblib.load(artifact_path)
    pipeline = artifact["pipeline"]
    table = parquet.read_table(validation_path, columns=list(ERROR_COLUMNS))
    rows = table.to_pylist()
    texts = [row["text"] for row in rows]
    labels = [int(row["label"]) for row in rows]
    predictions = pipeline.predict(texts).tolist()
    probabilities = pipeline.predict_proba(texts)[:, 1].tolist()

    errors: list[dict[str, Any]] = []
    for index, (row, prediction, probability) in enumerate(
        zip(rows, predictions, probabilities),
        start=1,
    ):
        if prediction == row["label"]:
            continue
        error_type = (
            "false_positive" if row["label"] == 0 else "false_negative"
        )
        full_text = row["text"]
        errors.append({
            "validation_row": index,
            "error_type": error_type,
            "true_label": int(row["label"]),
            "predicted_label": int(prediction),
            "predicted_malicious_probability": float(probability),
            "display_text": full_text[:DISPLAY_TEXT_LIMIT],
            "text_was_truncated_for_display": len(full_text) > DISPLAY_TEXT_LIMIT,
            "text": full_text,
            "metadata": {
                key: row.get(key) for key in ERROR_COLUMNS
                if key != "text"
            },
            "review_status": "pending",
            "review_decision": None,
        })

    report = {
        "experiment": artifact["experiment"],
        "baseline_configuration": {
            "feature_column": artifact["feature_column"],
            "target_column": artifact["target_column"],
            "seed": artifact["seed"],
            "pipeline_config": artifact["pipeline_config"],
            "decision_threshold": 0.5,
        },
        "validation": {
            "path": validation_path.as_posix(),
            "sha256": _sha256(validation_path),
            "rows": len(rows),
        },
        "source_artifacts": {
            "model_path": artifact_path.as_posix(),
            "model_sha256": _sha256(artifact_path),
            "validation_report_path": validation_report_path.as_posix(),
            "validation_report_sha256": _sha256(validation_report_path),
        },
        "error_counts": {
            "false_positive": sum(
                error["error_type"] == "false_positive" for error in errors
            ),
            "false_negative": sum(
                error["error_type"] == "false_negative" for error in errors
            ),
            "total": len(errors),
        },
        "errors": errors,
        "metadata_summary": {
            field: _count_by_metadata(errors, field)
            for field in ("source", "category", "severity")
        },
        "observations_and_hypotheses": _observations(errors),
        "near_duplicate_context": {
            "review_report": "reports/near-duplicate-review.md",
            "cross_split_candidates": 23,
            "interpretation": (
                "These are heuristic candidates, not confirmed leakage. "
                "Candidates involving train/validation could make validation "
                "performance optimistic if they are genuine template copies."
            ),
        },
        "test_split_loaded": False,
        "labels_or_threshold_changed": False,
    }
    return report


def write_review_worksheet(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# Validation error review worksheet",
        "",
        "Review status is pending for every row. Do not relabel automatically.",
        "Use the full text in the JSON report only when necessary.",
        "",
        "| Validation row | Error | True | Predicted | Malicious probability | Category | Source | Review decision |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    for error in report["errors"]:
        metadata = error["metadata"]
        lines.append(
            f"| {error['validation_row']} | {error['error_type']} | "
            f"{error['true_label']} | {error['predicted_label']} | "
            f"{error['predicted_malicious_probability']:.6f} | "
            f"{metadata.get('category', '')} | {metadata.get('source', '')} | "
            "pending |"
        )
    lines.extend([
        "",
        "Suggested review questions:",
        "",
        "- Is the dataset label defensible from the text and context?",
        "- Is this a lexical false alarm or a genuinely ambiguous example?",
        "- Does the error resemble another reviewed example?",
        "- Should the example be retained, flagged for investigation, or "
        "excluded only through a separately documented data decision?",
        "",
        "This worksheet records review decisions only; it does not change labels "
        "or the trained artifact.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--worksheet", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_validation_errors(
        args.artifact, args.validation, args.validation_report
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.worksheet.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_review_worksheet(report, args.worksheet)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
