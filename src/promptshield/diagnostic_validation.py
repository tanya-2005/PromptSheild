"""Validate manually authored diagnostic records without changing them."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ALLOWED_LABELS = {0, 1}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_REVIEW_STATUSES = {"approved", "review"}
REQUIRED_FIELDS = {
    "id",
    "text",
    "label",
    "category",
    "rationale",
    "difficulty",
    "authoring_provenance",
    "version",
    "review_status",
}


@dataclass(frozen=True)
class DiagnosticValidationReport:
    rows: int
    valid_rows: int
    excluded_for_review: int
    duplicate_ids: int
    invalid_rows: list[str]
    label_counts: dict[int, int]

    @property
    def valid(self) -> bool:
        return not self.invalid_rows


def validate_records(
    records: Iterable[dict[str, Any]],
) -> DiagnosticValidationReport:
    """Validate records and exclude, but do not count, examples under review."""
    rows = 0
    valid_rows = 0
    excluded_for_review = 0
    duplicate_ids = 0
    invalid_rows: list[str] = []
    label_counts: Counter[int] = Counter()
    seen_ids: set[str] = set()

    for row_number, record in enumerate(records, start=1):
        rows += 1
        errors: list[str] = []
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            errors.append(f"missing required fields: {sorted(missing)}")

        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            errors.append("id must be a non-empty string")
        elif record_id in seen_ids:
            duplicate_ids += 1
            errors.append("duplicate id")
        else:
            seen_ids.add(record_id)

        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append("text must be a non-empty string")

        status = record.get("review_status")
        if status not in ALLOWED_REVIEW_STATUSES:
            errors.append("review_status must be approved or review")

        label = record.get("label")
        if status == "approved":
            if label not in ALLOWED_LABELS:
                errors.append("approved records must have label 0 or 1")
        elif status == "review":
            if label is not None:
                errors.append("records under review must have a null label")
            else:
                excluded_for_review += 1

        if record.get("difficulty") not in ALLOWED_DIFFICULTIES:
            errors.append("difficulty must be easy, medium, or hard")

        if errors:
            invalid_rows.append(f"row {row_number}: {', '.join(errors)}")
            continue
        if status == "review":
            continue
        valid_rows += 1
        label_counts[label] += 1

    return DiagnosticValidationReport(
        rows=rows,
        valid_rows=valid_rows,
        excluded_for_review=excluded_for_review,
        duplicate_ids=duplicate_ids,
        invalid_rows=invalid_rows,
        label_counts=dict(sorted(label_counts.items())),
    )


def validate_jsonl(path: Path) -> DiagnosticValidationReport:
    """Validate one JSON object per line and reject malformed JSON explicitly."""
    records: list[dict[str, Any]] = []
    invalid_json: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_json.append(f"line {line_number}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(record, dict):
                invalid_json.append(f"line {line_number}: record must be an object")
                continue
            records.append(record)

    report = validate_records(records)
    if not invalid_json:
        return report
    return DiagnosticValidationReport(
        rows=report.rows + len(invalid_json),
        valid_rows=report.valid_rows,
        excluded_for_review=report.excluded_for_review,
        duplicate_ids=report.duplicate_ids,
        invalid_rows=invalid_json + report.invalid_rows,
        label_counts=report.label_counts,
    )
