"""Validate the small, explicit data contract used by PromptShield."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ALLOWED_LABELS = {0, 1}
ALLOWED_SPLITS = {"train", "validation", "test"}
REQUIRED_COLUMNS = {"text", "label"}


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    label_counts: dict[int, int]
    duplicate_texts: int
    invalid_rows: list[str]

    @property
    def valid(self) -> bool:
        return not self.invalid_rows


def validate_csv(path: Path) -> ValidationReport:
    """Validate a CSV without modifying it or silently dropping bad rows."""
    label_counts: Counter[int] = Counter()
    seen_texts: Counter[str] = Counter()
    invalid_rows: list[str] = []
    rows = 0

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            return ValidationReport(
                rows=0,
                label_counts={},
                duplicate_texts=0,
                invalid_rows=[f"missing required columns: {sorted(missing)}"],
            )

        for row_number, row in enumerate(reader, start=2):
            rows += 1
            text = (row.get("text") or "").strip()
            raw_label = (row.get("label") or "").strip()
            errors: list[str] = []
            if not text:
                errors.append("text is empty")
            try:
                label = int(raw_label)
            except ValueError:
                errors.append("label is not an integer")
                label = None
            if label is not None and label not in ALLOWED_LABELS:
                errors.append("label must be 0 or 1")
            split = (row.get("split") or "").strip()
            if split and split not in ALLOWED_SPLITS:
                errors.append("split must be train, validation, or test")

            if errors:
                invalid_rows.append(f"row {row_number}: {', '.join(errors)}")
                continue
            label_counts[label] += 1
            seen_texts[text] += 1

    duplicate_texts = sum(count - 1 for count in seen_texts.values() if count > 1)
    return ValidationReport(
        rows=rows,
        label_counts=dict(sorted(label_counts.items())),
        duplicate_texts=duplicate_texts,
        invalid_rows=invalid_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    report = validate_csv(args.csv_path)
    print(json.dumps({
        "rows": report.rows,
        "label_counts": report.label_counts,
        "duplicate_texts": report.duplicate_texts,
        "invalid_rows": report.invalid_rows,
        "valid": report.valid,
    }, indent=2))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

