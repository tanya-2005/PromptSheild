"""Inspect downloaded PromptShield Parquet splits without changing them."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")
EXPECTED_COLUMNS = (
    "text",
    "label",
    "category",
    "source",
    "severity",
    "group_id",
    "augmented",
    "tags",
)


@dataclass(frozen=True)
class SplitReport:
    rows: int
    columns: list[str]
    label_counts: dict[str, int]
    exact_duplicate_rows: int
    missing_group_ids: int
    unique_group_ids: int
    augmented_rows: int
    source_counts: dict[str, int]
    category_counts: dict[str, int]


@dataclass(frozen=True)
class DatasetInspection:
    splits: dict[str, SplitReport]
    cross_split_exact_text_overlap: dict[str, int]
    cross_split_group_overlap: dict[str, int]
    near_duplicate_candidate_pairs: int
    near_duplicate_cross_split_pairs: int
    near_duplicate_method: str


@dataclass(frozen=True)
class NearDuplicateCandidate:
    left_split: str
    left_row: int
    left_label: int
    left_category: str
    right_split: str
    right_row: int
    right_label: int
    right_category: str
    similarity: float
    left_text: str
    right_text: str


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _split_report(rows: list[dict[str, Any]], columns: list[str]) -> SplitReport:
    texts = Counter(_normalise_text(str(row["text"])) for row in rows)
    return SplitReport(
        rows=len(rows),
        columns=columns,
        label_counts=dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        exact_duplicate_rows=sum(count - 1 for count in texts.values() if count > 1),
        missing_group_ids=sum(not row.get("group_id") for row in rows),
        unique_group_ids=len({row.get("group_id") for row in rows}),
        augmented_rows=sum(bool(row.get("augmented")) for row in rows),
        source_counts=dict(Counter(str(row.get("source")) for row in rows)),
        category_counts=dict(Counter(str(row.get("category")) for row in rows)),
    )


def _near_duplicate_counts(
    split_rows: dict[str, list[dict[str, Any]]],
) -> tuple[int, int]:
    """Find conservative candidates; this is a review signal, not a label."""
    buckets: dict[str, list[tuple[str, str]]] = {}
    for split, rows in split_rows.items():
        for row in rows:
            text = _normalise_text(str(row["text"]))
            buckets.setdefault(text[:50], []).append((split, text))

    total = cross_split = 0
    for values in buckets.values():
        for index, (left_split, left_text) in enumerate(values):
            for right_split, right_text in values[index + 1 :]:
                if left_text == right_text:
                    continue
                if SequenceMatcher(None, left_text, right_text).ratio() >= 0.95:
                    total += 1
                    cross_split += left_split != right_split
    return total, cross_split


def near_duplicate_candidates(
    split_rows: dict[str, list[dict[str, Any]]],
    cross_split_only: bool = False,
) -> list[NearDuplicateCandidate]:
    """Return review candidates without changing or filtering source rows."""
    buckets: dict[str, list[tuple[str, int, dict[str, Any], str]]] = {}
    for split, rows in split_rows.items():
        for row_number, row in enumerate(rows, start=1):
            text = _normalise_text(str(row["text"]))
            buckets.setdefault(text[:50], []).append(
                (split, row_number, row, text)
            )

    candidates: list[NearDuplicateCandidate] = []
    for values in buckets.values():
        for index, (left_split, left_row, left, left_text) in enumerate(values):
            for right_split, right_row, right, right_text in values[index + 1 :]:
                if left_text == right_text:
                    continue
                if cross_split_only and left_split == right_split:
                    continue
                similarity = SequenceMatcher(
                    None, left_text, right_text
                ).ratio()
                if similarity < 0.95:
                    continue
                candidates.append(
                    NearDuplicateCandidate(
                        left_split=left_split,
                        left_row=left_row,
                        left_label=int(left["label"]),
                        left_category=str(left.get("category", "")),
                        right_split=right_split,
                        right_row=right_row,
                        right_label=int(right["label"]),
                        right_category=str(right.get("category", "")),
                        similarity=round(similarity, 6),
                        left_text=str(left["text"]),
                        right_text=str(right["text"]),
                    )
                )
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.left_split,
            candidate.left_row,
            candidate.right_split,
            candidate.right_row,
        ),
    )


def write_near_duplicate_report(
    output: Path,
    split_rows: dict[str, list[dict[str, Any]]],
) -> list[NearDuplicateCandidate]:
    """Write a bounded, human-readable report for cross-split review."""
    candidates = near_duplicate_candidates(split_rows, cross_split_only=True)
    lines = [
        "# Cross-split near-duplicate review",
        "",
        "This report is a review aid, not an automatic deletion decision.",
        "Candidates use normalized text, a shared first-50-character bucket, and",
        "SequenceMatcher similarity >= 0.95. Text is truncated to 500 characters",
        "per side to avoid unnecessary exposure of long payloads.",
        "",
        f"Candidates: {len(candidates)}",
        "",
        "## Decision policy",
        "",
        "- Confirmed exact duplicates across splits: remove from the training "
        "workflow only after recording the row IDs; preserve source files.",
        "- Same attack template with harmless wording changes: retain, but "
        "report as a limitation and use group-aware evaluation.",
        "- Heuristic-only matches: retain until manually confirmed.",
        "- Any conflicting labels: do not resolve automatically; investigate.",
        "",
    ]
    for index, candidate in enumerate(candidates, start=1):
        left_text = candidate.left_text.strip().replace("\n", " ")
        right_text = candidate.right_text.strip().replace("\n", " ")
        lines.extend(
            [
                f"### Candidate {index}",
                "",
                f"- Left: `{candidate.left_split}` row "
                f"{candidate.left_row}; label={candidate.left_label}; "
                f"category=`{candidate.left_category}`",
                f"- Right: `{candidate.right_split}` row "
                f"{candidate.right_row}; label={candidate.right_label}; "
                f"category=`{candidate.right_category}`",
                f"- Similarity: `{candidate.similarity:.6f}`",
                f"- Recommended disposition: retain pending manual confirmation; "
                f"do not delete automatically.",
                "",
                f"- Left text: `{left_text[:500]}`",
                f"- Right text: `{right_text[:500]}`",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")
    return candidates


def inspect_records(
    split_rows: dict[str, list[dict[str, Any]]],
    split_columns: dict[str, list[str]],
) -> DatasetInspection:
    reports = {
        split: _split_report(split_rows[split], split_columns[split])
        for split in SPLITS
    }
    text_sets = {
        split: {_normalise_text(str(row["text"])) for row in rows}
        for split, rows in split_rows.items()
    }
    group_sets = {
        split: {row.get("group_id") for row in rows}
        for split, rows in split_rows.items()
    }
    text_overlap: dict[str, int] = {}
    group_overlap: dict[str, int] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            key = f"{left}:{right}"
            text_overlap[key] = len(text_sets[left] & text_sets[right])
            group_overlap[key] = len(group_sets[left] & group_sets[right])
    near_total, near_cross = _near_duplicate_counts(split_rows)
    return DatasetInspection(
        splits=reports,
        cross_split_exact_text_overlap=text_overlap,
        cross_split_group_overlap=group_overlap,
        near_duplicate_candidate_pairs=near_total,
        near_duplicate_cross_split_pairs=near_cross,
        near_duplicate_method=(
            "whitespace/lowercase normalization, same first-50-character bucket, "
            "SequenceMatcher ratio >= 0.95; candidates require manual review"
        ),
    )


def inspect_parquet_directory(data_dir: Path) -> DatasetInspection:
    """Read only the three expected Parquet files from a local directory."""
    import pyarrow.parquet as parquet

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_columns: dict[str, list[str]] = {}
    for split in SPLITS:
        path = data_dir / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing expected split: {path}")
        table = parquet.read_table(path)
        columns = list(table.column_names)
        if columns != list(EXPECTED_COLUMNS):
            raise ValueError(
                f"{path} columns differ from expected schema: {columns}"
            )
        split_rows[split] = table.to_pylist()
        split_columns[split] = columns
    return inspect_records(split_rows, split_columns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    output: Path,
    data_dir: Path,
    inspection: DatasetInspection,
    revision: str,
) -> None:
    files = {}
    for split in SPLITS:
        path = data_dir / f"{split}.parquet"
        files[split] = {
            "path": str(path.as_posix()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {
        "dataset": "neuralchemy/Prompt-injection-dataset",
        "configuration": "core",
        "revision": revision,
        "source_url": (
            "https://huggingface.co/datasets/neuralchemy/"
            "Prompt-injection-dataset"
        ),
        "license_declared_by_dataset_card": "Apache-2.0",
        "files": files,
        "inspection": asdict(inspection),
        "notes": [
            "Raw Parquet files are local-only and ignored by Git.",
            "Dataset-card counts and leakage claims were independently checked "
            "against the downloaded files.",
            "Near-duplicate counts are conservative review candidates, not "
            "confirmed leakage.",
        ],
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--review-report", type=Path)
    args = parser.parse_args()
    inspection = inspect_parquet_directory(args.data_dir)
    write_manifest(args.manifest, args.data_dir, inspection, args.revision)
    if args.review_report:
        import pyarrow.parquet as parquet

        split_rows = {
            split: parquet.read_table(
                args.data_dir / f"{split}.parquet"
            ).to_pylist()
            for split in SPLITS
        }
        write_near_duplicate_report(args.review_report, split_rows)
    print(json.dumps(asdict(inspection), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
