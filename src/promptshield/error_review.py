"""Privacy-preserving human review workflow for DeBERTa validation errors.

This module prepares and completes a human review of a chosen source's false
positives (by default ``wildguard_judgecomp``) at the unchanged decision
threshold. It exists to help distinguish possible label noise from genuine
model errors without assuming either explanation.

It follows the repository's existing privacy convention:

- Raw prompt text appears ONLY in a local-only review packet that is listed in
  ``.gitignore`` and must never be committed, printed, or logged.
- The tracked review worksheet and the tracked summary carry row identifiers,
  dataset metadata, model scores, and aggregate decision counts only. They
  contain no prompt text.

Reviewers fill in ``decision`` and ``evidence`` for each row in the local
packet, either by hand or through ``promptshield.review_app``, a local web
interface that reads and writes this same packet. ``summarize_review`` then
produces an aggregate-only summary that reports review completeness honestly:
it stays ``incomplete`` until every row has a valid, evidence-backed decision.
Nothing here relabels data, retrains a model, tunes a threshold, or changes
API or gateway behaviour.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

from promptshield.deberta_decision_audit import (
    APPROVED_VALIDATION_PATH,
    DEFAULT_BATCH_SIZE,
)
from promptshield.deberta_error_analysis import (
    _canonical_validation_path,
    load_validation_records,
)
from promptshield.deberta_error_investigation import _class_1_probabilities
from promptshield.inference import DEFAULT_THRESHOLD, DebertaInference

# The single label a reviewer may assign. Each label states a distinct
# hypothesis so the review separates label noise from model error.
REVIEW_LABELS: dict[str, str] = {
    "likely_mislabeled": (
        "The text actually contains an injection, jailbreak, or instruction "
        "override, so the dataset's benign label appears wrong and the "
        "model's malicious flag may be justified."
    ),
    "genuine_model_false_positive": (
        "The text is a legitimate benign request and the benign label is "
        "correct; the model over-reacted to wording, topic, or formatting."
    ),
    "ambiguous_insufficient_context": (
        "The isolated text is not enough to judge; the decision depends on "
        "context that is not present in this record."
    ),
    "other": (
        "None of the above descriptions fit; the evidence note explains why."
    ),
}

ANNOTATION_GUIDE: list[str] = [
    "Read the full prompt text in the local packet, then choose exactly one "
    "label that best explains why the model flagged this benign-labeled row.",
    "For likely_mislabeled, quote the specific span that shows an injection, "
    "jailbreak, or override attempt.",
    "For genuine_model_false_positive, name the lexical or topical trigger "
    "that likely misled the model into flagging a benign request.",
    "For ambiguous_insufficient_context, state exactly what is missing, such "
    "as conversation history, a system prompt, or the dataset rubric.",
    "Every decision requires a short, evidence-based note citing either the "
    "text or the missing context. Do not decide from metadata alone.",
    "Record decisions in the local review packet. Do not edit dataset labels, "
    "model predictions, the threshold, or any trained artifact.",
]

# Local-only packet. Contains raw prompt text and is git-ignored.
DEFAULT_PACKET_PATH = Path("reports/manual-deberta-wildguard-review.json")
# Tracked worksheet. Row identifiers and metadata only, never prompt text.
DEFAULT_WORKSHEET_PATH = Path(
    "reports/deberta-wildguard-review-worksheet.md"
)


def select_source_false_positives(
    labels: list[int],
    metadata: list[dict[str, Any]],
    probabilities: list[float],
    threshold: float,
    source: str,
) -> list[dict[str, Any]]:
    """Select one source's false positives as text-free review records.

    Pure and inference-free so it can be unit-tested. ``validation_row`` is the
    1-based row position in ``validation.parquet`` so a reviewer can
    cross-reference the tracked worksheet against the local packet.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if not (len(labels) == len(metadata) == len(probabilities)):
        raise ValueError("labels, metadata, and probabilities must align")

    selected: list[dict[str, Any]] = []
    for offset, (label, meta, probability) in enumerate(
        zip(labels, metadata, probabilities)
    ):
        predicted = 1 if probability >= threshold else 0
        if label == 0 and predicted == 1 and meta.get("source") == source:
            selected.append({
                "validation_row": offset + 1,
                "true_label": 0,
                "predicted_label": 1,
                "predicted_malicious_probability": float(probability),
                "threshold": threshold,
                "category": meta.get("category"),
                "source": meta.get("source"),
                "severity": meta.get("severity"),
                "augmented": meta.get("augmented"),
            })
    return selected


def collect_source_false_positives(
    validation_path: Path,
    checkpoint_dir: Path,
    source: str,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run validation-only inference and attach prompt text for one source.

    Returns the selected false-positive rows (each carrying its prompt text for
    the local packet) plus aggregate provenance. Reads only the approved
    validation path and never touches the test split.
    """
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
    probabilities = _class_1_probabilities(logits)
    selected = select_source_false_positives(
        labels, metadata, probabilities, threshold, source
    )
    for record in selected:
        record["text"] = texts[record["validation_row"] - 1]

    provenance = {
        "validation_path": validation_path.as_posix(),
        "validation_rows": len(labels),
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "threshold": threshold,
        "source_filter": source,
        "selected_false_positives": len(selected),
        "total_false_positives_all_sources": sum(
            label == 0 and probability >= threshold
            for label, probability in zip(labels, probabilities)
        ),
    }
    return selected, provenance


def build_review_packet(
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the local-only packet with blank decision fields per row."""
    packet_rows: list[dict[str, Any]] = []
    for row in rows:
        packet_rows.append({
            **row,
            "decision": None,
            "evidence": None,
            "reviewed_by": None,
        })
    return {
        "experiment": "deberta_source_human_review_packet",
        "local_only": True,
        "contains_raw_prompt_text": True,
        "handling": (
            "This file holds raw validation prompt text for local human "
            "review only. It is listed in .gitignore and must never be "
            "committed, printed, logged, or sent to an external service."
        ),
        "review_labels": REVIEW_LABELS,
        "annotation_guide": ANNOTATION_GUIDE,
        "provenance": provenance,
        "row_count": len(packet_rows),
        "rows": packet_rows,
        "test_split_loaded": False,
        "labels_or_predictions_changed": False,
    }


def render_review_worksheet(packet: dict[str, Any]) -> str:
    """Render the tracked worksheet: identifiers and metadata, never text."""
    provenance = packet.get("provenance", {})
    lines = [
        "# DeBERTa source false-positive review worksheet",
        "",
        "This tracked worksheet lists only row identifiers, dataset metadata,",
        "and model scores. It contains no prompt text. The full text lives in",
        "the local-only packet, which is git-ignored and must not be committed.",
        "",
        f"- Source under review: `{provenance.get('source_filter')}`",
        f"- Decision threshold: `{provenance.get('threshold')}` (unmodified)",
        f"- Rows to review: `{packet.get('row_count')}`",
        "- Review status: pending until a human fills in every row of the",
        "  local packet.",
        "",
        "Review labels (assign exactly one per row, with evidence):",
        "",
    ]
    for label, description in REVIEW_LABELS.items():
        lines.append(f"- `{label}`: {description}")
    lines.extend(["", "Annotation guide:", ""])
    lines.extend(f"{index}. {step}" for index, step in enumerate(
        ANNOTATION_GUIDE, start=1
    ))
    lines.extend([
        "",
        "| Validation row | Category | Severity | Malicious probability | Decision |",
        "|---:|---|---|---:|---|",
    ])
    for row in packet.get("rows", []):
        lines.append(
            f"| {row['validation_row']} | {row.get('category')} | "
            f"{row.get('severity')} | "
            f"{row['predicted_malicious_probability']:.6f} | pending |"
        )
    lines.extend([
        "",
        "This worksheet records review decisions only. It does not change",
        "dataset labels, model predictions, the threshold, or any artifact.",
        "",
    ])
    return "\n".join(lines)


def summarize_review(packet: dict[str, Any]) -> dict[str, Any]:
    """Aggregate-only summary of reviewer decisions, honestly incomplete.

    Never copies prompt text or reviewer evidence into the summary. The review
    is reported complete only when every row carries a valid label, a
    non-empty evidence note, AND the name of the human reviewer who made the
    call. A decision with no named reviewer is reported separately as
    ``unattributed_rows`` and never counted toward completion, so machine- or
    script-generated judgments cannot be mistaken for human review. Decisions
    that adopted a machine suggestion after explicit reviewer confirmation
    are counted like any other human decision, but reported separately under
    ``assisted_confirmed_rows`` for transparency.
    """
    decision_counts = {label: 0 for label in REVIEW_LABELS}
    pending_rows = 0
    invalid_decisions: list[dict[str, Any]] = []
    missing_evidence_rows: list[int] = []
    unattributed_rows: list[int] = []
    assisted_confirmed_rows = 0

    for row in packet.get("rows", []):
        decision = row.get("decision")
        evidence = row.get("evidence")
        reviewer = row.get("reviewed_by")
        if decision is None:
            pending_rows += 1
            continue
        if decision not in REVIEW_LABELS:
            invalid_decisions.append({
                "validation_row": row.get("validation_row"),
                "decision": decision,
            })
            continue
        if not (isinstance(evidence, str) and evidence.strip()):
            missing_evidence_rows.append(row.get("validation_row"))
            continue
        if not (isinstance(reviewer, str) and reviewer.strip()):
            unattributed_rows.append(row.get("validation_row"))
            continue
        decision_counts[decision] += 1
        if row.get("decision_source") == "assisted_confirmed":
            assisted_confirmed_rows += 1

    total_rows = packet.get("row_count", len(packet.get("rows", [])))
    decided_rows = sum(decision_counts.values())
    complete = (
        pending_rows == 0
        and not invalid_decisions
        and not missing_evidence_rows
        and not unattributed_rows
        and decided_rows == total_rows
    )
    provenance = packet.get("provenance", {})
    return {
        "experiment": "deberta_source_human_review_summary",
        "local_only": False,
        "contains_raw_prompt_text": False,
        "source_filter": provenance.get("source_filter"),
        "threshold": provenance.get("threshold"),
        "total_rows": total_rows,
        "decided_rows": decided_rows,
        "pending_rows": pending_rows,
        "invalid_decisions": invalid_decisions,
        "missing_evidence_rows": missing_evidence_rows,
        "unattributed_rows": unattributed_rows,
        "assisted_confirmed_rows": assisted_confirmed_rows,
        "decision_counts": decision_counts,
        "review_complete": complete,
        "completion_note": (
            "Every row has a valid, evidence-backed decision attributed to a "
            "named human reviewer."
            if complete
            else (
                "Review is incomplete. No conclusion about label noise versus "
                "model error can be drawn until a human decides every row."
            )
        ),
        "labels_or_predictions_changed": False,
        "test_split_loaded": False,
        "limitations": [
            "This summary aggregates reviewer decisions only; it contains no "
            "prompt text and no reviewer evidence verbatim.",
            "Decision counts are review opinions, not corrected labels. No "
            "example was relabeled and no model artifact was changed.",
            "A decision with no named reviewer is treated as unattributed and "
            "does not count toward review completion.",
            "Decisions counted in assisted_confirmed_rows were recorded by a "
            "named reviewer who explicitly confirmed a machine-generated, "
            "unverified suggestion after reading the prompt; the suggestion "
            "itself is never a decision.",
            "Review outcomes are validation-only and do not establish "
            "held-out test performance.",
        ],
    }


def render_review_summary(summary: dict[str, Any]) -> str:
    """Render the tracked aggregate-only summary, never including text."""
    counts = summary.get("decision_counts", {})
    lines = [
        "# DeBERTa source false-positive review summary",
        "",
        "Aggregate decision counts only. This summary contains no prompt text",
        "and no reviewer evidence verbatim.",
        "",
        f"- Source under review: `{summary.get('source_filter')}`",
        f"- Decision threshold: `{summary.get('threshold')}` (unmodified)",
        f"- Total rows: `{summary.get('total_rows')}`",
        f"- Decided rows: `{summary.get('decided_rows')}`",
        f"- Pending rows: `{summary.get('pending_rows')}`",
        f"- Review complete: `{str(summary.get('review_complete')).lower()}`",
        "",
        "Decision counts:",
        "",
    ]
    for label in REVIEW_LABELS:
        lines.append(f"- `{label}`: {counts.get(label, 0)}")
    invalid = summary.get("invalid_decisions", [])
    missing = summary.get("missing_evidence_rows", [])
    unattributed = summary.get("unattributed_rows", [])
    lines.extend([
        "",
        f"- Invalid decisions: `{len(invalid)}`",
        f"- Decided rows missing evidence: `{len(missing)}`",
        f"- Decisions without a named reviewer (not counted): `{len(unattributed)}`",
        f"- Decisions confirmed from machine suggestions: "
        f"`{summary.get('assisted_confirmed_rows', 0)}`",
        "",
        summary.get("completion_note", ""),
        "",
        "Limitations:",
        "",
    ])
    lines.extend(f"- {item}" for item in summary.get("limitations", []))
    lines.extend([
        "",
        "This summary records review outcomes only. It does not change labels,",
        "predictions, the threshold, or any trained artifact.",
        "",
    ])
    return "\n".join(lines)


def ensure_packet_writable(path: Path, *, overwrite: bool = False) -> None:
    """Raise if ``path`` exists and already holds reviewer decisions.

    Guards human work against being silently overwritten by a re-run of
    ``prepare``. ``overwrite=True`` (the ``--force`` flag) discards it.
    """
    if not path.exists() or overwrite:
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if _has_reviewer_decisions(existing):
        raise ValueError(
            f"refusing to overwrite {path.as_posix()}: it already contains "
            "reviewer decisions; pass --force to discard them or choose a "
            "different --packet path"
        )


def write_local_packet(
    packet: dict[str, Any], path: Path, *, overwrite: bool = False
) -> None:
    """Write the raw-text packet to a local-only path.

    Refuses to overwrite an existing packet that already holds reviewer
    decisions, so re-running ``prepare`` cannot silently destroy human work.
    """
    ensure_packet_writable(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")


def _has_reviewer_decisions(packet: dict[str, Any]) -> bool:
    """True when any row carries a recorded reviewer decision."""
    return any(
        row.get("decision") is not None
        for row in packet.get("rows", [])
    )


def _ensure_packet_is_gitignored(path: Path) -> bool:
    """Best-effort check that the packet path is covered by .gitignore.

    Patterns are matched as gitignore globs, so a rule such as
    ``reports/manual-deberta-*-review.json`` correctly covers the packet.
    """
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        return False
    patterns = [
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    normalized = path.as_posix()
    return any(
        fnmatch.fnmatch(normalized, pattern)
        or fnmatch.fnmatch(normalized, pattern.lstrip("/"))
        for pattern in patterns
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build the review packet")
    prepare.add_argument("--validation", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--source", type=str, default="wildguard_judgecomp")
    prepare.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    prepare.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )
    prepare.add_argument("--packet", type=Path, default=DEFAULT_PACKET_PATH)
    prepare.add_argument(
        "--worksheet", type=Path, default=DEFAULT_WORKSHEET_PATH
    )
    prepare.add_argument(
        "--force",
        action="store_true",
        help="overwrite a packet that already contains reviewer decisions",
    )

    summarize = subparsers.add_parser(
        "summarize", help="aggregate reviewer decisions"
    )
    summarize.add_argument("--packet", type=Path, required=True)
    summarize.add_argument("--summary", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "prepare":
            # Check before spending inference time: never burn minutes of
            # compute only to refuse the write.
            ensure_packet_writable(args.packet, overwrite=args.force)
            rows, provenance = collect_source_false_positives(
                args.validation,
                args.checkpoint,
                args.source,
                threshold=args.threshold,
                batch_size=args.batch_size,
            )
            packet = build_review_packet(rows, provenance)
            write_local_packet(packet, args.packet, overwrite=args.force)
            args.worksheet.parent.mkdir(parents=True, exist_ok=True)
            args.worksheet.write_text(
                render_review_worksheet(packet), encoding="utf-8"
            )
            ignored = _ensure_packet_is_gitignored(args.packet)
            # Aggregate-only console output: never print prompt text.
            print(json.dumps({
                "packet": args.packet.as_posix(),
                "packet_gitignored": ignored,
                "worksheet": args.worksheet.as_posix(),
                "rows_to_review": packet["row_count"],
                "total_false_positives_all_sources": provenance[
                    "total_false_positives_all_sources"
                ],
                "threshold": args.threshold,
                "test_split_loaded": False,
                "note": (
                    "Fill decision and evidence for every row in the local "
                    "packet, then run the summarize command."
                ),
            }, indent=2))
            if not ignored:
                print(
                    "warning: packet path is not covered by .gitignore; do not "
                    "commit it",
                    file=sys.stderr,
                )
            return 0

        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        summary = summarize_review(packet)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            render_review_summary(summary), encoding="utf-8"
        )
        summary_json = args.summary.with_suffix(".json")
        summary_json.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        # Aggregate-only console output: never print prompt text or evidence.
        print(json.dumps({
            "summary_md": args.summary.as_posix(),
            "summary_json": summary_json.as_posix(),
            "review_complete": summary["review_complete"],
            "decided_rows": summary["decided_rows"],
            "pending_rows": summary["pending_rows"],
            "decision_counts": summary["decision_counts"],
            "test_split_loaded": False,
        }, indent=2))
        return 0
    except Exception as error:
        print(
            f"error review failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
