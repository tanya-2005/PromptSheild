"""Local-only decision store for the PromptShield human review packet.

This module owns reading and writing reviewer decisions in the git-ignored
review packet that ``promptshield.error_review`` prepares. It is deliberately
separate from the web layer so every rule about saving, resuming, validating,
and overwriting decisions can be unit-tested without a server.

Integrity properties:

- A decision is only accepted with a valid review label, a substantive
  evidence note, and the name of the human reviewer who made it. Nothing in
  this module invents decisions: a row stays ``pending`` until a human records
  one, so an automated caller cannot silently turn machine output into a
  "human" review.
- An existing decision is never overwritten unless the caller explicitly
  passes ``overwrite=True`` (the UI asks for confirmation first).
- Writes are atomic (temp file + ``os.replace``), so a crash mid-save can
  never leave the packet truncated or half-written.
- Aggregate progress is delegated to ``error_review.summarize_review`` and is
  therefore free of prompt text and reviewer evidence.

Nothing here reads the test split, retrains a model, changes the threshold or
decision rule, or sends prompt text anywhere.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from promptshield.error_review import REVIEW_LABELS

# Minimum length for an evidence note on a *new* save. This is an input guard
# against placeholder answers like "ok"; the definition of "decided" used for
# progress and completion stays "non-empty evidence" so the store and
# ``summarize_review`` always agree on how many rows are done.
MIN_EVIDENCE_LENGTH = 10


class DecisionConflictError(Exception):
    """Raised when a recorded decision would be overwritten without consent."""


def load_packet(path: Path) -> dict[str, Any]:
    """Load the local review packet from ``path``."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"review packet not found: {resolved.as_posix()}")
    packet = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(packet, dict) or not isinstance(packet.get("rows"), list):
        raise ValueError("review packet must be a JSON object with a 'rows' list")
    return packet


def save_packet(path: Path, packet: dict[str, Any]) -> None:
    """Atomically replace the packet at ``path``.

    The data is written to a temporary file in the same directory, flushed to
    disk, and then moved into place, so readers never observe a partial write.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    directory = str(resolved.parent) if resolved.parent != Path("") else "."
    temporary = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=directory,
        prefix=".review-packet-",
        suffix=".tmp",
        delete=False,
    )
    try:
        with temporary:
            json.dump(packet, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary.name, resolved)
    except BaseException:
        # Never leave a stray temp file holding prompt text behind.
        try:
            os.unlink(temporary.name)
        except OSError:
            pass
        raise


def find_row_index(packet: dict[str, Any], validation_row: int) -> int | None:
    """Return the index of ``validation_row`` in the packet, or ``None``."""
    for index, row in enumerate(packet.get("rows", [])):
        if row.get("validation_row") == validation_row:
            return index
    return None


def validate_decision_payload(
    validation_row: int,
    decision: str,
    evidence: str,
    reviewed_by: str,
) -> None:
    """Raise ``ValueError`` unless the payload is a complete, valid decision."""
    if not isinstance(validation_row, int) or isinstance(validation_row, bool):
        raise ValueError("validation_row must be an integer")
    if decision not in REVIEW_LABELS:
        raise ValueError(
            "decision must be one of: " + ", ".join(sorted(REVIEW_LABELS))
        )
    if not isinstance(evidence, str) or len(evidence.strip()) < MIN_EVIDENCE_LENGTH:
        raise ValueError(
            f"evidence must be at least {MIN_EVIDENCE_LENGTH} characters "
            "and cite the text or the missing context"
        )
    if not isinstance(reviewed_by, str) or not reviewed_by.strip():
        raise ValueError("reviewed_by must name the human reviewer")


def row_is_decided(row: dict[str, Any]) -> bool:
    """True when a row carries a complete human decision.

    Mirrors the "decided" rule in ``error_review.summarize_review``: a valid
    label, non-empty evidence, and a named reviewer.
    """
    decision = row.get("decision")
    evidence = row.get("evidence")
    reviewer = row.get("reviewed_by")
    return (
        decision in REVIEW_LABELS
        and isinstance(evidence, str)
        and bool(evidence.strip())
        and isinstance(reviewer, str)
        and bool(reviewer.strip())
    )


def _validate_assisted_confirmation(value: dict[str, Any]) -> None:
    """Guard the assisted-save audit record.

    An assisted confirmation must reference a suggestion that is itself a
    valid decision label: abstaining suggestions (``needs_manual_review``)
    can never be "accepted", so they can never be stored as an audit record.
    This keeps a script from laundering machine abstentions into decisions.
    """
    if not isinstance(value, dict):
        raise ValueError("assisted_confirmation must be a mapping")
    suggested_label = value.get("suggested_label")
    if suggested_label not in REVIEW_LABELS:
        raise ValueError(
            "assisted_confirmation.suggested_label must be one of the review "
            "labels; an abstention cannot be accepted as a decision"
        )


def record_decision(
    packet: dict[str, Any],
    validation_row: int,
    decision: str,
    evidence: str,
    reviewed_by: str,
    *,
    overwrite: bool = False,
    assisted_confirmation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of ``packet`` with one row's decision recorded.

    Raises ``ValueError`` for an invalid payload or an unknown row, and
    ``DecisionConflictError`` when the row already holds a decision and
    ``overwrite`` is not set. The input packet is never mutated.

    ``assisted_confirmation`` records that the reviewer explicitly confirmed
    a machine suggestion before saving (the web interface is the only caller
    that sets it, and only after an affirmative per-row confirmation). It is
    stored purely as an audit marker: the decision still requires a valid
    label, substantive evidence, and a named human reviewer, and it counts
    toward completion exactly like any other decision. Manual saves carry
    ``decision_source="manual"``.
    """
    validate_decision_payload(validation_row, decision, evidence, reviewed_by)
    if assisted_confirmation is not None:
        _validate_assisted_confirmation(assisted_confirmation)
    index = find_row_index(packet, validation_row)
    if index is None:
        raise ValueError(
            f"validation row {validation_row} is not in this review packet"
        )

    rows = list(packet.get("rows", []))
    existing = rows[index]
    if existing.get("decision") is not None and not overwrite:
        raise DecisionConflictError(
            f"validation row {validation_row} already has a recorded decision; "
            "confirm the overwrite to replace it"
        )

    rows[index] = {
        **existing,
        "decision": decision,
        "evidence": evidence.strip(),
        "reviewed_by": reviewed_by.strip(),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "decision_source": (
            "assisted_confirmed" if assisted_confirmation else "manual"
        ),
        "assisted_confirmation": assisted_confirmation,
    }
    return {**packet, "rows": rows}


def next_pending_row(packet: dict[str, Any]) -> int | None:
    """Return the ``validation_row`` of the first undecided row, or ``None``.

    This is what lets the interface resume: rows already reviewed by a human
    are skipped, including rows whose decision lacks reviewer attribution.
    """
    for row in packet.get("rows", []):
        if not row_is_decided(row):
            return row.get("validation_row")
    return None


def pending_rows(packet: dict[str, Any]) -> list[int]:
    """Return every ``validation_row`` that still needs a human decision."""
    return [
        row.get("validation_row")
        for row in packet.get("rows", [])
        if not row_is_decided(row)
    ]
