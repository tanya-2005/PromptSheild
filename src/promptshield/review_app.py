"""Local-only web interface for the PromptShield human review packet.

Streamlit is not installed in this project, so this interface is built on the
FastAPI and uvicorn versions already pinned in ``requirements.txt``: it adds no
new dependency. It serves a single self-contained HTML page (inline CSS and
JavaScript, no CDN assets and no external fonts) so that loading the page
makes no network request beyond the local server itself.

The interface shows one prompt at a time with its dataset label and model
prediction, lets a named human reviewer pick one of the four review labels,
records an evidence note, and saves with "Save & Next". It resumes at the
first undecided row, preserves every existing decision, and refuses to
overwrite a completed review unless the reviewer confirms.

Each row also carries a machine-generated, unverified *preliminary
suggestion* from ``promptshield.review_suggestions``: deterministic local
rules over the prompt text and dataset metadata, never over the DeBERTa
prediction itself. The reviewer may accept a suggestion as their own
decision, but only after ticking an explicit confirmation that they have
read the prompt; the server re-derives the suggestion and rejects any accept
that does not match it, so suggestions can never be bulk-applied or saved
without a named human taking responsibility for them.

Privacy and integrity properties, enforced by this module and its store:

- The server binds to a loopback address only (default ``127.0.0.1``); a
  non-loopback host is rejected, so prompt text never leaves the machine.
- Prompt text is only ever rendered into the page and returned by
  ``GET /api/row``; the state, progress, and summary endpoints are aggregate
  and text-free. Suggestion payloads cite rule identifiers only, never text.
- The interface never invents a decision. A save requires a valid label, a
  substantive evidence note, and the reviewer's name; nothing here marks
  machine output as a human review or claims completion before a human has
  decided every row.
- No external LLM or network service is called. The test split is never read,
  and the threshold and decision rule are untouched.
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from promptshield.error_review import (
    ANNOTATION_GUIDE,
    DEFAULT_PACKET_PATH,
    REVIEW_LABELS,
    summarize_review,
)
from promptshield.review_store import (
    DecisionConflictError,
    find_row_index,
    load_packet,
    next_pending_row,
    record_decision,
    save_packet,
)
from promptshield.review_suggestions import suggest_row

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8501


class DecisionRequest(BaseModel):
    """A single reviewer decision submitted by the local interface.

    ``assisted_accept`` marks a save that adopts the row's machine
    suggestion; such a save additionally requires ``confirmation=True`` (the
    reviewer attests to having read the prompt) and the server re-derives
    the suggestion to check the submitted label matches it. Manual saves
    leave both fields at their defaults and behave exactly as before.
    """

    validation_row: int
    decision: str
    evidence: str
    reviewed_by: str
    overwrite: bool = False
    assisted_accept: bool = False
    confirmation: bool = False


def _is_loopback(host: str) -> bool:
    """True only for addresses that cannot route off this machine."""
    if host in {"localhost"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _state_payload(packet: dict[str, Any]) -> dict[str, Any]:
    """Aggregate progress only: counts and the next row, never prompt text."""
    summary = summarize_review(packet)
    return {
        "total_rows": summary["total_rows"],
        "decided_rows": summary["decided_rows"],
        "pending_rows": summary["pending_rows"],
        "decision_counts": summary["decision_counts"],
        "review_complete": summary["review_complete"],
        "completion_note": summary["completion_note"],
        "labels": REVIEW_LABELS,
        "annotation_guide": ANNOTATION_GUIDE,
        "next_pending_row": next_pending_row(packet),
    }


def create_review_app(
    packet_path: Path,
    host: str = DEFAULT_HOST,
) -> FastAPI:
    """Build the local review interface over one packet file."""
    if not _is_loopback(host):
        raise ValueError(
            "review interface may only bind a loopback address so prompt text "
            f"stays on this machine; got {host!r}"
        )

    app = FastAPI(title="PromptShield local review")
    app.state.packet_path = Path(packet_path)
    app.state.host = host

    def _read_packet() -> dict[str, Any]:
        try:
            return load_packet(app.state.packet_path)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # Static, fully self-contained page: inline CSS/JS, no external assets.
        return REVIEW_PAGE

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        # Aggregate progress only; never includes prompt text.
        return _state_payload(_read_packet())

    @app.get("/api/rows")
    def rows() -> dict[str, Any]:
        # Navigation order only: row identifiers, never prompt text.
        packet = _read_packet()
        return {
            "rows": [
                record.get("validation_row") for record in packet.get("rows", [])
            ]
        }

    @app.get("/api/row/{validation_row}")
    def row(validation_row: int) -> dict[str, Any]:
        # The one endpoint that returns prompt text. It is served over the
        # loopback interface only, for display in the local page.
        packet = _read_packet()
        index = find_row_index(packet, validation_row)
        if index is None:
            raise HTTPException(
                status_code=404,
                detail=f"validation row {validation_row} is not in this packet",
            )
        record = packet["rows"][index]
        return {
            "validation_row": record.get("validation_row"),
            "text": record.get("text"),
            "true_label": record.get("true_label"),
            "predicted_label": record.get("predicted_label"),
            "malicious_probability": record.get(
                "predicted_malicious_probability"
            ),
            "threshold": record.get("threshold"),
            "category": record.get("category"),
            "source": record.get("source"),
            "severity": record.get("severity"),
            "decision": record.get("decision"),
            "evidence": record.get("evidence"),
            "reviewed_by": record.get("reviewed_by"),
            "decision_source": record.get("decision_source"),
            # Machine-generated, unverified, deterministic per row; cites
            # rule ids only and never contains prompt text. Deliberately
            # recomputed on every read (never persisted), so no machine
            # output can linger in the packet as if it were a decision.
            "suggestion": suggest_row(record),
        }

    @app.post("/api/decision")
    def decision(payload: DecisionRequest) -> dict[str, Any]:
        packet = _read_packet()
        assisted: dict[str, Any] | None = None
        if payload.assisted_accept:
            # Server-side enforcement of the accept-suggestion contract: the
            # suggestion is re-derived here (deterministically), so the UI
            # cannot accept a stale or hand-picked label under this path,
            # and an abstaining suggestion can never be accepted at all.
            if not payload.confirmation:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "accepting a machine suggestion requires explicit "
                        "reviewer confirmation"
                    ),
                )
            index = find_row_index(packet, payload.validation_row)
            if index is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"validation row {payload.validation_row} is not in "
                        "this review packet"
                    ),
                )
            suggestion = suggest_row(packet["rows"][index])
            if not suggestion["adoptable"]:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "this row's suggestion is an abstention and cannot be "
                        "accepted as a decision; choose a label manually"
                    ),
                )
            if payload.decision != suggestion["suggested_label"]:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "decision does not match the machine suggestion; "
                        "record a manual decision instead"
                    ),
                )
            assisted = {
                "suggested_label": suggestion["suggested_label"],
                "confidence": suggestion["confidence"],
                "matched_rule_ids": suggestion["matched_rule_ids"],
                "engine": suggestion["engine"],
                "confirmed_by_reviewer": True,
            }
        try:
            updated = record_decision(
                packet,
                payload.validation_row,
                payload.decision,
                payload.evidence,
                payload.reviewed_by,
                overwrite=payload.overwrite,
                assisted_confirmation=assisted,
            )
        except DecisionConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        save_packet(app.state.packet_path, updated)
        return _state_payload(updated)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET_PATH)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    try:
        app = create_review_app(args.packet, host=args.host)
    except Exception as error:
        print(
            f"review interface failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    import uvicorn

    # Aggregate-only console output: the packet path is printed, never text.
    print(
        f"PromptShield local review UI on http://{args.host}:{args.port}  "
        f"(packet: {app.state.packet_path.as_posix()})\n"
        "Serves prompt text on the loopback interface only. Stop with Ctrl+C.",
        file=sys.stderr,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# ---------------------------------------------------------------------------
# Self-contained page. No external stylesheets, scripts, fonts, or images:
# every byte the browser needs is inline, so nothing is fetched from a network
# service and no prompt text can leak through a subresource request.
# ---------------------------------------------------------------------------

REVIEW_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PromptShield local review</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    margin: 0 auto; padding: 1.25rem; max-width: 62rem; line-height: 1.45;
  }
  header { display: flex; justify-content: space-between; align-items: baseline;
           flex-wrap: wrap; gap: .5rem; }
  h1 { font-size: 1.15rem; margin: 0; }
  .meta, .guide { color: #666; font-size: .85rem; }
  @media (prefers-color-scheme: dark) { .meta, .guide { color: #aaa; } }
  .progress { font-variant-numeric: tabular-nums; }
  .prompt {
    white-space: pre-wrap; word-break: break-word;
    background: rgba(127,127,127,.14); border: 1px solid rgba(127,127,127,.35);
    border-radius: 6px; padding: 1rem; margin: 1rem 0;
  }
  .stats { display: flex; gap: 1.25rem; flex-wrap: wrap; font-size: .9rem; }
  .stats b { font-weight: 600; }
  fieldset { border: 1px solid rgba(127,127,127,.4); border-radius: 6px;
             margin: 1rem 0; }
  legend { font-size: .9rem; font-weight: 600; padding: 0 .35rem; }
  label.opt { display: block; margin: .4rem 0; cursor: pointer; }
  label.opt input { margin-right: .5rem; }
  label.checked { font-weight: 600; }
  textarea, input[type=text] { width: 100%; margin-top: .35rem; font: inherit;
                               padding: .5rem; }
  textarea { min-height: 5.5rem; resize: vertical; }
  .field-label { font-size: .9rem; font-weight: 600; display: block; }
  .bar { display: flex; gap: .6rem; align-items: center; margin-top: 1rem;
         flex-wrap: wrap; }
  button { padding: .5rem 1rem; font-size: .95rem; font: inherit;
           cursor: pointer; border-radius: 6px;
           border: 1px solid rgba(127,127,127,.6); background: transparent; }
  button.primary { background: #2563eb; border-color: #2563eb; color: #fff; }
  button:disabled { cursor: not-allowed; opacity: .45; }
  #status { font-size: .9rem; }
  #status.error { color: #b00020; }
  @media (prefers-color-scheme: dark) { #status.error { color: #ff8a80; } }
  #already { font-size: .85rem; color: #b45309; }
  @media (prefers-color-scheme: dark) { #already { color: #ffcc80; } }
  .suggestion {
    border: 1px dashed rgba(127,127,127,.55); border-radius: 6px;
    padding: .75rem 1rem; margin: 1rem 0;
  }
  .suggestion .badge-machine {
    display: inline-block; font-size: .75rem; font-weight: 700;
    letter-spacing: .02em; text-transform: uppercase;
    color: #92400e; border: 1px solid #92400e; border-radius: 4px;
    padding: .1rem .4rem; margin-right: .5rem;
  }
  @media (prefers-color-scheme: dark) {
    .suggestion .badge-machine { color: #ffcc80; border-color: #ffcc80; }
  }
  .suggestion .sug-label { font-weight: 700; }
  .suggestion .sug-conf { font-size: .85rem; color: #666; margin-left: .5rem; }
  @media (prefers-color-scheme: dark) { .suggestion .sug-conf { color: #aaa; } }
  .suggestion p { margin: .45rem 0 0; font-size: .9rem; }
  .suggestion .sug-notice { color: #666; font-style: italic; }
  @media (prefers-color-scheme: dark) { .suggestion .sug-notice { color: #aaa; } }
  .suggestion .confirm { display: block; margin-top: .6rem; font-size: .9rem;
                         cursor: pointer; }
  .suggestion .confirm input { margin-right: .45rem; }
  .sug-actions { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .6rem; }
  .guide ol { margin: .35rem 0; padding-left: 1.25rem; }
</style>
</head>
<body>
<header>
  <h1>PromptShield &mdash; local human review</h1>
  <span id="progress" class="progress meta">&hellip;</span>
</header>

<p class="meta">Runs entirely on this machine. Prompts are served on the
loopback interface only and are never sent to an external service.</p>

<div class="stats" id="stats"></div>
<p class="meta" id="position"></p>

<div class="prompt" id="prompt"></div>

<div class="suggestion" id="suggestion" hidden>
  <span class="badge-machine">Machine-generated &middot; unverified</span>
  <span class="sug-label" id="sugLabel"></span>
  <span class="sug-conf" id="sugConfidence"></span>
  <p class="sug-reason" id="sugReason"></p>
  <p class="sug-notice" id="sugNotice"></p>
  <label class="confirm" id="confirmWrap"><input type="checkbox"
    id="confirmSuggestion"> I have read the full prompt above and confirm
    this suggested label as my own reviewed decision.</label>
  <div class="sug-actions">
    <button id="acceptSuggestion" class="primary" type="button"
            disabled>Accept suggestion</button>
    <button id="rejectSuggestion" type="button">Reject &amp; choose another
      label</button>
    <button id="markUncertain" type="button">Mark uncertain</button>
  </div>
</div>

<fieldset>
  <legend>Review label</legend>
  <div id="labels"></div>
</fieldset>

<label class="field-label" for="evidence">Evidence-based justification
  <span class="meta">(required &mdash; cite the text or the missing
  context)</span></label>
<textarea id="evidence" placeholder="e.g. Quotes an override span, or names the missing context that changes the reading"></textarea>

<label class="field-label" for="reviewer" style="margin-top:.75rem">Reviewer name
  <span class="meta">(required &mdash; a decision with no name is not counted
  as human review)</span></label>
<input type="text" id="reviewer" autocomplete="off" placeholder="your name">

<div class="bar">
  <button id="prev">&larr; Previous</button>
  <button id="next">Skip &rarr;</button>
  <button id="save" class="primary">Save &amp; Next</button>
  <span id="already" hidden></span>
  <span id="status" class="meta"></span>
</div>

<div class="guide">
  <p><strong>Annotation guide</strong></p>
  <ol id="guide"></ol>
</div>

<script>
"use strict";
(function () {
  var state = null;          // aggregate progress from /api/state
  var order = [];            // validation_row values, worksheet order
  var position = 0;          // index into order
  var current = null;        // row payload currently displayed

  function $(id) { return document.getElementById(id); }

  function setStatus(message, isError) {
    var node = $("status");
    node.textContent = message || "";
    node.className = isError ? "error" : "meta";
  }

  function renderProgress() {
    $("progress").textContent = state.decided_rows + " / " + state.total_rows +
      " reviewed" + (state.review_complete ? " — complete" : "");
    var counts = state.decision_counts;
    var parts = [];
    Object.keys(counts).forEach(function (label) {
      parts.push(label + ": " + counts[label]);
    });
    $("stats").textContent = "";
    parts.forEach(function (text) {
      var span = document.createElement("span");
      span.textContent = text;
      $("stats").appendChild(span);
    });
    var guide = $("guide");
    guide.textContent = "";
    state.annotation_guide.forEach(function (step) {
      var li = document.createElement("li");
      li.textContent = step;
      guide.appendChild(li);
    });
  }

  function renderLabels() {
    var box = $("labels");
    box.textContent = "";
    Object.keys(state.labels).forEach(function (value) {
      var label = document.createElement("label");
      label.className = "opt";
      var radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "decision";
      radio.value = value;
      var text = document.createElement("span");
      // textContent, never innerHTML: prompt-derived values are untrusted.
      text.textContent = value + " — " + state.labels[value];
      label.appendChild(radio);
      label.appendChild(text);
      box.appendChild(label);
    });
  }

  function selectDecision(value) {
    var radios = document.querySelectorAll('input[name=decision]');
    for (var i = 0; i < radios.length; i++) {
      radios[i].checked = (radios[i].value === value);
      radios[i].parentNode.className = radios[i].checked ? "opt checked" : "opt";
    }
  }

  function selectedDecision() {
    var checked = document.querySelector('input[name=decision]:checked');
    return checked ? checked.value : null;
  }

  function suggestionOf(row) {
    return row && row.suggestion ? row.suggestion : null;
  }

  function refreshAcceptEnabled() {
    // Accept is only offered for adoptable suggestions, and only after the
    // reviewer ticks the read-the-prompt confirmation. There is no bulk
    // accept anywhere in this interface: one row, one confirmation.
    var sug = suggestionOf(current);
    var adoptable = !!(sug && sug.adoptable);
    $("acceptSuggestion").disabled =
      !adoptable || !$("confirmSuggestion").checked;
  }

  function renderSuggestion(row) {
    var sug = suggestionOf(row);
    var card = $("suggestion");
    if (!sug) { card.hidden = true; return; }
    card.hidden = false;
    // textContent only: suggestion values are untrusted like all
    // prompt-derived data, and reasons must never carry markup.
    $("sugLabel").textContent = sug.suggested_label;
    $("sugConfidence").textContent = "confidence: " + sug.confidence;
    $("sugReason").textContent = sug.reason;
    $("sugNotice").textContent = sug.notice;
    $("confirmSuggestion").checked = false;
    $("confirmWrap").hidden = !sug.adoptable;
    $("acceptSuggestion").textContent = sug.adoptable
      ? "Accept suggestion"
      : "Suggestion cannot be accepted — judge manually";
    refreshAcceptEnabled();
  }

  function showRow(index) {
    if (index < 0) { index = 0; }
    if (index >= order.length) { index = order.length - 1; }
    position = index;
    var rowId = order[position];
    $("position").textContent = "Row " + (position + 1) + " of " + order.length;
    $("save").disabled = true;
    fetch("/api/row/" + rowId).then(function (response) {
      if (!response.ok) { throw new Error("row " + rowId + " unavailable"); }
      return response.json();
    }).then(function (row) {
      current = row;
      // textContent only: the prompt may contain markup or script-like text.
      $("prompt").textContent = row.text;
      var labelNames = { 0: "benign", 1: "malicious" };
      $("stats").textContent = "";
      [
        ["Dataset label", labelNames[row.true_label] + " (" + row.true_label + ")"],
        ["Model prediction", labelNames[row.predicted_label] + " (" + row.predicted_label + ")"],
        ["Malicious probability", Number(row.malicious_probability).toFixed(6)],
        ["Threshold", String(row.threshold)],
        ["Category", row.category == null ? "—" : String(row.category)],
        ["Source", row.source == null ? "—" : String(row.source)]
      ].forEach(function (pair) {
        var span = document.createElement("span");
        var b = document.createElement("b");
        b.textContent = pair[0] + ": ";
        span.appendChild(b);
        span.appendChild(document.createTextNode(pair[1]));
        $("stats").appendChild(span);
      });
      selectDecision(row.decision);
      $("evidence").value = row.evidence || "";
      if (row.reviewed_by) { $("reviewer").value = row.reviewed_by; }
      renderSuggestion(row);
      var already = $("already");
      if (row.decision) {
        already.textContent = "Already reviewed by " + row.reviewed_by +
          " — saving will ask to overwrite.";
        already.hidden = false;
      } else {
        already.hidden = true;
      }
      $("save").disabled = false;
      setStatus("");
    }).catch(function (error) {
      setStatus(String(error), true);
      $("save").disabled = true;
    });
  }

  function adoptState(next) {
    state = next;
    renderProgress();
    renderLabels();
  }

  function postDecision(body) {
    return fetch("/api/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (response) {
      if (response.status === 409) {
        var ok = window.confirm(
          "Row " + body.validation_row + " already has a recorded " +
          "decision. Overwrite it?");
        if (ok) {
          var retry = {};
          Object.keys(body).forEach(function (key) { retry[key] = body[key]; });
          retry.overwrite = true;
          return postDecision(retry);
        }
        setStatus("Kept the existing decision.", true);
        return null;
      }
      if (!response.ok) {
        return response.json().then(function (payload) {
          throw new Error(payload.detail || "save failed");
        });
      }
      return response.json();
    });
  }

  function finishSave(next, savedMessage) {
    if (!next) {
      // Conflict declined: re-enable whatever the current row allows.
      $("save").disabled = false;
      refreshAcceptEnabled();
      return;
    }
    adoptState(next);
    if (state.next_pending_row != null) {
      setStatus(savedMessage, false);
      var resume = order.indexOf(state.next_pending_row);
      showRow(resume >= 0 ? resume : position + 1);
    } else {
      setStatus("Every row has a human decision. Review is complete.", false);
      $("save").disabled = true;
      $("acceptSuggestion").disabled = true;
    }
  }

  function evidenceAndReviewer() {
    var evidence = $("evidence").value;
    var reviewer = $("reviewer").value;
    if (evidence.trim().length < 10) {
      setStatus("Provide an evidence-based justification (at least 10 characters).", true);
      return null;
    }
    if (!reviewer.trim()) { setStatus("Enter your reviewer name.", true); return null; }
    return { evidence: evidence, reviewed_by: reviewer };
  }

  function save() {
    var decision = selectedDecision();
    if (!decision) { setStatus("Choose exactly one review label.", true); return; }
    var fields = evidenceAndReviewer();
    if (!fields) { return; }
    $("save").disabled = true;
    $("acceptSuggestion").disabled = true;
    postDecision({
      validation_row: current.validation_row,
      decision: decision,
      evidence: fields.evidence,
      reviewed_by: fields.reviewed_by,
      overwrite: false
    }).then(function (next) { finishSave(next, "Decision saved."); })
      .catch(function (error) {
        $("save").disabled = false;
        refreshAcceptEnabled();
        setStatus(String(error), true);
      });
  }

  function acceptSuggestion() {
    var sug = suggestionOf(current);
    if (!sug || !sug.adoptable) {
      setStatus("This row has no acceptable suggestion; choose a label manually.", true);
      return;
    }
    if (!$("confirmSuggestion").checked) {
      setStatus("Tick the confirmation box after reading the prompt; the suggestion is not accepted automatically.", true);
      return;
    }
    var fields = evidenceAndReviewer();
    if (!fields) { return; }
    $("save").disabled = true;
    $("acceptSuggestion").disabled = true;
    // The server re-derives the suggestion and rejects any mismatch, and
    // records the accept as decision_source=assisted_confirmed.
    postDecision({
      validation_row: current.validation_row,
      decision: sug.suggested_label,
      evidence: fields.evidence,
      reviewed_by: fields.reviewed_by,
      overwrite: false,
      assisted_accept: true,
      confirmation: true
    }).then(function (next) { finishSave(next, "Suggestion accepted and saved."); })
      .catch(function (error) {
        showRow(position);
        setStatus(String(error), true);
      });
  }

  function rejectSuggestion() {
    $("confirmSuggestion").checked = false;
    selectDecision(null);
    setStatus("Suggestion rejected. Choose one of the four labels, add evidence, then Save.", false);
    var first = document.querySelector("#labels input");
    if (first) { first.focus(); }
  }

  function markUncertain() {
    // "Uncertain" is itself a human decision recorded under
    // ambiguous_insufficient_context, with the usual evidence and name.
    selectDecision("ambiguous_insufficient_context");
    setStatus("Uncertain selected. Add your evidence, then Save to record ambiguous_insufficient_context.", false);
    $("evidence").focus();
  }

  $("save").addEventListener("click", save);
  $("acceptSuggestion").addEventListener("click", acceptSuggestion);
  $("rejectSuggestion").addEventListener("click", rejectSuggestion);
  $("markUncertain").addEventListener("click", markUncertain);
  $("confirmSuggestion").addEventListener("change", refreshAcceptEnabled);
  $("prev").addEventListener("click", function () { showRow(position - 1); });
  $("next").addEventListener("click", function () { showRow(position + 1); });
  document.addEventListener("change", function (event) {
    if (event.target.name === "decision") { selectDecision(event.target.value); }
  });

  fetch("/api/state").then(function (response) { return response.json(); })
    .then(function (next) {
      adoptState(next);
      fetch("/api/rows").then(function (response) { return response.json(); })
        .then(function (list) {
          order = list.rows;
          var start = 0;
          if (next.next_pending_row != null) {
            var found = order.indexOf(next.next_pending_row);
            if (found >= 0) { start = found; }
          }
          showRow(start);
        });
    }).catch(function (error) { setStatus(String(error), true); });
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
