"""Local-only prompt-scanner frontend for the PromptShield gateway.

Streamlit is not installed in this project, so the scanner is built the same
way as ``promptshield.review_app``: a small FastAPI application that serves
one fully self-contained HTML page (inline CSS and JavaScript, no CDN
assets, no external fonts, no build step) on top of the ``fastapi`` and
``uvicorn`` versions already pinned in ``requirements.txt``. It adds no new
dependency.

The frontend is a thin, local proxy over the existing backend — it does not
reimplement the model, the classifier threshold, or the gateway policy:

- ``POST /api/scan`` forwards ``{"prompt": ...}`` to the backend's
  ``POST /gateway`` (same request contract, reusing ``PromptRequest``) and
  returns the gateway payload unchanged, so the page shows the gateway
  decision and the raw classifier prediction exactly as the backend
  produced them.
- ``GET /api/health`` forwards to the backend ``GET /health`` and reports
  whether the backend was reachable at all; an unreachable backend is never
  dressed up as a healthy one.
- ``GET /api/insights`` serves a frozen, aggregate-only copy of the figures
  from the tracked validation reports (model comparison, error counts,
  calibration study, gateway trade-offs, limitations). It reads no Parquet,
  no checkpoint, and no review packet.

Privacy and integrity properties, enforced here:

- The backend URL must be a loopback address and the server refuses to bind
  anything but a loopback address, so scanned prompt text never leaves the
  machine.
- Prompt text is only ever forwarded to the loopback backend inside the
  scan request. It is never logged, printed, persisted, or embedded in any
  page, insights payload, or health payload.
- A failed security check is fail-closed in the UI: when the backend is
  unreachable or returns an error, the scan endpoint responds ``503`` with
  the gateway's own fixed detail string, which the page displays as
  "treated as review". No code path in this module can turn a failed scan
  into an allow.
- No external service or LLM is called; the transport is the Python
  standard library. Thresholds, gateway policy, and data files are untouched.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from promptshield.api import PromptRequest
from promptshield.gateway import UNAVAILABLE_DETAIL
from promptshield.inference import MAX_PROMPT_LENGTH

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8502
DEFAULT_TIMEOUT_SECONDS = 60.0

# Shown next to the health fields so a loaded model file is never mistaken
# for production readiness.
HEALTH_HONESTY_NOTE = (
    "reports only whether the local model file loaded on this machine; "
    "it is not a production-readiness or security claim"
)

UNREACHABLE_NOTE = (
    "backend unreachable — start promptshield.api on the loopback interface "
    "and refresh"
)

INSIGHTS_NOTE = (
    "Frozen aggregate copy of the tracked validation-only reports listed "
    "below. Validation figures do not establish held-out test performance; "
    "the test split has never been loaded; no model was retrained and no "
    "threshold or gateway policy was changed to produce these numbers."
)

LIMITATIONS: tuple[str, ...] = (
    "The detector does not guarantee security and does not prevent all "
    "prompt injections; novel or obfuscated attacks can be allowed.",
    "Every figure is validation-only (941 rows) and does not establish "
    "held-out test performance; the test split has never been loaded.",
    "The class-1 model score is uncalibrated: it is not a probability or a "
    "confidence, and no calibration claim is made.",
    "On validation the 0.90 block band contained 42 benign false positives "
    "(an availability cost) and 9 malicious false negatives passed below "
    "it (a residual miss rate).",
    "The band_v1 band defaults (0.70 / 0.90) were read off a validation "
    "grid for illustration; no optimality is claimed.",
    "The 27-row human review of wildguard false positives is still "
    "incomplete; no row has been relabeled by any tool.",
    "The score's calibration abstention study was an honest negative: the "
    "best pre-registered policy fell short of its target, so no abstention "
    "rule was adopted from it.",
    "Generalization is known to be poor: a single pre-registered evaluation "
    "on 200 public benign chat prompts blocked 162/200 (81.0%) at the 0.90 "
    "rule, far above the 10.3% validation benign-FPR benchmark, so the "
    "checkpoint over-triggers on ordinary benign chat; the proposed "
    "hard-negative mitigation was implemented but its training never ran, "
    "so no fix is claimed here (see reports/benign-chat-eval-20261019T1640"
    "18682950Z.md)",
)


def _is_loopback(host: str) -> bool:
    """True only for addresses that cannot route off this machine."""
    if host in {"localhost"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class BackendUnreachable(Exception):
    """The loopback backend could not be contacted. Carries no prompt data."""


Transport = Callable[[str, str, Any], tuple[int, Any]]


def _urllib_transport(timeout: float) -> Transport:
    """Standard-library JSON transport; no third-party HTTP client."""

    def transport(method: str, url: str, payload: Any) -> tuple[int, Any]:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                status = int(response.status)
        except urllib.error.HTTPError as error:
            body = error.read()
            status = int(error.code)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            # Type name only: never propagate low-level error strings that
            # could embed request content in a log or response.
            raise BackendUnreachable(type(error).__name__) from error
        try:
            parsed = json.loads(body) if body else {}
        except ValueError:
            parsed = {}
        return status, parsed

    return transport


# ---------------------------------------------------------------------------
# Aggregate-only figures, frozen from the tracked validation reports. Values
# are copied verbatim from the report files named in "sources"; nothing here
# reads data files at runtime and nothing here contains prompt text.
# ---------------------------------------------------------------------------

INSIGHTS: dict[str, Any] = {
    "note": INSIGHTS_NOTE,
    "test_split_loaded": False,
    "validation_rows": 941,
    "classifier_threshold": 0.90,
    "model_comparison": {
        "metric_note": (
            "validation-only; DeBERTa decisions use the deployed class-1 "
            "softmax >= 0.90 rule; latency is one batch timed with "
            "perf_counter on one machine, not a capacity benchmark"
        ),
        "rows": [
            {"model": "Logistic Regression (TF-IDF)", "accuracy": 0.9416,
             "macro_f1": 0.9397, "benign_fpr": 0.1130,
             "malicious_fnr": 0.0169, "mean_ms_per_text": 0.029},
            {"model": "Linear SVM (TF-IDF)", "accuracy": 0.9564,
             "macro_f1": 0.9554, "benign_fpr": 0.0737,
             "malicious_fnr": 0.0206, "mean_ms_per_text": 0.020},
            {"model": "DeBERTa v3 small (served model)", "accuracy": 0.9458,
             "macro_f1": 0.9442, "benign_fpr": 0.1032,
             "malicious_fnr": 0.0169, "mean_ms_per_text": 308.8},
        ],
    },
    "decision_audit": {
        "disagreements": 15,
        "rows": 941,
        "threshold_rule": {"accuracy": 0.9458, "macro_f1": 0.9442},
        "argmax_rule": {"accuracy": 0.9384, "macro_f1": 0.9362},
        "note": (
            "the deployed 0.90 threshold rule was not tuned or changed; "
            "the audit only compares it with argmax"
        ),
    },
    "error_analysis": {
        "false_positives": 42,
        "false_negatives": 9,
        "note": (
            "errors concentrate in specific dataset source/category groups; "
            "several per-source rates are statistically unstable, and high-"
            "confidence false positives may reflect label noise or genuine "
            "model bias — this tool cannot distinguish the two and relabels "
            "nothing"
        ),
    },
    "calibration": {
        "expected_calibration_error": 0.0562,
        "brier_score": 0.0550,
        "note": (
            "scores are treated as scores, not calibrated probabilities; "
            "the distribution is bimodal; the pre-registered abstention "
            "grid's best policy caught 31.4% of errors at 3.1% abstention, "
            "short of the 50% target, so no calibration or abstention claim "
            "is made"
        ),
    },
    "gateway_policy": {
        "policy_id": "band_v1",
        "allow_max": 0.70,
        "block_min": 0.90,
        "review_band_tradeoff": {
            "rows_in_band": 11,
            "fail_open": {"false_positives": 42, "false_negatives": 9},
            "fail_closed": {"false_positives": 49, "false_negatives": 5},
            "note": (
                "fail-open lets every ambiguous prompt through (more false "
                "negatives); fail-closed blocks them all (more false "
                "positives); human review sits between and assumes review "
                "capacity. The choice is a product risk decision, and the "
                "band defaults are illustrative, not optimal"
            ),
        },
    },
    "sources": (
        "reports/validation-model-comparison-20261017T180053566240Z.md",
        "reports/deberta-decision-audit-20261017T172138315024Z.md",
        "reports/deberta-error-analysis-20261017T174228255256Z.md",
        "reports/deberta-confidence-calibration-20261018T140720575767Z.md",
        "reports/source-reliability-policy-20261018T153548645826Z.md",
        "reports/benign-chat-eval-20261019T164018682950Z.md",
        "docs/gateway-policy.md",
    ),
    "limitations": LIMITATIONS,
}


def create_frontend(
    backend_url: str = DEFAULT_BACKEND_URL,
    transport: Transport | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> FastAPI:
    """Build the local scanner page over one loopback backend."""
    parsed = urlsplit(backend_url)
    if parsed.scheme not in {"http", "https"} or not _is_loopback(
        parsed.hostname or ""
    ):
        raise ValueError(
            "frontend may only proxy to a loopback backend so prompt text "
            f"stays on this machine; got {backend_url!r}"
        )
    base = backend_url.rstrip("/")
    app = FastAPI(title="PromptShield local scanner")
    app.state.backend_url = base
    app.state.transport = transport or _urllib_transport(timeout)

    def _forward(method: str, path: str, payload: Any) -> tuple[int, Any]:
        try:
            return app.state.transport(method, base + path, payload)
        except BackendUnreachable as error:
            raise HTTPException(503, UNAVAILABLE_DETAIL) from error

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # Static, fully self-contained page: inline CSS/JS, no external
        # assets, and no prompt-derived values baked in.
        return SCANNER_PAGE

    @app.post("/api/scan")
    def scan(request: PromptRequest) -> dict[str, Any]:
        # Same request contract as POST /gateway (reused PromptRequest); the
        # payload passes through unchanged so the page shows the backend's
        # real gateway decision and classifier prediction.
        status, body = _forward(
            "POST", "/gateway", {"prompt": request.prompt}
        )
        if status >= 400:
            detail = body.get("detail") if isinstance(body, dict) else None
            if status == 503:
                # Fail-closed canonical wording, whatever the backend sent.
                raise HTTPException(503, UNAVAILABLE_DETAIL)
            if not isinstance(detail, str):
                detail = (
                    "prompt failed backend validation"
                    if status == 422
                    else "backend rejected the request"
                )
            raise HTTPException(status, detail)
        if not isinstance(body, dict) or "gateway_decision" not in body:
            # A malformed success body is treated as a failed check, never
            # as an allow.
            raise HTTPException(503, UNAVAILABLE_DETAIL)
        return body

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            status, body = app.state.transport(
                "GET", base + "/health", None
            )
        except BackendUnreachable:
            return {
                "backend_reachable": False,
                "loaded": None,
                "device": None,
                "model_id": None,
                "load_error": None,
                "note": UNREACHABLE_NOTE,
            }
        reachable = status < 400 and isinstance(body, dict)
        return {
            "backend_reachable": reachable,
            "loaded": body.get("loaded") if reachable else None,
            "device": body.get("device") if reachable else None,
            "model_id": body.get("model_id") if reachable else None,
            "load_error": body.get("load_error") if reachable else None,
            "note": HEALTH_HONESTY_NOTE,
        }

    @app.get("/api/insights")
    def insights() -> dict[str, Any]:
        # Aggregate constants only; reads no data, model, or review files.
        return INSIGHTS

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    if not _is_loopback(args.host):
        print(
            "frontend may only bind a loopback address so prompt text stays "
            f"on this machine; got {args.host!r}",
            file=sys.stderr,
        )
        return 1
    try:
        app = create_frontend(args.backend, timeout=args.timeout)
    except ValueError as error:
        print(f"frontend failed: {error}", file=sys.stderr)
        return 1

    import uvicorn

    # Aggregate-only console output: no prompt text, ever.
    print(
        f"PromptShield scanner on http://{args.host}:{args.port}  "
        f"(backend: {app.state.backend_url})\n"
        "Prompts are proxied to the loopback backend only. Stop with Ctrl+C.",
        file=sys.stderr,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# ---------------------------------------------------------------------------
# Self-contained scanner page. No external stylesheets, scripts, fonts, or
# images: every byte the browser needs is inline, so nothing is fetched from
# a network service and no prompt text can leak through a subresource
# request. All values are rendered with textContent, never innerHTML.
# ---------------------------------------------------------------------------

SCANNER_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PromptShield &mdash; local prompt scanner</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    margin: 0 auto; padding: 1.25rem; max-width: 64rem; line-height: 1.5;
    background: Canvas; color: CanvasText;
  }
  header { display: flex; justify-content: space-between; align-items: baseline;
           flex-wrap: wrap; gap: .5rem; margin-bottom: .25rem; }
  h1 { font-size: 1.2rem; margin: 0; }
  h2 { font-size: 1rem; margin: 0 0 .5rem; }
  .badge-local {
    display: inline-block; font-size: .72rem; font-weight: 700;
    letter-spacing: .03em; text-transform: uppercase; color: #2563eb;
    border: 1px solid #2563eb; border-radius: 4px; padding: .12rem .45rem;
  }
  @media (prefers-color-scheme: dark) { .badge-local { color: #93c5fd; border-color: #93c5fd; } }
  .meta { color: #666; font-size: .87rem; }
  @media (prefers-color-scheme: dark) { .meta { color: #aaa; } }
  .card {
    border: 1px solid rgba(127,127,127,.35); border-radius: 10px;
    padding: 1rem 1.1rem; margin: 1rem 0; background: rgba(127,127,127,.05);
  }
  textarea { width: 100%; min-height: 8rem; resize: vertical; font: inherit;
             padding: .6rem; border-radius: 6px;
             border: 1px solid rgba(127,127,127,.5); background: transparent;
             color: inherit; margin-top: .4rem; }
  .bar { display: flex; gap: .6rem; align-items: center; margin-top: .7rem;
         flex-wrap: wrap; }
  button { padding: .55rem 1.2rem; font-size: .95rem; font: inherit;
           cursor: pointer; border-radius: 6px;
           border: 1px solid rgba(127,127,127,.6); background: transparent;
           color: inherit; }
  button.primary { background: #2563eb; border-color: #2563eb; color: #fff; }
  button:disabled { cursor: not-allowed; opacity: .45; }
  #counter { font-size: .8rem; }
  #counter.over { color: #b00020; font-weight: 700; }
  .banner { border-radius: 8px; padding: .7rem .9rem; margin: .8rem 0;
            font-size: .92rem; }
  .banner.error { border: 1px solid #b00020; color: #b00020;
                  background: rgba(176,0,32,.07); font-weight: 600; }
  .banner.warn { border: 1px solid #b45309; color: #92400e;
                 background: rgba(180,83,9,.08); }
  .banner.info { border: 1px solid rgba(127,127,127,.5); }
  @media (prefers-color-scheme: dark) {
    .banner.error { color: #ff8a80; border-color: #ff8a80;
                    background: rgba(255,138,128,.08); }
    .banner.warn { color: #ffcc80; border-color: #ffcc80;
                   background: rgba(255,204,128,.08); }
  }
  #decisionBadge { display: inline-block; font-size: 1.05rem; font-weight: 800;
                   letter-spacing: .04em; text-transform: uppercase;
                   border-radius: 8px; padding: .35rem .9rem; }
  #decisionBadge.allow  { color: #15803d; border: 2px solid #15803d; }
  #decisionBadge.review { color: #b45309; border: 2px solid #b45309; }
  #decisionBadge.block  { color: #b00020; border: 2px solid #b00020; }
  #decisionBadge.none   { color: #666; border: 2px solid rgba(127,127,127,.5); }
  @media (prefers-color-scheme: dark) {
    #decisionBadge.allow  { color: #4ade80; border-color: #4ade80; }
    #decisionBadge.review { color: #fbbf24; border-color: #fbbf24; }
    #decisionBadge.block  { color: #f87171; border-color: #f87171; }
    #decisionBadge.none   { color: #aaa; }
  }
  #decisionText { margin: .6rem 0 0; font-size: .95rem; }
  table { border-collapse: collapse; width: 100%; font-size: .88rem;
          margin-top: .7rem; }
  th, td { text-align: left; padding: .35rem .5rem;
           border-bottom: 1px solid rgba(127,127,127,.25);
           vertical-align: top; }
  th { font-weight: 600; white-space: nowrap; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  ul { margin: .4rem 0; padding-left: 1.3rem; }
  li { margin: .25rem 0; font-size: .9rem; }
  code { font-size: .85em; background: rgba(127,127,127,.15);
         border-radius: 4px; padding: 0 .3rem; }
  .kv { display: grid; grid-template-columns: max-content 1fr; gap: .2rem .8rem;
        font-size: .9rem; margin-top: .6rem; }
  .kv dt { font-weight: 600; }
  .kv dd { margin: 0; }
  @media (max-width: 40rem) { .kv { grid-template-columns: 1fr; } }
  footer { margin: 1.5rem 0 1rem; font-size: .82rem; color: #666; }
  @media (prefers-color-scheme: dark) { footer { color: #aaa; } }
  .spin { display: inline-block; width: .8em; height: .8em;
          border: 2px solid rgba(127,127,127,.4); border-top-color: #2563eb;
          border-radius: 50%; animation: r 1s linear infinite;
          vertical-align: -.1em; margin-right: .4em; }
  @keyframes r { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>PromptShield &mdash; local prompt scanner</h1>
  <span class="badge-local">Local only &middot; no external services</span>
</header>
<p class="meta">Scans run against the local DeBERTa gateway on this machine.
Prompts are proxied to the loopback backend only, never stored, and never
sent to an external service or LLM.</p>

<div class="card">
  <h2>Scan a prompt</h2>
  <label for="prompt" class="meta">Prompt text</label>
  <textarea id="prompt" maxlength="4096"
    placeholder="Paste a prompt to check, e.g. a message you received that may be trying to override instructions…"></textarea>
  <div class="bar">
    <button id="scan" class="primary" type="button">Scan</button>
    <button id="clear" type="button">Clear</button>
    <span id="counter" class="meta">0 / 4096</span>
  </div>
  <div id="banner" class="banner info" hidden></div>
</div>

<div class="card" id="resultCard" hidden>
  <h2>Result</h2>
  <span id="decisionBadge" class="none">&hellip;</span>
  <p id="decisionText" class="meta">Run a scan to see the gateway decision.</p>
  <dl class="kv">
    <dt>Gateway decision</dt>
    <dd id="gatewayDecision">&mdash;</dd>
    <dt>Classifier prediction</dt>
    <dd id="classifierDecision">&mdash;</dd>
    <dt>Model score</dt>
    <dd><span id="modelScore">&mdash;</span>
        <span class="meta">class-1 softmax output &mdash; an
        <strong>uncalibrated score, not a probability or
        confidence</strong></span></dd>
    <dt>Gateway reasons</dt>
    <dd id="reasons">&mdash;</dd>
    <dt>Injection markers matched</dt>
    <dd id="markers">&mdash;</dd>
    <dt>Policy</dt>
    <dd id="policy">&mdash;</dd>
  </dl>
</div>

<div class="card">
  <h2>System status</h2>
  <div class="bar" style="margin-top:0">
    <button id="refreshHealth" type="button">Refresh</button>
    <span id="healthReachable" class="meta">&hellip;</span>
  </div>
  <dl class="kv">
    <dt>Model loaded</dt>
    <dd id="healthLoaded">&mdash;</dd>
    <dt>Device</dt>
    <dd id="healthDevice">&mdash;</dd>
    <dt>Checkpoint</dt>
    <dd id="healthModel">&mdash;</dd>
    <dt>Load error</dt>
    <dd id="healthError">&mdash;</dd>
  </dl>
  <p class="meta" id="healthNote"></p>
</div>

<div class="card">
  <h2>Model insights <span class="meta">(validation-only aggregates)</span></h2>
  <p class="meta" id="insightsNote"></p>
  <table>
    <thead><tr><th>Model</th><th>Accuracy</th><th>Macro F1</th>
      <th>Benign FPR</th><th>Malicious FNR</th></tr></thead>
    <tbody id="cmpBody"></tbody>
  </table>
  <p class="meta" id="cmpNote"></p>
  <dl class="kv">
    <dt>Calibration</dt>
    <dd id="calib">&mdash;</dd>
    <dt>Validation errors</dt>
    <dd id="errors">&mdash;</dd>
    <dt>Gateway trade-off</dt>
    <dd id="tradeoff">&mdash;</dd>
  </dl>
  <h2 style="margin-top:.9rem">Known limitations</h2>
  <ul id="limitations"></ul>
  <p class="meta" id="sources"></p>
</div>

<footer>
  This tool is a research prototype. It does not guarantee security, does
  not prevent all prompt injections, and an "Allow" verdict only means the
  local model scored the prompt low and found no known injection marker.
  Scores are uncalibrated; every published figure is validation-only.
</footer>

<script>
"use strict";
(function () {
  function $(id) { return document.getElementById(id); }

  var DECISION_COPY = {
    allow: {
      title: "Allow",
      text: "The gateway scored this prompt below the review band and the " +
        "known-phrase rules found no strong injection marker, so the prompt " +
        "would be let through. This is not a safety guarantee: the " +
        "classifier missed some malicious validation prompts, and it does " +
        "not recognize novel or obfuscated attacks."
    },
    review: {
      title: "Review",
      text: "A human should read this prompt before it is used. Either the " +
        "score fell in the ambiguous band between the allow and block " +
        "thresholds, or a known injection phrasing matched at a low score. " +
        "The gateway deliberately abstains instead of guessing."
    },
    block: {
      title: "Block",
      text: "The model score is at or above the 0.90 block threshold, which " +
        "is the unchanged classifier decision. On validation this band was " +
        "mostly genuinely malicious, but it also blocked 42 of 941 benign " +
        "validation prompts, so a block can be a false positive."
    }
  };

  function showBanner(kind, message) {
    var banner = $("banner");
    banner.hidden = false;
    banner.className = "banner " + kind;
    banner.textContent = message;
  }

  function hideBanner() { $("banner").hidden = true; }

  function resetResult() {
    $("resultCard").hidden = true;
    var badge = $("decisionBadge");
    badge.className = "none";
    badge.textContent = "\\u2026";
    ["gatewayDecision", "classifierDecision", "modelScore", "reasons",
     "markers", "policy"].forEach(function (id) {
      $(id).textContent = "\\u2014";
    });
  }

  function listText(values) {
    return values && values.length ? values.join(", ") : "none";
  }

  function showResult(body) {
    $("resultCard").hidden = false;
    var decision = body.gateway_decision;
    var copy = DECISION_COPY[decision] || {
      title: String(decision),
      text: "The backend returned an unrecognized decision; treat this " +
        "prompt as review until the backend is checked."
    };
    var badge = $("decisionBadge");
    badge.className = DECISION_COPY[decision] ? decision : "none";
    badge.textContent = copy.title;
    $("decisionText").textContent = copy.text;
    // textContent only: every value below is untrusted backend output.
    $("gatewayDecision").textContent =
      decision + " (reasons: " + listText(body.reasons) + ")";
    $("classifierDecision").textContent =
      body.classifier_decision + " at threshold " +
      body.classifier_threshold +
      " \\u2014 the raw classifier output, separate from the gateway decision";
    $("modelScore").textContent =
      Number(body.malicious_probability).toFixed(6);
    $("reasons").textContent = listText(body.reasons);
    $("markers").textContent = listText(body.marker_rule_ids) +
      " \\u2014 deterministic known-phrase rules; they escalate to review, " +
      "never to block";
    var policy = body.policy || {};
    $("policy").textContent = "band_v1 \\u2014 allow below " +
      policy.allow_max + ", block at or above " + policy.block_min +
      "; band defaults are illustrative, not proven optimal";
  }

  function showScanFailure(detail) {
    resetResult();
    $("resultCard").hidden = false;
    var badge = $("decisionBadge");
    badge.className = "review";
    badge.textContent = "Treated as Review";
    $("decisionText").textContent =
      "The security check did not complete, so this prompt is NOT allowed. " +
      "Treat it as review and retry once the local backend is running. " +
      "A failed check is never shown as Allow.";
    $("gatewayDecision").textContent = "unavailable \\u2014 check failed";
    $("classifierDecision").textContent = "not available";
    $("modelScore").textContent = "not available";
    $("reasons").textContent = "check unavailable";
    $("markers").textContent = "not evaluated";
    $("policy").textContent = "fail-closed: a failed check is treated as review";
    showBanner("error", "Scan failed: " + detail);
  }

  function scan() {
    var prompt = $("prompt").value;
    if (!prompt.trim()) {
      resetResult();
      showBanner("error", "Enter a prompt to scan (validation error \\u2014 " +
        "nothing was sent).");
      return;
    }
    if (prompt.length > 4096) {
      resetResult();
      showBanner("error", "Prompt is too long (maximum 4096 characters " +
        "\\u2014 nothing was sent).");
      return;
    }
    hideBanner();
    resetResult();  // a stale Allow must never linger during a new scan
    $("scan").disabled = true;
    $("resultCard").hidden = false;
    $("decisionBadge").className = "none";
    $("decisionBadge").textContent = "";
    var spin = document.createElement("span");
    spin.className = "spin";
    $("decisionBadge").appendChild(spin);
    $("decisionBadge").appendChild(document.createTextNode("Scanning\\u2026"));
    $("decisionText").textContent =
      "Running the local gateway on this machine\\u2026";
    fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: prompt })
    }).then(function (response) {
      return response.json().then(function (body) {
        return { status: response.status, body: body };
      }).catch(function () {
        return { status: response.status, body: null };
      });
    }).then(function (res) {
      $("scan").disabled = false;
      if (res.status === 200 && res.body &&
          typeof res.body.gateway_decision === "string") {
        showResult(res.body);
        return;
      }
      var detail = res.body && typeof res.body.detail === "string"
        ? res.body.detail : "backend returned an unusable response";
      showScanFailure(detail);
    }).catch(function () {
      $("scan").disabled = false;
      showScanFailure("network error \\u2014 the frontend could not reach " +
        "its backend");
    });
  }

  function loadHealth() {
    fetch("/api/health").then(function (r) { return r.json(); })
      .then(function (h) {
        $("healthReachable").textContent = h.backend_reachable
          ? "backend reachable (loopback)" : "backend NOT reachable";
        $("healthReachable").style.color = h.backend_reachable ? "" : "#b00020";
        $("healthLoaded").textContent =
          h.loaded === null || h.loaded === undefined ? "\\u2014" :
          (h.loaded ? "yes" : "no \\u2014 scans will fail closed");
        $("healthDevice").textContent = h.device || "\\u2014";
        $("healthModel").textContent = h.model_id || "\\u2014";
        $("healthError").textContent = h.load_error || "none";
        $("healthNote").textContent = h.note || "";
      }).catch(function () {
        $("healthReachable").textContent =
          "frontend health endpoint unreachable";
        $("healthNote").textContent = "";
      });
  }

  function fill(key, text) { $(key).textContent = text; }

  function loadInsights() {
    fetch("/api/insights").then(function (r) { return r.json(); })
      .then(function (data) {
        fill("insightsNote", data.note || "");
        var tbody = $("cmpBody");
        tbody.textContent = "";
        (data.model_comparison.rows || []).forEach(function (row) {
          var tr = document.createElement("tr");
          [row.model, row.accuracy.toFixed(4), row.macro_f1.toFixed(4),
           row.benign_fpr.toFixed(4), row.malicious_fnr.toFixed(4)
          ].forEach(function (value, i) {
            var td = document.createElement("td");
            if (i > 0) { td.className = "num"; }
            td.textContent = value;
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
        fill("cmpNote", data.model_comparison.metric_note || "");
        var calib = data.calibration || {};
        fill("calib", "ECE " + calib.expected_calibration_error +
          ", Brier " + calib.brier_score + " \\u2014 " + (calib.note || ""));
        var errors = data.error_analysis || {};
        fill("errors", errors.false_positives + " false positives, " +
          errors.false_negatives + " false negatives on 941 validation " +
          "rows \\u2014 " + (errors.note || ""));
        var band = (data.gateway_policy || {}).review_band_tradeoff || {};
        fill("tradeoff", band.rows_in_band +
          " validation rows fall in the review band; resolving them " +
          "fail-open gives " + band.fail_open.false_positives + " FP / " +
          band.fail_open.false_negatives + " FN, fail-closed gives " +
          band.fail_closed.false_positives + " FP / " +
          band.fail_closed.false_negatives + " FN \\u2014 " +
          (band.note || ""));
        var list = $("limitations");
        list.textContent = "";
        (data.limitations || []).forEach(function (item) {
          var li = document.createElement("li");
          li.textContent = item;
          list.appendChild(li);
        });
        fill("sources", "Sources: " + (data.sources || []).join("; ") +
          (data.test_split_loaded
            ? "" : " \\u2014 test split loaded: no"));
      }).catch(function () {
        fill("insightsNote", "insights unavailable from the local frontend");
      });
  }

  $("scan").addEventListener("click", scan);
  $("clear").addEventListener("click", function () {
    $("prompt").value = "";
    $("counter").textContent = "0 / 4096";
    $("counter").className = "meta";
    hideBanner();
    resetResult();
  });
  $("prompt").addEventListener("input", function () {
    var length = $("prompt").value.length;
    $("counter").textContent = length + " / 4096";
    $("counter").className = length > 4096 ? "meta over" : "meta";
  });
  $("refreshHealth").addEventListener("click", loadHealth);

  loadHealth();
  loadInsights();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
