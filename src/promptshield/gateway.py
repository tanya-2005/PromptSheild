"""Local protective gateway policy for PromptShield inference.

This module implements the ``band_v1`` gateway decision rule documented in
``docs/gateway-policy.md``: a risk-triage wrapper that turns one uncalibrated
model score plus the deterministic strong injection-marker rules into one of
three decisions — ``allow``, ``review``, or ``block``. It is a policy layer,
not a classifier change.

Integrity rules enforced here:

- Pure policy: probabilities and marker rule IDs in, plain dicts out. No
  clocks, randomness, environment reads, filesystem, or network access, and
  no prompt text is ever returned, stored, or logged — reasons are fixed
  constant strings and marker IDs are pattern names, never text spans.
- The gateway never modifies the classifier: ``DebertaInference``, its
  ``0.90`` decision threshold, and the ``/predict`` contract are untouched.
  ``block_min`` defaults to exactly the deployed threshold, so ``block``
  means "the classifier already says malicious"; ``classifier_decision`` is
  echoed separately in API responses so the two decisions cannot be conflated.
- Failure is fail-closed by contract: the API layer maps an unavailable
  model or an inference error to HTTP 503 with a fixed detail string; the
  gateway never emits ``allow`` without a completed score and marker check.
- The band defaults (``allow_max = 0.70``, ``block_min = 0.90``) are read off
  the pre-registered validation abstention grid for illustration only. No
  optimality is claimed, and all supporting figures are validation-only.

The gateway is a triage aid, not a security guarantee: it cannot establish
that an allowed prompt contains no injection, and adversarial or novel
injections can pass through the score bands alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

POLICY_ID = "band_v1"
HYBRID_POLICY_ID = "hybrid_experimental_v1"

ALLOW = "allow"
REVIEW = "review"
BLOCK = "block"
DECISIONS = (ALLOW, REVIEW, BLOCK)

DEFAULT_ALLOW_MAX = 0.70
DEFAULT_BLOCK_MIN = 0.90

# Fixed reason codes. They are constants, never derived from prompt text, so
# a gateway response or stats payload cannot leak raw input.
REASON_BLOCK_BAND = "score_at_or_above_block_min"
REASON_REVIEW_BAND = "score_in_review_band"
REASON_STRONG_MARKER = "strong_injection_marker_matched"
REASON_BELOW_ALLOW_MAX = "score_below_allow_max"
REASON_HYBRID_AGREEMENT = "hybrid_models_agree"
REASON_HYBRID_DISAGREEMENT = "hybrid_models_disagree"

# Fixed 503 detail. Callers must treat this outcome as ``review``: a security
# check that cannot run never silently allows a prompt.
UNAVAILABLE_DETAIL = (
    "gateway security check unavailable; treat this prompt as review, "
    "do not allow it"
)

POLICY_STATUS = (
    "fixed local band policy with defaults chosen from a validation-only "
    "abstention grid for illustration; not claimed optimal and not "
    "established on held-out test data"
)

MARKER_ENGINE_NOTICE = (
    "Strong injection-marker rules reused from promptshield.review_suggestions "
    "(deterministic, local, pattern-name IDs only). Markers escalate allow to "
    "review and never block, because the rules cannot tell mention from use."
)


@dataclass(frozen=True)
class GatewayPolicy:
    """The ``band_v1`` decision rule (see ``docs/gateway-policy.md``)."""

    allow_max: float = DEFAULT_ALLOW_MAX
    block_min: float = DEFAULT_BLOCK_MIN

    def __post_init__(self) -> None:
        if not 0 < self.allow_max < self.block_min < 1:
            raise ValueError(
                "gateway policy requires 0 < allow_max < block_min < 1"
            )

    @property
    def policy_id(self) -> str:
        return POLICY_ID

    def decide(
        self,
        malicious_probability: float,
        strong_marker_ids: Sequence[str] = (),
    ) -> tuple[str, list[str]]:
        """Map one score and the matched strong marker IDs to a decision.

        Returns ``(decision, reasons)``. ``decision`` is ``block`` at or
        above ``block_min``, ``review`` inside the band or below it when a
        strong marker matched, and ``allow`` otherwise. Reasons are fixed
        code strings; the matched marker IDs themselves are returned by the
        caller in its own field, never inside the reason strings.
        """
        if malicious_probability >= self.block_min:
            return BLOCK, [REASON_BLOCK_BAND]
        if malicious_probability >= self.allow_max:
            return REVIEW, [REASON_REVIEW_BAND]
        markers = sorted({
            str(marker).strip()
            for marker in strong_marker_ids
            if str(marker).strip()
        })
        if markers:
            return REVIEW, [REASON_STRONG_MARKER]
        return ALLOW, [REASON_BELOW_ALLOW_MAX]

    def decide_hybrid(
        self,
        malicious_probability: float,
        svm_decision: str,
        strong_marker_ids: Sequence[str] = (),
    ) -> tuple[str, list[str]]:
        """Apply the experimental DeBERTa/SVM agreement policy.

        DeBERTa's review band is evaluated first and remains review regardless
        of the SVM result. Outside that band, agreement blocks or allows and
        disagreement reviews. Strong markers retain their existing
        allow-to-review precedence and never override block.
        """
        if not 0.0 <= malicious_probability <= 1.0:
            raise ValueError("malicious_probability must be within [0, 1]")
        if svm_decision not in {"benign", "malicious"}:
            raise ValueError("svm_decision must be benign or malicious")
        if self.allow_max <= malicious_probability < self.block_min:
            return REVIEW, [REASON_REVIEW_BAND]

        deberta_decision = (
            "malicious"
            if malicious_probability >= self.block_min
            else "benign"
        )
        if deberta_decision == svm_decision:
            if deberta_decision == "malicious":
                return BLOCK, [REASON_HYBRID_AGREEMENT]
            markers = sorted({
                str(marker).strip()
                for marker in strong_marker_ids
                if str(marker).strip()
            })
            if markers:
                return REVIEW, [REASON_STRONG_MARKER]
            return ALLOW, [REASON_HYBRID_AGREEMENT]
        return REVIEW, [REASON_HYBRID_DISAGREEMENT]

    def describe(self) -> dict[str, Any]:
        """Text-free policy description for API responses and stats."""
        return {
            "policy_id": POLICY_ID,
            "allow_max": self.allow_max,
            "block_min": self.block_min,
            "status": POLICY_STATUS,
        }


@dataclass
class GatewayCounters:
    """Process-local aggregate decision counts. Contains no prompt text.

    Counts are plain integers kept in memory only: nothing is persisted,
    written to disk, or logged, and the mapping is reset with the process.
    """

    allow: int = 0
    review: int = 0
    block: int = 0
    unavailable: int = 0

    def record(self, decision: str) -> None:
        if decision not in DECISIONS:
            raise ValueError(f"unknown gateway decision: {decision!r}")
        setattr(self, decision, getattr(self, decision) + 1)

    def record_unavailable(self) -> None:
        self.unavailable += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy_id": POLICY_ID,
            "allow": self.allow,
            "review": self.review,
            "block": self.block,
            "unavailable": self.unavailable,
            "note": (
                "aggregate integer counts only; no prompt text is logged, "
                "stored, or persisted"
            ),
        }


def evaluate(
    policy: GatewayPolicy,
    counters: GatewayCounters,
    malicious_probability: float,
    strong_marker_ids: Sequence[str] = (),
    classifier_decision: str = "",
    classifier_threshold: float = 0.0,
) -> dict[str, Any]:
    """Apply the policy to one scored prompt and record the decision.

    The returned payload is text-free by construction: it echoes the score,
    the classifier's own decision and threshold, fixed reason codes, and the
    matched marker rule IDs (pattern names) — never the prompt itself.
    """
    if not 0.0 <= malicious_probability <= 1.0:
        raise ValueError("malicious_probability must be within [0, 1]")
    decision, reasons = policy.decide(malicious_probability, strong_marker_ids)
    counters.record(decision)
    payload: dict[str, Any] = {
        "gateway_decision": decision,
        "reasons": reasons,
        "malicious_probability": malicious_probability,
        "classifier_decision": classifier_decision,
        "classifier_threshold": classifier_threshold,
        "classifier_threshold_status": (
            "configurable experimental threshold selected from validation "
            "experiments, not independent test results"
        ),
        "marker_rule_ids": sorted({
            str(m).strip() for m in strong_marker_ids if str(m).strip()
        }),
        "marker_engine_notice": MARKER_ENGINE_NOTICE,
        "policy": policy.describe(),
    }
    return payload


def evaluate_hybrid(
    policy: GatewayPolicy,
    counters: GatewayCounters,
    malicious_probability: float,
    svm_decision: str,
    strong_marker_ids: Sequence[str] = (),
    classifier_decision: str = "",
    classifier_threshold: float = 0.0,
) -> dict[str, Any]:
    """Apply the experimental hybrid policy and record its decision."""
    decision, reasons = policy.decide_hybrid(
        malicious_probability, svm_decision, strong_marker_ids
    )
    counters.record(decision)
    return {
        "gateway_decision": decision,
        "reasons": reasons,
        "malicious_probability": malicious_probability,
        "classifier_decision": classifier_decision,
        "classifier_threshold": classifier_threshold,
        "classifier_threshold_status": (
            "configurable experimental threshold selected from validation "
            "experiments, not independent test results"
        ),
        "svm_decision": svm_decision,
        "gateway_mode": HYBRID_POLICY_ID,
        "marker_rule_ids": sorted({
            str(m).strip() for m in strong_marker_ids if str(m).strip()
        }),
        "marker_engine_notice": MARKER_ENGINE_NOTICE,
        "policy": {
            **policy.describe(),
            "policy_id": HYBRID_POLICY_ID,
            "status": (
                "experimental agreement policy; not production-ready and "
                "not established on held-out test data"
            ),
        },
    }
