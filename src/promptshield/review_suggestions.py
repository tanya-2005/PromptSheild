"""Deterministic local preliminary suggestions for the human review packet.

This module produces a *triage aid* for each review row: a suggested label, a
short reason, and an explicit machine-generated / unverified marker. It is
deliberately small, deterministic, and offline so every suggestion can be
unit-tested and explained.

Integrity rules enforced here:

- The DeBERTa prediction, its probability, and the threshold are never inputs.
  A model's own output is not independent evidence about its own
  correctness, so it cannot support a suggestion about its own error.
- Only two input families are consulted: lexical/structural patterns over the
  prompt text, and dataset metadata fields whose semantics are documented
  in-repo (currently only an injection-type ``category`` on a benign-labeled
  row, which is a metadata contradiction independent of any model).
  ``severity`` and ``augmented`` are not used because this repository does
  not document what values they carry.
- When the rules cannot support any conclusion the engine abstains with
  ``needs_manual_review`` or ``ambiguous_insufficient_context`` instead of
  inventing one. Absence of known markers is weak negative evidence, never a
  proof of benignity, so it only ever yields a low-confidence suggestion.
- Reasons cite rule identifiers only; they never quote prompt text, so a
  suggestion payload cannot leak raw text. Nothing here prints, logs, writes,
  or transmits anything: pure functions in, plain dicts out.
- Same input row, same output: no clocks, randomness, or environment reads.

The suggestions are unverified machine output. They do not establish whether
any example is mislabeled; only a named human reviewer reading the full
prompt can record a decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from promptshield.error_review import REVIEW_LABELS

# Suggestion-only state: rows where no conclusion is supported. It is NOT one
# of the four decision labels and can never be saved as a human decision.
NEEDS_MANUAL_REVIEW = "needs_manual_review"

ENGINE_ID = "deterministic_local_rules_v1"

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_NONE = "none"

MACHINE_NOTICE = (
    "Machine-generated preliminary suggestion, unverified. It is a triage aid "
    "from local deterministic rules, not a judgment about the label; only a "
    "human reading the prompt can decide."
)
SELF_EVIDENCE_EXCLUSION = (
    "The DeBERTa prediction and its probability are excluded by design: a "
    "model's own output is not independent evidence about its own correctness."
)


@dataclass(frozen=True)
class _TextRule:
    """A named family of literal/regex patterns searched in prompt text."""

    rule_id: str
    weight: str  # "strong" -> can suggest likely_mislabeled, "weak" -> abstain
    patterns: tuple[str, ...]


_STRONG_RULES: tuple[_TextRule, ...] = (
    _TextRule(
        rule_id="instruction_override",
        weight="strong",
        patterns=(
            r"\bignor(?:e|es|ed|ing)\s+(?:all\s+|any\s+|your\s+|the\s+)*"
            r"(?:previous|prior|above|earlier|preceding|initial)\s+(?:instructions?|prompts?|"
            r"directives?|messages?|inputs?|rules?|guidelines?)\b",
            r"\bdisregard(?:s|ed|ing)?\s+(?:all\s+|any\s+|your\s+|the\s+)*(?:previous|prior|"
            r"above|earlier|initial)\s+(?:instructions?|prompts?|rules?|"
            r"guidelines?)\b",
            r"\bforget\s+(?:all\s+)?(?:your|the|previous|prior)\s+"
            r"(?:instructions?|rules?|guidelines?|prompts?)\b",
            r"\boverride\s+(?:your|these|the|all)\s+(?:instructions?|"
            r"guidelines?|rules?|programming|restrictions?|guardrails?)\b",
            r"\b(?:do\s+not|don'?t)\s+(?:follow|obey)\s+(?:your|the|previous|"
            r"any)\s+(?:instructions?|rules?|guidelines?)\b",
            r"\bignore\s+(?:your|its|the)\s+(?:programming|training|"
            r"guardrails?|safety\s+(?:rules|guidelines|instructions))\b",
            # Flexible form of the canonical override template: an override
            # verb within a short, sentence-bounded window of an
            # instruction-like noun ("ignore the instructions above",
            # "disregard those guidelines"). Verbless forms such as "follow
            # the instructions above" are deliberately not matched, and
            # message/response nouns are excluded because "ignore my
            # previous response" is common in benign multi-turn phrasing.
            r"\b(?:ignor(?:e|es|ed|ing)|disregard(?:s|ed|ing)?|forget|"
            r"forgot(?:ten)?|overrid(?:e|es|ed|ing))\b[^.?!]{0,48}"
            r"\b(?:instructions?|prompts?|rules?|guidelines?|directives?|"
            r"policies)\b",
        ),
    ),
    _TextRule(
        rule_id="system_prompt_exfiltration",
        weight="strong",
        patterns=(
            r"\b(?:reveal|show|print|display|repeat|output|give|provide)"
            r"\s+(?:me\s+)?(?:your|the)\s+(?:system|initial|original|"
            r"developer|hidden)\s+(?:prompt|instructions?|message|"
            r"guidelines?)\b",
            r"\bwhat\s+(?:is|are)\s+(?:your|the)\s+(?:system|initial|"
            r"original|developer)\s+(?:prompt|instructions?|guidelines?)\b",
        ),
    ),
    _TextRule(
        rule_id="jailbreak_lexicon",
        weight="strong",
        patterns=(
            r"\bjailbreak(?:ed|ing)?\b",
            r"\bdeveloper\s+mode\b",
            r"\bdo\s+anything\s+now\b",
            r"\bmaintenance\s+(?:prompt|mode)\b",
            r"\bevil\s+mode\b",
        ),
    ),
    _TextRule(
        rule_id="encoded_command",
        weight="strong",
        patterns=(
            r"\bdecode\s+(?:the\s+)?following(?:\s+text)?(?:\s+and|\s*,)?\s*"
            r"(?:execute|run|obey|follow|apply)\b",
            r"\b(?:execute|run)\s+(?:the\s+)?(?:decoded|base64)\s+(?:text|"
            r"message|command|instructions?|content)\b",
        ),
    ),
)

_WEAK_RULES: tuple[_TextRule, ...] = (
    _TextRule(
        rule_id="persona_escape",
        weight="weak",
        patterns=(
            r"\byou\s+are\s+now\b",
            r"\bpretend\s+(?:you\s+are|to\s+be)\b",
            r"\bact\s+as\s+if\s+you\s+(?:have|are|can|were)\b",
            r"\broleplay\s+as\b",
            r"\bfrom\s+now\s+on\s+you\s+(?:are|will|must)\b",
        ),
    ),
    _TextRule(
        rule_id="boundary_statement",
        weight="weak",
        patterns=(
            r"\b(?:you|ai\s+assistant)\s+(?:have|has)\s+no\s+(?:restrictions?|"
            r"rules?|limits?|filters?|guidelines?)\b",
            r"\bwithout\s+(?:any\s+)?(?:restrictions?|limitations?|rules?|"
            r"constraints?|boundaries)\b",
            r"\brelease\s+(?:your|the)\s+(?:restrictions?|inhibitions?|"
            r"guardrails?)\b",
            r"\b(?:free\s+of|free\s+from|not\s+bound\s+by|no\s+longer\s+bound"
            r"\s+by)\s+(?:any\s+)?(?:rules?|restrictions?|limitations?|"
            r"filters?|guidelines?)\b",
            r"\b(?:unrestricted|unfiltered|uncensored)\b",
        ),
    ),
    _TextRule(
        rule_id="obfuscation_blob",
        weight="weak",
        patterns=(
            r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{48,}={0,2}(?![A-Za-z0-9+/])",
            r"\b[0-9a-fA-F]{48,}\b",
        ),
    ),
)

# Compiled separately because blob matching deliberately skips the
# hyphen-collapsed text form (see ``suggest_row``).
_OBFUSCATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in next(
        rule.patterns for rule in _WEAK_RULES
        if rule.rule_id == "obfuscation_blob"
    )
)

# Standard benign task framings. Consulted ONLY when no marker matched, and
# they carry low confidence because absence of markers is negative evidence.
_BENIGN_RULES: tuple[_TextRule, ...] = (
    _TextRule(
        rule_id="benign_translation",
        weight="benign",
        patterns=(
            r"\btranslate\s+(?:this|the\s+following|the\s+text|it|these)\b",
            r"\btranslate\b[^.?!]{0,60}\binto\s+(?:english|french|spanish|"
            r"german|italian|portuguese|russian|japanese|chinese|korean|"
            r"hindi|arabic|urdu|dutch|latin)\b",
        ),
    ),
    _TextRule(
        rule_id="benign_summarization",
        weight="benign",
        patterns=(
            r"\bsummari[sz]e\s+(?:this|the\s+following|the\s+text|the\s+"
            r"article|the\s+paragraph)\b",
            r"\b(?:give|write)\s+(?:me\s+)?a\s+(?:short\s+)?summary\s+of\s+"
            r"(?:this|the\s+following)\b",
        ),
    ),
    _TextRule(
        rule_id="benign_math",
        weight="benign",
        patterns=(
            r"\bsolve\s+(?:this|the\s+following)?\s*(?:simple\s+|basic\s+|"
            r"math(?:ematical)?)?\s*equation\b",
            r"\bwhat\s+is\s+\d{1,5}\s*[\+\-\*\/x\u00d7]\s*\d{1,5}\b",
        ),
    ),
    _TextRule(
        rule_id="benign_writing",
        weight="benign",
        patterns=(
            r"\bwrite\s+(?:me\s+)?a\s+(?:short\s+|simple\s+|one-paragraph\s+)?"
            r"(?:poem|haiku|essay|story|paragraph|outline|email|letter|list)\b",
        ),
    ),
    _TextRule(
        rule_id="benign_coding_help",
        weight="benign",
        patterns=(
            r"\b(?:debug|fix)\s+(?:this|the\s+following)\s+(?:code|function|"
            r"script|error)\b",
            r"\bwrite\s+(?:me\s+)?a\s+(?:function|script|program|python\s+"
            r"script)\b",
        ),
    ),
    _TextRule(
        rule_id="benign_explanation",
        weight="benign",
        patterns=(r"\bexplain\s+(?:how|what|why|the\s+concept\s+of)\b",),
    ),
)

# Downgrades a marker match when the text frames injections as something to
# detect or teach about rather than as an instruction to obey: rules cannot
# tell "mention" from "use", so the engine abstains instead of guessing.
_DEFENSIVE_PATTERNS: tuple[str, ...] = (
    r"\b(?:how|what)\s+(?:would|do|does)\s+(?:you|i|we)\s+(?:detect|"
    r"identify|spot)\b",
    r"\b(?:detect|identify|spot|recogni[sz]e|flag)\s+(?:prompt\s+)?"
    r"(?:injections?|jailbreaks?|override\s+attempts?|attacks?)\b",
    r"\b(?:examples?|signs?|anatomy)\s+of\s+(?:a\s+|an\s+)?(?:prompt\s+"
    r"injection|injection|jailbreak)\b",
    r"\bdefend\s+against\b",
)

# Documented injection-type values for the ``category`` metadata field. When
# such a value sits on a benign-labeled row the dataset metadata contradicts
# itself, which is model-independent evidence for ``likely_mislabeled``.
_INJECTION_CATEGORIES = frozenset({
    "injection", "prompt_injection", "attack", "jailbreak", "adversarial",
    "adversary", "malicious",
})


def _compile(rules: tuple[_TextRule, ...]) -> list[tuple[str, list[re.Pattern[str]]]]:
    return [
        (
            rule.rule_id,
            [re.compile(p, re.IGNORECASE) for p in rule.patterns],
        )
        for rule in rules
    ]


_STRONG_COMPILED = _compile(_STRONG_RULES)
_WEAK_COMPILED = _compile(_WEAK_RULES)
_BENIGN_COMPILED = _compile(_BENIGN_RULES)
_DEFENSIVE_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in _DEFENSIVE_PATTERNS
]

# Deterministic, content-preserving normalizations used only for matching;
# the raw text itself is never returned from this module.
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_INNER_HYPHEN = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")


def _lexical_forms(text: str) -> list[str]:
    """Zero-width-stripped and hyphen-collapsed variants for rule matching.

    Catches trivial evasion such as zero-width joiners between words or
    hyphen-split tokens ("ig-nore"). Deduplicated so identical forms are
    matched once.
    """
    stripped = _ZERO_WIDTH.sub("", text)
    collapsed = _INNER_HYPHEN.sub("", stripped)
    forms: list[str] = []
    for form in (stripped, collapsed):
        if form and form not in forms:
            forms.append(form)
    return forms


def _matched_forms(
    compiled: list[tuple[str, list[re.Pattern[str]]]], forms: list[str]
) -> list[str]:
    """Rule ids matching any form, first-seen order preserved."""
    hits: list[str] = []
    for form in forms:
        for rule_id in _matched(compiled, form):
            if rule_id not in hits:
                hits.append(rule_id)
    return hits


def _matched(
    compiled: list[tuple[str, list[re.Pattern[str]]]], text: str
) -> list[str]:
    """Rule ids whose patterns appear in ``text`` (input order preserved)."""
    hits = []
    for rule_id, patterns in compiled:
        if any(pattern.search(text) for pattern in patterns):
            hits.append(rule_id)
    return hits


def matched_strong_marker_rules(text: str) -> list[str]:
    """Public, text-free view of the strong injection-marker rules.

    Returns the IDs of the strong marker rules (instruction override,
    system-prompt exfiltration, jailbreak lexicon, encoded commands) whose
    patterns match ``text``, after the same zero-width stripping and
    hyphen collapsing used by ``suggest_row``. Rule IDs are pattern names,
    never text spans, so the result cannot leak prompt content. Pure and
    deterministic; intended for the protective gateway, which escalates a
    matched prompt to human review (never to an automatic block) because
    the rules cannot tell "mention" from "use".
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return _matched_forms(_STRONG_COMPILED, _lexical_forms(text))


def _metadata_markers(row: dict[str, Any]) -> list[str]:
    """Model-independent metadata contradictions (see module docstring)."""
    category = row.get("category")
    if not isinstance(category, str):
        return []
    if category.strip().lower() not in _INJECTION_CATEGORIES:
        return []
    if row.get("true_label") == 0:
        return ["metadata_marks_injection"]
    return []


def suggest_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-generated, unverified preliminary suggestion.

    The result never contains prompt text: reasons cite rule ids only, so the
    payload can be shown in the local UI and asserted free of raw text in
    tests. ``suggested_label`` is one of the four decision labels (when the
    rules support a hypothesis) or ``needs_manual_review`` (when they do
    not); ``adoptable`` states which case applies.
    """
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        text = ""

    meta_hits = _metadata_markers(row)
    forms = _lexical_forms(text) if text else []
    strong = _matched_forms(_STRONG_COMPILED, forms)
    weak = _matched_forms(_WEAK_COMPILED, forms)
    # Obfuscation blobs are matched only on the zero-width-stripped form:
    # hyphen collapsing could fabricate long alnum runs out of hyphenated
    # word lists, which would be a false signal.
    stripped = _ZERO_WIDTH.sub("", text) if text else ""
    if "obfuscation_blob" in weak and (
        not stripped or not any(
            pattern.search(stripped) for pattern in _OBFUSCATION_PATTERNS
        )
    ):
        weak.remove("obfuscation_blob")
    benign = _matched_forms(_BENIGN_COMPILED, forms)
    defensive = any(
        pattern.search(form) for form in forms for pattern in _DEFENSIVE_COMPILED
    )

    matched: list[str] = [*meta_hits, *strong, *weak]
    if meta_hits:
        label = "likely_mislabeled"
        support = len(meta_hits) + len(strong)
        confidence = CONFIDENCE_HIGH if support >= 2 else CONFIDENCE_MEDIUM
        parts = [
            "row metadata marks an injection-type category while the row is "
            "benign-labeled (a dataset contradiction, not a model output)"
        ]
        if strong:
            parts.append("text also matches " + ", ".join(strong))
        reason = "; ".join(parts) + "."
    elif strong and defensive:
        # A marker is present but the text is about detecting injections:
        # mention versus use cannot be decided by rules, so abstain.
        label = "ambiguous_insufficient_context"
        confidence = CONFIDENCE_MEDIUM
        reason = (
            "marker rule(s) matched (" + ", ".join(strong) + ") but the text "
            "frames them as detection or education about injections; mention "
            "versus use needs human judgment"
        )
    elif strong:
        label = "likely_mislabeled"
        confidence = CONFIDENCE_HIGH if len(strong) >= 2 else CONFIDENCE_MEDIUM
        reason = (
            "text matches injection marker rule(s): "
            + ", ".join(strong)
            + "; independent of the model's own prediction"
        )
    elif weak:
        matched = [*meta_hits, *strong, *weak]
        label = "ambiguous_insufficient_context"
        confidence = CONFIDENCE_LOW
        reason = (
            "only weak signal rule(s) matched ("
            + ", ".join(weak)
            + "); too ambiguous to infer a label without human judgment"
        )
    elif benign:
        matched = [*meta_hits, *strong, *weak, *benign]
        label = "genuine_model_false_positive"
        confidence = CONFIDENCE_LOW
        tasks = ", ".join(rule_id.replace("benign_", "") for rule_id in benign)
        reason = (
            "no override, exfiltration, persona-escape, jailbreak, or "
            "obfuscation markers; text frames a standard " + tasks
            + " request. Absence of markers is weak negative evidence only"
        )
    elif not text:
        label = NEEDS_MANUAL_REVIEW
        confidence = CONFIDENCE_NONE
        reason = (
            "row has no usable text for rule matching; manual judgment "
            "required"
        )
    else:
        label = NEEDS_MANUAL_REVIEW
        confidence = CONFIDENCE_NONE
        reason = (
            "no independent text or metadata markers and no standard-task "
            "framing; label correctness cannot be inferred by local rules"
        )

    if defensive and strong:
        matched = [*matched, "defensive_framing"]

    return {
        "machine_generated": True,
        "verified": False,
        "engine": ENGINE_ID,
        "suggested_label": label,
        "confidence": confidence,
        "reason": reason,
        "matched_rule_ids": matched,
        "adoptable": label in REVIEW_LABELS,
        "notice": MACHINE_NOTICE,
        "excluded_evidence": SELF_EVIDENCE_EXCLUSION,
    }
