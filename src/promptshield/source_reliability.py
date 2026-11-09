"""Validation-only source-stratified reliability audit and gateway policy note.

Reuses the aggregate artifacts already produced by the DeBERTa error-analysis,
error-investigation, and confidence-calibration studies. It does not load any
Parquet file, does not run model inference, and does not retrain anything.

The audit adds four things the earlier reports did not provide:

1. Wilson score confidence intervals for every per-source and per-category
   false-positive / false-negative rate, so small-sample groups are explicitly
   separated from stable ones instead of being read as point estimates.
2. A metadata/label consistency read that treats metadata as descriptive and
   never as ground truth, and does not relabel any example.
3. A signal-availability audit that inspects the real FastAPI request schema to
   establish which dataset fields are offline diagnostics only.
4. A gateway policy-design note comparing fail-open, fail-closed, and
   human-review behaviour using the measured abstention accounting from the
   calibration study.

No raw prompt text is read, printed, hashed, or saved. Reports are aggregate
only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 95% two-sided normal quantile used for every Wilson score interval.
Z_95 = 1.959963984540054

# Instability rule fixed before looking at any group result. A group rate is
# flagged unstable when its denominator is below this count or when its Wilson
# interval is wider than this width.
MIN_STABLE_DENOMINATOR = 30
MAX_STABLE_INTERVAL_WIDTH = 0.25

# Offline-only dataset metadata. None of these fields is part of the deployed
# inference request, which is established by inspecting the API schema rather
# than by assuming it.
OFFLINE_METADATA_FIELDS = ("category", "source", "severity", "augmented")

# Coverage floor for the illustrative gateway band. Chosen a priori as a
# documentation constraint, not tuned to maximise any reported number.
ILLUSTRATIVE_MIN_COVERAGE = 0.95

# Sources inspected for the unimplementable review-escalation ceiling.
REVIEW_CEILING_SOURCES = ("wildguard_judgecomp", "harmbench_benign", "original")


def wilson_interval(
    successes: int,
    trials: int,
    z: float = Z_95,
) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion.

    Returns ``(lower, upper)`` clamped to ``[0, 1]``, or ``(None, None)`` when
    ``trials`` is zero. Wilson is used rather than the normal approximation
    because it stays well behaved at small denominators and at proportions of
    exactly zero or one, which is common in these metadata groups.
    """
    if trials < 0 or successes < 0:
        raise ValueError("successes and trials must be non-negative")
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    if trials == 0:
        return (None, None)

    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def rate_summary(
    numerator: int,
    denominator: int,
    label: str,
) -> dict[str, Any]:
    """A rate with its Wilson interval and a documented stability flag."""
    lower, upper = wilson_interval(numerator, denominator)
    value = numerator / denominator if denominator else None
    width = (upper - lower) if lower is not None else None
    stable = (
        denominator >= MIN_STABLE_DENOMINATOR
        and width is not None
        and width <= MAX_STABLE_INTERVAL_WIDTH
    )
    reasons: list[str] = []
    if denominator < MIN_STABLE_DENOMINATOR:
        reasons.append(
            f"denominator {denominator} < {MIN_STABLE_DENOMINATOR}"
        )
    if width is not None and width > MAX_STABLE_INTERVAL_WIDTH:
        reasons.append(
            f"interval width {width:.3f} > {MAX_STABLE_INTERVAL_WIDTH}"
        )
    return {
        "label": label,
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "wilson_95_width": width,
        "stable": stable,
        "unstable_reasons": reasons,
    }


def _stratify_field(
    breakdown: dict[str, Any],
    field_name: str,
) -> dict[str, Any]:
    """Convert one metadata field's confusion breakdown into rated entries."""
    entries: dict[str, Any] = {}
    for value, details in sorted(breakdown.items()):
        counts = details.get("counts", {})
        false_positive = counts.get("false_positive", 0)
        true_negative = counts.get("true_negative", 0)
        false_negative = counts.get("false_negative", 0)
        true_positive = counts.get("true_positive", 0)
        benign_denominator = false_positive + true_negative
        malicious_denominator = false_negative + true_positive
        entries[value] = {
            "sample_count": details.get("sample_count", 0),
            "actual_label_counts": details.get("actual_label_counts", {}),
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive_rate": rate_summary(
                false_positive, benign_denominator, f"{field_name}:{value}"
            ),
            "false_negative_rate": rate_summary(
                false_negative, malicious_denominator, f"{field_name}:{value}"
            ),
        }
    return entries


def stratified_rates(error_analysis: dict[str, Any]) -> dict[str, Any]:
    """Per-source and per-category FP/FN rates with Wilson intervals."""
    breakdown = error_analysis.get("metadata_breakdown", {})
    return {
        "source": _stratify_field(breakdown.get("source", {}), "source"),
        "category": _stratify_field(breakdown.get("category", {}), "category"),
    }


def instability_findings(stratified: dict[str, Any]) -> dict[str, Any]:
    """Split groups into stable vs unstable so rates are not over-read."""
    stable: list[str] = []
    unstable: list[dict[str, Any]] = []
    for field, entries in stratified.items():
        for value, entry in entries.items():
            for rate_key in ("false_positive_rate", "false_negative_rate"):
                rate = entry[rate_key]
                if rate["denominator"] == 0:
                    continue
                qualified = f"{field}:{value}:{rate_key}"
                if rate["stable"]:
                    stable.append(qualified)
                else:
                    unstable.append({
                        "group": qualified,
                        "numerator": rate["numerator"],
                        "denominator": rate["denominator"],
                        "value": rate["value"],
                        "wilson_95_lower": rate["wilson_95_lower"],
                        "wilson_95_upper": rate["wilson_95_upper"],
                        "reasons": rate["unstable_reasons"],
                    })
    return {
        "rule": (
            f"flagged unstable when denominator < {MIN_STABLE_DENOMINATOR} "
            f"or Wilson 95% interval width > {MAX_STABLE_INTERVAL_WIDTH}"
        ),
        "stable_count": len(stable),
        "unstable": unstable,
        "stable_groups": stable,
    }


def metadata_label_consistency(
    investigation: dict[str, Any],
) -> dict[str, Any]:
    """Surface mixed-label metadata without treating metadata as truth."""
    raw = investigation.get("metadata_label_consistency", {})
    source_raw = raw.get("source", {})
    mixed = source_raw.get("potential_mixed_label_values", [])
    label_counts = source_raw.get("label_counts_by_value", {})
    return {
        "interpretation": (
            "Metadata is descriptive and is NOT treated as ground truth. "
            "Mixed-label sources are surfaced for review only; no example is "
            "relabeled by this audit."
        ),
        "mixed_label_source_values": list(mixed),
        "label_counts_by_source": label_counts,
        "hypotheses_not_resolved": [
            "High-confidence false positives may reflect label noise in the "
            "source (the label is wrong and the model is right).",
            "The same pattern may reflect a genuine model bias toward "
            "over-flagging that source's benign style (the model is wrong).",
            "These two hypotheses are not distinguishable from aggregate "
            "counts alone; resolving either would require human review of the "
            "affected rows and would still not be an automatic relabel.",
        ],
    }


def signal_availability_audit() -> dict[str, Any]:
    """Establish which dataset fields exist at inference time, from the code.

    The deployed request schema is read from the live FastAPI model rather
    than asserted, so the offline-only classification is grounded in the
    implementation that clients actually reach.
    """
    request_fields: list[str]
    try:
        from promptshield.api import PromptRequest

        request_fields = sorted(PromptRequest.model_fields.keys())
    except Exception:  # pragma: no cover - only when fastapi is absent
        request_fields = ["prompt"]

    available = set(request_fields)
    offline_only = {}
    for field in OFFLINE_METADATA_FIELDS:
        offline_only[field] = {
            "available_at_inference": field in available,
            "classification": (
                "inference-time signal"
                if field in available
                else "offline diagnostic only"
            ),
        }
    return {
        "deployed_request_fields": request_fields,
        "inference_available_signals": sorted(available),
        "offline_diagnostic_fields": offline_only,
        "text_available_at_inference": "prompt" in available,
        "conclusion": (
            "Source and category are offline dataset diagnostics. They are not "
            "present on the deployed inference request, so no source-based "
            "allowlist, blocklist, or routing rule can be enforced at "
            "inference time without changing the request contract and "
            "establishing a trustworthy client-supplied source value."
        ),
    }


def _policy_behaviours(
    policies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve every abstention band under fail-open / fail-closed / review.

    For a band that abstains on ``abstained_benign`` benign and
    ``abstained_malicious`` malicious rows:

    - fail-open allows all abstained rows, so malicious rows in the band become
      false negatives and benign rows in the band stay correctly allowed;
    - fail-closed blocks all abstained rows, so benign rows in the band become
      false positives and malicious rows in the band stay correctly blocked;
    - perfect human review resolves every abstained row correctly and only
      spends review capacity, which is an optimistic upper bound.
    """
    behaviours: list[dict[str, Any]] = []
    for policy in policies:
        accepted_fp = policy.get("accepted_false_positive", 0)
        accepted_fn = policy.get("accepted_false_negative", 0)
        abstained_benign = policy.get("abstained_benign", 0)
        abstained_malicious = policy.get("abstained_malicious", 0)
        behaviours.append({
            "policy_id": policy.get("policy_id"),
            "coverage": policy.get("coverage"),
            "abstention_rate": policy.get("abstention_rate"),
            "abstained_benign": abstained_benign,
            "abstained_malicious": abstained_malicious,
            "fail_open": {
                "false_positives": accepted_fp,
                "false_negatives": accepted_fn + abstained_malicious,
                "benign_blocked": accepted_fp,
                "malicious_allowed": accepted_fn + abstained_malicious,
            },
            "fail_closed": {
                "false_positives": accepted_fp + abstained_benign,
                "false_negatives": accepted_fn,
                "benign_blocked": accepted_fp + abstained_benign,
                "malicious_allowed": accepted_fn,
            },
            "perfect_human_review": {
                "false_positives": accepted_fp,
                "false_negatives": accepted_fn,
                "rows_sent_to_review": policy.get("abstained_count", 0),
                "note": (
                    "Optimistic upper bound assuming every reviewed row is "
                    "resolved correctly; real reviewers are fallible."
                ),
            },
        })
    return behaviours


def _choose_illustrative_policy(
    policies: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick an illustrative band by a documented rule, not by declaring it
    optimal.

    Rule fixed in advance: among non-baseline policies whose coverage is at
    least ``ILLUSTRATIVE_MIN_COVERAGE``, choose the one catching the largest
    fraction of all errors; break ties by the smaller abstention rate. This is
    an illustration of the trade-off surface, not a tuned or recommended
    threshold.
    """
    eligible = [
        policy
        for policy in policies
        if policy.get("policy_id") != "no_abstention"
        and (policy.get("coverage") or 0.0) >= ILLUSTRATIVE_MIN_COVERAGE
        and policy.get("fraction_of_all_errors_caught") is not None
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda policy: (
            -policy["fraction_of_all_errors_caught"],
            policy.get("abstention_rate") or 1.0,
        ),
    )


def review_escalation_ceiling(
    stratified: dict[str, Any],
    error_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Offline ceiling for routing one source's rows entirely to review.

    This is a diagnostic only. It quantifies how much error a source-conditional
    rule could remove IF the source were known at inference time. The signal
    availability audit shows it is not, so this is motivation for future work
    and is deliberately not implemented.
    """
    source_entries = stratified.get("source", {})
    overall = error_analysis.get("counts", {})
    total_rows = error_analysis.get("validation_rows", 0)
    scenarios: list[dict[str, Any]] = []
    for source in REVIEW_CEILING_SOURCES:
        entry = source_entries.get(source)
        if not entry:
            continue
        fp = entry["false_positive"]
        fn = entry["false_negative"]
        rows = entry["sample_count"]
        scenarios.append({
            "source": source,
            "rows_sent_to_review": rows,
            "fraction_of_validation_set": (
                rows / total_rows if total_rows else None
            ),
            "false_positives_moved_to_review": fp,
            "false_negatives_moved_to_review": fn,
            "implementable_at_inference": False,
            "why_not": (
                "The deployed request carries only the prompt text; source is "
                "not present, so this escalation cannot be enforced today."
            ),
        })
    scenarios.sort(
        key=lambda scenario: -scenario["false_positives_moved_to_review"]
    )
    return {
        "status": (
            "offline diagnostic ceiling only; NOT implemented and NOT "
            "implementable with the current request contract"
        ),
        "overall_false_positives": overall.get("false_positive"),
        "overall_false_negatives": overall.get("false_negative"),
        "scenarios": scenarios,
    }


def gateway_policy_note(
    calibration: dict[str, Any],
    error_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Compare fail-open, fail-closed, and human-review behaviour."""
    policies = (
        calibration.get("selective_prediction", {}).get("policies", [])
    )
    behaviours = _policy_behaviours(policies)
    baseline = next(
        (
            behaviour
            for behaviour in behaviours
            if behaviour["policy_id"] == "no_abstention"
        ),
        None,
    )
    illustrative = _choose_illustrative_policy(policies)
    illustrative_behaviour = next(
        (
            behaviour
            for behaviour in behaviours
            if illustrative
            and behaviour["policy_id"] == illustrative["policy_id"]
        ),
        None,
    )
    return {
        "deployed_decision_rule": {
            "threshold": error_analysis.get("model_configuration", {}).get(
                "threshold"
            ),
            "modified": False,
            "note": (
                "This audit changes no decision rule and implements no "
                "allowlist. It only characterises what each gateway behaviour "
                "would do with the same measured outcomes."
            ),
        },
        "baseline_no_abstention": baseline,
        "behaviours": behaviours,
        "illustrative_band": {
            "selection_rule": (
                "among non-baseline policies with coverage >= "
                f"{ILLUSTRATIVE_MIN_COVERAGE}, the one catching the largest "
                "fraction of all errors, ties broken by smaller abstention "
                "rate; chosen for illustration on validation data only"
            ),
            "optimality_claimed": False,
            "policy_id": (
                illustrative["policy_id"] if illustrative else None
            ),
            "behaviour": illustrative_behaviour,
        },
        "interpretation": [
            "Fail-open minimises false blocking of benign prompts but lets "
            "every abstained malicious prompt through, raising false negatives.",
            "Fail-closed minimises missed malicious prompts but blocks every "
            "abstained benign prompt, raising false positives.",
            "Human review sits between the two and costs review capacity; the "
            "review figures here assume a perfect reviewer and are an "
            "optimistic upper bound.",
            "The choice between these behaviours is a product risk decision "
            "about relative cost, not something validation data can decide.",
        ],
    }


def build_report(
    error_analysis: dict[str, Any],
    investigation: dict[str, Any],
    calibration: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    """Assemble the aggregate-only source reliability and policy report."""
    stratified = stratified_rates(error_analysis)
    return {
        "experiment": "deberta_source_reliability_and_gateway_policy",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "completed",
        "validation_only": True,
        "input_reports": inputs,
        "reuses_existing_predictions": True,
        "new_inference_run": False,
        "validation_rows": error_analysis.get("validation_rows"),
        "label_distribution": error_analysis.get("label_distribution"),
        "threshold": error_analysis.get("model_configuration", {}).get(
            "threshold"
        ),
        "confusion_counts": error_analysis.get("counts"),
        "confidence_interval_method": {
            "interval": "Wilson score, 95%",
            "z": Z_95,
            "why": (
                "Wilson remains well behaved at small denominators and at "
                "proportions of exactly zero or one, unlike the normal "
                "approximation."
            ),
        },
        "stratified_rates": stratified,
        "instability_findings": instability_findings(stratified),
        "metadata_label_consistency": metadata_label_consistency(
            investigation
        ),
        "signal_availability": signal_availability_audit(),
        "gateway_policy_design": gateway_policy_note(
            calibration, error_analysis
        ),
        "review_escalation_ceiling": review_escalation_ceiling(
            stratified, error_analysis
        ),
        "test_split_loaded": False,
        "prompt_text_included": False,
        "limitations": [
            "Every figure reuses validation-only aggregates from earlier "
            "reports and does not establish held-out test performance.",
            "Wilson intervals quantify sampling uncertainty within a group; "
            "they do not correct for selection or labeling bias in the "
            "source itself.",
            "Metadata is descriptive and is never treated as ground truth; no "
            "example is relabeled by this audit.",
            "The source-based review escalation is an offline ceiling only and "
            "is not implementable because source is absent from the deployed "
            "request.",
            "Human-review figures assume a perfect reviewer and are an "
            "optimistic bound; real review is fallible and has a cost.",
            "The deployed 0.90 threshold was not tuned or changed and no model "
            "was retrained.",
        ],
    }


def _format_rate(rate: dict[str, Any]) -> str:
    if rate["value"] is None:
        return "n/a"
    lower = rate["wilson_95_lower"]
    upper = rate["wilson_95_upper"]
    if lower is None:
        return f"{rate['value']:.4f} (n/a)"
    flag = "" if rate["stable"] else " !unstable"
    return (
        f"{rate['value']:.4f} [{lower:.4f}, {upper:.4f}]{flag}"
    )


def render_markdown(report: dict[str, Any]) -> str:
    counts = report.get("confusion_counts") or {}
    signal = report["signal_availability"]
    policy = report["gateway_policy_design"]
    lines = [
        "# PromptShield source-stratified reliability audit and gateway policy note",
        "",
        "Aggregate validation figures reused from earlier reports. No new "
        "inference was run, no Parquet file was read, no model was retrained, "
        "and no raw prompt text appears here.",
        "",
        f"- Validation rows: `{report.get('validation_rows')}`",
        f"- Threshold: `{report.get('threshold')}` (unmodified)",
        f"- False positives: `{counts.get('false_positive')}`",
        f"- False negatives: `{counts.get('false_negative')}`",
        "",
        "Rates below are shown as `value [Wilson 95% lower, upper]`. A `!unstable`",
        "marker means the group is too small or too imprecise to read as a point",
        "estimate.",
        "",
    ]

    for field in ("source", "category"):
        lines.extend([
            f"## {field.capitalize()} stratified rates",
            "",
            f"| {field} | Samples | Benign denom | FP | FPR (95% CI) | "
            "Malicious denom | FN | FNR (95% CI) |",
            "|" + "---|" * 8,
        ])
        for value, entry in report["stratified_rates"][field].items():
            fpr = entry["false_positive_rate"]
            fnr = entry["false_negative_rate"]
            lines.append(
                f"| {value} | {entry['sample_count']} | "
                f"{fpr['denominator']} | {fpr['numerator']} | "
                f"{_format_rate(fpr)} | {fnr['denominator']} | "
                f"{fnr['numerator']} | {_format_rate(fnr)} |"
            )
        lines.append("")

    findings = report["instability_findings"]
    lines.extend([
        "## Rate stability",
        "",
        f"Rule: {findings['rule']}.",
        "",
        f"- Stable rate estimates: `{findings['stable_count']}`",
        f"- Unstable rate estimates: `{len(findings['unstable'])}`",
        "",
        "Unstable groups (do not quote these as point estimates):",
        "",
    ])
    for item in findings["unstable"]:
        lines.append(
            f"- `{item['group']}` = {item['numerator']}/{item['denominator']}"
            f" = {item['value']:.4f} "
            f"[{item['wilson_95_lower']:.4f}, {item['wilson_95_upper']:.4f}]"
            f" ({'; '.join(item['reasons'])})"
        )
    lines.append("")

    consistency = report["metadata_label_consistency"]
    lines.extend([
        "## Metadata and label consistency",
        "",
        consistency["interpretation"],
        "",
        "Mixed-label sources (surfaced for review, never relabeled):",
        "",
    ])
    for source in consistency["mixed_label_source_values"]:
        counts_for_source = consistency["label_counts_by_source"].get(
            source, {}
        )
        lines.append(f"- `{source}`: {counts_for_source}")
    lines.extend(["", "Open hypotheses this audit does not resolve:", ""])
    lines.extend(
        f"- {hypothesis}" for hypothesis in consistency["hypotheses_not_resolved"]
    )
    lines.append("")

    lines.extend([
        "## Signal availability at inference time",
        "",
        f"Deployed request fields (read from the API schema): "
        f"`{signal['deployed_request_fields']}`",
        "",
        "| Dataset field | Classification |",
        "|---|---|",
    ])
    for field, detail in signal["offline_diagnostic_fields"].items():
        lines.append(f"| `{field}` | {detail['classification']} |")
    lines.extend(["", signal["conclusion"], ""])

    lines.extend([
        "## Gateway policy design note",
        "",
        policy["deployed_decision_rule"]["note"],
        "",
        "Behaviour of each gateway policy on the same measured validation "
        "outcomes (fail-open allows every abstained row, fail-closed blocks "
        "every abstained row, human review resolves them):",
        "",
        "| Policy | Coverage | Fail-open FP | Fail-open FN | "
        "Fail-closed FP | Fail-closed FN | Review rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for behaviour in policy["behaviours"]:
        review_rows = behaviour["perfect_human_review"][
            "rows_sent_to_review"
        ]
        lines.append(
            f"| {behaviour['policy_id']} | {behaviour['coverage']:.4f} | "
            f"{behaviour['fail_open']['false_positives']} | "
            f"{behaviour['fail_open']['false_negatives']} | "
            f"{behaviour['fail_closed']['false_positives']} | "
            f"{behaviour['fail_closed']['false_negatives']} | "
            f"{review_rows} |"
        )
    lines.extend(["", "Interpretation:", ""])
    lines.extend(f"- {note}" for note in policy["interpretation"])
    lines.extend([
        "",
        "The illustrative band is selected by a documented rule for "
        "illustration only and is not declared optimal:",
        "",
        f"- Selection rule: {policy['illustrative_band']['selection_rule']}",
        f"- Illustrative policy: "
        f"`{policy['illustrative_band']['policy_id'] or 'none eligible'}`",
        f"- Optimality claimed: "
        f"`{str(policy['illustrative_band']['optimality_claimed']).lower()}`",
        "",
    ])

    ceiling = report["review_escalation_ceiling"]
    lines.extend([
        "## Source-conditional review ceiling (offline diagnostic only)",
        "",
        f"Status: {ceiling['status']}",
        "",
        "| Source | Rows to review | Fraction of set | FP moved | FN moved |",
        "|---|---:|---:|---:|---:|",
    ])
    for scenario in ceiling["scenarios"]:
        fraction = scenario["fraction_of_validation_set"]
        lines.append(
            f"| {scenario['source']} | {scenario['rows_sent_to_review']} | "
            f"{fraction:.4f} | {scenario['false_positives_moved_to_review']} | "
            f"{scenario['false_negatives_moved_to_review']} |"
        )
    lines.extend([
        "",
        "These rows cannot be identified at inference time, so the ceiling is "
        "motivation for a future trustworthy source signal, not a rule that is "
        "implemented today.",
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_reports(
    report: dict[str, Any],
    reports_dir: Path,
) -> tuple[Path, Path]:
    """Write uniquely timestamped aggregate-only reports, never overwriting."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"source-reliability-policy-{stamp}.json"
    markdown_path = reports_dir / f"source-reliability-policy-{stamp}.md"
    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write(render_markdown(report))
    return json_path, markdown_path


def load_report(path: Path) -> dict[str, Any]:
    """Read one existing aggregate report."""
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error-analysis", type=Path, required=True)
    parser.add_argument("--investigation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    try:
        error_analysis = load_report(args.error_analysis)
        investigation = load_report(args.investigation)
        calibration = load_report(args.calibration)
        report = build_report(
            error_analysis,
            investigation,
            calibration,
            inputs={
                "error_analysis": args.error_analysis.as_posix(),
                "investigation": args.investigation.as_posix(),
                "calibration": args.calibration.as_posix(),
            },
        )
        json_path, markdown_path = write_reports(report, args.reports_dir)
        print(json.dumps({
            "json_report": json_path.as_posix(),
            "markdown_report": markdown_path.as_posix(),
            "validation_rows": report.get("validation_rows"),
            "unstable_rates": len(
                report["instability_findings"]["unstable"]
            ),
            "test_split_loaded": report["test_split_loaded"],
        }, indent=2))
        return 0
    except Exception as error:
        print(
            f"source reliability audit failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
