"""Validation-only confidence calibration and selective-prediction study.

Characterizes DeBERTa class-1 softmax scores as scores (calibration is not
established), computes reliability/ECE/Brier summaries under an explicitly
defined equal-width binning scheme, and evaluates pre-registered abstention
bands for offline selective prediction. No retraining, no threshold change,
and no raw prompt text in any output.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from promptshield.baseline import load_labeled_parquet
from promptshield.deberta_decision_audit import (
    APPROVED_VALIDATION_PATH,
    DEFAULT_BATCH_SIZE,
    _dependency_version,
    _model_configuration,
    hash_checkpoint_files,
)
from promptshield.deberta_error_investigation import (
    _class_1_probabilities,
    _confidence_summary,
)
from promptshield.inference import DEFAULT_THRESHOLD, DebertaInference

DEFAULT_BIN_COUNT = 10
BASELINE_POLICY_ID = "no_abstention"
ERROR_REDUCTION_FRACTION = 0.5
CONFUSION_GROUPS = (
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
)
# The abstention grid is fixed in source and was not adapted to any results.
BENIGN_ACCEPT_MAX_CANDIDATES = ("decision_threshold", 0.80, 0.70, 0.50)
MALICIOUS_ACCEPT_MIN_CANDIDATES = ("decision_threshold", 0.95, 0.99)


def validate_scores(labels: list[int], scores: list[float]) -> None:
    """Reject malformed labels or scores before any metric calculation."""
    if not labels:
        raise ValueError("validation split contains no rows")
    if any(
        isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1)
        for label in labels
    ):
        raise ValueError("labels must be integers in {0, 1}")
    if len(scores) != len(labels):
        raise ValueError(
            f"received {len(scores)} scores; expected {len(labels)}"
        )
    for position, score in enumerate(scores, start=1):
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"score {position} is not a finite number")
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"score {position} is outside [0, 1]")


def reliability_bins(
    labels: list[int],
    scores: list[float],
    bin_count: int = DEFAULT_BIN_COUNT,
) -> list[dict[str, Any]]:
    """Equal-width bins on [0, 1]; final bin is closed on the right.

    Bin ``i`` covers ``[i / bin_count, (i + 1) / bin_count)``, except the last
    bin which covers ``[(bin_count - 1) / bin_count, 1.0]``. A score exactly
    on an interior boundary goes to the upper bin; a score of exactly 1.0
    goes to the final bin.
    """
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")
    validate_scores(labels, scores)
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bin_count)]
    for label, score in zip(labels, scores):
        index = min(int(float(score) * bin_count), bin_count - 1)
        buckets[index].append((float(score), label))

    bins: list[dict[str, Any]] = []
    for index, members in enumerate(buckets):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        closing = "]" if index == bin_count - 1 else ")"
        base = {
            "index": index,
            "lower": lower,
            "upper": upper,
            "interval": f"[{lower:.2f}, {upper:.2f}{closing}",
        }
        if not members:
            bins.append({
                **base,
                "count": 0,
                "fraction": 0.0,
                "mean_score": None,
                "empirical_positive_rate": None,
                "signed_gap": None,
                "absolute_gap": None,
            })
            continue
        count = len(members)
        mean_score = sum(score for score, _ in members) / count
        positive_rate = sum(label for _, label in members) / count
        signed_gap = mean_score - positive_rate
        bins.append({
            **base,
            "count": count,
            "fraction": count / len(labels),
            "mean_score": mean_score,
            "empirical_positive_rate": positive_rate,
            "signed_gap": signed_gap,
            "absolute_gap": abs(signed_gap),
        })
    return bins


def expected_calibration_error(
    bins: list[dict[str, Any]],
    total: int,
) -> float:
    """Weighted mean of per-bin absolute gaps: sum (n_b / N) * |gap_b|."""
    if total <= 0:
        raise ValueError("total must be positive")
    return sum(
        (entry["count"] / total) * entry["absolute_gap"]
        for entry in bins
        if entry["count"]
    )


def brier_score(labels: list[int], scores: list[float]) -> float:
    """Mean squared difference between score and binary label."""
    validate_scores(labels, scores)
    return sum(
        (float(score) - label) ** 2 for label, score in zip(labels, scores)
    ) / len(labels)


def _group_statistics(
    indexes: list[int],
    labels: list[int],
    scores: list[float],
) -> dict[str, Any]:
    return _confidence_summary([scores[index] for index in indexes]) | {
        "sample_count": len(indexes)
    }


def score_distributions(
    labels: list[int],
    scores: list[float],
    predictions: list[int],
) -> dict[str, Any]:
    """Confidence summaries overall, by true label, and by confusion group."""
    validate_scores(labels, scores)
    if len(predictions) != len(labels):
        raise ValueError("predictions must align with labels")
    groups: dict[str, list[int]] = {name: [] for name in CONFUSION_GROUPS}
    for index, (label, prediction) in enumerate(zip(labels, predictions)):
        if label == 1 and prediction == 1:
            groups["true_positive"].append(index)
        elif label == 0 and prediction == 0:
            groups["true_negative"].append(index)
        elif label == 0 and prediction == 1:
            groups["false_positive"].append(index)
        else:
            groups["false_negative"].append(index)
    return {
        "overall": _confidence_summary(scores),
        "by_true_label": {
            "benign": _confidence_summary(
                [score for label, score in zip(labels, scores) if label == 0]
            ),
            "malicious": _confidence_summary(
                [score for label, score in zip(labels, scores) if label == 1]
            ),
        },
        "by_confusion_group": {
            name: _group_statistics(indexes, labels, scores)
            for name, indexes in groups.items()
        },
    }


def evaluate_abstention(
    labels: list[int],
    scores: list[float],
    decision_threshold: float = DEFAULT_THRESHOLD,
    benign_accept_max: float = DEFAULT_THRESHOLD,
    malicious_accept_min: float = DEFAULT_THRESHOLD,
    policy_id: str | None = None,
) -> dict[str, Any]:
    """Account one abstention policy on validation data.

    A prediction is accepted as benign when ``score < benign_accept_max`` and
    accepted as malicious when ``score >= malicious_accept_min``; every other
    score abstains (routed to human review). The deployed decision threshold
    is not modified: accepted decisions follow the deployed rule by
    construction because ``benign_accept_max <= threshold <=
    malicious_accept_min``.
    """
    if not 0 < decision_threshold < 1:
        raise ValueError("decision_threshold must be between 0 and 1")
    if not 0.0 <= benign_accept_max <= decision_threshold:
        raise ValueError(
            "benign_accept_max must be in [0, decision_threshold]"
        )
    if not decision_threshold <= malicious_accept_min <= 1.0:
        raise ValueError(
            "malicious_accept_min must be in [decision_threshold, 1]"
        )
    validate_scores(labels, scores)

    true_positive = false_positive = 0
    true_negative = false_negative = 0
    abstained = abstained_benign = abstained_malicious = 0
    abstained_deployed_errors = 0
    for label, score in zip(labels, scores):
        score = float(score)
        if score < benign_accept_max:
            if label == 0:
                true_negative += 1
            else:
                false_negative += 1
        elif score >= malicious_accept_min:
            if label == 1:
                true_positive += 1
            else:
                false_positive += 1
        else:
            abstained += 1
            if label == 0:
                abstained_benign += 1
            else:
                abstained_malicious += 1
            # An abstained case counts as an error caught only when the
            # unmodified deployed rule (score vs decision_threshold) would
            # have misclassified it.
            deployed_prediction = 1 if score >= decision_threshold else 0
            if deployed_prediction != label:
                abstained_deployed_errors += 1

    accepted = true_positive + false_positive + true_negative + false_negative
    total = len(labels)
    accepted_errors = false_positive + false_negative
    if policy_id is None:
        policy_id = (
            BASELINE_POLICY_ID
            if (benign_accept_max == decision_threshold
                and malicious_accept_min == decision_threshold)
            else f"b{benign_accept_max:.2f}_m{malicious_accept_min:.2f}"
        )
    return {
        "policy_id": policy_id,
        "benign_accept_max": benign_accept_max,
        "malicious_accept_min": malicious_accept_min,
        "accepted_true_positive": true_positive,
        "accepted_true_negative": true_negative,
        "accepted_false_positive": false_positive,
        "accepted_false_negative": false_negative,
        "accepted_count": accepted,
        "abstained_count": abstained,
        "abstained_benign": abstained_benign,
        "abstained_malicious": abstained_malicious,
        "coverage": accepted / total,
        "abstention_rate": abstained / total,
        "accepted_case_error_rate": (
            accepted_errors / accepted if accepted else None
        ),
        "abstained_deployed_error_rate": (
            abstained_deployed_errors / abstained if abstained else None
        ),
        "total_errors_not_reviewed": accepted_errors,
        "fraction_of_all_errors_caught": (
            abstained_deployed_errors / (accepted_errors + abstained_deployed_errors)
            if (accepted_errors + abstained_deployed_errors)
            else None
        ),
    }


def evaluate_policy_grid(
    labels: list[int],
    scores: list[float],
    decision_threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Evaluate the pre-registered grid; baseline (no abstention) first."""
    policies: list[dict[str, Any]] = []
    for benign_value in BENIGN_ACCEPT_MAX_CANDIDATES:
        for malicious_value in MALICIOUS_ACCEPT_MIN_CANDIDATES:
            benign_max = (
                decision_threshold
                if benign_value == "decision_threshold"
                else float(benign_value)
            )
            malicious_min = (
                decision_threshold
                if malicious_value == "decision_threshold"
                else float(malicious_value)
            )
            policies.append(evaluate_abstention(
                labels,
                scores,
                decision_threshold=decision_threshold,
                benign_accept_max=benign_max,
                malicious_accept_min=malicious_min,
            ))
    return policies


def _error_key(policy: dict[str, Any]) -> float:
    value = policy["accepted_case_error_rate"]
    return float("inf") if value is None else value


def pareto_frontier(policies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep policies not dominated on (coverage up, accepted error down)."""
    frontier: list[dict[str, Any]] = []
    for candidate in policies:
        dominated = any(
            other["coverage"] >= candidate["coverage"]
            and _error_key(other) <= _error_key(candidate)
            and (
                other["coverage"] > candidate["coverage"]
                or _error_key(other) < _error_key(candidate)
            )
            for other in policies
            if other is not candidate
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def smallest_abstention_for_error_reduction(
    policies: list[dict[str, Any]],
    baseline: dict[str, Any],
    relative_reduction: float = ERROR_REDUCTION_FRACTION,
) -> dict[str, Any] | None:
    """Illustrative rule, pre-registered: smallest abstention rate achieving
    at least ``relative_reduction`` relative reduction of the baseline
    accepted-case error rate. Returns None when the rule selects nothing."""
    baseline_error = baseline["accepted_case_error_rate"]
    if baseline_error is None or baseline_error == 0:
        return None
    target = baseline_error * (1.0 - relative_reduction)
    eligible = [
        policy
        for policy in policies
        if policy["policy_id"] != baseline["policy_id"]
        and policy["accepted_case_error_rate"] is not None
        and policy["accepted_case_error_rate"] <= target
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda policy: (
            policy["abstention_rate"],
            -policy["coverage"],
            policy["policy_id"],
        ),
    )


def build_report(
    labels: list[int],
    scores: list[float],
    predictions: list[int],
    decision_threshold: float = DEFAULT_THRESHOLD,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> dict[str, Any]:
    """Assemble the aggregate-only calibration and selective-prediction report."""
    validate_scores(labels, scores)
    if len(predictions) != len(labels):
        raise ValueError("predictions must align with labels")
    bins = reliability_bins(labels, scores, bin_count=bin_count)
    policies = evaluate_policy_grid(labels, scores, decision_threshold)
    baseline = policies[0]
    frontier = pareto_frontier(policies)
    reference = smallest_abstention_for_error_reduction(policies, baseline)
    return {
        "validation_rows": len(labels),
        "validation_only": True,
        "score_definition": (
            "Class-1 softmax output of the saved DeBERTa checkpoint under the "
            "deployed preprocessing. Treated as an uncalibrated score; this "
            "study characterizes reliability and does not establish or "
            "improve calibration."
        ),
        "reliability": {
            "binning_scheme": (
                f"{bin_count} equal-width bins on [0, 1]; bin i covers "
                "[i/bin_count, (i+1)/bin_count) and the final bin is closed "
                "on the right; scores on interior boundaries go to the upper "
                "bin"
            ),
            "bin_count": bin_count,
            "bins": bins,
            "expected_calibration_error": expected_calibration_error(
                bins, len(labels)
            ),
            "brier_score": brier_score(labels, scores),
            "ece_definition": (
                "sum over non-empty bins of (bin count / total) * |mean "
                "score - empirical positive rate|"
            ),
            "brier_definition": (
                "mean over examples of (score - binary label)^2"
            ),
        },
        "score_distributions": score_distributions(labels, scores, predictions),
        "selective_prediction": {
            "decision_threshold": decision_threshold,
            "decision_threshold_modified": False,
            "policy_definition": (
                "accept as benign when score < benign_accept_max; accept as "
                "malicious when score >= malicious_accept_min; abstain "
                "(send to human review) otherwise"
            ),
            "grid_status": (
                "fixed in source code before this run; not adapted to "
                "observed results"
            ),
            "policies": policies,
            "baseline_policy_id": BASELINE_POLICY_ID,
            "pareto_frontier_policy_ids": [
                policy["policy_id"] for policy in frontier
            ],
            "illustrative_reference_rule": (
                "smallest abstention rate achieving at least a 50% relative "
                "reduction in accepted-case error rate versus the "
                "no-abstention baseline"
            ),
            "illustrative_reference_policy_id": (
                reference["policy_id"] if reference else None
            ),
            "optimality_claimed": False,
            "selection_notes": [
                "No candidate is declared optimal; deployment choice "
                "requires review-capacity input and independent confirmation.",
                "All candidates were scored on the same validation data used "
                "to describe them, so any selection is optimistically biased.",
            ],
        },
        "limitations": [
            "This study uses validation data only and does not establish held-out test performance.",
            "Class-1 softmax outputs are treated as scores; reliability summaries do not make them calibrated probabilities.",
            "ECE depends on the binning scheme; equal-width binning leaves sparse high-score bins whose rates are unstable at this sample size.",
            "Candidate policies are described on the same validation data used to score them, so any selection is optimistically biased.",
            "Abstention moves cases to human review; review capacity and cost are not modeled here.",
            "The deployed decision threshold was not tuned or changed; no model was retrained.",
        ],
        "test_split_loaded": False,
        "prompt_text_included": False,
    }


def run_study(
    validation_path: Path,
    checkpoint_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> dict[str, Any]:
    """Run the study against validation.parquet and a local checkpoint only."""
    resolved = validation_path.resolve(strict=False)
    if resolved != APPROVED_VALIDATION_PATH:
        raise ValueError(
            "study accepts only the repository-approved "
            "data/raw/validation.parquet path"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")
    texts, labels = load_labeled_parquet(validation_path)
    service = DebertaInference(
        checkpoint_dir, threshold=threshold, device="auto"
    )
    service.load()
    logits = service.predict_logits_batch(
        texts, batch_size=batch_size, enforce_prompt_limit=False
    )
    if len(logits) != len(labels):
        raise ValueError(
            f"model returned {len(logits)} logit rows; expected {len(labels)}"
        )
    scores = _class_1_probabilities(logits)
    predictions = [1 if score >= threshold else 0 for score in scores]
    study = build_report(labels, scores, predictions, threshold, bin_count)
    return {
        "experiment": "deberta_validation_confidence_calibration",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "completed",
        "validation_only": True,
        "validation_path": validation_path.as_posix(),
        "label_distribution": {
            "0": labels.count(0),
            "1": labels.count(1),
        },
        "model_configuration": _model_configuration(service),
        "checkpoint": {
            "directory": checkpoint_dir.as_posix(),
            "files": hash_checkpoint_files(checkpoint_dir),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": getattr(service._torch, "__version__", None),
            "transformers": _dependency_version("transformers"),
            "device": service.device,
            "batch_size": batch_size,
        },
        **study,
        "test_split_loaded": False,
        "prompt_text_included": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    reliability = report["reliability"]
    selective = report["selective_prediction"]
    distributions = report["score_distributions"]
    lines = [
        "# PromptShield DeBERTa validation calibration and selective prediction",
        "",
        "Aggregate validation metrics only; no prompt text is included. "
        "Class-1 softmax outputs are treated as uncalibrated scores; this "
        "study does not establish or improve calibration.",
        "",
        f"- Validation rows: `{report['validation_rows']}`",
        f"- Decision threshold: `{selective['decision_threshold']}` "
        "(unmodified)",
        f"- Expected calibration error: "
        f"`{reliability['expected_calibration_error']:.6f}`",
        f"- Brier score: `{reliability['brier_score']:.6f}`",
        "",
        "## Reliability bins",
        "",
        f"Binning: {reliability['binning_scheme']}. Empty bins are omitted "
        "here but present in the JSON report.",
        "",
        "| Interval | Count | Fraction | Mean score | Empirical positive rate | Signed gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for entry in reliability["bins"]:
        if not entry["count"]:
            continue
        lines.append(
            f"| {entry['interval']} | {entry['count']} | "
            f"{entry['fraction']:.4f} | {entry['mean_score']:.6f} | "
            f"{entry['empirical_positive_rate']:.6f} | "
            f"{entry['signed_gap']:+.6f} |"
        )
    lines.extend([
        "",
        "## Score distributions",
        "",
        "| Group | Samples | Mean | Median | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    groups = {
        "overall": distributions["overall"],
        "true benign": distributions["by_true_label"]["benign"],
        "true malicious": distributions["by_true_label"]["malicious"],
        **distributions["by_confusion_group"],
    }
    for name, summary in groups.items():
        if not summary["sample_count"]:
            lines.append(f"| {name} | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {name} | {summary['sample_count']} | {summary['mean']:.6f} | "
            f"{summary['median']:.6f} | {summary['minimum']:.6f} | "
            f"{summary['maximum']:.6f} |"
        )
    lines.extend([
        "",
        "## Selective prediction (abstention bands)",
        "",
        "Policies: accept as benign when score < benign accept max; accept as "
        "malicious when score >= malicious accept min; abstain to human "
        "review otherwise. The grid was fixed in source before this run.",
        "",
        "| Policy | Benign accept <= | Malicious accept >= | Coverage | Abstention rate | Accepted error rate | Accepted FP | Accepted FN | Abstained benign | Abstained malicious | Errors caught |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for policy in selective["policies"]:
        error_rate = policy["accepted_case_error_rate"]
        caught = policy["fraction_of_all_errors_caught"]
        error_text = f"{error_rate:.6f}" if error_rate is not None else "n/a"
        caught_text = f"{caught:.4f}" if caught is not None else "n/a"
        lines.append(
            f"| {policy['policy_id']} | {policy['benign_accept_max']:.2f} | "
            f"{policy['malicious_accept_min']:.2f} | "
            f"{policy['coverage']:.4f} | {policy['abstention_rate']:.4f} | "
            f"{error_text} | "
            f"{policy['accepted_false_positive']} | "
            f"{policy['accepted_false_negative']} | "
            f"{policy['abstained_benign']} | "
            f"{policy['abstained_malicious']} | "
            f"{caught_text} |"
        )
    lines.extend([
        "",
        f"- Pareto frontier (coverage vs accepted error rate): "
        f"`{selective['pareto_frontier_policy_ids']}`",
        f"- Illustrative reference rule: {selective['illustrative_reference_rule']}",
        f"- Illustrative reference policy: "
        f"`{selective['illustrative_reference_policy_id'] or 'none'}`",
        "- No candidate is declared optimal. All candidates were scored on "
        "the same validation data that describes them, so any selection is "
        "optimistically biased and requires independent confirmation.",
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
    """Write uniquely timestamped reports without overwriting existing files."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = reports_dir / f"deberta-confidence-calibration-{stamp}.json"
    markdown_path = (
        reports_dir / f"deberta-confidence-calibration-{stamp}.md"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    for path, contents in (
        (json_path, json.dumps(report, indent=2) + "\n"),
        (markdown_path, render_markdown(report)),
    ):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(contents)
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--bins", type=int, default=DEFAULT_BIN_COUNT)
    args = parser.parse_args()
    try:
        report = run_study(
            args.validation,
            args.checkpoint,
            batch_size=args.batch_size,
            bin_count=args.bins,
        )
        json_path, markdown_path = write_reports(report, args.reports_dir)
        print(json.dumps({
            "json_report": json_path.as_posix(),
            "markdown_report": markdown_path.as_posix(),
            "validation_rows": report["validation_rows"],
            "ece": report["reliability"]["expected_calibration_error"],
            "brier": report["reliability"]["brier_score"],
            "test_split_loaded": report["test_split_loaded"],
        }, indent=2))
        return 0
    except Exception as error:
        print(
            f"calibration study failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
