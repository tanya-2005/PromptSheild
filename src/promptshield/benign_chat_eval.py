"""Benign-chat generalization evaluation: single pre-registered run.

Protocol frozen in ``docs/benign-chat-eval-proposal.md`` (2026-10-19):

- Two pinned public datasets (Databricks Dolly, OpenAssistant oasst1) scored
  with the frozen checkpoint and gateway policy, 100 eligible benign prompts
  each (n=200 total).
- OASST1 eligibility requires positive benign-quality evidence from the
  documented reviewer annotations, not merely an English tag or absent flags.
- Deduplication runs against ``train.parquet`` and ``validation.parquet``
  only; ``data/raw/test.parquet`` is never read (path guards enforce this).
- Aggregate-only outputs: console output and tracked reports never contain
  prompt text. Raw files live under gitignored ``data/raw-eval/`` and are
  never redistributed.
- The checkpoint, threshold (0.90), and gateway policy (band_v1) are used
  unchanged; the run executes exactly once.

Run once, from the repository root:

    $env:PYTHONPATH = "src"
    & .\\.venv-deberta\\Scripts\\python.exe -m promptshield.benign_chat_eval
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from promptshield.gateway import GatewayPolicy
from promptshield.inference import (
    DEFAULT_THRESHOLD,
    MAX_LENGTH,
    MAX_PROMPT_LENGTH,
    DebertaInference,
)
from promptshield.review_suggestions import matched_strong_marker_rules

# --- Frozen protocol constants (docs/benign-chat-eval-proposal.md) ---

PROTOCOL_DOCUMENT = "docs/benign-chat-eval-proposal.md"
PROTOCOL_FROZEN = "2026-10-19"
SAMPLE_PER_DATASET = 100
REFERENCE_VALIDATION_FP = 42
REFERENCE_VALIDATION_BENIGN = 407
REFERENCE_RATE = REFERENCE_VALIDATION_FP / REFERENCE_VALIDATION_BENIGN
REFERENCE_VALIDATION_WILSON = (0.0773, 0.1366)  # from the proposal

FREEZE: dict[str, dict[str, str]] = {
    "dolly": {
        "commit": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
        "license": "cc-by-sa-3.0",
    },
    "oasst1": {
        "commit": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "license": "apache-2.0",
    },
}
AUDIT_CHECKPOINT_SHA256 = (
    "2cefb8e40ade17940899ef523c7dea2a1bd1813ba0373edda84481a32c783614"
)

# --- Eligibility rules (frozen protocol, step 3) ---

DOLLY_MIN_CHARS = 4
OASST_MIN_CHARS = 10
OASST_REVIEW_MIN = 3
OASST_QUALITY_MIN = 0.5
OASST_NEGATIVE_LABELS = (
    "spam",
    "lang_mismatch",
    "pii",
    "not_appropriate",
    "hate_speech",
    "sexual_content",
    "toxicity",
    "violence",
)
OASST_DETOXIFY_LIMITS = {"toxicity": 0.2, "severe_toxicity": 0.1}

# --- Deduplication (frozen protocol, step 4) ---

NEAR_DUPLICATE_RATIO = 0.95
BUCKET_CHARS = 50

# --- Allowed paths (test-split access is structurally impossible) ---

TEST_FILE_NAME = "test.parquet"
ALLOWED_EVAL_FILES = {
    "data/raw-eval/dolly/databricks-dolly-15k.jsonl",
    "data/raw-eval/oasst1/train.parquet",
}
ALLOWED_REFERENCE_FILES = {
    "data/raw/train.parquet",
    "data/raw/validation.parquet",
}
MANIFEST_PATH = Path("data/eval-manifest.json")
MODEL_DIR = Path("models/deberta-v3-small")
CONTROL_SCRIPT = Path("scripts/fp_benign_diagnostic.py")
SCORE_BUCKETS = ((0.0, 0.10), (0.10, 0.70), (0.70, 0.90), (0.90, 1.0))


class EvaluationBlocker(RuntimeError):
    """Raised when the protocol forbids proceeding (no substitution allowed)."""


# --- Pure helpers (importable without torch or pyarrow) ---


def normalize_text(text: str) -> str:
    lowered = text.lower()
    collapsed = " ".join(lowered.split())
    return collapsed.strip(" \t\n\r.,;:!?\"'()[]{}<>")


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be within [0, total]")
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    half_width = (
        z
        * (
            proportion * (1 - proportion) / total
            + z_squared / (4 * total * total)
        )
        ** 0.5
    ) / denominator
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    if lower < 1e-12:
        lower = 0.0
    if upper > 1 - 1e-12:
        upper = 1.0
    return lower, upper


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _label_value(labels: Any, name: str) -> float | None:
    """Read one reviewer label from the documented encodings.

    The oasst1 parquet stores ``labels`` as a columnar dict of parallel
    lists (``{"name": [...], "value": [...], "count": [...]}``); the JSONL
    export stores a list of ``{name, value, count}`` records. Dict-of-dicts
    is accepted as well.
    """
    if labels is None:
        return None
    if isinstance(labels, dict):
        names = labels.get("name")
        values = labels.get("value")
        if isinstance(names, (list, tuple)) and isinstance(
            values, (list, tuple)
        ):
            for label_name, label_value in zip(names, values):
                if label_name == name:
                    return _as_float(label_value)
            return None
        entry = labels.get(name)
        if isinstance(entry, dict):
            return _as_float(entry.get("value"))
        if isinstance(entry, (list, tuple)):
            return _as_float(entry[0]) if entry else None
        return _as_float(entry)
    if isinstance(labels, (list, tuple)):
        for entry in labels:
            if isinstance(entry, dict) and entry.get("name") == name:
                return _as_float(entry.get("value"))
    return None


def dolly_eligible(record: dict[str, Any]) -> tuple[bool, str | None]:
    text = record.get("instruction")
    if not isinstance(text, str) or not text.strip():
        return False, "missing_instruction"
    if len(text.strip()) < DOLLY_MIN_CHARS:
        return False, "instruction_too_short"
    return True, None


def oasst_eligible(record: dict[str, Any]) -> tuple[bool, str | None]:
    if record.get("role") != "prompter":
        return False, "not_prompter"
    if record.get("lang") != "en":
        return False, "not_english"
    if record.get("parent_id") is not None:
        return False, "not_root_message"
    if record.get("deleted"):
        return False, "deleted"
    tree_state = record.get("tree_state")
    if tree_state is not None and tree_state != "ready_for_export":
        return False, "tree_not_ready_for_export"
    if record.get("review_result") is not True:
        return False, "review_not_passed"
    review_count = record.get("review_count") or 0
    if review_count < OASST_REVIEW_MIN:
        return False, "insufficient_review_count"

    labels = record.get("labels")
    for name in OASST_NEGATIVE_LABELS:
        value = _label_value(labels, name)
        if value is not None and value > 0:
            return False, f"reviewer_flagged_{name}"
    quality = _label_value(labels, "quality")
    if quality is None:
        return False, "quality_label_absent"
    if quality < OASST_QUALITY_MIN:
        return False, "quality_below_threshold"

    detoxify = record.get("detoxify")
    if not isinstance(detoxify, dict):
        return False, "detoxify_scores_missing"
    for name, limit in OASST_DETOXIFY_LIMITS.items():
        value = detoxify.get(name)
        if not isinstance(value, (int, float)):
            return False, "detoxify_scores_missing"
        if value >= limit:
            return False, f"detoxify_{name}_at_or_above_limit"

    text = record.get("text")
    if not isinstance(text, str) or len(text.strip()) < OASST_MIN_CHARS:
        return False, "text_too_short"
    return True, None


def build_reference_index(texts: Iterable[str]) -> dict[str, Any]:
    exact: set[str] = set()
    buckets: dict[str, list[str]] = {}
    for text in texts:
        normalized = normalize_text(text)
        exact.add(normalized)
        buckets.setdefault(normalized[:BUCKET_CHARS], []).append(normalized)
    return {"exact": exact, "buckets": buckets}


def classify_reference(normalized: str, reference: dict[str, Any]) -> str:
    if normalized in reference["exact"]:
        return "exact_match_reference"
    for other in reference["buckets"].get(normalized[:BUCKET_CHARS], ()):
        if (
            SequenceMatcher(None, normalized, other).ratio()
            >= NEAR_DUPLICATE_RATIO
        ):
            return "near_match_reference"
    return "keep"


def process_records(
    records: Iterable[dict[str, Any]],
    *,
    eligibility_fn: Callable[[dict[str, Any]], tuple[bool, str | None]],
    text_getter: Callable[[dict[str, Any]], str],
    reference: dict[str, Any],
    seen: set[str],
    duplicate_reason: str,
    counters: Counter,
) -> list[tuple[str, str]]:
    kept: list[tuple[str, str]] = []
    for record in records:
        eligible, reason = eligibility_fn(record)
        if not eligible:
            counters[reason] += 1
            continue
        text = text_getter(record)
        # Execution-time rule (disclosed in the report): the API validates
        # prompts at MAX_PROMPT_LENGTH characters and returns 422 before the
        # gateway is reached, so longer rows cannot produce a gateway decision.
        if len(text) > MAX_PROMPT_LENGTH:
            counters["exceeds_serving_char_limit"] += 1
            continue
        normalized = normalize_text(text)
        if normalized in seen:
            counters[duplicate_reason] += 1
            continue
        seen.add(normalized)
        classification = classify_reference(normalized, reference)
        if classification != "keep":
            counters[classification] += 1
            continue
        kept.append((normalized, text))
    return kept


def drop_cross_duplicates(
    candidates: Sequence[tuple[str, str]],
    other_norms: set[str],
    counters: Counter,
) -> list[tuple[str, str]]:
    kept: list[tuple[str, str]] = []
    for normalized, text in candidates:
        if normalized in other_norms:
            counters["duplicate_cross_dataset"] += 1
            continue
        kept.append((normalized, text))
    return kept


def select_deterministic(
    candidates: Sequence[tuple[str, str]], sample_size: int
) -> list[tuple[str, str]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
            item[0],
        ),
    )
    return ordered[:sample_size]


def ensure_sample(eligible_counts: dict[str, int]) -> None:
    for name, count in eligible_counts.items():
        if count < SAMPLE_PER_DATASET:
            raise EvaluationBlocker(
                f"insufficient eligible rows for {name}: {count} < "
                f"{SAMPLE_PER_DATASET}; the frozen protocol forbids "
                "substituting datasets or relaxing eligibility rules"
            )


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise EvaluationBlocker(f"manifest not found: {manifest_path.as_posix()}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise EvaluationBlocker(f"manifest is not valid JSON: {error}") from error
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        raise EvaluationBlocker("manifest has no datasets section")
    for key, frozen in FREEZE.items():
        entry = datasets.get(key)
        if not isinstance(entry, dict):
            raise EvaluationBlocker(f"manifest is missing dataset entry: {key}")
        if entry.get("commit") != frozen["commit"]:
            raise EvaluationBlocker(f"manifest commit mismatch for {key}")
        if str(entry.get("license", "")).lower() != frozen["license"]:
            raise EvaluationBlocker(f"manifest license mismatch for {key}")
        if not str(entry.get("attribution", "")).strip():
            raise EvaluationBlocker(f"manifest attribution missing for {key}")
        file_value = str(entry.get("file", ""))
        if Path(file_value).name == TEST_FILE_NAME:
            raise EvaluationBlocker("manifest must never reference the test split")
        if not file_value.replace("\\", "/").startswith("data/raw-eval/"):
            raise EvaluationBlocker(f"manifest file outside data/raw-eval: {key}")
        file_path = Path(file_value)
        if not file_path.is_file():
            raise EvaluationBlocker(f"manifest file missing: {file_value}")
        if file_path.stat().st_size != entry.get("bytes"):
            raise EvaluationBlocker(f"manifest size mismatch: {file_value}")
        if _sha256_of(file_path) != str(entry.get("sha256", "")).lower():
            raise EvaluationBlocker(f"manifest hash mismatch: {file_value}")
    return data


def assert_run_paths(
    dolly_path: str,
    oasst_path: str,
    train_path: str,
    validation_path: str,
) -> None:
    for label, value in (("dolly", dolly_path), ("oasst1", oasst_path)):
        if Path(value).name == TEST_FILE_NAME:
            raise EvaluationBlocker(f"{label} input must never be the test split")
        if Path(value).as_posix() not in ALLOWED_EVAL_FILES:
            raise EvaluationBlocker(f"{label} input path is not allowed: {value}")
    for label, value in (("train", train_path), ("validation", validation_path)):
        if Path(value).name == TEST_FILE_NAME:
            raise EvaluationBlocker(f"{label} input must never be the test split")
        if Path(value).as_posix() not in ALLOWED_REFERENCE_FILES:
            raise EvaluationBlocker(f"{label} path is not allowed: {value}")


def operational_decision(
    block_interval: tuple[float, float],
) -> dict[str, Any]:
    lower, upper = block_interval
    if lower > REFERENCE_RATE:
        outcome = (
            "b_new_slice_materially_worse_than_observed_validation_benchmark"
        )
    elif upper < REFERENCE_RATE:
        outcome = "c_new_slice_better_than_observed_validation_benchmark"
    else:
        outcome = "a_new_slice_interval_contains_observed_validation_benchmark"
    return {
        "kind": "operational_go_no_go_rule_not_a_formal_equality_test",
        "reference_rate": REFERENCE_RATE,
        "reference_wilson_95": [
            REFERENCE_VALIDATION_WILSON[0],
            REFERENCE_VALIDATION_WILSON[1],
        ],
        "reference_note": (
            "10.3% (42/407) is the observed validation-slice FPR benchmark, "
            "not an estimate of the true population FPR; overlapping "
            "intervals are not evidence of equality between populations"
        ),
        "new_evaluation_wilson_95": [block_interval[0], block_interval[1]],
        "outcome": outcome,
    }


def summarize_outcomes(
    scores: Sequence[float],
    gateway_decisions: Sequence[str],
    classifier_decisions: Sequence[str],
    marker_lists: Sequence[Sequence[str]],
) -> dict[str, Any]:
    total = len(scores)
    if total == 0:
        raise ValueError("cannot summarize zero rows")
    decision_counts = Counter(gateway_decisions)
    gateway: dict[str, Any] = {}
    for decision in ("allow", "review", "block"):
        count = decision_counts.get(decision, 0)
        interval = wilson_interval(count, total)
        gateway[decision] = {
            "count": count,
            "rate": count / total,
            "wilson_95": [interval[0], interval[1]],
        }
    marker_counts: Counter = Counter()
    rows_with_marker = 0
    for markers in marker_lists:
        if markers:
            rows_with_marker += 1
        marker_counts.update(markers)
    bucket_counts = [0] * len(SCORE_BUCKETS)
    for score in scores:
        for index, (low, high) in enumerate(SCORE_BUCKETS):
            if (score >= low if index == 0 else score > low) and score <= high:
                bucket_counts[index] += 1
                break
    return {
        "n": total,
        "gateway": gateway,
        "classifier": dict(Counter(classifier_decisions)),
        "score": {
            "min": min(scores),
            "max": max(scores),
            "mean": statistics.fmean(scores),
            "median": statistics.median(scores),
        },
        "score_buckets": {
            f"[{low:.2f},{high:.2f}]": count
            for (low, high), count in zip(SCORE_BUCKETS, bucket_counts)
        },
        "rows_with_strong_marker": rows_with_marker,
        "marker_rule_counts": dict(sorted(marker_counts.items())),
    }


def load_control_prompts(repo_root: Path) -> tuple[str, ...]:
    """Load the frozen positive-control prompts (no torch required)."""
    path = repo_root / CONTROL_SCRIPT
    if not path.is_file():
        raise EvaluationBlocker(f"control script missing: {CONTROL_SCRIPT}")
    spec = importlib.util.spec_from_file_location(
        "promptshield_fp_control", path
    )
    if spec is None or spec.loader is None:
        raise EvaluationBlocker("control script could not be opened")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = getattr(module, "MALICIOUS_CASES", None)
    if not cases or not all(
        isinstance(case, str) and case.strip() for case in cases
    ):
        raise EvaluationBlocker("control prompts are missing or invalid")
    return tuple(cases)


def load_dolly_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_oasst_records(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=None)
    return table.to_pylist()


def load_reference_texts(train_path: Path, validation_path: Path) -> list[str]:
    import pyarrow.parquet as pq

    texts: list[str] = []
    for path in (train_path, validation_path):
        table = pq.read_table(path, columns=["text"])
        texts.extend(table.column("text").to_pylist())
    return texts


def count_truncated_prompts(service: DebertaInference, texts: Sequence[str]) -> int:
    tokenizer = service._tokenizer  # local-only tokenizer from the frozen load
    if tokenizer is None:
        raise EvaluationBlocker("tokenizer unavailable for truncation check")
    truncated = 0
    for text in texts:
        token_ids = tokenizer(
            text, add_special_tokens=True, truncation=False
        )["input_ids"]
        if len(token_ids) > MAX_LENGTH:
            truncated += 1
    return truncated


def build_report(sections: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "report": "benign-chat-eval",
        "created_utc": datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        ),
        "protocol": {
            "document": PROTOCOL_DOCUMENT,
            "frozen": PROTOCOL_FROZEN,
            "single_run": True,
        },
        "test_split_loaded": False,
    }
    report.update(sections)
    report["privacy"] = {
        "raw_prompts_in_report": False,
        "raw_prompts_on_console": False,
        "raw_data_redistributed": False,
    }
    return report


def write_report(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    stamp = report["created_utc"]
    json_path = reports_dir / f"benign-chat-eval-{stamp}.json"
    md_path = reports_dir / f"benign-chat-eval-{stamp}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(format_markdown(report), encoding="utf-8")
    return json_path, md_path


def format_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    metrics = report["metrics"]
    decision = report["operational_decision"]
    integrity = report["integrity"]
    lines.append("# PromptShield benign-chat generalization evaluation")
    lines.append("")
    lines.append(
        f"Single pre-registered run of the frozen protocol "
        f"[{PROTOCOL_DOCUMENT}]({PROTOCOL_DOCUMENT}) (frozen "
        f"{PROTOCOL_FROZEN}). Aggregate metrics only; no prompt text is "
        "included. `test_split_loaded`: `false`."
    )
    lines.append("")
    lines.append("## Frozen configuration")
    lines.append("")
    lines.append(
        f"- Checkpoint: `{report['model']['checkpoint']}` "
        f"(safetensors sha256 matches decision audit: "
        f"`{integrity['checkpoint_hash_matches_decision_audit']}`)"
    )
    lines.append(
        f"- Classifier threshold: `{report['model']['threshold']}`; "
        f"gateway policy: `{report['gateway']['policy_id']}` "
        f"(allow_max `{report['gateway']['allow_max']}`, "
        f"block_min `{report['gateway']['block_min']}`)"
    )
    lines.append(f"- Device: `{report['model']['device']}`")
    lines.append("")
    lines.append("## Inputs and licensing")
    lines.append("")
    lines.append(
        "| Dataset | Commit | License | Rows | File sha256 (first 16) |"
    )
    lines.append("|---|---|---|---:|---|")
    for entry in report["inputs"]["datasets"]:
        lines.append(
            f"| {entry['name']} | `{entry['commit'][:12]}` | "
            f"{entry['license']} | {entry['source_rows']} | "
            f"`{entry['sha256'][:16]}` |"
        )
    lines.append("")
    lines.append("Attribution:")
    for entry in report["inputs"]["datasets"]:
        lines.append(f"- {entry['attribution']}")
    lines.append("")
    lines.append("## Eligibility and exclusions (first-fail counts)")
    lines.append("")
    for name, block in report["eligibility_counts"].items():
        lines.append(
            f"- **{name}**: source rows {block['source_rows']}, "
            f"candidates retained after eligibility and deduplication "
            f"{block['candidates_retained']}"
        )
        for reason, count in sorted(block["excluded"].items()):
            lines.append(f"  - `{reason}`: {count}")
    lines.append("")
    lines.append("## Deduplication and selection")
    lines.append("")
    dedup = report["deduplication"]
    lines.append(f"Method: {dedup['method']}")
    dedup_keys = (
        "duplicate_within",
        "duplicate_cross_dataset",
        "exact_match_reference",
        "near_match_reference",
    )
    for name, counts in dedup["counts"].items():
        dedup_only = {
            key: value for key, value in counts.items() if key in dedup_keys
        }
        rendered = (
            ", ".join(f"{key}: {value}" for key, value in sorted(dedup_only.items()))
            or "no duplicates found"
        )
        lines.append(f"- **{name}**: {rendered}")
    selection = report["selection"]
    lines.append("")
    lines.append(
        f"Selection: {selection['method']}; selected "
        f"{selection['dolly_selected']} Dolly + "
        f"{selection['oasst1_selected']} OASST1 rows "
        f"(n={metrics['n']})."
    )
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Decision | Count | Rate | Wilson 95% CI |")
    lines.append("|---|---:|---:|---|")
    for decision_name in ("allow", "review", "block"):
        cell = metrics["gateway"][decision_name]
        lines.append(
            f"| {decision_name} | {cell['count']} | {cell['rate']:.4f} | "
            f"[{cell['wilson_95'][0]:.4f}, {cell['wilson_95'][1]:.4f}] |"
        )
    lines.append("")
    score = metrics["score"]
    lines.append(
        f"Class-1 score: min {score['min']:.4f}, median {score['median']:.4f}, "
        f"mean {score['mean']:.4f}, max {score['max']:.4f}."
    )
    lines.append(
        f"Classifier decisions: {metrics['classifier']}. Rows with a strong "
        f"marker: {metrics['rows_with_strong_marker']}; marker counts: "
        f"{metrics['marker_rule_counts']}."
    )
    lines.append(
        f"Score buckets: {metrics['score_buckets']}. Prompts truncated at "
        f"{MAX_LENGTH} tokens: {metrics['truncated_at_max_length']}."
    )
    lines.append("")
    lines.append("Exploratory per-dataset block counts (n=100 each; "
                 "subgroup rates are not supported at this size):")
    for name, block in report["exploratory_by_dataset"].items():
        cell = block["gateway"]["block"]
        lines.append(
            f"- {name}: {cell['count']}/100 blocked "
            f"(rate {cell['rate']:.4f}, CI "
            f"[{cell['wilson_95'][0]:.4f}, {cell['wilson_95'][1]:.4f}])"
        )
    lines.append("")
    lines.append("## Operational go/no-go outcome")
    lines.append("")
    lines.append(
        f"- Observed validation benchmark: {REFERENCE_VALIDATION_FP}/"
        f"{REFERENCE_VALIDATION_BENIGN} = {REFERENCE_RATE:.4f} "
        f"(Wilson 95% CI "
        f"[{REFERENCE_VALIDATION_WILSON[0]:.4f}, "
        f"{REFERENCE_VALIDATION_WILSON[1]:.4f}]) — an observed slice value, "
        "not a population estimate."
    )
    lines.append(
        f"- New evaluation block rate: {metrics['gateway']['block']['rate']:.4f} "
        f"(Wilson 95% CI "
        f"[{decision['new_evaluation_wilson_95'][0]:.4f}, "
        f"{decision['new_evaluation_wilson_95'][1]:.4f}])."
    )
    lines.append(f"- Outcome: **{decision['outcome']}** "
                 f"({decision['kind'].replace('_', ' ')}).")
    lines.append("")
    lines.append("## Integrity and privacy")
    lines.append("")
    for key, value in integrity.items():
        lines.append(f"- {key}: {value}")
    lines.append(
        "- raw_prompts_in_report: false; raw_prompts_on_console: false; "
        "raw_data_redistributed: false."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(report["limitations"])
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path.cwd()
    assert_run_paths(args.dolly, args.oasst, args.train, args.validation)
    manifest = verify_manifest(MANIFEST_PATH)

    checkpoint_file = MODEL_DIR / "model.safetensors"
    if not checkpoint_file.is_file():
        raise EvaluationBlocker("frozen checkpoint is missing")
    checkpoint_hash = _sha256_of(checkpoint_file)
    checkpoint_matches_audit = checkpoint_hash == AUDIT_CHECKPOINT_SHA256
    if not checkpoint_matches_audit:
        raise EvaluationBlocker(
            "checkpoint hash does not match the decision-audit record"
        )

    reference_texts = load_reference_texts(Path(args.train), Path(args.validation))
    reference = build_reference_index(reference_texts)

    dolly_records = load_dolly_records(Path(args.dolly))
    dolly_counters: Counter = Counter()
    dolly_candidates = process_records(
        dolly_records,
        eligibility_fn=dolly_eligible,
        text_getter=lambda record: record["instruction"],
        reference=reference,
        seen=set(),
        duplicate_reason="duplicate_within",
        counters=dolly_counters,
    )

    oasst_records = load_oasst_records(Path(args.oasst))
    oasst_counters: Counter = Counter()
    oasst_candidates = process_records(
        oasst_records,
        eligibility_fn=oasst_eligible,
        text_getter=lambda record: record["text"],
        reference=reference,
        seen=set(),
        duplicate_reason="duplicate_within",
        counters=oasst_counters,
    )
    dolly_norms = {normalized for normalized, _ in dolly_candidates}
    oasst_candidates = drop_cross_duplicates(
        oasst_candidates, dolly_norms, oasst_counters
    )

    ensure_sample(
        {
            "dolly": len(dolly_candidates),
            "oasst1": len(oasst_candidates),
        }
    )

    dolly_selected = select_deterministic(dolly_candidates, SAMPLE_PER_DATASET)
    oasst_selected = select_deterministic(oasst_candidates, SAMPLE_PER_DATASET)
    selected_texts = [text for _, text in dolly_selected]
    selected_texts += [text for _, text in oasst_selected]

    control_texts = list(load_control_prompts(repo_root))

    service = DebertaInference(MODEL_DIR, threshold=DEFAULT_THRESHOLD, device="auto")
    service.load()
    policy = GatewayPolicy()

    predictions = service.predict_batch(selected_texts, batch_size=args.batch_size)
    scores = [prediction.malicious_probability for prediction in predictions]
    classifier_decisions = [prediction.decision for prediction in predictions]
    marker_lists = [matched_strong_marker_rules(text) for text in selected_texts]
    gateway_decisions = [
        policy.decide(score, markers)[0]
        for score, markers in zip(scores, marker_lists)
    ]

    control_predictions = service.predict_batch(
        control_texts, batch_size=args.batch_size
    )
    control_blocks = sum(
        policy.decide(
            prediction.malicious_probability,
            matched_strong_marker_rules(text),
        )[0]
        == "block"
        for text, prediction in zip(control_texts, control_predictions)
    )
    if control_blocks != len(control_texts):
        raise EvaluationBlocker(
            f"positive control failed: {control_blocks}/"
            f"{len(control_texts)} blocked; aborting without metrics report"
        )

    truncated = count_truncated_prompts(service, selected_texts)

    overall = summarize_outcomes(
        scores, gateway_decisions, classifier_decisions, marker_lists
    )
    overall["truncated_at_max_length"] = truncated
    exploratory = {
        "dolly": summarize_outcomes(
            scores[:SAMPLE_PER_DATASET],
            gateway_decisions[:SAMPLE_PER_DATASET],
            classifier_decisions[:SAMPLE_PER_DATASET],
            marker_lists[:SAMPLE_PER_DATASET],
        ),
        "oasst1": summarize_outcomes(
            scores[SAMPLE_PER_DATASET:],
            gateway_decisions[SAMPLE_PER_DATASET:],
            classifier_decisions[SAMPLE_PER_DATASET:],
            marker_lists[SAMPLE_PER_DATASET:],
        ),
    }
    block_interval = tuple(overall["gateway"]["block"]["wilson_95"])

    def dataset_counts(records: list[dict[str, Any]], counters: Counter) -> dict[str, Any]:
        excluded = {key: value for key, value in sorted(counters.items())}
        excluded_total = sum(counts for counts in counters.values())
        return {
            "source_rows": len(records),
            "excluded": excluded,
            "excluded_total": excluded_total,
            "candidates_retained": len(records) - excluded_total,
        }

    report = build_report(
        {
            "model": {
                "checkpoint": MODEL_DIR.as_posix(),
                "safetensors_sha256": checkpoint_hash,
                "checkpoint_hash_matches_decision_audit": True,
                "threshold": service.threshold,
                "max_length_tokens": MAX_LENGTH,
                "max_prompt_length_chars": MAX_PROMPT_LENGTH,
                "device": service.device,
            },
            "gateway": policy.describe(),
            "inputs": {
                "manifest": MANIFEST_PATH.as_posix(),
                "datasets": [
                    {
                        "name": entry["name"],
                        "commit": entry["commit"],
                        "license": entry["license"],
                        "license_source": entry["license_source"],
                        "attribution": entry["attribution"],
                        "file": entry["file"],
                        "bytes": entry["bytes"],
                        "sha256": entry["sha256"],
                        "source_rows": len(dolly_records)
                        if key == "dolly"
                        else len(oasst_records),
                    }
                    for key, entry in manifest["datasets"].items()
                ],
            },
            "integrity": {
                "manifest_verified_before_scoring": True,
                "revision_pinned_and_verified": True,
                "license_metadata_verified": True,
                "attribution_recorded": True,
                "checkpoint_hash_matches_decision_audit": True,
                "positive_control": {
                    "expected_blocks": len(control_texts),
                    "actual_blocks": control_blocks,
                    "passed": True,
                },
            },
            "eligibility_rules": {
                "dolly": [
                    "instruction present, non-empty, >= 4 chars",
                    "serving char limit <= MAX_PROMPT_LENGTH (execution-time "
                    "rule mirroring API validation, disclosed)",
                ],
                "oasst1": [
                    "role=prompter, lang=en, parent_id=null (root), "
                    "deleted=false, tree_state=ready_for_export",
                    "review_result=true and review_count>=3",
                    f"reviewer negative labels all 0: "
                    f"{', '.join(OASST_NEGATIVE_LABELS)}",
                    f"reviewer quality label >= {OASST_QUALITY_MIN} "
                    "(positive benign-quality evidence)",
                    "detoxify toxicity < 0.2 and severe_toxicity < 0.1",
                    f"text >= {OASST_MIN_CHARS} chars",
                    "serving char limit <= MAX_PROMPT_LENGTH (execution-time "
                    "rule mirroring API validation, disclosed)",
                ],
            },
            "eligibility_counts": {
                "dolly": dataset_counts(dolly_records, dolly_counters),
                "oasst1": dataset_counts(oasst_records, oasst_counters),
            },
            "deduplication": {
                "method": (
                    "lowercase + whitespace collapse + edge punctuation "
                    "strip; exact set match and same-first-"
                    f"{BUCKET_CHARS}-char bucket with SequenceMatcher ratio "
                    f">= {NEAR_DUPLICATE_RATIO}; deduplicated against "
                    "train.parquet and validation.parquet only (test split "
                    "never read) plus within-dataset and cross-dataset "
                    "checks"
                ),
                "counts": {
                    "dolly": dict(sorted(dolly_counters.items())),
                    "oasst1": dict(sorted(oasst_counters.items())),
                },
                "reference_texts_indexed": len(reference_texts),
            },
            "selection": {
                "method": (
                    "sort eligible deduplicated candidates by "
                    "sha256(normalized text) ascending, take first "
                    f"{SAMPLE_PER_DATASET} per dataset"
                ),
                "dolly_selected": len(dolly_selected),
                "oasst1_selected": len(oasst_selected),
                "dolly_candidates_after_dedup": len(dolly_candidates),
                "oasst1_candidates_after_dedup": len(oasst_candidates),
            },
            "metrics": overall,
            "exploratory_by_dataset": exploratory,
            "operational_decision": operational_decision(block_interval),
            "limitations": (
                "Public dataset prompts are not production traffic: "
                "English-only, single-turn, and drawn from two sources; "
                "n=200 supports only an overall block-rate estimate "
                "(roughly +/-4-7 percentage points), not subgroup, recall, "
                "calibration, or non-English claims. Eligibility requires "
                "reviewer-confirmed benign quality, but crowd annotations "
                "are imperfect. Intervals overlapping the validation "
                "benchmark do not prove the rates are equal. This slice is "
                "single-use: if mitigation is ever pursued it becomes "
                "development data and cannot support post-fix claims."
            ),
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dolly",
        default=str(Path("data/raw-eval/dolly/databricks-dolly-15k.jsonl")),
    )
    parser.add_argument(
        "--oasst", default=str(Path("data/raw-eval/oasst1/train.parquet"))
    )
    parser.add_argument(
        "--train", default=str(Path("data/raw/train.parquet"))
    )
    parser.add_argument(
        "--validation", default=str(Path("data/raw/validation.parquet"))
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--reports-dir", default=str(Path("reports"))
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
        json_path, md_path = write_report(report, Path(args.reports_dir))
    except EvaluationBlocker as blocker:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "protocol": PROTOCOL_DOCUMENT,
                    "detail": str(blocker),
                    "test_split_loaded": False,
                },
                indent=2,
            )
        )
        return 2
    summary = {
        "status": "completed",
        "protocol": PROTOCOL_DOCUMENT,
        "n": report["metrics"]["n"],
        "gateway": report["metrics"]["gateway"],
        "score": report["metrics"]["score"],
        "eligibility_counts": report["eligibility_counts"],
        "deduplication_counts": report["deduplication"]["counts"],
        "operational_decision": report["operational_decision"],
        "positive_control": report["integrity"]["positive_control"],
        "reports": [json_path.as_posix(), md_path.as_posix()],
        "test_split_loaded": False,
        "raw_prompts_printed": False,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
