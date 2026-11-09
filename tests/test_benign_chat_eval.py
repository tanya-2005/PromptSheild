"""Tests for the benign-chat evaluation protocol helpers.

These tests exercise only pure functions: eligibility rules, deduplication,
deterministic selection, interval math, aggregation, and path/manifest guards.
They do not load a model, read parquet files, or touch any dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from promptshield.benign_chat_eval import (
    FREEZE,
    EvaluationBlocker,
    assert_run_paths,
    build_reference_index,
    classify_reference,
    dolly_eligible,
    drop_cross_duplicates,
    ensure_sample,
    normalize_text,
    oasst_eligible,
    operational_decision,
    process_records,
    select_deterministic,
    summarize_outcomes,
    verify_manifest,
    wilson_interval,
)

DOLLY_FILE = "data/raw-eval/dolly/databricks-dolly-15k.jsonl"
OASST_FILE = "data/raw-eval/oasst1/train.parquet"


def make_oasst_record(**overrides: object) -> dict:
    record = {
        "role": "prompter",
        "lang": "en",
        "parent_id": None,
        "deleted": False,
        "tree_state": "ready_for_export",
        "review_result": True,
        "review_count": 3,
        "text": "How does photosynthesis work in plants?",
        "labels": [
            {"name": "spam", "value": 0.0, "count": 3},
            {"name": "lang_mismatch", "value": 0.0, "count": 3},
            {"name": "pii", "value": 0.0, "count": 3},
            {"name": "not_appropriate", "value": 0.0, "count": 3},
            {"name": "hate_speech", "value": 0.0, "count": 3},
            {"name": "sexual_content", "value": 0.0, "count": 3},
            {"name": "toxicity", "value": 0.0, "count": 3},
            {"name": "violence", "value": 0.0, "count": 3},
            {"name": "quality", "value": 0.75, "count": 3},
        ],
        "detoxify": {"toxicity": 0.01, "severe_toxicity": 0.0},
    }
    record.update(overrides)
    return record


def test_normalize_text_collapses_space_and_edges() -> None:
    assert normalize_text("  Hello   WORLD!  ") == "hello world"
    assert normalize_text("Line one\nLine two.") == "line one line two"


def test_wilson_interval_zero_false_positives() -> None:
    lower, upper = wilson_interval(0, 200)
    assert lower == 0.0
    assert 0.018 < upper < 0.019


def test_wilson_interval_matches_validation_reference() -> None:
    lower, upper = wilson_interval(42, 407)
    assert abs(lower - 0.0773) < 5e-4
    assert abs(upper - 0.1366) < 5e-4


def test_dolly_eligibility() -> None:
    assert dolly_eligible({"instruction": "Summarize this paragraph."}) == (
        True,
        None,
    )
    assert dolly_eligible({"instruction": "   "}) == (False, "missing_instruction")
    assert dolly_eligible({"instruction": "hi"}) == (
        False,
        "instruction_too_short",
    )


def test_oasst_eligible_accepts_reviewer_confirmed_benign() -> None:
    assert oasst_eligible(make_oasst_record()) == (True, None)


def test_oasst_eligible_accepts_dict_form_labels() -> None:
    record = make_oasst_record(
        labels={
            "spam": 0.0,
            "hate_speech": 0.0,
            "quality": {"value": 0.9, "count": 3},
        }
    )
    assert oasst_eligible(record) == (True, None)


def test_oasst_eligible_accepts_columnar_parquet_labels() -> None:
    columnar = {
        "name": ["spam", "hate_speech", "quality", "toxicity"],
        "value": [0.0, 0.0, 0.9166, 0.0],
        "count": [3, 3, 3, 3],
    }
    assert oasst_eligible(make_oasst_record(labels=columnar)) == (True, None)
    flagged = {
        "name": ["spam", "hate_speech", "quality", "toxicity"],
        "value": [0.0, 0.0, 0.9166, 0.5],
        "count": [3, 3, 3, 3],
    }
    assert oasst_eligible(make_oasst_record(labels=flagged)) == (
        False,
        "reviewer_flagged_toxicity",
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"role": "assistant"}, "not_prompter"),
        ({"lang": "de"}, "not_english"),
        ({"parent_id": "parent-1"}, "not_root_message"),
        ({"deleted": True}, "deleted"),
        ({"tree_state": "reviewed"}, "tree_not_ready_for_export"),
        ({"review_result": False}, "review_not_passed"),
        ({"review_result": None}, "review_not_passed"),
        ({"review_count": 2}, "insufficient_review_count"),
        ({"text": "short"}, "text_too_short"),
    ],
)
def test_oasst_eligibility_rejections(overrides: dict, reason: str) -> None:
    assert oasst_eligible(make_oasst_record(**overrides)) == (False, reason)


def test_oasst_rejects_negative_reviewer_flags() -> None:
    labels = [
        {"name": "spam", "value": 0.0},
        {"name": "hate_speech", "value": 0.16},
        {"name": "quality", "value": 0.9},
    ]
    assert oasst_eligible(make_oasst_record(labels=labels)) == (
        False,
        "reviewer_flagged_hate_speech",
    )


def test_oasst_requires_positive_quality_evidence() -> None:
    labels = [{"name": "spam", "value": 0.0}]
    assert oasst_eligible(make_oasst_record(labels=labels)) == (
        False,
        "quality_label_absent",
    )
    labels_with_low_quality = [{"name": "quality", "value": 0.25}]
    assert oasst_eligible(
        make_oasst_record(labels=labels_with_low_quality)
    ) == (False, "quality_below_threshold")


def test_oasst_requires_low_automated_toxicity() -> None:
    assert oasst_eligible(
        make_oasst_record(
            detoxify={"toxicity": 0.2, "severe_toxicity": 0.0}
        )
    ) == (False, "detoxify_toxicity_at_or_above_limit")
    assert oasst_eligible(
        make_oasst_record(
            detoxify={"toxicity": 0.0, "severe_toxicity": 0.1}
        )
    ) == (False, "detoxify_severe_toxicity_at_or_above_limit")
    assert oasst_eligible(make_oasst_record(detoxify={})) == (
        False,
        "detoxify_scores_missing",
    )


def test_reference_dedup_detects_exact_and_near_matches() -> None:
    reference = build_reference_index(
        [
            "please summarize the following article about renewable energy "
            "policy in europe today",
            "What is the capital of France?",
        ]
    )
    assert (
        classify_reference(
            normalize_text(
                "Please summarize the following article about renewable "
                "energy policy in Europe today."
            ),
            reference,
        )
        == "exact_match_reference"
    )
    assert (
        classify_reference(
            normalize_text(
                "please summarize the following article about renewable "
                "energy policy in europe today please"
            ),
            reference,
        )
        == "near_match_reference"
    )
    assert (
        classify_reference(
            normalize_text("Describe how a heat pump works."), reference
        )
        == "keep"
    )


def test_process_records_counts_each_exclusion_reason() -> None:
    reference = build_reference_index(["what is the capital of france?"])
    records = [
        {"instruction": "Summarize the plot of a short story."},
        {"instruction": "how do solar panels generate electricity?"},
        {"instruction": "what is the capital of France?"},  # exact reference
        {"instruction": "hi"},  # too short
        {"instruction": "Summarize a short story plot."},  # near? distinct
        {"instruction": "x" * 4100},  # serving char limit
    ]
    counter: dict = {}
    counters = Counter(counter)
    kept = process_records(
        records,
        eligibility_fn=dolly_eligible,
        text_getter=lambda record: record["instruction"],
        reference=reference,
        seen=set(),
        duplicate_reason="duplicate_within",
        counters=counters,
    )
    assert len(kept) == 3
    assert counters["exact_match_reference"] == 1
    assert counters["instruction_too_short"] == 1
    assert counters["exceeds_serving_char_limit"] == 1


def test_process_records_deduplicates_within_input() -> None:
    counters = Counter()
    kept = process_records(
        [
            {"instruction": "Explain photosynthesis."},
            {"instruction": "  explain PHOTOSYNTHESIS  "},
        ],
        eligibility_fn=dolly_eligible,
        text_getter=lambda record: record["instruction"],
        reference=build_reference_index([]),
        seen=set(),
        duplicate_reason="duplicate_within",
        counters=counters,
    )
    assert len(kept) == 1
    assert counters["duplicate_within"] == 1


def test_cross_dataset_duplicates_are_dropped() -> None:
    counters = Counter()
    kept = drop_cross_duplicates(
        [("same prompt", "Same prompt"), ("unique prompt", "Unique prompt")],
        {"same prompt"},
        counters,
    )
    assert kept == [("unique prompt", "Unique prompt")]
    assert counters["duplicate_cross_dataset"] == 1


def test_deterministic_selection_is_order_independent_and_stable() -> None:
    items = [
        (normalize_text(f"prompt number {index}"), f"prompt number {index}")
        for index in range(20)
    ]
    forward = select_deterministic(items, 5)
    backward = select_deterministic(list(reversed(items)), 5)
    assert [norm for norm, _ in forward] == [norm for norm, _ in backward]
    assert len(forward) == 5


def test_ensure_sample_blocks_instead_of_substituting() -> None:
    ensure_sample({"dolly": 100, "oasst1": 100})
    with pytest.raises(EvaluationBlocker, match="insufficient eligible rows"):
        ensure_sample({"dolly": 99, "oasst1": 100})


def test_assert_run_paths_rejects_test_split_and_unknown_paths() -> None:
    assert_run_paths(DOLLY_FILE, OASST_FILE, "data/raw/train.parquet",
                     "data/raw/validation.parquet")
    with pytest.raises(EvaluationBlocker, match="test split"):
        assert_run_paths(
            "data/raw/test.parquet",
            OASST_FILE,
            "data/raw/train.parquet",
            "data/raw/validation.parquet",
        )
    with pytest.raises(EvaluationBlocker, match="test split"):
        assert_run_paths(
            DOLLY_FILE,
            OASST_FILE,
            "data/raw/test.parquet",
            "data/raw/validation.parquet",
        )
    with pytest.raises(EvaluationBlocker, match="not allowed"):
        assert_run_paths(
            DOLLY_FILE,
            OASST_FILE,
            "data/raw/other.parquet",
            "data/raw/validation.parquet",
        )
    with pytest.raises(EvaluationBlocker, match="not allowed"):
        assert_run_paths(
            "data/raw-eval/other.jsonl",
            OASST_FILE,
            "data/raw/train.parquet",
            "data/raw/validation.parquet",
        )


def test_verify_manifest_requires_hashes_licenses_and_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dolly_path = tmp_path / "data/raw-eval/dolly/data.jsonl"
    oasst_path = tmp_path / "data/raw-eval/oasst1/data.parquet"
    dolly_path.parent.mkdir(parents=True)
    oasst_path.parent.mkdir(parents=True)
    dolly_path.write_bytes(b"dolly bytes")
    oasst_path.write_bytes(b"oasst bytes")

    def entry(key: str, path: Path) -> dict:
        return {
            "commit": FREEZE[key]["commit"],
            "license": FREEZE[key]["license"].upper(),
            "attribution": "Example attribution",
            "file": path.relative_to(tmp_path).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {
        "datasets": {
            "dolly": entry("dolly", dolly_path),
            "oasst1": entry("oasst1", oasst_path),
        }
    }
    manifest_path = tmp_path / "data/eval-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest(manifest_path)["datasets"]["dolly"]["license"]

    oasst_path.write_bytes(b"tampered")
    with pytest.raises(EvaluationBlocker, match="size mismatch"):
        verify_manifest(manifest_path)


def test_verify_manifest_rejects_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data/raw-eval/dolly"
    data_dir.mkdir(parents=True)
    dolly_path = data_dir / "data.jsonl"
    dolly_path.write_bytes(b"x")
    manifest = {
        "datasets": {
            "dolly": {
                "commit": "0" * 40,
                "license": FREEZE["dolly"]["license"],
                "attribution": "Example",
                "file": "data/raw-eval/dolly/data.jsonl",
                "bytes": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(),
            },
            "oasst1": {
                "commit": FREEZE["oasst1"]["commit"],
                "license": FREEZE["oasst1"]["license"],
                "attribution": "Example",
                "file": "data/raw-eval/dolly/data.jsonl",
                "bytes": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(),
            },
        }
    }
    manifest_path = tmp_path / "data/eval-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EvaluationBlocker, match="commit mismatch"):
        verify_manifest(manifest_path)


def test_summarize_outcomes_aggregates_counts_and_intervals() -> None:
    summary = summarize_outcomes(
        [0.95, 0.85, 0.5, 0.2],
        ["block", "review", "allow", "allow"],
        ["malicious", "benign", "benign", "benign"],
        [[], ["ignore_previous"], [], []],
    )
    assert summary["n"] == 4
    assert summary["gateway"]["block"]["count"] == 1
    assert summary["gateway"]["block"]["rate"] == 0.25
    assert summary["gateway"]["review"]["count"] == 1
    assert summary["gateway"]["allow"]["count"] == 2
    assert summary["classifier"] == {"malicious": 1, "benign": 3}
    assert summary["score"]["min"] == 0.2
    assert summary["score"]["max"] == 0.95
    assert summary["rows_with_strong_marker"] == 1
    assert summary["marker_rule_counts"] == {"ignore_previous": 1}
    assert summary["score_buckets"] == {
        "[0.00,0.10]": 0,
        "[0.10,0.70]": 2,
        "[0.70,0.90]": 1,
        "[0.90,1.00]": 1,
    }


def test_operational_decision_zones() -> None:
    contains = operational_decision((0.066, 0.149))
    assert contains["outcome"].startswith("a_")
    worse = operational_decision((0.195, 0.314))
    assert worse["outcome"].startswith("b_")
    better = operational_decision((0.0, 0.019))
    assert better["outcome"].startswith("c_")
    assert contains["kind"] == "operational_go_no_go_rule_not_a_formal_equality_test"
