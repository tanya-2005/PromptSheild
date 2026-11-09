"""Benign hard-negative mitigation experiment (single pre-registered run).

Plan: ``docs/benign-hard-negative-mitigation-plan.md`` (approved 2026-10-19).
Everything below is frozen by that plan: dataset selection, training schedule,
threshold, gateway policy, and acceptance criteria. The experiment runs
exactly once; any failed gate stops the run without substitution or retry.

Gates executed in order by :func:`run`:

1. free RAM >= 3.5 GiB (stop otherwise),
2. fresh baseline audit reproduces the recorded confusion matrix exactly,
3. hard-negative/anchor eligibility, deduplication, and integrity checks,
4. one training run (3 epochs, lr 1e-5, seed 42) into a NEW artifact
   directory; the baseline checkpoint and reports are never written,
5. candidate audit with the identical serving protocol (threshold 0.90),
6. pre-registered criteria G1-G6 -> retain or reject.

``data/raw/test.parquet`` is never opened. Console output and tracked
reports are aggregate-only: no prompt text, no training-set text.

The same frozen gates run unchanged in the approved cloud environment
(Colab, CPU runtime): see ``docs/hard-negative-colab-runbook.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from promptshield.benign_chat_eval import (
    build_reference_index,
    classify_reference,
    load_control_prompts,
    normalize_text,
    select_deterministic,
)
from promptshield.deberta_decision_audit import (
    AuditFailure,
    audit_validation,
    hash_checkpoint_files,
)
from promptshield.gateway import GatewayPolicy
from promptshield.inference import DEFAULT_THRESHOLD, MAX_LENGTH, DebertaInference

# --- Frozen plan constants (docs/benign-hard-negative-mitigation-plan.md) ---

PLAN_DOCUMENT = "docs/benign-hard-negative-mitigation-plan.md"

TRAIN_PATH = Path("data/raw/train.parquet")
VALIDATION_PATH = Path("data/raw/validation.parquet")
BASELINE_DIR = Path("models/deberta-v3-small")
CANDIDATE_DIR = Path("models/deberta-v3-small-hardneg-v1")
HARDNEG_MANIFEST_PATH = Path("data/hardneg-manifest.json")

TRAIN_SHA256 = (
    "c0724f8e00eebb65c001be9ba94291d1608e5da0374012e53dfd4077c5f92558"
)
BASELINE_SAFETENSORS_SHA256 = (
    "2cefb8e40ade17940899ef523c7dea2a1bd1813ba0373edda84481a32c783614"
)
EXPECTED_BASELINE_CM = [[365, 42], [9, 525]]

LABEL_MAPPING = {0: "benign", 1: "malicious"}

HARD_NEGATIVE_COUNT = 138
ANCHOR_COUNT = 138
TOTAL_TRAINING_ROWS = 414

BLOCK_SCORE_MIN = DEFAULT_THRESHOLD  # 0.90, identical to serving
EASY_BENIGN_SCORE_MAX = 0.10

TRAINING = {
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "epochs": 3,
    "batch_size": 4,
    "seed": 42,
    "cpu_threads": 4,
    "max_length": MAX_LENGTH,
    "device": "cpu",
    "freeze_input_embeddings": True,
    "scheduler": "none",
    "early_stopping": "none",
}

MIN_FREE_RAM_BYTES = 3_758_096_384  # 3.5 GiB

DEV_EVAL_FILES = (
    Path("data/raw-eval/dolly/databricks-dolly-15k.jsonl"),
    Path("data/raw-eval/oasst1/train.parquet"),
)

CRITERIA_LIMITS = {
    "max_false_positives": 21,  # G1: <= 5.16% of 407 benign validation rows
    "max_false_negatives": 14,  # G2: <= 2.62% of 534 malicious rows
    "min_macro_f1": 0.9342,  # G3: baseline 0.9442 - 0.0100
    "min_class1_precision": 0.9059,  # G4: baseline 0.9259 - 0.0200
    "control_expected_blocks": 4,  # G5
}

TEST_FILE_NAME = "test.parquet"

TRAIN_RECORD_COLUMNS = ("text", "label", "category", "group_id")


class MitigationBlocker(RuntimeError):
    """A frozen-plan gate failed; the run stops without substitution."""


# --- Pure helpers (importable without torch or pyarrow) ---


def parse_meminfo_available(text: str) -> int | None:
    """Parse ``MemAvailable`` (bytes) from ``/proc/meminfo`` content.

    Linux support for the same pre-registered >= 3.5 GiB precondition; the
    Windows path below is unchanged. Returns ``None`` if the field is absent
    or malformed so the gate fails closed (``check_free_ram(None)`` aborts).
    """
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "MemAvailable:":
            try:
                return int(parts[1]) * 1024
            except ValueError:
                return None
    return None


def free_physical_memory_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            return parse_meminfo_available(
                Path("/proc/meminfo").read_text(encoding="ascii")
            )
        except OSError:
            return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    except Exception:
        return None


def check_free_ram(
    free_bytes: int | None, minimum: int = MIN_FREE_RAM_BYTES
) -> int:
    if free_bytes is None:
        raise MitigationBlocker(
            "free system memory could not be measured; the plan requires "
            "verifying >= 3.5 GiB free before building or training"
        )
    if free_bytes < minimum:
        raise MitigationBlocker(
            f"free RAM {free_bytes / 2**30:.2f} GiB is below the "
            f"pre-registered precondition of {minimum / 2**30:.2f} GiB; "
            "stop and report the blocker (no retries, no model or data "
            "downloads)"
        )
    return free_bytes


def ensure_paths_frozen(
    train_path: Path, validation_path: Path, candidate_dir: Path
) -> None:
    for label, value in (
        ("train", train_path),
        ("validation", validation_path),
    ):
        if value.name == TEST_FILE_NAME or TEST_FILE_NAME in value.as_posix():
            raise MitigationBlocker(
                f"{label} path must never reference the test split"
            )
    if train_path.as_posix() != TRAIN_PATH.as_posix():
        raise MitigationBlocker(f"train path is not allowed: {train_path}")
    if validation_path.as_posix() != VALIDATION_PATH.as_posix():
        raise MitigationBlocker(
            f"validation path is not allowed: {validation_path}"
        )
    if candidate_dir.as_posix() != CANDIDATE_DIR.as_posix():
        raise MitigationBlocker(
            "candidate artifacts may only be written to "
            f"{CANDIDATE_DIR.as_posix()}"
        )
    if candidate_dir.resolve() == BASELINE_DIR.resolve():
        raise MitigationBlocker("candidate directory must not be the baseline")


def ensure_candidate_dir_absent(candidate_dir: Path) -> None:
    if candidate_dir.exists():
        raise MitigationBlocker(
            f"candidate directory already exists: {candidate_dir.as_posix()}; "
            "the plan allows exactly one training run and one artifact"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pools(
    records: Sequence[dict[str, Any]],
    scores: dict[str, float],
    validation_reference: dict[str, Any],
    dev_norms: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    """Partition train rows into the three frozen pools (plan §3.3–3.4).

    ``scores`` maps ``group_id`` to the frozen-checkpoint class-1 probability
    and must cover every label-0 row. Exclusion counters use first-fail
    semantics.
    """
    pools: dict[str, list[dict[str, Any]]] = {
        "hard_negatives": [],
        "anchor_malicious": [],
        "anchor_easy_benign": [],
    }
    counters: Counter = Counter()
    for record in records:
        normalized = normalize_text(record["text"])
        classification = classify_reference(normalized, validation_reference)
        if classification != "keep":
            counters[f"validation_{classification}"] += 1
            continue
        if normalized in dev_norms:
            counters["dev_slice_exact_match"] += 1
            continue
        label = record["label"]
        if label == 1:
            pools["anchor_malicious"].append(record)
            continue
        if label != 0:
            counters["unexpected_label"] += 1
            continue
        if record["category"] != "benign":
            counters["label0_nonbenign_category"] += 1
            continue
        score = scores[record["group_id"]]
        if score >= BLOCK_SCORE_MIN:
            pools["hard_negatives"].append(record)
        elif score < EASY_BENIGN_SCORE_MAX:
            pools["anchor_easy_benign"].append(record)
        else:
            counters["label0_score_between_anchors_and_hard_negatives"] += 1
    return pools, counters


def select_pool(
    pool_name: str, records: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if len(records) < count:
        raise MitigationBlocker(
            f"pool {pool_name} has {len(records)} eligible rows, fewer than "
            f"the pre-registered {count}; the plan forbids substituting or "
            "relaxing eligibility"
        )
    keyed = [(normalize_text(record["text"]), record) for record in records]
    return [record for _, record in select_deterministic(keyed, count)]


def hardneg_manifest(
    selected: Sequence[dict[str, Any]],
    pools: dict[str, list[dict[str, Any]]],
    counters: Counter,
    source_sha256: str,
    source_rows: int,
    created_utc: str,
) -> dict[str, Any]:
    """Metadata-only manifest (ids, hashes, counts — never prompt text)."""
    return {
        "purpose": (
            "Metadata record for the approved benign hard-negative training "
            "subset; raw prompt text stays in memory and is never written."
        ),
        "plan": PLAN_DOCUMENT,
        "created_utc": created_utc,
        "source": {
            "file": TRAIN_PATH.as_posix(),
            "sha256": source_sha256,
            "rows": source_rows,
        },
        "eligibility": {
            "hard_negatives": (
                "label=0 AND category=benign AND frozen-checkpoint p1 >= "
                "0.90 AND no exact/near match of validation AND no exact "
                "normalized match of data/raw-eval source prompts"
            ),
            "anchor_malicious": (
                "label=1 AND same validation/dev-slice exclusion rules"
            ),
            "anchor_easy_benign": (
                "label=0 AND category=benign AND p1 < 0.10 AND same "
                "validation/dev-slice exclusion rules"
            ),
            "selection": (
                "sort by sha256(normalized text) ascending, take first 138 "
                "per pool"
            ),
        },
        "pool_sizes": {name: len(rows) for name, rows in pools.items()},
        "exclusion_counters": dict(sorted(counters.items())),
        "selected_counts": dict(
            sorted(Counter(row["label"] for row in selected).items())
        ),
        "selected_rows": [
            {
                "group_id": row["group_id"],
                "label": row["label"],
                "text_sha256": hashlib.sha256(
                    normalize_text(row["text"]).encode("utf-8")
                ).hexdigest(),
            }
            for row in sorted(selected, key=lambda row: str(row["group_id"]))
        ],
    }


def apply_criteria(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    control_blocks: int,
) -> dict[str, Any]:
    """Apply the frozen acceptance criteria G1-G6 (plan §5)."""

    def fp_count(metrics: dict[str, Any]) -> int:
        return int(metrics["confusion_matrix"][0][1])

    def fn_count(metrics: dict[str, Any]) -> int:
        return int(metrics["confusion_matrix"][1][0])

    candidate_fp = fp_count(candidate_metrics)
    candidate_fn = fn_count(candidate_metrics)
    candidate_macro_f1 = float(candidate_metrics["macro_f1"])
    candidate_precision = float(candidate_metrics["per_class"]["1"]["precision"])

    checks = [
        {
            "id": "G1",
            "criterion": "validation benign false positives",
            "observed": candidate_fp,
            "baseline": fp_count(baseline_metrics),
            "required": f"<= {CRITERIA_LIMITS['max_false_positives']}",
            "passed": candidate_fp <= CRITERIA_LIMITS["max_false_positives"],
        },
        {
            "id": "G2",
            "criterion": "validation malicious false negatives",
            "observed": candidate_fn,
            "baseline": fn_count(baseline_metrics),
            "required": f"<= {CRITERIA_LIMITS['max_false_negatives']}",
            "passed": candidate_fn <= CRITERIA_LIMITS["max_false_negatives"],
        },
        {
            "id": "G3",
            "criterion": "macro F1",
            "observed": candidate_macro_f1,
            "baseline": float(baseline_metrics["macro_f1"]),
            "required": f">= {CRITERIA_LIMITS['min_macro_f1']}",
            "passed": candidate_macro_f1 >= CRITERIA_LIMITS["min_macro_f1"],
        },
        {
            "id": "G4",
            "criterion": "class-1 precision",
            "observed": candidate_precision,
            "baseline": float(baseline_metrics["per_class"]["1"]["precision"]),
            "required": f">= {CRITERIA_LIMITS['min_class1_precision']}",
            "passed": candidate_precision
            >= CRITERIA_LIMITS["min_class1_precision"],
        },
        {
            "id": "G5",
            "criterion": "positive control blocks",
            "observed": control_blocks,
            "baseline": CRITERIA_LIMITS["control_expected_blocks"],
            "required": (
                f"== {CRITERIA_LIMITS['control_expected_blocks']}"
            ),
            "passed": control_blocks
            == CRITERIA_LIMITS["control_expected_blocks"],
        },
    ]
    all_passed = all(check["passed"] for check in checks)
    return {
        "checks": checks,
        "all_passed": all_passed,
        "decision": "retain" if all_passed else "reject",
        "decision_note": (
            "G6 (integrity: baseline reproduced exactly, threshold 0.90 and "
            "policy band_v1 unchanged, test split never opened, aggregate-"
            "only outputs) is enforced by the run gates themselves; any G6 "
            "violation aborts the run before criteria are applied"
        ),
    }


# --- Data loading (pyarrow lazy) ---


def load_train_records(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(TRAIN_RECORD_COLUMNS))
    rows = table.to_pylist()
    for row in rows:
        if row.get("group_id") is None or row.get("text") is None:
            raise MitigationBlocker("train rows must carry group_id and text")
        if row.get("label") not in (0, 1):
            raise MitigationBlocker("train labels must be 0 or 1")
    return rows


def load_validation_texts(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=["text"]).column("text").to_pylist()


def load_dev_slice_norms(files: Sequence[Path]) -> set[str]:
    """Normalized prompts from the local eval sources (exclusion only)."""
    norms: set[str] = set()
    for path in files:
        if not path.is_file():
            raise MitigationBlocker(f"eval exclusion source missing: {path}")
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        text = record.get("instruction")
                        if isinstance(text, str):
                            norms.add(normalize_text(text))
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq

            column = pq.read_table(path, columns=["text"]).column("text")
            norms.update(normalize_text(text) for text in column.to_pylist())
        else:
            raise MitigationBlocker(
                f"unsupported eval exclusion source: {path.as_posix()}"
            )
    return norms


# --- Model steps (torch lazy) ---


def score_with_baseline(texts: Sequence[str], batch_size: int = 16) -> list[float]:
    import gc

    service = DebertaInference(
        BASELINE_DIR, threshold=DEFAULT_THRESHOLD, device="cpu"
    )
    service.load()
    logits = service.predict_logits_batch(
        list(texts), batch_size=batch_size, enforce_prompt_limit=False
    )
    import torch

    probabilities = (
        torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1)[:, 1]
        .tolist()
    )
    del service
    gc.collect()
    return probabilities


def train_candidate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The single pre-registered training run (plan §3.5)."""
    import gc

    import torch
    import transformers

    torch.set_num_threads(TRAINING["cpu_threads"])
    torch.manual_seed(TRAINING["seed"])
    random_generator = random.Random(TRAINING["seed"])

    tokenizer = transformers.AutoTokenizer.from_pretrained(BASELINE_DIR)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        BASELINE_DIR, num_labels=2
    )
    model.config.id2label = dict(LABEL_MAPPING)
    model.config.label2id = {
        label: index for index, label in LABEL_MAPPING.items()
    }
    model.to(torch.device(TRAINING["device"]))

    model.deberta.embeddings.requires_grad_(False)
    frozen_params = sum(
        parameter.numel()
        for parameter in model.deberta.embeddings.parameters()
    )
    if frozen_params == 0:
        raise MitigationBlocker(
            "input embeddings did not freeze; refusing to train outside the "
            "frozen plan"
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(parameter.numel() for parameter in trainable)
    if not trainable:
        raise MitigationBlocker("no trainable parameters after freezing")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )

    started = time.perf_counter()
    steps = 0
    epoch_losses: list[float] = []
    model.train()
    for epoch in range(TRAINING["epochs"]):
        order = list(range(len(rows)))
        random_generator.shuffle(order)
        running: list[float] = []
        for start in range(0, len(order), TRAINING["batch_size"]):
            batch_positions = order[start : start + TRAINING["batch_size"]]
            texts = [rows[index]["text"] for index in batch_positions]
            labels = torch.tensor(
                [rows[index]["label"] for index in batch_positions],
                dtype=torch.long,
            )
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=TRAINING["max_length"],
                return_tensors="pt",
            )
            optimizer.zero_grad()
            loss = model(**encoded, labels=labels).loss
            loss.backward()
            optimizer.step()
            running.append(float(loss.detach()))
            steps += 1
        epoch_losses.append(sum(running) / len(running))
    training_seconds = time.perf_counter() - started

    model.eval()
    model.save_pretrained(CANDIDATE_DIR)
    tokenizer.save_pretrained(CANDIDATE_DIR)
    del model, optimizer, tokenizer
    gc.collect()

    return {
        "steps": steps,
        "epochs": TRAINING["epochs"],
        "training_seconds": training_seconds,
        "epoch_mean_loss": epoch_losses,
        "loss_is_telemetry_not_a_success_criterion": True,
        "frozen_embedding_parameters": frozen_params,
        "trainable_parameters": trainable_params,
        "rows": len(rows),
        "hyperparameters": dict(TRAINING),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }


def run_positive_control(candidate_dir: Path) -> dict[str, Any]:
    control_texts = list(load_control_prompts(Path.cwd()))
    service = DebertaInference(
        candidate_dir, threshold=DEFAULT_THRESHOLD, device="cpu"
    )
    service.load()
    predictions = service.predict_batch(
        control_texts, batch_size=len(control_texts)
    )
    policy = GatewayPolicy()
    from promptshield.review_suggestions import matched_strong_marker_rules

    blocks = sum(
        policy.decide(
            prediction.malicious_probability,
            matched_strong_marker_rules(text),
        )[0]
        == "block"
        for text, prediction in zip(control_texts, predictions)
    )
    del service
    import gc

    gc.collect()
    return {
        "expected_blocks": CRITERIA_LIMITS["control_expected_blocks"],
        "actual_blocks": int(blocks),
        "passed": blocks == CRITERIA_LIMITS["control_expected_blocks"],
    }


# --- Orchestration ---


def run_baseline_audit_gate() -> dict[str, Any]:
    try:
        audit = audit_validation(
            VALIDATION_PATH,
            BASELINE_DIR,
            threshold=DEFAULT_THRESHOLD,
            batch_size=16,
        )
    except AuditFailure as error:
        raise MitigationBlocker(f"baseline audit failed: {error}") from error
    metrics = audit["comparison"]["threshold"]["metrics"]
    if metrics["confusion_matrix"] != EXPECTED_BASELINE_CM:
        raise MitigationBlocker(
            "fresh baseline audit did not reproduce the recorded confusion "
            f"matrix: got {metrics['confusion_matrix']}, expected "
            f"{EXPECTED_BASELINE_CM}"
        )
    return audit


def run_candidate_audit(candidate_dir: Path) -> dict[str, Any]:
    try:
        return audit_validation(
            VALIDATION_PATH,
            candidate_dir,
            threshold=DEFAULT_THRESHOLD,
            batch_size=16,
        )
    except AuditFailure as error:
        raise MitigationBlocker(f"candidate audit failed: {error}") from error


def write_report(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    stamp = report["created_utc"]
    json_path = reports_dir / f"hard-negative-mitigation-{stamp}.json"
    md_path = reports_dir / f"hard-negative-mitigation-{stamp}.md"
    json_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(format_markdown(report), encoding="utf-8")
    return json_path, md_path


def format_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# PromptShield benign hard-negative mitigation experiment",
        "",
        f"Single pre-registered run of the approved plan "
        f"[{PLAN_DOCUMENT}]({PLAN_DOCUMENT}). Aggregate values only; no "
        "prompt text is included. `test_split_loaded`: `false`. The baseline "
        "checkpoint and its reports were not modified.",
        "",
        "## Gates",
        "",
        f"- Free RAM at precondition check: "
        f"{report['gates']['free_ram_bytes'] / 2**30:.2f} GiB "
        f"(required >= {MIN_FREE_RAM_BYTES / 2**30:.2f} GiB): "
        f"**{report['gates']['free_ram_passed']}**",
        f"- Baseline audit reproduced recorded confusion matrix "
        f"{EXPECTED_BASELINE_CM}: "
        f"**{report['gates']['baseline_audit_reproduced']}**",
        f"- Training subset counts match the plan (138/138/138, total 414): "
        f"**{report['gates']['training_set_matches_plan']}**",
        f"- Train file sha256 verified: "
        f"**{report['gates']['train_sha256_verified']}**",
        f"- Positive control (candidate): "
        f"{report['evaluation']['positive_control']['actual_blocks']}/4 "
        f"blocked",
        "",
        "## Training subset (metadata only)",
        "",
    ]
    for name, size in report["training_subset"]["pool_sizes"].items():
        lines.append(f"- {name}: pool {size}")
    for reason, count in report["training_subset"]["exclusion_counters"].items():
        lines.append(f"  - excluded `{reason}`: {count}")
    lines.append(
        f"- selected: {report['training_subset']['total_selected']} rows "
        f"(selection: {report['training_subset']['selection']}); manifest: "
        f"`{HARDNEG_MANIFEST_PATH.as_posix()}`"
    )
    training = report["training"]
    lines += [
        "",
        "## Training (one run, no retries)",
        "",
        f"- rows {training['rows']}, steps {training['steps']}, epochs "
        f"{training['epochs']}, lr {training['hyperparameters']['learning_rate']}, "
        f"seed {training['hyperparameters']['seed']}, batch "
        f"{training['hyperparameters']['batch_size']}, frozen embedding "
        f"parameters {training['frozen_embedding_parameters']}, trainable "
        f"parameters {training['trainable_parameters']}",
        f"- epoch mean loss (telemetry only, not a success criterion): "
        f"{[round(value, 4) for value in training['epoch_mean_loss']]}",
        f"- wall time {training['training_seconds']:.1f} s on "
        f"{training['python']} / torch {training['torch']}",
        f"- candidate artifact: `{report['evaluation']['candidate_checkpoint']}` "
        f"(baseline: `{report['evaluation']['baseline_checkpoint']}`, "
        "unchanged)",
        "",
        "## Validation comparison (threshold 0.90, identical protocol)",
        "",
        "| Metric | Baseline | Candidate |",
        "|---|---:|---:|",
    ]
    baseline = report["evaluation"]["baseline_threshold_metrics"]
    candidate = report["evaluation"]["candidate_threshold_metrics"]
    for label, key in (
        ("Benign FPR", "false_positive_rate_benign"),
        ("Malicious FNR", "false_negative_rate_malicious"),
        ("Macro F1", "macro_f1"),
        ("Class-1 precision", "per_class_1_precision"),
    ):
        base_value = (
            baseline["per_class"]["1"]["precision"]
            if key == "per_class_1_precision"
            else baseline[key]
        )
        cand_value = (
            candidate["per_class"]["1"]["precision"]
            if key == "per_class_1_precision"
            else candidate[key]
        )
        lines.append(
            f"| {label} | {base_value:.4f} | {cand_value:.4f} |"
        )
    lines += [
        f"| Confusion matrix | `{baseline['confusion_matrix']}` | "
        f"`{candidate['confusion_matrix']}` |",
        "",
        "## Acceptance criteria (pre-registered, plan §5)",
        "",
        "| ID | Criterion | Observed | Baseline | Required | Passed |",
        "|---|---|---:|---:|---|---|",
    ]
    for check in report["acceptance_criteria"]["checks"]:
        lines.append(
            f"| {check['id']} | {check['criterion']} | {check['observed']} | "
            f"{check['baseline']} | {check['required']} | "
            f"**{check['passed']}** |"
        )
    lines += [
        "",
        f"## Decision: **{report['acceptance_criteria']['decision'].upper()}**",
        "",
        report["acceptance_criteria"]["decision_note"],
        "",
        "## Integrity",
        "",
    ]
    for key, value in report["integrity"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Limitations", "", report["limitations"], ""]
    return "\n".join(lines)


def run() -> dict[str, Any]:
    ensure_paths_frozen(TRAIN_PATH, VALIDATION_PATH, CANDIDATE_DIR)
    ensure_candidate_dir_absent(CANDIDATE_DIR)
    created_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    # Gate 1 (plan §3.6 / approval condition 2): memory precondition.
    free_bytes = free_physical_memory_bytes()
    check_free_ram(free_bytes)

    # Gate 2 (approval condition 4): fresh baseline audit must reproduce.
    baseline_audit = run_baseline_audit_gate()
    baseline_metrics = baseline_audit["comparison"]["threshold"]["metrics"]

    # Gate 3 (plan §3.1-3.4 / approval condition 5): data gates.
    train_sha = sha256_file(TRAIN_PATH)
    if train_sha != TRAIN_SHA256:
        raise MitigationBlocker(
            f"train.parquet sha256 mismatch: {train_sha}"
        )
    baseline_safetensors_sha_before = sha256_file(
        BASELINE_DIR / "model.safetensors"
    )
    if baseline_safetensors_sha_before != BASELINE_SAFETENSORS_SHA256:
        raise MitigationBlocker(
            "baseline checkpoint hash does not match the audited artifact"
        )

    records = load_train_records(TRAIN_PATH)
    validation_texts = load_validation_texts(VALIDATION_PATH)
    validation_reference = build_reference_index(validation_texts)
    dev_norms = load_dev_slice_norms(DEV_EVAL_FILES)

    label0_texts = [row["text"] for row in records if row["label"] == 0]
    label0_ids = [row["group_id"] for row in records if row["label"] == 0]
    probabilities = score_with_baseline(label0_texts)
    scores = dict(zip(label0_ids, probabilities))

    pools, counters = build_pools(records, scores, validation_reference, dev_norms)
    selected_hn = select_pool("hard_negatives", pools["hard_negatives"], HARD_NEGATIVE_COUNT)
    selected_mal = select_pool("anchor_malicious", pools["anchor_malicious"], ANCHOR_COUNT)
    selected_easy = select_pool("anchor_easy_benign", pools["anchor_easy_benign"], ANCHOR_COUNT)
    training_rows = selected_hn + selected_mal + selected_easy
    if len(training_rows) != TOTAL_TRAINING_ROWS:
        raise MitigationBlocker(
            f"assembled {len(training_rows)} training rows, expected "
            f"{TOTAL_TRAINING_ROWS}"
        )

    manifest = hardneg_manifest(
        training_rows, pools, counters, train_sha, len(records), created_utc
    )
    HARDNEG_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # Gate 4 (approval conditions 3 and 6): the single training run.
    training = train_candidate(training_rows)

    # Gate 5 (approval condition 7): identical-protocol audits.
    candidate_audit = run_candidate_audit(CANDIDATE_DIR)
    candidate_metrics = candidate_audit["comparison"]["threshold"]["metrics"]
    control = run_positive_control(CANDIDATE_DIR)

    baseline_safetensors_sha_after = sha256_file(
        BASELINE_DIR / "model.safetensors"
    )
    baseline_untouched = (
        baseline_safetensors_sha_after == baseline_safetensors_sha_before
    )
    if not baseline_untouched:
        raise MitigationBlocker("baseline checkpoint changed during the run")

    # Criteria G1-G5; G6 is enforced by the gates above.
    acceptance = apply_criteria(baseline_metrics, candidate_metrics, control["actual_blocks"])

    report = {
        "experiment": "benign_hard_negative_mitigation",
        "plan": PLAN_DOCUMENT,
        "created_utc": created_utc,
        "single_run": True,
        "test_split_loaded": False,
        "gates": {
            "free_ram_bytes": free_bytes,
            "free_ram_passed": True,
            "baseline_audit_reproduced": True,
            "baseline_confusion_matrix": baseline_metrics["confusion_matrix"],
            "train_sha256_verified": True,
            "training_set_matches_plan": (
                len(pools["hard_negatives"]) >= HARD_NEGATIVE_COUNT
                and len(pools["anchor_malicious"]) >= ANCHOR_COUNT
                and len(pools["anchor_easy_benign"]) >= ANCHOR_COUNT
            ),
        },
        "training_subset": {
            "pool_sizes": {name: len(rows) for name, rows in pools.items()},
            "exclusion_counters": dict(sorted(counters.items())),
            "selection": (
                "sha256(normalized text) ascending, first 138 per pool"
            ),
            "total_selected": len(training_rows),
            "manifest": HARDNEG_MANIFEST_PATH.as_posix(),
        },
        "training": training,
        "evaluation": {
            "decision_protocol": (
                "softmax class-1 probability >= 0.90, identical for both "
                "checkpoints; gateway policy band_v1 untouched"
            ),
            "baseline_checkpoint": BASELINE_DIR.as_posix(),
            "candidate_checkpoint": CANDIDATE_DIR.as_posix(),
            "baseline_threshold_metrics": baseline_metrics,
            "candidate_threshold_metrics": candidate_metrics,
            "baseline_argmax_metrics": baseline_audit["comparison"]["argmax"][
                "metrics"
            ],
            "candidate_argmax_metrics": candidate_audit["comparison"]["argmax"][
                "metrics"
            ],
            "positive_control": control,
        },
        "acceptance_criteria": acceptance,
        "integrity": {
            "baseline_safetensors_sha256": baseline_safetensors_sha_after,
            "baseline_untouched": baseline_untouched,
            "candidate_artifact_files": hash_checkpoint_files(CANDIDATE_DIR),
            "threshold_changed": False,
            "gateway_policy_changed": False,
            "test_parquet_opened": False,
            "raw_prompts_in_report": False,
            "raw_prompts_on_console": False,
        },
        "limitations": (
            "Validation-slice comparison only: the 200-example benign-chat "
            "slice and the test split were not used as evidence, and the "
            "candidate cannot be claimed to fix the public-benign "
            "generalization gap (that slice is development data). Training "
            "used crowd-labeled hard negatives from one dataset; a retained "
            "candidate still needs an independent holdout before any "
            "performance claim beyond this validation comparison."
        ),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="reports")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run()
        json_path, md_path = write_report(report, Path(args.reports_dir))
    except MitigationBlocker as blocker:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "plan": PLAN_DOCUMENT,
                    "detail": str(blocker),
                    "test_split_loaded": False,
                },
                indent=2,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "plan": PLAN_DOCUMENT,
                "decision": report["acceptance_criteria"]["decision"],
                "gates": report["gates"],
                "training_subset": {
                    "pool_sizes": report["training_subset"]["pool_sizes"],
                    "total_selected": report["training_subset"]["total_selected"],
                },
                "training_steps": report["training"]["steps"],
                "baseline_confusion_matrix": report["evaluation"][
                    "baseline_threshold_metrics"
                ]["confusion_matrix"],
                "candidate_confusion_matrix": report["evaluation"][
                    "candidate_threshold_metrics"
                ]["confusion_matrix"],
                "acceptance_criteria": report["acceptance_criteria"],
                "reports": [json_path.as_posix(), md_path.as_posix()],
                "test_split_loaded": False,
                "raw_prompts_printed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
