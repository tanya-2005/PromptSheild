"""Reproducible TF-IDF and Logistic Regression baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import pyarrow.parquet as parquet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

FEATURE_COLUMN = "text"
TARGET_COLUMN = "label"
REQUIRED_COLUMNS = {FEATURE_COLUMN, TARGET_COLUMN}
SEED = 42
PIPELINE_CONFIG = {
    "tfidf": {
        "lowercase": True,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.98,
    },
    "logistic_regression": {
        "C": 1.0,
        "solver": "liblinear",
        "random_state": SEED,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labeled_parquet(path: Path) -> tuple[list[str], list[int]]:
    """Load one split and reject malformed rows without silently dropping them."""
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset split: {path}")
    table = parquet.read_table(path)
    columns = set(table.column_names)
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    records = table.select([FEATURE_COLUMN, TARGET_COLUMN]).to_pylist()
    texts: list[str] = []
    labels: list[int] = []
    errors: list[str] = []
    for row_number, record in enumerate(records, start=1):
        text = record[FEATURE_COLUMN]
        label = record[TARGET_COLUMN]
        row_errors: list[str] = []
        if not isinstance(text, str) or not text.strip():
            row_errors.append("text must be a non-empty string")
        if label not in (0, 1):
            row_errors.append("label must be 0 or 1")
        if row_errors:
            errors.append(f"row {row_number}: {', '.join(row_errors)}")
            continue
        texts.append(text)
        labels.append(int(label))
    if errors:
        raise ValueError(f"{path} contains invalid rows: {'; '.join(errors)}")
    if not texts:
        raise ValueError(f"{path} contains no valid rows")
    return texts, labels


def build_pipeline() -> Pipeline:
    """Build the fixed baseline pipeline without fitting it."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**PIPELINE_CONFIG["tfidf"])),
        ("classifier", LogisticRegression(
            **PIPELINE_CONFIG["logistic_regression"]
        )),
    ])


def calculate_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict[str, Any]:
    """Calculate binary metrics using label 1 as the positive class."""
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    benign_total = false_positive + true_negative
    malicious_total = false_negative + true_positive
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "per_class": {
            "0": {
                "precision": float(precision_score(
                    labels, predictions, pos_label=0, zero_division=0
                )),
                "recall": float(recall_score(
                    labels, predictions, pos_label=0, zero_division=0
                )),
                "f1": float(f1_score(
                    labels, predictions, pos_label=0, zero_division=0
                )),
                "support": int(sum(label == 0 for label in labels)),
            },
            "1": {
                "precision": float(precision_score(
                    labels, predictions, pos_label=1, zero_division=0
                )),
                "recall": float(recall_score(
                    labels, predictions, pos_label=1, zero_division=0
                )),
                "f1": float(f1_score(
                    labels, predictions, pos_label=1, zero_division=0
                )),
                "support": int(sum(label == 1 for label in labels)),
            },
        },
        "macro_f1": float(f1_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "confusion_matrix": matrix.tolist(),
        "support": {"0": int(benign_total), "1": int(malicious_total)},
        "false_positive_rate_benign": (
            float(false_positive / benign_total) if benign_total else None
        ),
        "false_negative_rate_malicious": (
            float(false_negative / malicious_total)
            if malicious_total else None
        ),
    }


def measure_latency(pipeline: Pipeline, texts: list[str]) -> dict[str, Any]:
    """Measure warm inference over the validation texts, excluding warm-up."""
    warmup_texts = texts[: min(10, len(texts))]
    pipeline.predict(warmup_texts)
    repetitions = 5
    start = time.perf_counter()
    for _ in range(repetitions):
        pipeline.predict(texts)
    elapsed = time.perf_counter() - start
    calls = repetitions * len(texts)
    return {
        "method": (
            "One warm-up prediction on up to 10 validation texts, followed by "
            f"{repetitions} full validation-batch predictions using "
            "time.perf_counter; reporting mean wall-clock time per text."
        ),
        "warmup_examples": len(warmup_texts),
        "measured_examples": len(texts),
        "repetitions": repetitions,
        "mean_ms_per_text": float(elapsed / calls * 1000),
    }


def train_and_validate(
    train_path: Path,
    validation_path: Path,
    artifact_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Fit on train, evaluate on validation, and persist the experiment."""
    train_texts, train_labels = load_labeled_parquet(train_path)
    validation_texts, validation_labels = load_labeled_parquet(validation_path)
    pipeline = build_pipeline()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(validation_texts).tolist()
    scores = calculate_metrics(validation_labels, predictions)
    report = {
        "experiment": "tfidf_logistic_regression_baseline",
        "feature_column": FEATURE_COLUMN,
        "target_column": TARGET_COLUMN,
        "excluded_feature_columns": [
            "category", "source", "severity", "group_id", "augmented", "tags"
        ],
        "seed": SEED,
        "pipeline_config": PIPELINE_CONFIG,
        "train": {
            "path": train_path.as_posix(),
            "sha256": _sha256(train_path),
            "rows": len(train_texts),
        },
        "validation": {
            "path": validation_path.as_posix(),
            "sha256": _sha256(validation_path),
            "rows": len(validation_texts),
        },
        "metrics": scores,
        "latency": measure_latency(pipeline, validation_texts),
        "test_split_loaded": False,
        "limitations": [
            "Validation metrics are not final held-out test metrics.",
            "This text classifier is not a production security guarantee.",
            "The primary dataset is not an independent real-world benchmark.",
        ],
    }
    artifact = {
        "pipeline": pipeline,
        "experiment": report["experiment"],
        "feature_column": FEATURE_COLUMN,
        "target_column": TARGET_COLUMN,
        "seed": SEED,
        "pipeline_config": PIPELINE_CONFIG,
        "train_sha256": report["train"]["sha256"],
        "validation_sha256": report["validation"]["sha256"],
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = train_and_validate(
        args.train, args.validation, args.artifact, args.report
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
