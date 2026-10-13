"""Reproducible TF-IDF and Linear SVM comparison experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from promptshield.baseline import (
    FEATURE_COLUMN,
    TARGET_COLUMN,
    calculate_metrics,
    load_labeled_parquet,
    measure_latency,
)

SEED = 42
PIPELINE_CONFIG = {
    "tfidf": {
        "lowercase": True,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.98,
    },
    "linear_svm": {
        "C": 1.0,
        "random_state": SEED,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pipeline() -> Pipeline:
    """Build the unfitted text-only TF-IDF and Linear SVM pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**PIPELINE_CONFIG["tfidf"])),
        ("classifier", LinearSVC(**PIPELINE_CONFIG["linear_svm"])),
    ])


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
    report = {
        "experiment": "tfidf_linear_svm_comparison",
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
        "metrics": calculate_metrics(validation_labels, predictions),
        "latency": measure_latency(pipeline, validation_texts),
        "test_split_loaded": False,
        "limitations": [
            "Validation metrics are not final held-out test metrics.",
            "This text classifier is not a production security guarantee.",
            "The primary dataset is not an independent real-world benchmark.",
            "LinearSVC decision scores are not calibrated probabilities.",
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
    import joblib

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
