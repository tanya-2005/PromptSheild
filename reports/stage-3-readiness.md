# Stage 3 baseline-readiness audit

This report records verified facts, proposed modeling decisions, and unresolved
questions. It does not contain model results.

## Verified workspace facts

- The workspace is not a Git repository. No Git commands that modify state were
  run, and no commit was created.
- The local raw files are `data/raw/train.parquet`,
  `data/raw/validation.parquet`, and `data/raw/test.parquet`.
- `data/raw/` was already excluded from `.gitignore`. The environment,
  model-artifact, and cache exclusions were added during this audit.
- The current `.venv` contains Python 3.14 and `pip`, but not `pyarrow`.
  Pylance also reports `pyarrow` as unresolved. Therefore this audit did not
  re-read the Parquet files; installing dependencies requires explicit approval.

The values below are the prior recorded inspection in `data/manifest.json`, not
new reads performed in the current environment:

| Split | Rows | Label 0 | Label 1 | Missing groups | Exact duplicate rows |
|---|---:|---:|---:|---:|---:|
| train | 4,391 | 1,741 | 2,650 | 0 | 0 |
| validation | 941 | 407 | 534 | 0 | 0 |
| test | 942 | 390 | 552 | 0 | 0 |

The recorded manifest reports zero normalized exact-text overlap and zero group
overlap for every split pair. It also reports 23 cross-split near-duplicate
candidates. These remain review signals, not confirmed leakage. The existing
report preserves source files and does not delete candidates automatically.

## Data-quality interpretation

The numeric label is the target; `category`, `source`, `severity`, `group_id`,
`augmented`, and `tags` are metadata. The manifest shows multiple sources and
categories in both classes, so metadata must not be used as model input.
`augmented: false` is a dataset field, not independent proof that examples are
real-world observations. A held-out split is also not an independent external
benchmark.

Before training, the Parquet audit should be rerun after `pyarrow` is installed
and should explicitly verify string text values, empty text, null metadata,
invalid labels, row counts, and the current near-duplicate candidate count.
The raw files must remain unchanged.

## Baseline protocol

1. Use only `text` as the feature and the existing binary `label` as the
   target. Do not pass metadata columns to the vectorizer.
2. Fit one scikit-learn `Pipeline` on the train split:
   `TfidfVectorizer` followed by `LogisticRegression`.
3. Start with lowercase text, default Unicode handling, word n-grams `(1, 2)`,
   `min_df=2`, `max_df=0.98`, and a bounded `max_features` only if memory
   requires it. Start with `C=1.0`, `solver="liblinear"`, and a fixed
   `random_state=42`. These are reproducible starting points, not optimized
   claims.
4. The Stage 5 baseline uses the default probability threshold of `0.5`.
   Validation was used for the first baseline evaluation; no threshold
   optimization was performed.
5. The test split remains untouched. It must be evaluated only after the
   pipeline and any threshold decisions are explicitly frozen.
6. The fitted pipeline, configuration, and input-file hashes are saved in
   `models/baseline.joblib` and `reports/baseline-validation.json`; both
   generated outputs are ignored by Git.

The positive class is label `1` (potentially malicious). Report precision,
recall, and F1 for both classes, macro F1, confusion matrix, support, benign
false-positive rate (`FP / (FP + TN)`), malicious false-negative rate
(`FN / (FN + TP)`), and latency measured over a stated batch and warm-up
procedure. Accuracy alone is insufficient because the costs of missed attacks
and false alarms differ.

For error analysis, record a stable split/row reference, true and predicted
labels, score, and metadata such as source/category for analysis only. Keep
long or sensitive text out of reports; use a short redacted description of the
error pattern.

## Diagnostic-set design (not created yet)

The diagnostic set should be a separate, manually authored JSONL file under a
future ignored path such as `data/diagnostic/`. It must not be merged with
`data/raw/` or any model split. No examples were created in this stage.

Each record must contain:

```json
{
  "id": "diag-v1-001",
  "text": "newly authored text",
  "label": 0,
  "category": "ordinary_benign",
  "rationale": "why the label is defensible",
  "difficulty": "easy",
  "authoring_provenance": "PromptShield-authored",
  "version": "v1",
  "review_status": "approved"
}
```

Allowed categories should cover ordinary benign requests, difficult benign
security discussion, direct overrides, protected-instruction extraction,
indirect-injection-style content, obfuscation, long benign documents, and
ambiguous cases. Examples should vary length and wording so that label is not
revealed by superficial shortcuts.

Records under `review_status: "review"` must have a null label and are excluded
from binary evaluation until a human resolves them. The new
`promptshield.diagnostic_validation` utility enforces this contract, rejects
duplicate IDs and invalid labels, and never silently drops malformed records.
Examples must be newly authored, individually reviewed, and not copied from
NeurAlchemy, BIPIA, InjecAgent, or another public benchmark.

This set is a targeted diagnostic, not an independent benchmark. It is subject
to author bias, small-sample uncertainty, and limited generalization. It must
not be used for threshold tuning if it is presented as a final diagnostic
result.

## Stage 5 status

- The baseline was trained on the training split and evaluated on validation.
- Validation results are recorded in `reports/baseline-validation.json`.
- The test split was not loaded.
- No external benchmark was downloaded.
- No raw Parquet file or split content was changed.
- No diagnostic examples were created.
- No test-set performance metric exists yet.
