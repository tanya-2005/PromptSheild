# PromptShield validation model comparison

Validation-only aggregate comparison; no prompt text is included.

- Validation rows: `941`

## Metrics

| Model | Accuracy | Macro F1 | Benign FPR | Malicious FNR | Mean ms/text |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.941552 | 0.939721 | 0.11302211302211303 | 0.016853932584269662 | 0.02881849096671309 |
| linear_svm | 0.956429 | 0.955355 | 0.07371007371007371 | 0.020599250936329586 | 0.019720191286686773 |
| deberta | 0.945802 | 0.944189 | 0.10319410319410319 | 0.016853932584269662 | 308.83828862909826 |

## Decision rules

- Logistic Regression and Linear SVM use their serialized pipeline.predict numeric class decisions.
- DeBERTa uses class-1 softmax output >= 0.90, matching the current inference adapter.
- LinearSVC decision_function values are not probabilities.
- No threshold was tuned and no model was retrained.

## Limitations

- All metrics are validation-only and do not establish held-out test performance.
- Latency depends on hardware, runtime, batch behavior, and process state; these measurements are not a production capacity benchmark.
- The three models have different score semantics and preprocessing/runtime costs, so score magnitudes are not directly comparable.
- Checkpoint and artifact identity do not prove historical training provenance.
