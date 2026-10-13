# PromptShield validation robustness benchmark

This report evaluates controlled, local transformations on the validation split only.
It is not an independent test result and contains no prompt text.

- Evaluation time (UTC): `2026-10-10T20:15:07.499476+00:00`
- Validation rows: `941`
- Seed: `42`
- Threshold: `0.9`

## Results

| Model | Status | Transformation | Category | Examples | Accuracy | Macro F1 | Benign FPR | Malicious recall | Flip rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | available | clean validation | reference | 941 | 0.9416 | 0.9397 | 0.1130 | 0.9831 | — |
| logistic_regression | available | whitespace_variation | candidate_label_preserving | 941 | 0.9416 | 0.9397 | 0.1130 | 0.9831 | 0.0000 |
| logistic_regression | available | harmless_formatting | candidate_label_preserving | 941 | 0.9416 | 0.9397 | 0.1130 | 0.9831 | 0.0000 |
| logistic_regression | available | neutral_context | exploratory | 941 | 0.8714 | 0.8620 | 0.2948 | 0.9981 | 0.0871 |
| linear_svm | available | clean validation | reference | 941 | 0.9564 | 0.9554 | 0.0737 | 0.9794 | — |
| linear_svm | available | whitespace_variation | candidate_label_preserving | 941 | 0.9564 | 0.9554 | 0.0737 | 0.9794 | 0.0000 |
| linear_svm | available | harmless_formatting | candidate_label_preserving | 941 | 0.9564 | 0.9554 | 0.0737 | 0.9794 | 0.0000 |
| linear_svm | available | neutral_context | exploratory | 941 | 0.8502 | 0.8371 | 0.3440 | 0.9981 | 0.1275 |
| deberta | unavailable | — | — | — | — | — | — | — | — |

Clean rows and every transformed row count are 941 in this run.
Clean metrics use the model's ordinary prediction convention. Transformed metrics use the copied validation labels; flip rate compares predictions by stable row order and identifier.

## Limitations

- All results use the validation split and are not independent test results.
- Transformations are controlled perturbations, not a complete adversarial threat model.
- Neutral-context results are exploratory because the added context may alter meaning.
- Candidate label-preserving transformations still have documented edge cases.
- Prediction flips indicate sensitivity, not necessarily a label error.
- No transformed copies were used for training.
- Prompt text is intentionally excluded from this report.
