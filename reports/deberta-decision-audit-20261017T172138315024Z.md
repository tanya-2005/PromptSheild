# PromptShield DeBERTa validation decision audit

This report contains validation metadata and aggregate metrics only; it does not contain prompt text.

- Validation rows: `941`
- Evaluation time (UTC): `2026-10-17T17:21:37.447219+00:00`
- Threshold: `0.9`
- Decision disagreements: `15` (0.015940)

## Decision-rule metrics

| Rule | Accuracy | Macro F1 | Benign FPR | Malicious FNR | Confusion matrix |
|---|---:|---:|---:|---:|---|
| argmax | 0.938363 | 0.936155 | 0.130221 | 0.009363 | `[[354, 53], [5, 529]]` |
| threshold | 0.945802 | 0.944189 | 0.103194 | 0.016854 | `[[365, 42], [9, 525]]` |

Per-class precision, recall, and F1 are available in the JSON report under each rule's `metrics.per_class` field.

## Provenance limitation

The checkpoint file hashes establish artifact identity from the time of this audit onward. They do not, by themselves, prove which historical training run produced the checkpoint.
