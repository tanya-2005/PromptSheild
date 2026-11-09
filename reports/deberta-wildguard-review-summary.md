# DeBERTa source false-positive review summary

Aggregate decision counts only. This summary contains no prompt text
and no reviewer evidence verbatim.

- Source under review: `wildguard_judgecomp`
- Decision threshold: `0.9` (unmodified)
- Total rows: `27`
- Decided rows: `0`
- Pending rows: `27`
- Review complete: `false`

Decision counts:

- `likely_mislabeled`: 0
- `genuine_model_false_positive`: 0
- `ambiguous_insufficient_context`: 0
- `other`: 0

- Invalid decisions: `0`
- Decided rows missing evidence: `0`

Review is incomplete. No conclusion about label noise versus model error can be drawn until a human decides every row.

Limitations:

- This summary aggregates reviewer decisions only; it contains no prompt text and no reviewer evidence verbatim.
- Decision counts are review opinions, not corrected labels. No example was relabeled and no model artifact was changed.
- Review outcomes are validation-only and do not establish held-out test performance.

This summary records review outcomes only. It does not change labels,
predictions, the threshold, or any trained artifact.
