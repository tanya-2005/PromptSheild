# DeBERTa source false-positive review worksheet

This tracked worksheet lists only row identifiers, dataset metadata,
and model scores. It contains no prompt text. The full text lives in
the local-only packet, which is git-ignored and must not be committed.

- Source under review: `wildguard_judgecomp`
- Decision threshold: `0.9` (unmodified)
- Rows to review: `27`
- Review status: pending until a human fills in every row of the
  local packet.

Review labels (assign exactly one per row, with evidence):

- `likely_mislabeled`: The text actually contains an injection, jailbreak, or instruction override, so the dataset's benign label appears wrong and the model's malicious flag may be justified.
- `genuine_model_false_positive`: The text is a legitimate benign request and the benign label is correct; the model over-reacted to wording, topic, or formatting.
- `ambiguous_insufficient_context`: The isolated text is not enough to judge; the decision depends on context that is not present in this record.
- `other`: None of the above descriptions fit; the evidence note explains why.

Annotation guide:

1. Read the full prompt text in the local packet, then choose exactly one label that best explains why the model flagged this benign-labeled row.
2. For likely_mislabeled, quote the specific span that shows an injection, jailbreak, or override attempt.
3. For genuine_model_false_positive, name the lexical or topical trigger that likely misled the model into flagging a benign request.
4. For ambiguous_insufficient_context, state exactly what is missing, such as conversation history, a system prompt, or the dataset rubric.
5. Every decision requires a short, evidence-based note citing either the text or the missing context. Do not decide from metadata alone.
6. Record decisions in the local review packet. Do not edit dataset labels, model predictions, the threshold, or any trained artifact.

| Validation row | Category | Severity | Malicious probability | Decision |
|---:|---|---|---:|---|
| 15 | benign |  | 0.999482 | pending |
| 18 | benign |  | 0.999209 | pending |
| 19 | benign |  | 0.999456 | pending |
| 20 | benign |  | 0.999269 | pending |
| 21 | benign |  | 0.999485 | pending |
| 22 | benign |  | 0.999411 | pending |
| 23 | benign |  | 0.999412 | pending |
| 24 | benign |  | 0.999512 | pending |
| 26 | benign |  | 0.999329 | pending |
| 27 | benign |  | 0.999345 | pending |
| 29 | benign |  | 0.999592 | pending |
| 36 | benign |  | 0.999613 | pending |
| 38 | benign |  | 0.999614 | pending |
| 39 | benign |  | 0.999613 | pending |
| 41 | benign |  | 0.999612 | pending |
| 42 | benign |  | 0.999612 | pending |
| 43 | benign |  | 0.999619 | pending |
| 45 | benign |  | 0.982664 | pending |
| 46 | benign |  | 0.924893 | pending |
| 47 | benign |  | 0.971383 | pending |
| 48 | benign |  | 0.969945 | pending |
| 49 | benign |  | 0.955425 | pending |
| 50 | benign |  | 0.986090 | pending |
| 53 | benign |  | 0.992374 | pending |
| 56 | benign |  | 0.982272 | pending |
| 58 | benign |  | 0.977676 | pending |
| 59 | benign |  | 0.997123 | pending |

This worksheet records review decisions only. It does not change
dataset labels, model predictions, the threshold, or any artifact.
