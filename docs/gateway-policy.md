# PromptShield protective gateway policy (design note)

## Experimental hybrid mode

The API supports an explicitly selectable `gateway_mode="hybrid"` for local
experimentation. It loads the existing DeBERTa checkpoint and
`models/linear-svm.joblib` once at startup; it does not retrain or modify either
artifact. DeBERTa's review band remains `review`. Outside that band, agreement
blocks or allows and disagreement returns `review`; strong-marker precedence is
unchanged. The default `gateway_mode="deberta"` remains the production path.

This mode is experimental only. It has not been validated on the held-out test
split and is not a production-readiness claim. The module-level API app uses
DeBERTa by default; select the experiment programmatically with
`create_app(gateway_mode="hybrid")`. No thresholds or model artifacts change.

### Measured comparison

The following results use the exact same frozen 200-example benign-chat set
(100 Dolly and 100 OASST1) and the existing 941-row validation dataset. The
200-example set was used during hybrid development and is therefore not an
independent final evaluation.

| Gateway | Benign allow | Benign review | Benign block | Benign block rate |
|---|---:|---:|---:|---:|
| DeBERTa | 30 | 8 | 162 | 81% |
| Hybrid | 21 | 81 | 98 | 49% |

| Validation metric | DeBERTa | Hybrid |
|---|---:|---:|
| Allow / review / block | 363 / 11 / 567 | 358 / 42 / 541 |
| Accuracy | 94.58% | 95.86% |
| Macro F1 | 94.42% | 95.77% |
| Benign false-positive rate | 10.32% | 5.65% |
| Malicious false-negative rate | 1.69% | 3.00% |

The hybrid reduces hard blocks and validation false positives, but increases
review volume and malicious false negatives. The existing malicious check had
only four positive controls; both gateways blocked all four. That small control
set cannot establish general malicious detection performance. The held-out
test split and its labels were not accessed.

Status: proposed policy, documented before implementation. This note fixes the
gateway decision rule, its rationale, its failure behaviour, and its known
costs before any gateway code is written. It changes no classifier behaviour:
the saved model, the `0.90` decision threshold, and the existing `/predict`
contract are untouched.

## Scope

The gateway is a local, single-prompt protective wrapper around the existing
DeBERTa inference service. It accepts a prompt and returns exactly one of
three decisions: `allow`, `review`, or `block`. It is a **risk triage layer**,
not a security guarantee: it cannot prove that an allowed prompt is free of
injection, and every figure below is validation-only.

Inputs to the decision, and nothing else:

1. The class-1 model score `s` from the unchanged inference path
   (`DebertaInference.predict`, threshold `0.90`, unmodified).
2. The set of strong injection-marker rule IDs matched by the existing
   deterministic lexical rules in `promptshield.review_suggestions` (the same
   rules that already power the human-review triage suggestions). The model's
   score plays no role in those rules, and the rules play no role in the
   model's score, so the two signals are independent.

Dataset metadata (`source`, `category`, `severity`, `augmented`) is
deliberately **not** an input: the source-reliability audit established that
these fields are absent from the deployed request contract and are offline
diagnostics only, so no source-based rule is implementable at inference time.

## Decision rule (band policy `band_v1`)

Defaults, fixed in source: `allow_max = 0.70`, `block_min = 0.90`
(`0 < allow_max < block_min < 1` is validated at construction).

| Condition (evaluated in order) | Decision | Reason code |
|---|---|---|
| `s >= block_min` (0.90) | `block` | `score_at_or_above_block_min` |
| `allow_max <= s < block_min` (0.70–0.90) | `review` | `score_in_review_band` |
| `s < allow_max` and at least one strong marker matched | `review` | `strong_injection_marker_matched` |
| `s < allow_max` and no strong marker matched | `allow` | `score_below_allow_max` |

`block_min` defaults to exactly the deployed classifier threshold, so the
gateway blocks precisely what the classifier already calls `malicious`; no
threshold is moved. A separate `classifier_decision` field is echoed in every
response so the gateway decision is never confused with the raw prediction.

Strong markers escalate `allow` to `review` only — they never block. The
review-suggestions engine documents that rules cannot tell "mention" from
"use" (e.g. educational text about injection phrasing), so blocking on a
lexical match alone would create avoidable false positives. Weak marker rules
do not escalate: they are documented as too ambiguous to support any
conclusion, and escalating on them would flood review capacity with noise.

## Rationale and measured trade-offs (validation-only)

All figures reuse the 941-row validation aggregates in
[reports/deberta-confidence-calibration-20261018T140720575767Z.md](../reports/deberta-confidence-calibration-20261018T140720575767Z.md)
and
[reports/source-reliability-policy-20261018T153548645826Z.md](../reports/source-reliability-policy-20261018T153548645826Z.md).
They do not establish held-out test performance.

- **`block` (s >= 0.90).** The `[0.90, 1.00]` band held 567 of 941 validation
  rows, of which 92.6% were genuinely malicious, so blocking preserves
  malicious recall. Cost (false positives): 42 validation benign rows scored
  in this band and would be blocked — an availability cost for benign users.
  Some of those are high-confidence (mean score 0.99) and concentrated in
  `wildguard_judgecomp` benign rows (FPR 0.82, flagged unstable), so part of
  the block cost may even be label noise rather than model error; this
  gateway does not resolve that question and relabels nothing.
- **`review` band (0.70 <= s < 0.90).** From the pre-registered `b0.70_m0.90`
  row of the abstention grid: 11 of 941 rows fall in this band; resolving
  them by blocking (`fail-closed`) would move the validation outcome from
  42 FP / 9 FN to 49 FP / 5 FN; resolving them by allowing (`fail-open`)
  leaves 42 FP / 9 FN. Human review sits between and costs review capacity;
  the grid's review figures assume a perfect reviewer and are an optimistic
  bound. The band exists because the score distribution is bimodal and the
  middle region is sparse and empirically mixed, not because 0.70 is proven
  optimal.
- **Marker escalation (`allow` -> `review`).** The 9 validation false
  negatives have scores from ~0.0005 to 0.86 (mean 0.40), so a score band
  alone can never catch them; a low-score prompt that matches a known
  injection phrasing ("ignore previous instructions", system-prompt
  exfiltration, jailbreak lexicon, encoded-command templates) is escalated
  to review instead of being silently allowed. How many of the 9 would
  actually be escalated is **unknown** — the rules were built and tested for
  the review packet, not measured against the FN rows, and they only
  recognize known phrasings, so novel or obfuscated injections still pass
  through the score bands alone.
- **`allow` (s < 0.70, no strong marker).** 357+ validation rows sit near
  score 0.0005 with an ~1.1% empirical positive rate in `[0.00, 0.10)`.
  Residual false negatives are unavoidable: some malicious prompts will be
  allowed. That is the documented cost of not blocking everything.

The band defaults were read off the pre-registered abstention grid for
illustration; no optimality is claimed, and any selection on the same
validation data that describes it is optimistically biased.

## Failure behaviour (fail-closed, explicit)

A security check that cannot run must never produce `allow`:

| Situation | Gateway behaviour |
|---|---|
| Model directory missing / load failure at startup | `POST /gateway` returns `503` with the fixed detail `gateway security check unavailable; treat this prompt as review, do not allow it`; `/health` reports the load error as today. |
| `RuntimeError` / `OSError` during inference | Same `503`, the error type name (not the message, not the prompt) is recorded on the service, matching `/predict`. |
| Invalid prompt (empty, whitespace-only, non-string, > 4096 chars) | `422` from the shared request model; the prompt is rejected before evaluation and is never allowed or scored. |
| Any `503` | The caller (or the human operating the gateway) must treat the prompt as `review`. The gateway never emits `allow` without a completed model score and marker check. |

A `503` is deliberately distinguished from `block`: the gateway does not
claim a benign-looking prompt is malicious; it reports that the check itself
failed and refuses to allow.

## Privacy

- The gateway stores and logs **no prompt text**: responses echo the score,
  decision, reason codes, and marker rule IDs only; reason codes are fixed
  constants and rule IDs are pattern names, never text spans.
- A small in-memory counter (`GET /gateway/stats`) tracks decision counts
  (`allow` / `review` / `block` / `unavailable`) as aggregate integers for
  operations. It is process-local, never persisted, reset on restart, and
  contains no text.
- No external service is called; inference is local-file-only, as in the
  existing API.
- HTTP `422` validation bodies echo the caller's own input back to that same
  caller, exactly as `/predict` already does; nothing is logged and no other
  party sees it.

## What this gateway is not

- It does not guarantee security and does not prevent all prompt injections.
  It is one triage decision over one classifier's uncalibrated score plus
  known-phrase rules; adversarial or novel injections will be allowed.
- It does not rewrite, sanitize, or sandbox prompts, and it does not
  authenticate or rate-limit callers.
- Its band figures are validation-only observations; held-out test
  performance has not been measured for any policy here.
- `review` is a verdict of "a human must look at this", not a punishment of
  the prompt; no automated action is taken on a reviewed prompt.
