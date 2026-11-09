# PromptShield source-stratified reliability audit and gateway policy note

Aggregate validation figures reused from earlier reports. No new inference was run, no Parquet file was read, no model was retrained, and no raw prompt text appears here.

- Validation rows: `941`
- Threshold: `0.9` (unmodified)
- False positives: `42`
- False negatives: `9`

Rates below are shown as `value [Wilson 95% lower, upper]`. A `!unstable`
marker means the group is too small or too imprecise to read as a point
estimate.

## Source stratified rates

| source | Samples | Benign denom | FP | FPR (95% CI) | Malicious denom | FN | FNR (95% CI) |
|---|---|---|---|---|---|---|---|
| hackaprompt | 291 | 0 | 0 | n/a | 291 | 0 | 0.0000 [0.0000, 0.0130] |
| harmbench | 14 | 0 | 0 | n/a | 14 | 7 | 0.5000 [0.2680, 0.7320] !unstable |
| harmbench_benign | 18 | 18 | 7 | 0.3889 [0.2031, 0.6138] !unstable | 0 | 0 | n/a |
| neuralchemy_v1 | 558 | 352 | 7 | 0.0199 [0.0097, 0.0405] | 206 | 2 | 0.0097 [0.0027, 0.0347] |
| original | 14 | 4 | 1 | 0.2500 [0.0456, 0.6994] !unstable | 10 | 0 | 0.0000 [0.0000, 0.2775] !unstable |
| wildguard_judgecomp | 46 | 33 | 27 | 0.8182 [0.6561, 0.9139] !unstable | 13 | 0 | 0.0000 [0.0000, 0.2281] !unstable |

## Category stratified rates

| category | Samples | Benign denom | FP | FPR (95% CI) | Malicious denom | FN | FNR (95% CI) |
|---|---|---|---|---|---|---|---|
| adversarial | 85 | 0 | 0 | n/a | 85 | 0 | 0.0000 [0.0000, 0.0432] |
| agent_manipulation | 3 | 0 | 0 | n/a | 3 | 0 | 0.0000 [0.0000, 0.5615] !unstable |
| benign | 396 | 396 | 35 | 0.0884 [0.0642, 0.1204] | 0 | 0 | n/a |
| code_execution | 1 | 0 | 0 | n/a | 1 | 0 | 0.0000 [0.0000, 0.7935] !unstable |
| context_confusion | 4 | 0 | 0 | n/a | 4 | 0 | 0.0000 [0.0000, 0.4899] !unstable |
| control | 5 | 0 | 0 | n/a | 5 | 1 | 0.2000 [0.0362, 0.6245] !unstable |
| direct_injection | 293 | 0 | 0 | n/a | 293 | 0 | 0.0000 [0.0000, 0.0129] |
| edge_case | 11 | 11 | 7 | 0.6364 [0.3538, 0.8483] !unstable | 0 | 0 | n/a |
| encoding | 40 | 0 | 0 | n/a | 40 | 0 | 0.0000 [0.0000, 0.0876] |
| encoding_obfuscation | 3 | 0 | 0 | n/a | 3 | 0 | 0.0000 [0.0000, 0.5615] !unstable |
| instruction_override | 2 | 0 | 0 | n/a | 2 | 0 | 0.0000 [0.0000, 0.6576] !unstable |
| jailbreak | 48 | 0 | 0 | n/a | 48 | 7 | 0.1458 [0.0725, 0.2717] |
| model_fingerprinting | 2 | 0 | 0 | n/a | 2 | 0 | 0.0000 [0.0000, 0.6576] !unstable |
| multi_turn | 9 | 0 | 0 | n/a | 9 | 0 | 0.0000 [0.0000, 0.2991] !unstable |
| payload_injection | 4 | 0 | 0 | n/a | 4 | 0 | 0.0000 [0.0000, 0.4899] !unstable |
| persona_replacement | 1 | 0 | 0 | n/a | 1 | 0 | 0.0000 [0.0000, 0.7935] !unstable |
| prompt_extraction | 2 | 0 | 0 | n/a | 2 | 1 | 0.5000 [0.0945, 0.9055] !unstable |
| prompt_injection | 1 | 0 | 0 | n/a | 1 | 0 | 0.0000 [0.0000, 0.7935] !unstable |
| rag_poisoning | 6 | 0 | 0 | n/a | 6 | 0 | 0.0000 [0.0000, 0.3903] !unstable |
| response_manipulation | 2 | 0 | 0 | n/a | 2 | 0 | 0.0000 [0.0000, 0.6576] !unstable |
| system_extraction | 3 | 0 | 0 | n/a | 3 | 0 | 0.0000 [0.0000, 0.5615] !unstable |
| system_manipulation | 2 | 0 | 0 | n/a | 2 | 0 | 0.0000 [0.0000, 0.6576] !unstable |
| token_smuggling | 5 | 0 | 0 | n/a | 5 | 0 | 0.0000 [0.0000, 0.4345] !unstable |
| training_extraction | 13 | 0 | 0 | n/a | 13 | 0 | 0.0000 [0.0000, 0.2281] !unstable |

## Rate stability

Rule: flagged unstable when denominator < 30 or Wilson 95% interval width > 0.25.

- Stable rate estimates: `8`
- Unstable rate estimates: `25`

Unstable groups (do not quote these as point estimates):

- `source:harmbench:false_negative_rate` = 7/14 = 0.5000 [0.2680, 0.7320] (denominator 14 < 30; interval width 0.464 > 0.25)
- `source:harmbench_benign:false_positive_rate` = 7/18 = 0.3889 [0.2031, 0.6138] (denominator 18 < 30; interval width 0.411 > 0.25)
- `source:original:false_positive_rate` = 1/4 = 0.2500 [0.0456, 0.6994] (denominator 4 < 30; interval width 0.654 > 0.25)
- `source:original:false_negative_rate` = 0/10 = 0.0000 [0.0000, 0.2775] (denominator 10 < 30; interval width 0.278 > 0.25)
- `source:wildguard_judgecomp:false_positive_rate` = 27/33 = 0.8182 [0.6561, 0.9139] (interval width 0.258 > 0.25)
- `source:wildguard_judgecomp:false_negative_rate` = 0/13 = 0.0000 [0.0000, 0.2281] (denominator 13 < 30)
- `category:agent_manipulation:false_negative_rate` = 0/3 = 0.0000 [0.0000, 0.5615] (denominator 3 < 30; interval width 0.561 > 0.25)
- `category:code_execution:false_negative_rate` = 0/1 = 0.0000 [0.0000, 0.7935] (denominator 1 < 30; interval width 0.793 > 0.25)
- `category:context_confusion:false_negative_rate` = 0/4 = 0.0000 [0.0000, 0.4899] (denominator 4 < 30; interval width 0.490 > 0.25)
- `category:control:false_negative_rate` = 1/5 = 0.2000 [0.0362, 0.6245] (denominator 5 < 30; interval width 0.588 > 0.25)
- `category:edge_case:false_positive_rate` = 7/11 = 0.6364 [0.3538, 0.8483] (denominator 11 < 30; interval width 0.495 > 0.25)
- `category:encoding_obfuscation:false_negative_rate` = 0/3 = 0.0000 [0.0000, 0.5615] (denominator 3 < 30; interval width 0.561 > 0.25)
- `category:instruction_override:false_negative_rate` = 0/2 = 0.0000 [0.0000, 0.6576] (denominator 2 < 30; interval width 0.658 > 0.25)
- `category:model_fingerprinting:false_negative_rate` = 0/2 = 0.0000 [0.0000, 0.6576] (denominator 2 < 30; interval width 0.658 > 0.25)
- `category:multi_turn:false_negative_rate` = 0/9 = 0.0000 [0.0000, 0.2991] (denominator 9 < 30; interval width 0.299 > 0.25)
- `category:payload_injection:false_negative_rate` = 0/4 = 0.0000 [0.0000, 0.4899] (denominator 4 < 30; interval width 0.490 > 0.25)
- `category:persona_replacement:false_negative_rate` = 0/1 = 0.0000 [0.0000, 0.7935] (denominator 1 < 30; interval width 0.793 > 0.25)
- `category:prompt_extraction:false_negative_rate` = 1/2 = 0.5000 [0.0945, 0.9055] (denominator 2 < 30; interval width 0.811 > 0.25)
- `category:prompt_injection:false_negative_rate` = 0/1 = 0.0000 [0.0000, 0.7935] (denominator 1 < 30; interval width 0.793 > 0.25)
- `category:rag_poisoning:false_negative_rate` = 0/6 = 0.0000 [0.0000, 0.3903] (denominator 6 < 30; interval width 0.390 > 0.25)
- `category:response_manipulation:false_negative_rate` = 0/2 = 0.0000 [0.0000, 0.6576] (denominator 2 < 30; interval width 0.658 > 0.25)
- `category:system_extraction:false_negative_rate` = 0/3 = 0.0000 [0.0000, 0.5615] (denominator 3 < 30; interval width 0.561 > 0.25)
- `category:system_manipulation:false_negative_rate` = 0/2 = 0.0000 [0.0000, 0.6576] (denominator 2 < 30; interval width 0.658 > 0.25)
- `category:token_smuggling:false_negative_rate` = 0/5 = 0.0000 [0.0000, 0.4345] (denominator 5 < 30; interval width 0.434 > 0.25)
- `category:training_extraction:false_negative_rate` = 0/13 = 0.0000 [0.0000, 0.2281] (denominator 13 < 30)

## Metadata and label consistency

Metadata is descriptive and is NOT treated as ground truth. Mixed-label sources are surfaced for review only; no example is relabeled by this audit.

Mixed-label sources (surfaced for review, never relabeled):

- `neuralchemy_v1`: {'0': 352, '1': 206}
- `original`: {'0': 4, '1': 10}
- `wildguard_judgecomp`: {'0': 33, '1': 13}

Open hypotheses this audit does not resolve:

- High-confidence false positives may reflect label noise in the source (the label is wrong and the model is right).
- The same pattern may reflect a genuine model bias toward over-flagging that source's benign style (the model is wrong).
- These two hypotheses are not distinguishable from aggregate counts alone; resolving either would require human review of the affected rows and would still not be an automatic relabel.

## Signal availability at inference time

Deployed request fields (read from the API schema): `['prompt']`

| Dataset field | Classification |
|---|---|
| `category` | offline diagnostic only |
| `source` | offline diagnostic only |
| `severity` | offline diagnostic only |
| `augmented` | offline diagnostic only |

Source and category are offline dataset diagnostics. They are not present on the deployed inference request, so no source-based allowlist, blocklist, or routing rule can be enforced at inference time without changing the request contract and establishing a trustworthy client-supplied source value.

## Gateway policy design note

This audit changes no decision rule and implements no allowlist. It only characterises what each gateway behaviour would do with the same measured outcomes.

Behaviour of each gateway policy on the same measured validation outcomes (fail-open allows every abstained row, fail-closed blocks every abstained row, human review resolves them):

| Policy | Coverage | Fail-open FP | Fail-open FN | Fail-closed FP | Fail-closed FN | Review rows |
|---|---:|---:|---:|---:|---:|---:|
| no_abstention | 1.0000 | 42 | 9 | 42 | 9 | 0 |
| b0.90_m0.95 | 0.9957 | 40 | 11 | 42 | 9 | 4 |
| b0.90_m0.99 | 0.9809 | 30 | 15 | 42 | 9 | 18 |
| b0.80_m0.90 | 0.9926 | 42 | 9 | 47 | 7 | 7 |
| b0.80_m0.95 | 0.9883 | 40 | 11 | 47 | 7 | 11 |
| b0.80_m0.99 | 0.9734 | 30 | 15 | 47 | 7 | 25 |
| b0.70_m0.90 | 0.9883 | 42 | 9 | 49 | 5 | 11 |
| b0.70_m0.95 | 0.9841 | 40 | 11 | 49 | 5 | 15 |
| b0.70_m0.99 | 0.9692 | 30 | 15 | 49 | 5 | 29 |
| b0.50_m0.90 | 0.9841 | 42 | 9 | 53 | 5 | 15 |
| b0.50_m0.95 | 0.9798 | 40 | 11 | 53 | 5 | 19 |
| b0.50_m0.99 | 0.9649 | 30 | 15 | 53 | 5 | 33 |

Interpretation:

- Fail-open minimises false blocking of benign prompts but lets every abstained malicious prompt through, raising false negatives.
- Fail-closed minimises missed malicious prompts but blocks every abstained benign prompt, raising false positives.
- Human review sits between the two and costs review capacity; the review figures here assume a perfect reviewer and are an optimistic upper bound.
- The choice between these behaviours is a product risk decision about relative cost, not something validation data can decide.

The illustrative band is selected by a documented rule for illustration only and is not declared optimal:

- Selection rule: among non-baseline policies with coverage >= 0.95, the one catching the largest fraction of all errors, ties broken by smaller abstention rate; chosen for illustration on validation data only
- Illustrative policy: `b0.70_m0.99`
- Optimality claimed: `false`

## Source-conditional review ceiling (offline diagnostic only)

Status: offline diagnostic ceiling only; NOT implemented and NOT implementable with the current request contract

| Source | Rows to review | Fraction of set | FP moved | FN moved |
|---|---:|---:|---:|---:|
| wildguard_judgecomp | 46 | 0.0489 | 27 | 0 |
| harmbench_benign | 18 | 0.0191 | 7 | 0 |
| original | 14 | 0.0149 | 1 | 0 |

These rows cannot be identified at inference time, so the ceiling is motivation for a future trustworthy source signal, not a rule that is implemented today.

## Limitations

- Every figure reuses validation-only aggregates from earlier reports and does not establish held-out test performance.
- Wilson intervals quantify sampling uncertainty within a group; they do not correct for selection or labeling bias in the source itself.
- Metadata is descriptive and is never treated as ground truth; no example is relabeled by this audit.
- The source-based review escalation is an offline ceiling only and is not implementable because source is absent from the deployed request.
- Human-review figures assume a perfect reviewer and are an optimistic bound; real review is fallible and has a cost.
- The deployed 0.90 threshold was not tuned or changed and no model was retrained.
