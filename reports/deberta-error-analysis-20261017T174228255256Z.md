# PromptShield DeBERTa validation error analysis

Aggregate validation analysis only; no prompt text is included.

- Validation rows: `941`
- Threshold: `0.9`
- True negatives: `365`
- False positives: `42`
- False negatives: `9`
- True positives: `525`

## Metadata breakdowns

### `category`

| Value | Samples | FP | FN | TP | TN | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| adversarial | 85 | 0 | 0 | 85 | 0 | n/a | 0.0 |
| agent_manipulation | 3 | 0 | 0 | 3 | 0 | n/a | 0.0 |
| benign | 396 | 35 | 0 | 0 | 361 | 0.08838383838383838 | n/a |
| code_execution | 1 | 0 | 0 | 1 | 0 | n/a | 0.0 |
| context_confusion | 4 | 0 | 0 | 4 | 0 | n/a | 0.0 |
| control | 5 | 0 | 1 | 4 | 0 | n/a | 0.2 |
| direct_injection | 293 | 0 | 0 | 293 | 0 | n/a | 0.0 |
| edge_case | 11 | 7 | 0 | 0 | 4 | 0.6363636363636364 | n/a |
| encoding | 40 | 0 | 0 | 40 | 0 | n/a | 0.0 |
| encoding_obfuscation | 3 | 0 | 0 | 3 | 0 | n/a | 0.0 |
| instruction_override | 2 | 0 | 0 | 2 | 0 | n/a | 0.0 |
| jailbreak | 48 | 0 | 7 | 41 | 0 | n/a | 0.14583333333333334 |
| model_fingerprinting | 2 | 0 | 0 | 2 | 0 | n/a | 0.0 |
| multi_turn | 9 | 0 | 0 | 9 | 0 | n/a | 0.0 |
| payload_injection | 4 | 0 | 0 | 4 | 0 | n/a | 0.0 |
| persona_replacement | 1 | 0 | 0 | 1 | 0 | n/a | 0.0 |
| prompt_extraction | 2 | 0 | 1 | 1 | 0 | n/a | 0.5 |
| prompt_injection | 1 | 0 | 0 | 1 | 0 | n/a | 0.0 |
| rag_poisoning | 6 | 0 | 0 | 6 | 0 | n/a | 0.0 |
| response_manipulation | 2 | 0 | 0 | 2 | 0 | n/a | 0.0 |
| system_extraction | 3 | 0 | 0 | 3 | 0 | n/a | 0.0 |
| system_manipulation | 2 | 0 | 0 | 2 | 0 | n/a | 0.0 |
| token_smuggling | 5 | 0 | 0 | 5 | 0 | n/a | 0.0 |
| training_extraction | 13 | 0 | 0 | 13 | 0 | n/a | 0.0 |

### `source`

| Value | Samples | FP | FN | TP | TN | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| hackaprompt | 291 | 0 | 0 | 291 | 0 | n/a | 0.0 |
| harmbench | 14 | 0 | 7 | 7 | 0 | n/a | 0.5 |
| harmbench_benign | 18 | 7 | 0 | 0 | 11 | 0.3888888888888889 | n/a |
| neuralchemy_v1 | 558 | 7 | 2 | 204 | 345 | 0.019886363636363636 | 0.009708737864077669 |
| original | 14 | 1 | 0 | 10 | 3 | 0.25 | 0.0 |
| wildguard_judgecomp | 46 | 27 | 0 | 13 | 6 | 0.8181818181818182 | 0.0 |

### `severity`

| Value | Samples | FP | FN | TP | TN | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
|  | 407 | 42 | 0 | 0 | 365 | 0.10319410319410319 | n/a |
| critical | 21 | 0 | 1 | 20 | 0 | n/a | 0.047619047619047616 |
| high | 168 | 0 | 7 | 161 | 0 | n/a | 0.041666666666666664 |
| low | 6 | 0 | 1 | 5 | 0 | n/a | 0.16666666666666666 |
| medium | 339 | 0 | 0 | 339 | 0 | n/a | 0.0 |

### `augmented`

| Value | Samples | FP | FN | TP | TN | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| false | 941 | 42 | 9 | 525 | 365 | 0.10319410319410319 | 0.016853932584269662 |

## Limitations

- This analysis uses validation data only and does not estimate held-out test performance.
- Metadata group rates can be unstable when sample_count is small.
- The threshold was not tuned or selected from this analysis.
- Checkpoint hashes identify the artifact captured during this run; they do not prove historical training provenance.
