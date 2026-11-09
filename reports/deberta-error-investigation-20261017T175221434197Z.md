# PromptShield DeBERTa validation error investigation

Aggregate confidence and metadata analysis only; no prompt text is included.

- Validation rows: `941`
- Threshold: `0.9`

## Confidence by confusion group

| Group | Samples | Mean class-1 score | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| true_positive | 525 | 0.998980916841207 | 0.9996009395643597 | 0.9063234851252135 | 0.9996326405266405 |
| true_negative | 365 | 0.02361800469671506 | 0.0005215454301095732 | 0.0004999785431752227 | 0.8932769871745087 |
| false_positive | 42 | 0.9903173221946391 | 0.9978811205304312 | 0.9248929054048124 | 0.9996185741082809 |
| false_negative | 9 | 0.401204081674278 | 0.47291093811683393 | 0.0005576518812120726 | 0.8586655558411417 |

## Requested source comparison

| Source | Samples | FP | FN | FPR | FNR | Mean score |
|---|---:|---:|---:|---:|---:|---:|
| wildguard_judgecomp | 46 | 27 | 0 | 0.8181818181818182 | 0.0 | 0.9042137176247521 |
| harmbench | 14 | 0 | 7 | n/a | 0.5 | 0.741056416218686 |
| harmbench_benign | 18 | 7 | 0 | 0.3888888888888889 | n/a | 0.6698007052199537 |
| neuralchemy_v1 | 558 | 7 | 2 | 0.019886363636363636 | 0.009708737864077669 | 0.3804533616128904 |

## Metadata/label consistency

- `category` values with both numeric labels: none
- `source` values with both numeric labels: ['neuralchemy_v1', 'original', 'wildguard_judgecomp']

## Limitations

- This investigation uses validation data only and does not establish held-out test performance.
- Confidence summaries describe model scores; they are not calibrated probabilities.
- Metadata is descriptive and is not treated as ground truth for labels.
- Small groups can produce unstable rates and confidence summaries.
- The threshold was not tuned or changed for this investigation.
