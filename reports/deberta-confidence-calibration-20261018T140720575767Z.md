# PromptShield DeBERTa validation calibration and selective prediction

Aggregate validation metrics only; no prompt text is included. Class-1 softmax outputs are treated as uncalibrated scores; this study does not establish or improve calibration.

- Validation rows: `941`
- Decision threshold: `0.9` (unmodified)
- Expected calibration error: `0.056247`
- Brier score: `0.055028`

## Reliability bins

Binning: 10 equal-width bins on [0, 1]; bin i covers [i/bin_count, (i+1)/bin_count) and the final bin is closed on the right; scores on interior boundaries go to the upper bin. Empty bins are omitted here but present in the JSON report.

| Interval | Count | Fraction | Mean score | Empirical positive rate | Signed gap |
|---|---:|---:|---:|---:|---:|
| [0.00, 0.10) | 356 | 0.3783 | 0.000583 | 0.011236 | -0.010653 |
| [0.10, 0.20) | 1 | 0.0011 | 0.166912 | 0.000000 | +0.166912 |
| [0.20, 0.30) | 1 | 0.0011 | 0.210457 | 0.000000 | +0.210457 |
| [0.40, 0.50) | 1 | 0.0011 | 0.472911 | 1.000000 | -0.527089 |
| [0.50, 0.60) | 2 | 0.0021 | 0.573589 | 0.000000 | +0.573589 |
| [0.60, 0.70) | 2 | 0.0021 | 0.616042 | 0.000000 | +0.616042 |
| [0.70, 0.80) | 4 | 0.0043 | 0.728276 | 0.500000 | +0.228276 |
| [0.80, 0.90) | 7 | 0.0074 | 0.840172 | 0.285714 | +0.554457 |
| [0.90, 1.00] | 567 | 0.6026 | 0.998339 | 0.925926 | +0.072413 |

## Score distributions

| Group | Samples | Mean | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| overall | 941 | 0.614548 | 0.999390 | 0.000500 | 0.999633 |
| true benign | 407 | 0.123376 | 0.000524 | 0.000500 | 0.999619 |
| true malicious | 534 | 0.988906 | 0.999600 | 0.000558 | 0.999633 |
| true_positive | 525 | 0.998981 | 0.999601 | 0.906323 | 0.999633 |
| true_negative | 365 | 0.023618 | 0.000522 | 0.000500 | 0.893277 |
| false_positive | 42 | 0.990317 | 0.997881 | 0.924893 | 0.999619 |
| false_negative | 9 | 0.401204 | 0.472911 | 0.000558 | 0.858666 |

## Selective prediction (abstention bands)

Policies: accept as benign when score < benign accept max; accept as malicious when score >= malicious accept min; abstain to human review otherwise. The grid was fixed in source before this run.

| Policy | Benign accept <= | Malicious accept >= | Coverage | Abstention rate | Accepted error rate | Accepted FP | Accepted FN | Abstained benign | Abstained malicious | Errors caught |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_abstention | 0.90 | 0.90 | 1.0000 | 0.0000 | 0.054198 | 42 | 9 | 0 | 0 | 0.0000 |
| b0.90_m0.95 | 0.90 | 0.95 | 0.9957 | 0.0043 | 0.052295 | 40 | 9 | 2 | 2 | 0.0392 |
| b0.90_m0.99 | 0.90 | 0.99 | 0.9809 | 0.0191 | 0.042254 | 30 | 9 | 12 | 6 | 0.2353 |
| b0.80_m0.90 | 0.80 | 0.90 | 0.9926 | 0.0074 | 0.052463 | 42 | 7 | 5 | 2 | 0.0392 |
| b0.80_m0.95 | 0.80 | 0.95 | 0.9883 | 0.0117 | 0.050538 | 40 | 7 | 7 | 4 | 0.0784 |
| b0.80_m0.99 | 0.80 | 0.99 | 0.9734 | 0.0266 | 0.040393 | 30 | 7 | 17 | 8 | 0.2745 |
| b0.70_m0.90 | 0.70 | 0.90 | 0.9883 | 0.0117 | 0.050538 | 42 | 5 | 7 | 4 | 0.0784 |
| b0.70_m0.95 | 0.70 | 0.95 | 0.9841 | 0.0159 | 0.048596 | 40 | 5 | 9 | 6 | 0.1176 |
| b0.70_m0.99 | 0.70 | 0.99 | 0.9692 | 0.0308 | 0.038377 | 30 | 5 | 19 | 10 | 0.3137 |
| b0.50_m0.90 | 0.50 | 0.90 | 0.9841 | 0.0159 | 0.050756 | 42 | 5 | 11 | 4 | 0.0784 |
| b0.50_m0.95 | 0.50 | 0.95 | 0.9798 | 0.0202 | 0.048807 | 40 | 5 | 13 | 6 | 0.1176 |
| b0.50_m0.99 | 0.50 | 0.99 | 0.9649 | 0.0351 | 0.038546 | 30 | 5 | 23 | 10 | 0.3137 |

- Pareto frontier (coverage vs accepted error rate): `['no_abstention', 'b0.90_m0.95', 'b0.90_m0.99', 'b0.80_m0.95', 'b0.80_m0.99', 'b0.70_m0.90', 'b0.70_m0.95', 'b0.70_m0.99']`
- Illustrative reference rule: smallest abstention rate achieving at least a 50% relative reduction in accepted-case error rate versus the no-abstention baseline
- Illustrative reference policy: `none`
- No candidate is declared optimal. All candidates were scored on the same validation data that describes them, so any selection is optimistically biased and requires independent confirmation.

## Limitations

- This study uses validation data only and does not establish held-out test performance.
- Class-1 softmax outputs are treated as scores; reliability summaries do not make them calibrated probabilities.
- ECE depends on the binning scheme; equal-width binning leaves sparse high-score bins whose rates are unstable at this sample size.
- Candidate policies are described on the same validation data used to score them, so any selection is optimistically biased.
- Abstention moves cases to human review; review capacity and cost are not modeled here.
- The deployed decision threshold was not tuned or changed; no model was retrained.
