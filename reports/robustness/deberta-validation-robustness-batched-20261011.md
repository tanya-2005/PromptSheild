# DeBERTa validation robustness benchmark (incomplete)

This is an explicit failure/availability record for a validation-only run. It
contains no model metrics and no prompt text.

- Status: `interrupted_cpu_runtime`
- Checkpoint loaded: `true`
- Synthetic batch smoke test: `passed`
- Device: `cpu`
- Batch size: `8`
- Expected validation rows: `941`
- Completed validation metrics: `none`
- Test split accessed: `false`
- Existing reports overwritten: `false`

The saved local `DebertaV2ForSequenceClassification` checkpoint loaded in the
isolated Python 3.13.16 environment. A synthetic two-prompt batch produced the
same predictions as single-item inference. The full validation benchmark was
then started with the existing transformations and threshold, but it remained
active for more than four minutes without completing and was stopped. Because
the benchmark does not persist partial counters, the number of rows evaluated
before interruption is intentionally not claimed.

Run the unchanged documented benchmark in the established compatible GPU
environment to obtain completed metrics. Keep the validation-only input,
threshold, transformations, and a recorded conservative batch size unchanged.
