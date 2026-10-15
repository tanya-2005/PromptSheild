# DeBERTa validation robustness runbook

This runbook evaluates the existing local artifacts only. It does not train a
model, select a threshold, or access the test split.

## Exact command

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
& .\.venv-deberta\Scripts\python.exe -m promptshield.robustness `
  --validation data\raw\validation.parquet `
  --baseline models\baseline.joblib `
  --svm models\linear-svm.joblib `
  --deberta models\deberta-v3-small `
  --deberta-batch-size 8 `
  --json-report reports\robustness\validation-robustness-deberta.json `
  --markdown-report reports\robustness\validation-robustness-deberta.md
```

The command must be run from the repository root. The isolated
`.venv-deberta` environment needs the pinned local inference dependencies
(`torch`, `transformers`, and `sentencepiece`), while the baseline artifacts
also require the project's compatible `joblib`, `pyarrow`, and
`scikit-learn` installation. Use a smaller batch size if CPU memory is
constrained. Do not install packages into `.venv` for this experiment.

## Required artifacts and output

Before running, verify these paths exist:

- `data\raw\validation.parquet`
- `models\baseline.joblib`
- `models\linear-svm.joblib`
- all required files under `models\deberta-v3-small\`

Use new report paths. Do not overwrite historical reports or incomplete-run
backups. The JSON and Markdown reports contain metadata and metrics, but never
prompt text.

## Completion checks

In the JSON report, confirm:

1. `validation_only` is `true` and `rows` is `941`.
2. Top-level `evaluation_status` is `"completed"` and
   `inference_complete` is `true`.
3. All three requested model names are present and have `available: true`.
4. Every model has `evaluation_status: "completed"`.
5. Every model has `completed_rows.clean_validation: 941`.
6. Each of `whitespace_variation`, `harmless_formatting`, and
   `neutral_context` has `examples_evaluated: 941` and non-null metrics.

Any unavailable model, invalid prediction, or short prediction array must make
the report incomplete. An incomplete report must not be used as a result.

## Metric and provenance interpretation

The current DeBERTa inference contract uses the class-1 softmax probability
with a `0.90` threshold. Historical training validation used
`argmax(logits)`, which is effectively a much lower boundary for a two-class
model. Therefore historical macro F1 values are not directly comparable to
this robustness report unless the decision convention is matched.

The local checkpoint's base model metadata does not prove which training run
created its weights. Record the checkpoint directory, model configuration,
dependency versions, and a hash manifest if provenance is required. Do not
describe the checkpoint as reproducing `reports/deberta-validation.json`
without an artifact path and weight hash linking the two.

## Safety checks

After the run, confirm the report files contain no prompt text. Search only
for structural keys and known metadata; do not print dataset rows:

```powershell
Select-String -Path reports\robustness\validation-robustness-deberta.json `
  -Pattern '"text"|"prompt"|"validation_only"|"evaluation_status"'
```

The only expected matches are the structural status fields; prompt-bearing
fields must be absent. Also confirm the command used
`data\raw\validation.parquet`, never `test.parquet`. The benchmark's path guard
rejects forbidden test-split paths, but the operator should still verify the
command before execution.
