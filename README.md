# PromptShield

PromptShield is an ML-powered prompt-injection detection system. The project
will begin with a reproducible text-classification baseline and later add an
API, UI, and configurable protective gateway.

## Current stage

The first TF-IDF + Logistic Regression baseline has been trained on the
primary training split and evaluated on validation data only. The test split
remains untouched. The readiness protocol and diagnostic-record schema are
documented in [reports/stage-3-readiness.md](reports/stage-3-readiness.md),
and the generated validation result is
[reports/baseline-validation.json](reports/baseline-validation.json).

See [docs/dataset-decision.md](docs/dataset-decision.md) for the dataset
recommendation and verification notes.

## Local validation

The CSV validator uses only the Python standard library:

```text
python -m unittest discover -s tests -v
python -m promptshield.data_validation path\to\data.csv
```

The expected tabular contract is:

- `text`: non-empty input text
- `label`: integer `0` (benign) or `1` (injection)
- optional `split`: `train`, `validation`, or `test`

The validator reports row counts, label counts, duplicate texts, and invalid
rows. It does not silently repair data.

After Stage 2 acquisition, local Parquet inspection can be rerun with the
installed `pyarrow` tooling:

```text
python -m promptshield.dataset_inspection data\raw `
  --manifest data\manifest.json `
  --revision <pinned-hugging-face-commit>
```

The raw Parquet files are intentionally ignored by Git. See
[data/manifest.json](data/manifest.json) for the pinned revision, file hashes,
schema, and inspection findings.

## Stage 5 baseline

The reproducible baseline uses only `text` to predict `label` with a
`TfidfVectorizer` and `LogisticRegression`. It fits on `train.parquet` and
evaluates on `validation.parquet`; it does not load the test split.

Run it from PowerShell with the project environment:

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m promptshield.baseline `
  --train data\raw\train.parquet `
  --validation data\raw\validation.parquet `
  --artifact models\baseline.joblib `
  --report reports\baseline-validation.json
```

The report contains per-class precision, recall, F1, macro F1, accuracy,
confusion matrix, false-positive and false-negative rates, support counts, and
the latency measurement method. The saved artifact contains the fitted
pipeline and the input-file hashes. The validation report is ignored because
it is a generated result; the raw data and model artifact are also ignored.

Run the complete test suite with:

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Validation errors can be reviewed without retraining or loading the test
split:

```powershell
$env:PYTHONPATH = "src"
& .\.venv\Scripts\python.exe -m promptshield.error_analysis `
  --artifact models\baseline.joblib `
  --validation data\raw\validation.parquet `
  --validation-report reports\baseline-validation.json `
  --report reports\validation-error-analysis.json `
  --worksheet reports\validation-error-review.md
```
