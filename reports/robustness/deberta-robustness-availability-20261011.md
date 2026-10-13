# DeBERTa robustness availability — 2026-10-11

The saved DeBERTa checkpoint was inspected but was **not evaluated** in the
current local environment.

## Verified checkpoint

- Model: `microsoft/deberta-v3-small`
- Saved revision: `a36c739020e01763fe789b4b85e2df55d6180012`
- Architecture: `DebertaV2ForSequenceClassification`
- Labels: two
- Required local files are present, including `model.safetensors`, the
  SentencePiece model, tokenizer files, and `config.json`.
- The checkpoint configuration records Transformers `4.49.0`.

## Runtime blocker

The selected project environment is Python 3.14.2 and does not import:

- `torch==2.6.0`
- `transformers==4.49.0`
- `sentencepiece==0.2.0`

No packages were installed and no model weights were downloaded.

## Next action

Use a compatible isolated environment, preferably the existing reproducible
Python 3.13/Colab setup, with the pinned runtime packages and the saved
checkpoint. Run the existing validation-only robustness CLI against the same
941-row validation split. Save a new DeBERTa-only report; do not overwrite
`validation-robustness.json`, its audit backup, or historical reports.

The run must retain the existing transformation contract: whitespace and
formatting are candidate label-preserving perturbations with limitations;
neutral-context insertion is exploratory; and no test split or raw prompt text
may be accessed or recorded.
