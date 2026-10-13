# DeBERTa robustness runtime blocked — 2026-10-11

The requested DeBERTa robustness evaluation did not start. No checkpoint was
loaded, no smoke test ran, and no validation rows were evaluated.

## Environment setup

- Created isolated environment: `.venv-deberta`
- Python: `3.13.16` x64
- Original `.venv`: unchanged
- Packages installed: none

## Pinned dependency resolution

The binary-only pip dry-run showed:

- `torch==2.6.0`: Python 3.13 Windows x64 wheel metadata available.
- `transformers==4.49.0`: Python 3.13-compatible wheel metadata available.
- `sentencepiece==0.2.0`: blocked; no matching Python 3.13 Windows x64
  distribution. pip offered `0.2.1` and `0.2.2` instead.

The task requires pinned versions and does not authorize silently changing
`sentencepiece`, so installation and inference stopped at this gate.

## Safety checks

- Test split was not accessed.
- No model weights were downloaded.
- No model was retrained.
- Existing benchmark reports, historical reports, and checkpoint files were
  not overwritten.

## Required next action

Use an approved runtime/package source that provides the exact pinned
`sentencepiece==0.2.0` build, or explicitly approve a separately reviewed
sentencepiece version change. Then install only the approved runtime packages,
load the local checkpoint with `local_files_only=True`, pass the synthetic
smoke test, and run the existing validation-only robustness CLI.
