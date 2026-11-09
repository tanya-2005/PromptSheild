# PromptShield finalization audit (2026-10-20)

Audit-only pass: no training, no downloads, no external API calls, and the
held-out `test.parquet` was never opened. All figures below are aggregate;
no raw prompt text appears in this file.

## What was verified

### Repository state

- Git HEAD `7afde57` ("Improve PromptShield robustness evaluation"), 3 commits.
  A large body of uncommitted work (gateway, frontend, evaluation modules,
  reports, docs) was preserved untouched; nothing was staged, committed, or
  pushed during this audit.
- Privacy scan of the working tree: no API keys, tokens, or credentials in
  any tracked or planned-to-be-tracked file. The only files under `reports\`
  containing raw prompt text are `manual-deberta-wildguard-review.json` and
  `validation-error-analysis.json`, both matched by `.gitignore` rules and
  therefore never tracked. `data/eval-manifest.json` is metadata-only
  (dataset commits, licenses, file hashes).
- `.gitignore` correctly excludes `data/raw/`, `data/raw-eval/`, `models/`,
  the review packets, and the generated `*-validation.json` reports.

### Automated tests

| Check | Result |
|---|---|
| `PYTHONPATH=src .venv\Scripts\python.exe -m pytest -q` | **399 passed**, 349 warnings (numpy/joblib/asyncio deprecations only), ~13 s |
| `.venv` environment | Python 3.14.2, pytest 9.1.1, httpx 0.28.1 |
| `.venv-deberta` environment | Python 3.13.16, torch 2.6.0+cpu, transformers 4.49.0 |
| `python -m unittest discover -s tests` | runs only **337** tests (silently skips every pytest-style module) — the README previously documented this command and now documents `pytest` instead |
| `git diff --check` | **exit 0**; only the pre-existing `LF → CRLF` warning on `.gitignore` |

### Live smoke test — backend (`promptshield.api`, real HTTP, loopback only)

1. **Environment without PyTorch** (`.venv`, port 8010): `/health` reported
   `loaded: false` with `load_error: "No module named 'torch'"`; `/predict`
   returned `503 model is unavailable for inference`; `/gateway` returned
   `503` with the canonical fail-closed detail (`gateway security check
   unavailable; treat this prompt as review, do not allow it`);
   `/gateway/stats` counted `unavailable: 1`. Failure handling is honest and
   fail-closed, exactly as documented.
2. **Torch environment** (`.venv-deberta`, port 8000, README command):
   the checkpoint loaded on CPU within the polling window; `/predict`
   returned `malicious_probability` / `decision` / `threshold` /
   `threshold_status`; `/gateway` returned the full `band_v1` payload
   (`gateway_decision`, fixed `reasons` codes, echoed
   `classifier_decision` + `classifier_threshold`, pattern-name
   `marker_rule_ids`, policy description); whitespace-only prompt → `422`;
   `/gateway/stats` returned aggregate integers only. As a live sanity check
   of the documented limitation, a trivially benign arithmetic prompt scored
   0.9993 and was **blocked** — the over-triggering described below is
   reproducible on the running system, not just in reports.

### Live smoke test — frontend (`promptshield.frontend`)

1. **Backend down** (dead port): `/api/health` reported
   `backend_reachable: false` with no fabricated load state; `/api/scan`
   returned `503` with the gateway's fail-closed detail, which the page
   renders as "Treated as Review", never "Allow".
2. **Backend up** (port 8000): the self-contained page served at
   `http://127.0.0.1:8502` (inline CSS/JS, no external resources);
   `/api/health` proxied truthfully; `/api/scan` returned the backend's real
   gateway decision unchanged; `/api/insights` returned the frozen
   aggregates, now including the benign-chat generalization limitation
   (8 entries after this audit's fix).
3. Method note: stale frontend/backend processes left running from earlier
   sessions were still bound to ports 8000/8502 and initially answered these
   probes. They were terminated and every result above was re-verified with
   freshly started processes.

### Documentation consistency (fixes applied in this pass)

- `README.md`: replaced the stale "no transformer training was run / no
  PyTorch installed" text (it contradicted the served checkpoint and the
  results table) with the verifiable status: the fine-tune ran on an
  external CUDA runtime, recorded in `reports/deberta-validation.json`
  (device `cuda`, pinned base revision `a36c739…`, lr 2e-5, 1 epoch,
  batch 2, seq len 128, seed 42), with the provenance caveat that hashes
  establish artifact identity only from the audit onward.
- `README.md`: the example `transformer_experiment` command previously showed
  `--artifact models\deberta-v3-small`, i.e. it would overwrite the frozen
  baseline; it is now marked reference-only and points at a new directory.
- `README.md`: test instructions now use `pytest` (with the new pinned
  `requirements-dev.txt`) instead of `python -m unittest discover`, which
  silently ran 45 fewer tests; removed the unittest line from the stdlib
  CSV-validator section for the same reason.
- `README.md`: added a "What a fresh clone contains" section (the dataset
  splits, model artifacts, and generated validation reports are Git-ignored,
  so a fresh clone runs the tests but cannot serve or re-evaluate the model
  without re-acquiring them) and a serving-environment note (`.venv-deberta`
  has PyTorch; a PyTorch-less environment fails closed with `503`).
- `README.md`: added the mitigation-status subsection — implemented, tests
  included, **training never ran**, no candidate checkpoint, no effect
  claimed.
- `src/promptshield/frontend.py`: the scanner's "Model insights" card
  previously disclosed only validation figures; it now also discloses the
  benign-chat generalization result and cites the aggregate report, so a
  person running the app sees the limitation without reading the README.
- `reports/stage-3-readiness.md`: added a dated "historical snapshot" note
  (it states, as stage-era facts, that the workspace was not a Git
  repository and that `pyarrow` was missing).

## Commands to run the project locally (PowerShell, repository root)

```powershell
# One-time setup
py -3.14 -m venv .venv                       # classical + API-capable env
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
# DeBERTa serving/evaluation needs torch; this machine uses a second env:
py -3.13 -m venv .venv-deberta
.\.venv-deberta\Scripts\python.exe -m pip install -r requirements.txt

# Tests (no dataset, no model needed)
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest

# Backend (model + /predict + /gateway) — needs models\deberta-v3-small\
.\.venv-deberta\Scripts\python.exe -m uvicorn promptshield.api:app --host 127.0.0.1 --port 8000

# Frontend (loopback-only scanner page)
.\.venv\Scripts\python.exe -m promptshield.frontend --backend http://127.0.0.1:8000 --host 127.0.0.1 --port 8502
# then open http://127.0.0.1:8502
```

Evaluation/experiment commands (all validation-only, all refuse
`test.parquet` paths) are listed verbatim in `README.md`: `baseline`,
`svm_experiment`, `model_comparison`, `deberta_decision_audit`,
`deberta_error_analysis`, `deberta_error_investigation`,
`confidence_calibration`, `source_reliability`, `error_review`,
`review_app`, `robustness`, `benign_chat_eval`.

## Validation-only results (unchanged, for reference)

- Logistic Regression: accuracy 0.9416, macro F1 0.9397, benign FPR 0.1130,
  malicious FNR 0.0169.
- Linear SVM: accuracy 0.9564, macro F1 0.9554, benign FPR 0.0737,
  malicious FNR 0.0206 (decision scores, not probabilities).
- DeBERTa (serving rule: class-1 score ≥ 0.90): accuracy 0.9458, macro F1
  0.9442, benign FPR 10.32% (42/407), malicious FNR 1.69% (9/534).
- Score outputs are uncalibrated; the pre-registered abstention study was an
  honest negative (best policy caught 31.4% of errors at 3.1% abstention,
  short of the 50% target).

## Known false-positive / generalization limitation

A single pre-registered benign-chat evaluation (200 public Dolly/OASST
prompts, frozen protocol, executed once) found **162/200 = 81.0% blocked**
(Wilson 95% CI 75.0–85.8%), vs. the 10.3% validation benign-FPR benchmark:
the checkpoint materially over-triggers on ordinary benign chat/instruction
prompts. The threshold and gateway policy were deliberately not changed in
response. This is disclosed in `README.md`, in the frontend insights card,
and in `reports/benign-chat-eval-20261019T164018682950Z.{md,json}`.

## Mitigation experiment status

Implemented, **not executed**. `src/promptshield/hard_negative_experiment.py`
plus 15 gate tests (`tests/test_hard_negative_experiment.py`) implement the
frozen plan `docs/benign-hard-negative-mitigation-plan.md`: eligibility and
deduplication gates, deterministic 414-row subset, warm-start training with
frozen embeddings, identical-protocol audits, positive control, and
pre-registered accept/reject criteria. The run stopped at its own memory
precondition (required ≥ 3.5 GiB free RAM before assembling training data;
2.75 GiB available). No training subset was built, no training ran, no
candidate checkpoint exists, the baseline checkpoint is byte-identical
(`model.safetensors` sha256 `2cefb8e4…`), and no mitigation effect is
claimed anywhere in the repository.

## Remaining reproducibility / environment limitations

- Model artifacts (`models/`) and dataset splits (`data/raw/`,
  `data/raw-eval/`) are Git-ignored by design: a fresh clone runs the full
  test suite but cannot serve or re-evaluate the model until they are
  re-acquired (`data/manifest.json` pins the dataset revision and file
  hashes). The DeBERTa checkpoint itself is not redistributable through this
  repository.
- The fine-tune is not reproducible on this laptop (~2 GB free RAM, no GPU);
  the Colab workflow notebook documents the procedure, but notebook outputs
  are not tracked, and checkpoint provenance is limited to the recorded
  training report plus hashes established from the decision audit onward.
- Commands are documented for PowerShell/Windows paths only.
- DeBERTa CPU inference is ~309 ms/text (one machine, not a capacity
  benchmark). All runtime numbers here are from this machine.
- The 27-row human label-noise review packet remains incomplete (`pending`);
  no row has been relabeled by any tool.
- Everything except the first 3 commits is uncommitted work; the repository
  as presented on GitHub depends on the owner committing it.

## Portfolio-readiness verdict

PromptShield is presentable as an internship portfolio project in its current
state: three trained classifiers, a served model with a documented decision
rule, a protective gateway with fail-closed behaviour, a local frontend,
extensive validation-only analysis with honest negative results, a
pre-registered external evaluation that exposed a real weakness, and a
tested-but-not-executed mitigation with explicit gates. There are no hidden
failures: the smoke tests reproduce both the working path and every
documented failure path.

Materially remaining actions (in order):

1. **Commit the work** (owner action; deliberately not done here, per the
   standing no-stage/no-commit rule).
2. Optionally execute the single mitigation run once ≥ 3.5 GiB free RAM is
   available, then report retain/reject by the pre-registered criteria.
3. Optionally, if external reviewers must run inference, publish a small
   release bundle (checkpoint + manifest hashes) outside the repo — the
   repository alone cannot serve the model.
