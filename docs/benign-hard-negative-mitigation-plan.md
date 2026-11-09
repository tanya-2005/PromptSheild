# Benign hard-negative mitigation — experiment plan (pre-approval)

Status: **planning only, 2026-10-19.** No training has been run, no data
downloaded, no source code changed, nothing staged/committed/pushed. This
document defines exactly one experiment, executed once after approval.

## 1. What was inspected

| Area | Finding |
|---|---|
| Training pipeline | `src/promptshield/transformer_experiment.py`: fine-tunes from the **HF base** (`microsoft/deberta-v3-small`, revision `a36c7390…`), seed 42, lr 2e-5, batch 2, 1 epoch, **no shuffling**, reports **argmax** metrics, and saves into `--artifact` (would overwrite the baseline if misused). It cannot warm-start from the local checkpoint and does not report the serving decision rule (0.90 threshold), so it cannot be reused as-is. |
| Label mappings | `{0: "benign", 1: "malicious"}` consistent across `baseline.py`, `transformer_experiment.py`, `inference.py`; all metrics treat label 1 as the positive class. Checkpoint `config.json` has **no** `id2label`/`num_labels` (must be set explicitly on any new save; baseline files untouched). |
| Dataset manifest | `data/manifest.json`: train 4,391 (label 0: 1,741 = 1,699 `category=benign` + 42 `edge_case`; label 1: 2,650, zero benign-category rows), sha256 pinned, cross-split exact/group overlap **0**, but **23 unreviewed near-duplicate candidate pairs** across splits. |
| Model artifacts | `models/deberta-v3-small/` (model.safetensors 567,598,552 B; audited sha256 `2cefb8e4…`; 141M params, vocab 128,100, 6 layers), `baseline.joblib`, `linear-svm.joblib`. `models/` is gitignored. |
| Evaluation tooling | `deberta_decision_audit.audit_validation()` + CLI `--checkpoint <dir>`: runs **any** local checkpoint on `data/raw/validation.parquet` only (path-locked), threshold rule `softmax[:,1] ≥ 0.90` plus argmax for reference, and emits via `calculate_metrics`: benign FPR, malicious FNR, per-class precision/recall, macro F1, confusion matrix. Reports are timestamped (no overwrites). `scripts/fp_benign_diagnostic.py` provides the 4/4 malicious positive control. |
| Baseline reference (threshold 0.90, validation) | FPR **42/407 = 10.32%**, FNR **9/534 = 1.69%**, macro F1 **0.9442**, class-1 precision **0.9259**, confusion matrix **[[365, 42], [9, 525]]** (from `reports/deberta-decision-audit-20261017T172138315024Z.json`). |

## 2. Read-only pre-check performed (this task, aggregates only)

Frozen checkpoint scored on train label-0 rows (n = 1,741):

- blocked (`p1 ≥ 0.90`): **159**; review band (0.70–0.90): 13; allowed (<0.70): 1,569
- restricted to `category=benign`: **138 blocked** (the hard-negative pool)
- score distribution is bimodal: median 0.0005, max 0.9996
- leakage checks: train label-0 vs validation → **0 exact, 0 near**; blocked pool vs validation → **0 exact, 0 near**
- dev-slice separation: the benign-chat evaluation recorded 3 Dolly prompts matching train+validation verbatim → explicit exclusion rule in §3.3

## 3. Experiment design (fixed before implementation)

### 3.1 Data source
Only `data/raw/train.parquet` (sha256 `c0724f8e…`, revision `7d70432d…`).
Validation is read only by the evaluation tooling (and, at build time, its
text column for exclusion checks) — never for training loss. `test.parquet`
and `data/raw-eval/` are never inputs.

### 3.2 Label semantics
`label == 0` = benign (dataset's training target), `label == 1` = malicious;
metrics use label 1 as positive. Benign training eligibility additionally
requires `category == "benign"` (excludes the 42 `edge_case` rows, which are
label 0 but not affirmatively benign).

### 3.3 Hard negatives (HN) — eligible training rows only
`label == 0 AND category == "benign"` AND frozen-checkpoint `p1 ≥ 0.90`
(same decision rule as serving) AND **not** an exact/near (≥0.95, same
first-50-char bucket) match of any validation text AND **not** an exact
normalized match of any `data/raw-eval/` source prompt.
Expected pool: **exactly 138 rows** (pre-measured today; exclusion checks are
0 for validation). Any deviation at build time → **stop and report** — no
relaxation, no substitution.

### 3.4 Anchors (protect recall; 1:1:1 balance)
- 138 malicious: `label == 1`, same exclusion rules,
- 138 easy benign: `label == 0 AND category == "benign" AND p1 < 0.10`,
  same exclusion rules.

Selection in both cases: ascending `sha256(normalized text)`, first k.
Normalized-exact dedup within the final set (group_ids are unique per
manifest). **Training set = 414 rows.** A metadata manifest
(`data/hardneg-manifest.json`: counts, hashes, selection rule, excluded
counts — **no text**) is written before training; mismatch with §2 numbers →
stop.

### 3.5 Training procedure (one run, no sweep)
- **Warm start from the frozen checkpoint files** (`models/deberta-v3-small`),
  not the HF base (no download needed; HF base is not cached locally).
- **Freeze input embeddings** (`DebertaModel.embeddings`, ~70% of params):
  trains encoder + pooler + classifier only — reduces RAM from ~3.1 GB to
  ~2.0 GB and limits lexical drift.
- AdamW, **lr 1e-5**, weight decay 0.01, no scheduler; **epochs 3**
  (≈311 optimizer steps at batch 4), batch 4, max_length 128, fp32, CPU,
  threads 4, **seed 42**, seeded per-epoch shuffle (the existing script does
  not shuffle; the new module will).
- Labels set to `{0: benign, 1: malicious}` on save; save to a **new
  directory** `models/deberta-v3-small-hardneg-v1/` — baseline directory
  never written.
- Exactly one training run; no early stopping, no hyperparameter choices
  after results are seen.

### 3.6 Compute requirements
141M params total, 42.6M trainable → peak RSS ≈ **1.9–2.3 GB**; training
≈ **3–10 min** on the i5-12500H (311 steps × ~0.5–1.5 s), two validation
audits ≈ 5–15 min. Precondition: **≥3.5 GB free RAM** at start (currently
3.2 GB — see blocker B1). No network needed.

## 4. Evaluation protocol (identical for both models)

- Tool: `deberta_decision_audit` CLI on `data/raw/validation.parquet`, run
  twice: `--checkpoint models/deberta-v3-small` (fresh baseline) and
  `--checkpoint models/deberta-v3-small-hardneg-v1`. Threshold **0.90 fixed
  for both**; threshold rule is primary, argmax reported only for
  completeness. Timestamped reports — existing reports untouched.
- **Execution sanity gate:** the fresh baseline audit must reproduce
  `[[365, 42], [9, 525]]` exactly; otherwise abort with no conclusions.
- Reported for both: benign FPR (+ FP count), malicious FNR and recall,
  class-1 precision, macro F1, confusion matrix, full per-class table,
  label supports. Secondary (no go/no-go weight): gateway band distribution
  (0.70/0.90) and the 4/4 positive control on the candidate.
- **Never used as evidence:** `test.parquet`; the 200-example dev slice
  (not run by default; if ever run, labeled exploratory/dev-only, never a
  holdout claim).

## 5. Go/no-go criteria (pre-registered — retain candidate iff ALL hold)

| # | Criterion | Threshold |
|---|---|---|
| G1 | Validation benign FPs | **≤ 21 / 407** (FPR ≤ 5.16%, ≥50% reduction from 42) |
| G2 | Validation malicious FNs | **≤ 14 / 534** (FNR ≤ 2.62%, ≤ +1.0 pp over 1.69%) |
| G3 | Macro F1 | **≥ 0.9342** (baseline − 0.0100) |
| G4 | Class-1 precision | **≥ 0.9059** (baseline − 0.0200) |
| G5 | Positive control (candidate) | 4/4 malicious override prompts blocked |
| G6 | Integrity | baseline audit reproduced exactly; threshold 0.90 and policy `band_v1` unchanged; `test.parquet` never opened; reports aggregate-only |

**Any failure → reject:** the baseline remains the only presented model, the
candidate artifact and reports are kept labeled "rejected candidate" for the
write-up, the outcome is documented honestly, and the project finishes.
No second training run, no threshold tuning, no policy changes — one
experiment by design.

## 6. Contamination and preservation guards

- Training inputs: train split only, with validation/dev-slice exclusion
  rules above; stopping rules instead of substitution if pools are short.
- Baseline preserved: separate artifact directory, timestamped reports,
  no edits to `models/deberta-v3-small/`, `models/` gitignored.
- Raw prompts: manifests contain ids/hashes/counts only; console output and
  tracked reports are aggregate-only (pattern from the existing audits).
- No staging, committing, or pushing in implementation either.

## 7. Blockers and risks

- **B1 (conditional, most likely): free RAM is 3.2 GB.** Training needs
  ~2.0–2.3 GB peak with frozen embeddings; precondition ≥3.5 GB at start
  (close applications first). If not met → stop and report rather than risk
  an OOM mid-run; fallback is the existing (untested) Colab notebook
  pattern, which would need separate approval.
- **B2: existing script unusable** for this experiment (trains from HF base,
  argmax reporting, artifact overwrite risk) → implementation requires a new
  module `src/promptshield/hard_negative_experiment.py` + tests, with guards:
  refuse any test path, refuse the baseline artifact path, aggregate-only
  output. Code changes are part of the implementation approval, not this
  task.
- **B3: under-correction risk** with a single fixed schedule (138 HNs,
  3 epochs, lr 1e-5): possible that G1 fails on the first try; accepted —
  the result would then be a documented negative finding, not a reason to
  start tuning.
- **B4: label noise** in crowd-labeled training data → mitigated by the
  strict `category=benign` restriction and by G2/G4 recall/precision gates.
- **B5:** checkpoint `config.json` lacks label metadata → candidate save
  must set `id2label`/`label2id` explicitly; baseline config untouched.

## 8. Execution order after approval

1. Implement `hard_negative_experiment.py` (+ unit tests): build step,
   training step, evaluation orchestration; path guards; aggregate-only I/O.
2. Build the 414-row set; verify counts against §2 (deviation → stop).
3. RAM precondition → train once → save candidate + metadata manifest.
4. Audits: baseline (repro gate), candidate; positive control; write
   `reports/hard-negative-mitigation-<ts>.{json,md}`.
5. Apply G1–G6 → decision recorded in the report + one README paragraph
   (honest result either way).
6. Full test suite + `git diff --check`; no stage/commit/push.
