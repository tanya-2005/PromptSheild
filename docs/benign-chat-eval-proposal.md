# Benign-chat generalization evaluation — proposal

Status: **protocol frozen 2026-10-19.** No evaluation data has been downloaded,
the test split has not been loaded, and no source code, threshold, model, or
gateway policy has been changed at freeze time. This document is the
pre-registration record for a single evaluation run; it must not be edited
after the first dataset download.

## Question

The validation split records an **observed** benign false-positive rate of
42/407 = 10.3% (95% Wilson CI 7.7%–13.7%). This 10.3% is a benchmark measured
on the current validation slice; it is **not** an estimate of the true
population false-positive rate, and the validation interval is reported
alongside the new interval. False positives there concentrate at high
confidence (mean class-1 score 0.9903, minimum 0.9249). A five-prompt local
diagnostic blocked 5/5 benign educational prompts at 0.9666–0.9990, which is
consistent with those numbers but too small to establish anything.

The evaluation asks one narrow question: **on ordinary, clearly benign
educational and conversational prompts from unrelated public datasets, is the
gateway block rate consistent with the documented ~10%, or materially worse?**

This is a generalization check of an existing, unchanged system. It is not a
model comparison, not a calibration study, and not a reason to retune.

## Datasets (verified 2026-10-19 against official dataset cards / HF API)

| Dataset | License | Availability | Label meaning | Representativeness | Verdict |
|---|---|---|---|---|---|
| `databricks/databricks-dolly-15k` | CC BY-SA 3.0 | Ungated; 15,011 rows; one JSONL; English | Every row is a human-written instruction in 8 benign everyday/educational categories; card states prompts were not written with generative AI and the data is usable commercially | Directly matches the observed failure style (`open_qa`, `brainstorming`, `closed_qa`) | **Primary** |
| `OpenAssistant/oasst1` | Apache-2.0 | Ungated; 84.4k train rows; 35 languages | `role=prompter` rows are human user prompts; per-message reviewer labels (`spam`, `hate_speech`, `not_appropriate`, `sexual_content`, `pii`, `quality`), `review_result`, `detoxify` scores, `lang`, `parent_id` | Real user chat, including educational requests ("explain X in simple terms") | **Secondary stratum** |
| `facebook/empathetic_dialogues` | CC BY-NC 4.0 (non-commercial) | Ungated; ~19k English supportive dialogs | Openers are benign by construction | Emotional-support dialogs only; single narrow style | Conditional; **skip by default** (NC license, low diversity) |

Considered and rejected:

- `lmsys/lmsys-chat-1m`: gated ("auto"); the raw dataset card returned HTTP 401
  without accepting terms; no license is declared in the official metadata;
  contains adversarial/unsafe prompts by construction. Not suitable now.
- `Anthropic/hh-rlhf`: MIT and ungated, but the official card describes
  preference pairs (`chosen`/`rejected`), states the data is not meant for
  supervised dialogue training, and explicitly warns that the harmlessness and
  red-team subsets contain offensive/harmful-topic content. A benign label
  cannot be established without heavy manual auditing; rejected for this
  purpose.
- DailyDialog and MSDialog mirrors checked under common HF ids returned 401
  (unavailable/gated at those ids).

No manually curated fallback is needed: two suitable, permissively licensed,
ungated datasets exist.

### Licensing considerations

- Dolly is CC BY-SA 3.0: attribution required, share-alike applies to
  redistribution of adapted material. Raw files stay local and are never
  redistributed, and only aggregate metrics are published, so share-alike is
  not triggered. Attribute "databricks-dolly-15k © Databricks, CC BY-SA 3.0"
  in the report and README.
- OASST1 is Apache-2.0: permissive; credit "OpenAssistant/oasst1, Apache-2.0".
- Aggregate-only reporting also avoids any question of redistributing dataset
  text, which is an additional reason for the privacy rules below.
- Do not use the CC BY-NC EmpatheticDialogues data unless a third stratum is
  explicitly needed and non-commercial scope is acceptable.

## Sample size: 200 (100 Dolly + 100 OASST1 roots)

Why 200 is the smallest defensible size:

- At an observed rate of 10%, the Wilson 95% CI is 6.6%–14.9% (±4.2 pp),
  which overlaps the validation CI 7.7%–13.7%.
- Zero observed false positives gives a 95% upper bound of 1.9% (rule of
  three would say 1.5%; Wilson is the conservative figure used here).
- A gross failure (25%) yields CI 19.5%–31.4%, clearly separated from the
  documented range. At 16% the lower bound is 11.6%, already above the
  validation point estimate.
- 100 prompts total would be too narrow: at 10%, CI 5.5%–17.4% cannot
  separate "documented" from "twice as bad". 200 is the minimum that can.
- Cost: one ~13 MB download and under a minute of CPU batch scoring.

What n = 200 supports:

- An overall benign gateway block-rate estimate with roughly ±4–7 pp
  precision, an upper bound when zero false positives occur, and a defensible
  "consistent with / materially worse than documented" decision.

What it cannot support:

- Per-dataset or per-category rates (n = 100 each: exploratory only, CI
  5.5%–17.4% at 10%).
- Any injection-detection/recall claim (no malicious rows).
- Calibration claims (ECE/Brier on 200 points are not meaningful).
- Non-English behavior, multi-turn behavior, or production-traffic behavior.
- Reuse after any future mitigation: this slice becomes development data.

## Execution-integrity preconditions

- `data/raw-eval/` is in `.gitignore` and verified with `git check-ignore`
  before the first download; raw prompts stay local.
- Execute once, after this protocol is frozen; no reruns after results are
  seen.
- If eligibility or licensing fails — including fewer eligible rows than the
  pre-registered minimum with no allowed same-dataset contingency — stop and
  report the blocker. Do not silently substitute datasets, relax eligibility
  rules, or lower the target n=200.
- The frozen checkpoint (`models/deberta-v3-small`), `DebertaInference`
  settings, `DEFAULT_THRESHOLD` (0.90), and gateway policy `band_v1` are used
  unchanged: no retraining, no threshold tuning, no policy changes.
- `data/raw/test.parquet` is never opened; deduplication runs only against
  `train.parquet` and `validation.parquet`.
- All console output and tracked reports are aggregate-only: counts, rates,
  score statistics, hashes, and rule IDs — never prompt text.

## Protocol

0. **Pre-register** this document before downloading anything.
1. **Privacy precondition:** add `data/raw-eval/` to `.gitignore` and verify
   with `git check-ignore` before the first download.
2. **Download pinned revisions** into `data/raw-eval/<dataset>/`:
   `databricks/databricks-dolly-15k` at commit
   `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` (JSONL) and
   `OpenAssistant/oasst1` at commit
   `fdf72ae0827c1cda404aff25b6603abec9e3399b` (parquet). Before scoring,
   verify that the resolved commit, license metadata, and file SHA-256 match
   the values recorded in `data/eval-manifest.json`, and record attribution
   text. Raw files stay local and are never redistributed.
3. **Build the slice (inclusion/exclusion):**
   - Dolly: use the `instruction` field only; exclude blank/short (< 4 chars)
     and normalized-duplicate instructions.
   - OASST1 (documented annotation semantics, per the official dataset card):
     `labels` values are reviewer-aggregate ratings and the ready parquet
     already excludes spam and deleted messages. A row is eligible only with
     **positive evidence of benign quality**, not merely an English tag or
     absent flags:
     - `role = prompter`, `lang = en`, `parent_id = null` (context-free root),
       `deleted = false`, `tree_state = ready_for_export`;
     - `review_result = true` and `review_count >= 3` (multiple independent
       reviews);
     - zero reviewer evidence for every negative label: `spam`,
       `lang_mismatch`, `pii`, `not_appropriate`, `hate_speech`,
       `sexual_content`, `toxicity`, `violence` (each value = 0);
     - positive quality evidence: `quality` is present with value ≥ 0.5 (at
       least half the aggregate review weight rates the message high-quality);
     - automated second signal: `detoxify.toxicity < 0.2` and
       `detoxify.severe_toxicity < 0.1`;
     - stripped text length ≥ 10 characters.
     Contingency (pre-registered; same dataset and identical rule): if fewer
     than 100 eligible roots exist, non-root English prompter messages
     satisfying the identical rule may be used and the deviation recorded.
     If still fewer than 100 eligible rows exist, stop and report the blocker
     instead of substituting a different dataset.
   - Select deterministically: sort qualifying rows by
     SHA-256(normalized text) and take the first 100 per dataset. This is
     seed-free and reproducible.
4. **Deduplication (counts only):**
   - Normalize (lowercase, collapse whitespace, strip edge punctuation).
   - Exact-match against `train.text` and `validation.text`; exclude matches.
   - Near-duplicate check using the existing manifest method (same
     first-50-normalized-character bucket, `difflib.SequenceMatcher` ratio
     ≥ 0.95); exclude candidates.
   - Pairwise dedup within the new slice (Dolly vs OASST1, and within each).
   - `test.parquet` is never opened. This is deliberate: preserving the
     untouched-test guarantee outranks test-side dedup. Residual leakage risk
     is low because the upstream training sources (`original`,
     `wildguard_judgecomp`, `harmbench`, `harmbench_benign`, `hackaprompt`,
     `neuralchemy_v1`) do not include these datasets.
5. **Score:** use a new guarded runner (code to be added in a later task; it
   does not exist yet) that calls `DebertaInference` directly on CPU (no HTTP,
   hence no access logs) and applies the unchanged `band_v1` policy to each
   score plus its matched marker rules. Positive control: the four malicious
   override prompts from `scripts/fp_benign_diagnostic.py` must all block; if
   any does not, abort the report as a pipeline error.
6. **Metrics (aggregate only):**
   - Primary: gateway block rate on benign prompts (= classifier false
     positive rate, because blocking requires score ≥ 0.90), with Wilson 95%
     CI.
   - Secondary: review rate, allow rate, score min/median/mean/max, fixed
     score buckets ([0, 0.1), [0.1, 0.7), [0.7, 0.9), [0.9, 1.0]), and
     counts of matched marker rule IDs.
   - Report per-dataset rates as exploratory only.
7. **Pre-registered operational decision rule (go/no-go).** This is an
   operational rule, not a formal statistical test of equality between
   populations; failure to distinguish the new interval from the validation
   benchmark is not evidence that the rates are equal. Report the new Wilson
   95% CI alongside the observed validation benchmark 10.3% (CI
   7.7%–13.7%). With the new interval [lo, hi]:
   - (a) interval contains 10.3% → no clear evidence of a difference at this
     sample size; **no mitigation**; document the limitation and finish the
     portfolio project.
   - (b) lo > 10.3% → materially worse than the observed validation
     benchmark; open a separate, small mitigation-design task. This
     evaluation then becomes development data and may not be reused to claim
     a post-fix improvement; a fresh holdout is required.
   - (c) hi < 10.3% → better than the observed benchmark; document and finish.
   - Run exactly once. Do not change the 0.90 threshold, the 0.70 band, or the
     gateway policy in response to the result.
8. **Deliverables:** `reports/benign-chat-eval-<timestamp>.json` and `.md`
   containing aggregate metrics, dataset metadata, input hashes, exclusion
   counts, and `"test_split_loaded": false`; plus one README paragraph stating
   what the measurement does and does not show.

## Privacy rules

- Raw prompts exist only under `data/raw-eval/` (gitignored, local).
- Console output contains counts, rates, score statistics, and rule IDs only;
  the runner never interpolates prompt text into prints or logs (pattern:
  `scripts/fp_benign_diagnostic.py`).
- No HTTP layer is used, so no server access logs contain prompts.
- Tracked reports contain no prompt text and no reversible text hashes. If
  per-row debugging is ever needed, it stays in the gitignored local directory
  with opaque ordinals only.
- The optional 20-row manual spot-check is read locally; only pass/fail counts
  are recorded.

## Risks

- **External validity:** public dataset prompts are not product traffic;
  English-only, single-turn framing. This bounds every conclusion.
- **Label noise:** OASST1 reviewer labels are crowd-sourced; Dolly's card
  itself says "caveat emptor" about contributor quality. Mitigated by the
  filter set, the deterministic sample, and a small local spot-check.
- **Contamination:** exact/near-duplicate matches are removed against
  train+validation only; the test split is never touched. Residual risk is low
  and is disclosed.
- **Post-hoc bias:** single pre-registered run, fixed decision zones, no
  reruns after seeing results.
- **Truncation:** the model caps at 128 tokens; the fraction of eval prompts
  exceeding that is reported.
- **License drift:** pin revisions and hashes; attribute both datasets.

## Recommendation

**GO** with two datasets (Dolly primary, OASST1 secondary) at n = 200, one
run, using the decision rule above. Skip the non-commercial third dataset and
the unsuitable candidates. If the result is consistent with the documented
~10%, the honest conclusion is that the false-positive cost is already
documented, not a new defect, and the portfolio project can be finished with
this limitation stated plainly. Only a clearly worse result justifies further
work, and that work is a separate mitigation task with a fresh holdout.
