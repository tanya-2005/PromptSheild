# Benign hard-negative mitigation — Colab execution runbook

Status: **preparation only, updated 2026-10-26. Training has NOT been run.**
The frozen plan is `docs/benign-hard-negative-mitigation-plan.md`; this runbook
only relocates its *unchanged* execution to Google Colab because local free RAM
(blocker B1) is below the pre-registered 3.5 GiB precondition (1.3 GiB
available at time of writing). The Colab fallback referenced in plan §7 needed
separate approval and has now been approved with the safeguards below. The
2026-10-26 update hardens the TorchVision/TorchAudio check; it does not change
the experiment, the pins, or the gates.

Nothing here changes the dataset selection, hyperparameters, threshold (0.90),
gateway policy (`band_v1`), acceptance criteria (G1–G6), or the one-run rule.
The single entry point is still `src/promptshield/hard_negative_experiment.py`
(the same module and its 15 tests); the notebook only stages files, verifies
the environment, reproduces the baseline audit, and invokes it.

The only code change made for this environment is Linux support in the
pre-existing RAM gate (`parse_meminfo_available` / `free_physical_memory_bytes`):
on Linux there is no `ctypes.windll`, so the gate previously could not measure
memory and failed closed. The 3.5 GiB threshold and fail-closed behavior are
unchanged; a test covers the parser.

## 1. What is transferred — and nothing else

Exact inventory (37 files across 7 categories; ~629 MB of files,
**425.8 MiB / 446,441,007 bytes zipped**):

| Path | Purpose | sha256 |
|---|---|---|
| `src/promptshield/` (whole package, source only) | experiment + audit code | — |
| `data/raw/train.parquet` | the only training source (plan §3.1) | `c0724f8e00eebb65c001be9ba94291d1608e5da0374012e53dfd4077c5f92558` |
| `data/raw/validation.parquet` | evaluation only, never training loss | `bbd417bfd0df1553dd769e814a0b62ca58ad5b1d08d153c90f66937a4af74deb` |
| `data/raw-eval/dolly/databricks-dolly-15k.jsonl` | §3.3 dev-slice exclusion source | `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec` |
| `data/raw-eval/oasst1/train.parquet` | §3.3 dev-slice exclusion source | `bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e` |
| `models/deberta-v3-small/` (7 files) | frozen baseline: warm start, scoring, audits | see table below |
| `scripts/fp_benign_diagnostic.py` | frozen 4/4 positive-control prompts (G5) | `e262d65ad966eac03777f77ff402d197f27a210ebd5f71400a8a03f8d600c244` |

Baseline checkpoint file hashes (from the frozen 2026-10-17 audit; the notebook
verifies them before *and* after the run):

| File | sha256 |
|---|---|
| `added_tokens.json` | `dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f` |
| `config.json` | `44e8fe36a9ea5659b039545691a0896e21ab5629f37b2c7f4480354433902f0d` |
| `model.safetensors` | `2cefb8e40ade17940899ef523c7dea2a1bd1813ba0373edda84481a32c783614` |
| `special_tokens_map.json` | `9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9` |
| `spm.model` | `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd` |
| `tokenizer.json` | `37fe3823ad5737f0033c832905c8d4089ed4008da79682ab3fd85c8ce8e5b040` |
| `tokenizer_config.json` | `e6c4c771911c211618a2d46488dc7e9e499f051773cd4ff4faeca6fe55fdf569` |

**Never transferred:** `data/raw/test.parquet` (or any file named
`test.parquet`), the rest of the repository, `.git/`, Git credentials, reports,
`docs/`, `tests/`, environment/secret files. The bundle step asserts these are
absent; the notebook re-asserts them after extraction. **No external model or
dataset is downloaded** in Colab — the notebook forces Hugging Face offline
mode and installs only pinned PyPI packages (`torch==2.6.0+cpu`,
`transformers==4.49.0`, `sentencepiece==0.2.1`, `pyarrow==22.0.0`,
`scikit-learn==1.8.0`, `joblib==1.5.3` — the versions recorded in the frozen
baseline audit). `torchvision` and `torchaudio` are intentionally **absent**:
PromptShield never imports them, and Colab's preinstalled builds are linked
against its own torch. Cell 2 removes both before **and after** the pinned
installs, then verifies distribution metadata, import specs, and loaded
submodules in that same cell. Cell 3 repeats the check before and after
importing Transformers, rejects cached Transformers availability flags or
stale `transformers.image_utils` references, and forces the same optional
vision path plus the concrete DeBERTa classes before Gate 0. Cell 3 prints
`guard build hardneg-env-v3`; Cells 4 and 5 refuse to run unless that exact
guard completed in the same runtime.

## 2. Local step — build the bundle (PowerShell, run from anywhere)

**Why this builder exists (2026-10-26):** the earlier `Compress-Archive`
bundle stored backslash (`\`) separators in the ZIP central directory. Linux
Colab does not treat `\` as a directory separator, so `extractall()` created
flat files named like `src\promptshield\...` and Cell 1 failed its required-file
assertion. `scripts/build_colab_bundle.py` writes POSIX `/` separators, checks
the exact allowlist, and verifies every pinned hash before publishing the ZIP.
If you still have the old ZIP, **do not upload it** — rebuild and replace it.

The builder reads directly from the repository and writes only the ZIP in
`%TEMP%`; no tracked or untracked project files are modified:

```powershell
$ErrorActionPreference = "Stop"
$repo   = "C:\Users\tanya\Documents\PromptSheild"
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python (Join-Path $repo "scripts\build_colab_bundle.py")
```

The script prints the archive path, byte size, SHA-256, and verification flags.
It fails closed (and removes the partial file) if any of these checks fail:

- exact 37-file allowlist (`src/promptshield/*.py`, 4 data files, 7 baseline
  checkpoint files, 1 control script);
- every ZIP entry name uses `/`, not `\`;
- ZIP CRC test;
- `test.parquet` absent;
- `.git`, `__pycache__`, `.venv`, and `.pyc` absent;
- all 12 pinned SHA-256 values match both source and ZIP members.

**Artifact rebuilt and verified 2026-10-26** (by the agent, using the command
above): `%TEMP%\promptshield-colab-bundle.zip`, 446,441,007 bytes
(425.8 MiB), sha256
`e87c1efb4a6e8394e6333bd0457c67eebeb0dc1e399a82c549d9d63c729d3d56`.
Verification results: ZIP CRC test pass; 37 file entries, all inside the
approved allowlist; zero `test.parquet`, `.git`, `__pycache__`, `.venv`, or
`.pyc` entries; all 12 pinned files (train, validation, both eval sources,
control script, 7 baseline checkpoint files) hash-identical to the frozen
values above; every entry name is POSIX `/`. If you rebuild the bundle later,
the timestamped contents are identical but the ZIP hash will differ; the
notebook re-verifies the content hashes that matter (train, validation,
baseline) after extraction.

## 3. Transfer to Colab

Upload `promptshield-colab-bundle.zip` (**~426 MiB**) to **your private Google
Drive**, recommended at `MyDrive/Colab Notebooks/promptshield-colab-bundle.zip`
(the notebook's default `BUNDLE_ZIP`). If you store it elsewhere, update
`BUNDLE_ZIP` in the Config cell. Do not share the folder. Drive is the only
intermediary; no repository, no credentials, and no other local files are
involved.

## 4. Colab steps (user-executed; nothing starts automatically)

1. **Re-upload the current notebook.** Open Colab
   (https://colab.research.google.com), choose *File → Upload notebook*, and
   select the local file `notebooks/hard-negative-colab.ipynb`, replacing the
   old Colab copy. Do **not** rebuild or re-upload the ZIP for this fix: the
   bundle is unchanged. After uploading, confirm the header says
   `Environment guard build: hardneg-env-v3 (2026-10-26)`, Cell 3 contains
   `ENV_GUARD_BUILD = 'hardneg-env-v3'`, and Cell 3 prints
   `environment verified; guard build hardneg-env-v3`.
2. **Runtime → Change runtime type → Hardware accelerator: CPU** (no GPU,
   no one-click GPU). Training and the audit protocol are CPU-frozen.
3. **Reset the runtime properly before running anything.** If any previous
   cell failed (including the TorchVision mismatch), choose
   **Runtime → Disconnect and delete runtime**, wait for the session to end,
   then reconnect. Do not merely use *Restart session* and do not reuse the
   old tab: a restart can preserve the process state that cached a broken
   `torchvision._C` or Transformers availability flag. After reconnecting,
   check that the runtime is new and CPU-only.
4. Run the cells strictly top to bottom:
   - **Config** — set `BUNDLE_ZIP` if your Drive path differs.
   - **Cell 1** — mounts Drive if needed (approve the pop-up), then extracts;
     verify layout, `train.parquet`/`validation.parquet`
     hashes, all 7 baseline checkpoint hashes; assert `test.parquet` and
     `.git` are absent.
   - **Cell 2** — remove Colab's optional `torchvision`/`torchaudio`, install
     the pinned dependencies (CPU torch wheel), remove the optional media
     packages again, and verify their metadata/specs/loaded modules are
     absent. It must print
     `Cell 2 verified: torchvision/torchaudio absent; ... guard build hardneg-env-v3`.
     Pip may print resolver warnings about unrelated preinstalled Colab
     packages; those are not the environment gate.
   - **Cell 3** — assert exact versions, CUDA unavailable, media packages
     absent before and after Transformers imports, Transformers did not cache
     them as available, `transformers.image_utils` imported, the concrete
     DeBERTa classes imported, offline mode, `promptshield` imported from
     `/content/promptshield/src`, and the **RAM gate (≥ 3.5 GiB)**. It must
     print `environment verified; guard build hardneg-env-v3`.
   - **Cell 4 (Gate 0)** — read-only baseline audit: must reproduce
     `[[365, 42], [9, 525]]` at threshold 0.90 or the cell fails → stop.
     Cell 4 also refuses to run unless Cells 2 and 3 completed in this same
     runtime.
   - **Cell 5** — **the single run** (`main()`): RAM gate → fresh baseline
     audit gate → hashes/selection gates (138/138/138 = 414 rows) → one
     training run → candidate audit → positive control → G1–G6. Cell 5 has
     the same same-runtime guard precondition.
   - **Cell 6** — re-verify baseline integrity; print the aggregate decision.
   - **Cell 7/8** — package and copy the return artifacts to Drive.
5. Stopping rules (any of these → **stop, record the output, report back;
   no retries, no tuning, no threshold or policy changes**):
   - Gate 0 or the in-run baseline gate does not reproduce `[[365, 42], [9, 525]]`;
   - any pin/version assertion fails;
   - Cell 2/3 reports a stale runtime, an old notebook, a `torchvision`/
     `torchaudio` package or submodule, a cached Transformers availability
     flag, or a Transformers import error (including
     `operator torchvision::nms does not exist`);
   - any pool count deviates from the plan (138/138/138);
   - `main()` returns blocked (`"status": "blocked"` JSON, exit code 2);
   - G1–G6 fail → the decision is **reject**, documented as a final negative
     result (keep the artifact labeled “rejected candidate”).

### If you see `operator torchvision::nms does not exist`

Colab preinstalls `torchvision`/`torchaudio` built against its own CUDA torch.
Installing `torch==2.6.0+cpu` leaves those optional packages mismatched, and
Transformers may import TorchVision while loading DeBERTa. Do **not** install
or pin TorchVision and do not change the torch version.

The earlier guard was insufficient because it checked only
`find_spec("torchvision")` and the exact top-level `sys.modules["torchvision"]`
key. A failed import can leave `torchvision._C` loaded, and Transformers caches
`is_torchvision_available()` when it is first imported; either state can survive
an uninstall and let a shallow check pass. The guard also did not force
`transformers.image_utils`, which is the module that actually imports
TorchVision, or the concrete DeBERTa classes used later.

The current Cells 2/3 close those gaps: metadata, import spec, every loaded
`torchvision.*`/`torchaudio.*` submodule, Transformers' cached availability
flags, stale `image_utils` references, and the DeBERTa class imports are all
checked before Gate 0. Cells 4 and 5 require the same `guard build
hardneg-env-v3` token from this runtime, so a stale Cell 3 output cannot satisfy
them.

If Cell 4 still reports this error after a green Cell 3, **stop; do not train**.
Record the full Cell 2 and Cell 3 output (including the guard build line) and
report back. It means the old notebook or an old runtime bypassed the guard;
do not retry with changed settings. The correct recovery is:

1. *File → Upload notebook* and replace the Colab copy with the current
   `notebooks/hard-negative-colab.ipynb`.
2. **Runtime → Disconnect and delete runtime**, wait, reconnect, and confirm
   CPU-only.
3. Run Config → Cell 1 → Cell 2 → Cell 3 only.
4. Confirm Cell 2 prints `guard build hardneg-env-v3` and Cell 3 prints
   `environment verified; guard build hardneg-env-v3`.
5. Only then run Cell 4. If any guard line is missing or different, stop and
   re-upload the notebook; do not run training.

## 5. Returning results

The artifact zip contains **only** `models/deberta-v3-small-hardneg-v1/`
(candidate, separate from the baseline), `data/hardneg-manifest.json`
(ids/hashes/counts only), and `reports/hard-negative-mitigation-<ts>.{json,md}`
(aggregate metrics/metadata only — no raw prompts, no `test.parquet`).

After downloading it to the local project:

1. Place the candidate at `models/deberta-v3-small-hardneg-v1/`; never touch
   `models/deberta-v3-small/` (re-hash it against the table above if in doubt).
2. Place the manifest at `data/hardneg-manifest.json` and the reports at
   `reports/` (existing reports are timestamped and must not be overwritten).
3. Record the decision (retain/reject) exactly as written in the report.
   A retained candidate is still **not** validated for deployment: the
   limitations in the plan (§5, report “Limitations”) stand — no performance
   claims beyond this validation comparison until an independent holdout exists.

## 6. Invariants verified by the notebook (summary)

- Bundle entries use POSIX `/` separators (portable on Linux Colab); the
  archive allowlist, CRC, and pinned hashes are verified locally before
  upload, and the notebook re-validates the layout before extraction.
- `test.parquet` never transferred, opened, loaded, or evaluated.
- Exactly one training run; existing candidate artifact blocks a rerun.
- Baseline checkpoint hash-verified before and after; never written.
- Threshold 0.90, policy `band_v1`, dataset selection, hyperparameters, and
  G1–G6 used unchanged from the frozen plan/module.
- Offline mode: no external model/dataset downloads.
- Optional media state: distribution metadata, import specs, loaded
  `torchvision.*`/`torchaudio.*` submodules, Transformers cached availability
  flags, and `transformers.image_utils` are checked before Gate 0; Cells 4/5
  require the same `hardneg-env-v3` guard token from this runtime.
- Console and report output aggregate-only; no raw prompts, secrets, or
  personal data logged or published.
