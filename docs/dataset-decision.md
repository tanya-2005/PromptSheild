# Dataset decision

## Recommendation

Use **[NeurAlchemy Prompt Injection and Jailbreak Detection
Dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset)**,
starting with its `core` configuration, subject to a license and data review
before downloading or redistributing it.

This is the best fit for PromptShield's first real experiment because its
dataset card explicitly describes:

- binary `label` values (`1` malicious, `0` benign);
- a `core` configuration intended for classical ML;
- `text`, `label`, `category`, `source`, `severity`, `group_id`, `augmented`,
  and `tags` fields;
- separate train, validation, and test splits;
- original and augmented examples, with a stated group-aware split policy;
- both malicious examples and benign/hard-negative examples.

The card reports 4,391 train, 941 validation, and 942 test rows for `core`
(6,274 rows total), and approximately 60% malicious / 40% benign. These are
publisher-reported figures, not results independently verified by PromptShield.
The card also reports that `full` has 14,036 training rows because it contains
augmentation; we will not use `full` for the first TF-IDF experiment.

## Evidence and license

The canonical source URL is the Hugging Face dataset page above. Its card
declares **Apache License 2.0** and lists several upstream sources with their
own terms (including CC BY 4.0, MIT, and research-derived data). Therefore,
“Apache-2.0” should not be treated as proof that every upstream example has
identical redistribution terms. Before acquiring the data, inspect the
dataset revision, its files, and the upstream terms recorded in the card.
Keep the source revision and attribution in experiment metadata.

The dataset card is unusually detailed, but it is still a self-authored
dataset description. In particular, the reported benchmark scores and
“zero leakage” claim are not PromptShield measurements and must not appear as
our model results.

## Alternatives considered

### deepset/prompt-injections

[Canonical page](https://huggingface.co/datasets/deepset/prompt-injections).
The Hugging Face metadata exposes a small, easy-to-audit schema: `text` and
`label`, with 546 train and 116 test examples. The label is documented in
common usage as `0` benign and `1` attack. It is useful as a smoke-test or
secondary benchmark, but it is too small for the primary portfolio
experiment. Its metadata is also inconsistent: the page's tags declare
Apache-2.0 while the dataset card's `dataset_info` declares CC BY 4.0. That
license ambiguity should be resolved with the publisher before reuse.

### Dataset catalogs and newer benchmarks

The [prompt-injection-datasets catalog](https://github.com/rossja/prompt-injection-datasets)
is useful for discovery, but it is a catalog rather than a single training
dataset. Newer multimodal or agent-focused benchmarks may be valuable later,
but they are not the simplest starting point for a transparent TF-IDF
baseline.

## Risks to test explicitly

1. **Duplicates and near-duplicates:** variants of the same attack can cross
   a naive random split and inflate metrics.
2. **Group leakage:** if `group_id` is present, split by group, never by row.
3. **Synthetic-language bias:** attack prompts may contain obvious phrases
   such as “ignore previous instructions,” making lexical baselines look
   stronger than they are.
4. **Class imbalance:** the reported 60/40 distribution is moderate, but all
   metrics must include per-class counts and a confusion matrix.
5. **Scope mismatch:** prompt injection, jailbreaks, harmful requests, and
   indirect document injections are related but not identical tasks.
6. **Temporal/source leakage:** preserve source and revision metadata, and
   reserve an adversarial/difficult-benign evaluation set that is not used for
   fitting or threshold selection.

## Stage 2 acquisition record

Stage 2 acquired only the three `core` Parquet files at revision
`7d70432dfcf47a821612cbf9d34e9d9e3ad20e75`. The raw files are stored locally
under `data/raw/` and ignored by Git. Their file sizes and SHA-256 hashes are
recorded in [data/manifest.json](../data/manifest.json).

The actual files contain:

| Split | Rows | Label 0 | Label 1 | Augmented | Unique groups |
|---|---:|---:|---:|---:|---:|
| train | 4,391 | 1,741 | 2,650 | 0 | 4,391 |
| validation | 941 | 407 | 534 | 0 | 941 |
| test | 942 | 390 | 552 | 0 | 942 |

All splits have the same eight columns:
`text`, `label`, `category`, `source`, `severity`, `group_id`, `augmented`,
and `tags`. The existing CSV validator was run against each downloaded split
through a temporary CSV projection; all three were valid, with zero invalid
rows and zero exact duplicate texts within each split.

The cross-split checks found zero exact text overlap and zero group overlap for
every split pair. A conservative near-duplicate heuristic found 48 candidate
pairs in total, including 23 across split boundaries. These are not confirmed
leaks: the heuristic groups texts by their first 50 normalized characters and
uses a `SequenceMatcher` ratio of at least 0.95. They require manual review
before any model results are interpreted.

Every row is marked `augmented: false`, but the data still contains multiple
sources and many category names. In particular, some label-0 rows have
categories such as `adversarial`, `edge_case`, or `encoding` rather than the
literal `benign` category. This is not automatically wrong: labels define the
binary task, while categories describe the example. It does mean that future
analysis must use the numeric label as the target and treat category as
metadata, not as a substitute label.

No model has been trained and no performance metric has been produced.
