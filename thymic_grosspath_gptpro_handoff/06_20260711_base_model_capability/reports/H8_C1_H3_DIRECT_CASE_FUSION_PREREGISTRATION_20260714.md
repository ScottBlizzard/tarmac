# H8 C1-H3 Direct Case-Embedding Fusion Preregistration

Date locked: 2026-07-15, before any H8 classifier was fitted or evaluated

Experiment ID: `H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714`

Evidence basis: `ScottBlizzard/tarmac@ff81fe4c98fd5f1b9d2bcfd53f4379e5741e38e8`

## Decision and scope

H8 is the single terminal current-cohort classifier experiment authorized by
the final cross-representation oracle-gap audit. It tests whether a small
disease head can use same-case information retained in two different frozen
visual representations but discarded by their final probabilities.

The endpoint is unchanged: one selected gross photograph per case; A/AB/B1 is
low risk and B2/B3/TC is high risk; threshold 0.5; 100% coverage. No source,
subtype, text, concept, confidence, probability, logit, margin, correctness,
quality, or external-cohort feature enters the H8 head.

This is an exploratory internal acquisition-shift experiment on a repeatedly
used 591-case cohort. A positive result cannot establish multicenter,
prospective, clinical, or histological validity.

## Pre-metric implementation amendments

The GPT Pro specification assumed that the C1 dense bank had been deleted and
that a timm model cache would be used to regenerate it. Server preflight before
any H8 fitting found the opposite state:

- the exact C1 dense bank is present at
  `/root/thymic_feature_banks_20260711/phase2_siglipl512_whole_crop_quadrants6_512_internal`;
- its `591 x 6 x 1024 x 1024` float16 tensor has SHA-256
  `ff10fbf255cb0da08be19566b321947479c321ac1c7e5dee2e5158c8cc68c3ee`;
- its config and metadata SHA-256 values are respectively
  `6788fd540a6e2f54bd3d5f204528804625f1f1868524042e7a6c0dbf3d9edd55`
  and `8bf1a1746ddd6b2b3000160761093283ac5f10f2c33b5dc42fde0f456118f176`;
- the assumed timm snapshot directory is absent.

H8 therefore locks and consumes the already materialized C1 tensor instead of
regenerating it. Byte-locking the exact tensor used by C1 is stricter and less
operationally ambiguous than regeneration from a model cache. It does not
change a case, feature, checkpoint, split, label, or model decision.

The PE dense bank is absent. It is regenerated only in memory, one case at a
time, from the existing 591-image max2048 cache. Raw PE tokens are discarded
after the fold-specific 128-dimensional embeddings are produced. No new dense
bank is written.

A one-case, pre-metric reproduction test established the required numerical
mode:

- C1 extracted versus locked probability absolute error: `5.96e-8`;
- H3 extracted versus locked probability absolute error in the original H3
  inference mode: `6.94e-17`;
- forcing PyTorch deterministic algorithms during PE feature extraction changed
  the H3 probability by about `2.02e-4` and is therefore prohibited for frozen
  extraction.

The frozen PE extractor retains the exact existing bfloat16 autocast path,
fixed seed, cuDNN benchmark disabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Strict deterministic algorithms are enabled after extraction for the new FP32
H8 heads. This separates upstream byte-level reproduction from new-head
training determinism.

## Immutable data and assets

- Registry:
  `/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv`,
  SHA-256 `ef2d4e16b041038e36bd165c80d460b527c0c09eba19250b38f50d67199c62af`.
- Split:
  `/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv`,
  SHA-256 `48610c996298f8af317d547681665e0be20e4361aa61de66675efc51bb954545`.
- Cases: 591 unique; batch1 117, batch2 168, third batch 306.
- Risk totals: low 368, high 223.
- Subtypes: A 44, AB 262, B1 62, B2 89, B3 24, TC 110.
- All 591 cached image paths must be readable.
- PE checkpoint SHA-256:
  `47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1`.
- PE source snapshot tree SHA-256:
  `4e00f80f27fc360591f94eb125ed8ea18b86237ae59e5284abf3bde2ddebdbea`;
  declared revision `3e352cca660658d4b5c90f42a7808b11469e4c66`.
- All fold-specific C1 and H3 checkpoints and the locked C1, C2, and H3
  prediction files are hash-locked by the server manifest.
- The 108- and 162-case consumed stress cohorts are prohibited from extraction,
  fitting, selection, thresholding, gating, or rescue analysis.

## Frozen branch embeddings

For each outer fold and case:

- C1 uses its locked six-view SigLIP-L dense tensor and frozen fold-specific
  gated pooler to produce the normalized pre-classifier embedding
  `c in R^256`.
- H3 uses its six deterministic semantic views, frozen PE-Spatial-L14-448
  encoder, valid-token mask, and frozen fold-specific masked-gated pooler to
  produce the normalized pre-classifier embedding `h in R^128`.
- Both upstream encoders, poolers, normalizers, and classifiers remain frozen.
- C1 and H3 branch probabilities reconstructed from these embeddings must match
  every corresponding locked OOF prediction within `1e-5`. Any larger error is
  an integrity failure and stops H8 before training.

Only small per-case embedding shards are retained. C1 and H3 dense tokens are
never copied into a new persistent bank.

## Candidate head

For each sample:

```text
c_bar = c / max(L2(c), 1e-6)
h_bar = h / max(L2(h), 1e-6)
x = concat(c_bar, h_bar)
u = GELU(Linear(384, 16)(x))
z = LayerNorm(16)(u)
logits = Linear(16, 2)(Dropout(0.10)(z))
p_high = softmax(logits)[1]
```

The trainable parameter count is exactly 6,226. Linear weights use Xavier
uniform initialization with gain 1.0; linear biases are zero; LayerNorm scale
is one and bias is zero. There is no branch gate, attention layer, residual
path, temperature, calibration, or learned threshold.

## Fixed controls

Every learned control uses the identical 6,226-parameter implementation,
optimizer, sampler, seed, folds, and early stopping:

| Tag | Input |
| --- | --- |
| `C1_ONLY_PADDED` | `concat(c_bar, zeros(128))` |
| `H3_ONLY_PADDED` | `concat(zeros(256), h_bar)` |
| `C1_H3_EXACT` | `concat(c_bar_i, h_bar_i)` |
| `C1_H3_SAME_SOURCE_DERANGED` | `concat(c_bar_i, h_bar_pi(i))` |

The derangement is built independently within each outer fold, split role, and
source. Cases are ordered by
`SHA256("H8|fold|split|case_id")`, and H3 embeddings are cyclically shifted by
one. Every group must contain at least two cases, no case may retain its own H3
embedding, and labels/subtypes cannot influence the mapping.

Locked C1, C2, H3, and the fixed 0.5 C1 plus 0.5 H3 probability average are
descriptive comparators. Only `C1_H3_EXACT` can advance.

## Optimization

- Loss: unweighted two-class cross-entropy.
- Sampler: inverse-frequency source by binary-risk replacement sampler,
  calculated from outer-training cases only; `num_samples=len(train)`.
- AdamW: learning rate `3e-4`, weight decay `1e-4`.
- Batch size 32; FP32 head.
- Maximum 80 epochs; patience 12.
- Validation selection: balanced accuracy at threshold 0.5; earliest epoch wins
  every tie.
- Cosine annealing with `T_max=80`.
- Global gradient clipping: L2 norm 5.0.
- Primary seed: `20260714`; every control resets to the same seed.
- Conditional confirmation seed: `20260715` only after every primary and
  secondary gate passes.
- cuDNN benchmark off; deterministic algorithms on during new-head training.

## Leakage-safe partitions

Source-LODO is primary and must run first:

| Fold | Held source | Validation master fold among non-held cases |
| ---: | --- | ---: |
| 1 | batch1 | 2 |
| 2 | batch2 | 3 |
| 3 | third_batch | 4 |

For every fold, the upstream checkpoint excludes the held source, the H8 head
fits only outer-training cases, the sampler uses only those cases, and early
stopping uses only the predefined validation subset. No target-source cohort
statistic alters the model. Samplewise L2 normalization and LayerNorm are the
only normalizations introduced by H8.

Five-fold is forbidden unless all primary gates pass. It then uses the locked
five-fold C1/H3 checkpoints and the same architecture and optimization.

## Primary gates

All P1-P13 must pass simultaneously:

| Gate | Requirement |
| --- | --- |
| P1 | 591/591 predictions at threshold 0.5 |
| P2 | overall BAcc at least 0.7739 |
| P3 | TP at least 164/223 |
| P4 | TN at least 299/368 |
| P5 | B1 correct at least 40/62 |
| P6 | B2 correct at least 59/89 |
| P7 | third-batch B2 correct at least 18/29 |
| P8 | BAcc delta versus H3 positive in at least two of three held sources |
| P9 | no held-source delta below -0.0200 and minimum-source BAcc at least 0.7381 |
| P10 | exact BAcc at least 0.0100 above both padded branch controls |
| P11 | exact minus deranged BAcc at least 0.0100, positive in at least two sources, paired 95% CI lower bound above zero |
| P12 | exact minus locked H3 BAcc at least 0.0200 and paired 95% CI lower bound above zero |
| P13 | net correct gain versus H3 within B1 plus B2 at least seven, while P5 and P6 pass |

Paired intervals use 20,000 bootstrap replicates sampled within source by risk
strata. They are repeated-cohort stability diagnostics, not independent
confirmatory intervals.

## Conditional gates

If and only if P1-P13 all pass, five-fold must satisfy all of the following:

- BAcc at least 0.7903;
- sensitivity at least 176/223 and specificity at least 285/368;
- B1 correct at least 40/62 and B2 correct at least 60/89;
- no test fold BAcc below 0.7000;
- exact fusion at least 0.0100 above both padded branch controls;
- exact fusion at least 0.0100 above derangement, with paired 95% CI lower
  bound above zero;
- 591/591 coverage at threshold 0.5.

Only after that may seed 20260715 rerun source-LODO. It must pass P1-P13 again,
and the mean H3 BAcc delta across the two seeds must be at least 0.0200. Seeds
cannot be ensembled or selected post hoc.

## Required outputs and privacy

The aggregate report must include overall, source, subtype, source-by-subtype,
B1, B2, and third-batch-B2 metrics; confusion counts; rescue/harm and McNemar
discordance versus H3 and C2; control deltas; stratified bootstrap summaries;
best epochs; parameter counts; runtime; peak GPU allocation; peak resident
memory; and peak new disk use.

Images, paths, case identifiers, fold membership, embeddings, checkpoints,
per-case predictions, derangement maps, and raw histories remain server-only.
Only code, hashes, aggregate metrics, gate decisions, and the aggregate report
may enter GitHub.

New persistent output must remain below 1 GiB. Extraction batch size is fixed at
one. C1 and PE encoders cannot be resident simultaneously. Resolution, views,
token count, head width, loss, threshold, and sampler cannot be changed to
avoid a failure.

## Hard stopping rule

At the first integrity failure or failed preregistered gate, write exactly:

`STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT`

Do not run the next stage, inspect the consumed external cohorts for rescue, or
initiate another seed, threshold, fusion, router, pooling, loss, augmentation,
backbone, or architecture experiment on these 591 cases.
