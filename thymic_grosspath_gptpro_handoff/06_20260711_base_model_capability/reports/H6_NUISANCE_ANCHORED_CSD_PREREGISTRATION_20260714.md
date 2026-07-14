# H6 Nuisance-Anchored Common-Specific Decomposition Preregistration

Date: 2026-07-14

## Decision

Run one bounded fixed-data experiment, `H6_NUISANCE_ANCHORED_CSD_20260714`,
with one required identifying control. H6 changes the final H3 PE-Spatial
classifier, not the image set, risk labels, threshold, coverage policy, encoder,
or evaluation splits.

The primary method is rank-1 nuisance-anchored common-specific decomposition
(`NA-CSD`). The matched control is acquisition-source-only CSD
(`SOURCE_ONLY_CSD`). At inference, both discard all environment-specific
parameters and use only the common classifier at threshold 0.5 and 100%
coverage.

Primary literature basis: Piratla, Netrapalli and Sarawagi, *Efficient Domain
Generalization via Common-Specific Low-Rank Decomposition*, ICML 2020,
<https://proceedings.mlr.press/v119/piratla20a.html>.

## Locked Development Cohort

The authoritative input is the exact 591-case registry used to create H3:

- server path:
  `/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv`;
- SHA-256:
  `ef2d4e16b041038e36bd165c80d460b527c0c09eba19250b38f50d67199c62af`;
- source totals: batch1 117, batch2 168, third batch 306;
- subtype totals: A 44, AB 262, B1 62, B2 89, B3 24, TC 110.

The previously proposed
`task7_four_domain_master_registry.csv` is prohibited for H6 because it has 861
rows and includes the 108- and 162-case consumed external stress sets. Those 270
cases cannot enter extraction, training, environment construction, checkpoint
selection, thresholding, or advancement.

The locked split is:

- path:
  `/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv`;
- SHA-256:
  `48610c996298f8af317d547681665e0be20e4361aa61de66675efc51bb954545`.

## Locked Representation and References

The frozen encoder is PE-Spatial-L14-448. The six views are `whole`, `crop`,
and four crop quadrants, with at most 1,024 valid tokens per view. The expected
regenerated bank hashes are:

- dense features:
  `e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34`;
- valid mask:
  `af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c`;
- spatial shapes:
  `14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f`.

The regenerated bank is materialized under
`/root/thymic_h6_nuisance_anchored_csd_20260714/pe_dense_bank` because the
`/workspace` OSS mount does not support NumPy memmap random writes. This is a
storage-location change only; the manifest rejects any byte-level hash drift.
The PE code is a revision-named source snapshot rather than a Git work tree;
its 48 non-bytecode files are locked by tree SHA-256
`4e00f80f27fc360591f94eb125ed8ea18b86237ae59e5284abf3bde2ddebdbea`
in addition to revision `3e352cca660658d4b5c90f42a7808b11469e4c66`.

The four case-level reference files are locked by the integrity manifest. Their
current SHA-256 values are:

| Reference | SHA-256 |
|---|---|
| H3 five-fold | `f321e4faa34ca6212acbb0acea63bc9efbc4e485a90ea376c21c154bef9643ad` |
| H3 source-LODO | `9d706a7f308dd834a9fd773205a008b300eb559750651b8367b11b66de2f0504` |
| C2 five-fold | `4890740c02f1bce33d229be6c0e71f11f11c7118e1b39c6e9e8c7bb6ae841895` |
| C2 source-LODO | `f58035975cabc147fb7a7a8e0ee67ccddcd9f54e0d813c27048f1a594a0242f3` |

All contain 591 unique, exactly aligned cases. Metrics recomputed from them
match the existing reports.

## Outer-Fold-Clean Nuisance Construction

For each outer fold independently:

1. Use only outer-training cases.
2. Regress the 156 fixed Haar features on intercept, binary risk, and available
   acquisition-source indicators.
3. Standardize training residuals and fit one full-SVD principal component.
4. Orient the component by the sum of its loadings.
5. Within each source by risk stratum, sort by `(PC1, case_id)` and assign the
   lower half to bin 0 and upper half to bin 1; an odd median enters bin 1.
6. Define an environment as acquisition source by nuisance bin.

No validation or test case affects regression, scaling, PCA, orientation,
boundaries, sampling, or fitting. Validation environment IDs are transformed
through the training fit and use the validation risk label only for a diagnostic
specific-head comparison. That comparison never selects a checkpoint or enters
the deployed model. Test and deployment use no environment ID.

All three source-LODO and five five-fold partitions passed the preregistered
minimum of four low-risk and four high-risk cases per nuisance environment.

## Model and Optimization

The H3 input normalization, 1,024-to-128 projection, GELU, projection dropout,
view embeddings, masked gated pooling, and output normalization are unchanged.
H3 also used dropout 0.10 immediately before its final linear classifier; H6
retains this dropout and replaces only the linear readout.

The readout is:

`W_e = W_common + gamma_e * W_specific`, with the same decomposition for bias.

Both weight matrices are `2 x 128`. The primary nuisance environments initialize
`gamma` to -1 for bin 0 and +1 for bin 1. The source-only control has no bin, so
its otherwise unspecified coefficients are deterministically initialized with
`linspace(-1,+1)` in canonical source order. This implementation completion is
reported explicitly and is not attributed to the original CSD paper.

The locked loss is:

`0.5 * CE(common) + 0.5 * CE(environment-specific) + 0.05 * orthonormality`.

AdamW uses learning rate `3e-4`, weight decay `1e-4` except on biases and gamma,
batch size 8, source-by-risk inverse-frequency replacement sampling, at most 80
epochs, patience 12, global gradient clipping 5.0, and primary seed 20260714.
Strict deterministic CUDA matrix multiplication uses
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Only common-head validation balanced accuracy selects the checkpoint, followed
lexicographically by sensitivity and then earlier epoch.

## Evaluation and Stop Rule

Source-LODO is primary; canonical five-fold OOF is secondary. The exact gates
are those in the complete GPT Pro response dated 2026-07-14 and are implemented
in `analyze_task7_h6_nuisance_csd_20260714.py`, including:

- LODO BAcc at least 0.7641;
- sensitivity at least 0.7354 and specificity at least 0.7800;
- B1 at least 38/62 and B2 at least 59/89;
- minimum source BAcc at least 0.7381;
- at least two sources improve versus H3, with no source loss beyond 0.015;
- paired source-by-risk bootstrap requirements versus H3 and C2;
- improvement over `SOURCE_ONLY_CSD` sufficient to identify nuisance anchoring;
- five-fold retention requirements;
- 591/591 coverage at threshold 0.5.

Seed 20260715 is run only if every primary and secondary gate passes. Fishr is
permitted only if every exact mechanistic trigger passes. Otherwise fixed-data
visual development stops under this preregistration.

## Evidence Ceiling

Even a passing result is exploratory fixed-data engineering evidence and
internal acquisition-batch robustness. It is not independent external,
multicenter, prospective, or clinical validation.
