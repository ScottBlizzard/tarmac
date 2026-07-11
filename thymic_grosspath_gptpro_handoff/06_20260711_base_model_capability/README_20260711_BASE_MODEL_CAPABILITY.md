# 2026-07-11 Task7 Base-Model Capability Update

Repository: https://github.com/ScottBlizzard/tarmac

This folder is the latest handoff for the project's primary scientific objective: improving the 100%-coverage, image-only forced classifier for thymic gross pathology Task7.

Task7 definition:

- Low risk: A / AB / B1.
- High risk: B2 / B3 / thymic carcinoma.

Selective release, rejection, and physician-review workflows remain downstream safety layers. They are not substitutes for a stronger base classifier and are not the optimization target in this update.

## Read First

1. `GPTPRO_PROMPT_20260711_POST_EXPERIMENT.md`
   - Ready-to-use English prompt for a critical post-experiment audit and next decision.

2. `reports/Task7_Base_Model_Capability_Experiments_20260711.md`
   - Full internal experiment ledger from runs 206-369.
   - Canonical five-fold OOF and three-source LODO protocols.
   - Positive, negative, fusion, bootstrap, doctor-concept, and resource-cleanup results.
   - Final internal candidate lock and its limitations.

3. `scripts/phase2_fresh_external_candidate_lock_20260711.csv`
   - Machine-readable lock for the two candidates allowed into a new independent external blind test.
   - Includes fixed metrics, thresholds, member definitions, and prediction hashes.

4. `reports/FRESH_EXTERNAL_BLIND_TEST_PROTOCOL_20260711.md`
   - Exact cohort, image-selection, blinding, hashing, reporting, and interpretation rules for the next external test.

5. `reports/AI Pathology Model Improvement.md`
   - The broad research-lead plan that motivated this experiment wave.

6. `scripts/`
   - Complete local 2026-07-11 implementation set: registries, dense-token extraction, LoRA, contrastive learning, structured pooling, SAM optimization, fusion search, bootstrap, nested thresholds, error analysis, and queue/recovery scripts.

## Data Boundary

- Internal development: 591 cases.
  - batch1: 117.
  - batch2: 168.
  - third_batch: 306.
- Historical strict external: 108 cases.
- Newer external: 162 deduplicated cases.

The 108-case and 162-case external cohorts were both inspected in Phase 1. They are now consumed audit sets. Phase 2 did not use them for training, selection, thresholding, fusion, or evaluation. Any Phase-2 generalization claim requires a genuinely fresh external cohort.

## Main Positive Result

The only clear representation-level gain in this broad wave was:

`SigLIP-L @ 512 px + six deterministic views + all dense patch tokens + low-capacity gated pooling`

The six views are whole image, foreground crop, and four crop quadrants from the same primary image.

Locked single-model candidate C1:

- Five-fold OOF: BAcc 0.7477, AUC 0.8240, sensitivity 0.7399, specificity 0.7554.
- Source-LODO: BAcc 0.7397, AUC 0.8072, sensitivity 0.7130, specificity 0.7663.
- LODO source BAcc: 0.7145 / 0.7024 / 0.7329.
- Versus the prior two-view SigLIP-L@512 model, OOF BAcc improved by 0.0460; paired bootstrap 95% CI [0.0096, 0.0817]. LODO BAcc was unchanged within uncertainty.

Locked two-model candidate C2:

`AIMv2 MixStyle run 253 + C1, equal probability average`

- Five-fold OOF: BAcc 0.7514, AUC 0.8377.
- Source-LODO: BAcc 0.7441, AUC 0.8108.
- LODO minimum-source BAcc: 0.7083.
- LODO B1/B2 risk accuracy: 0.5000 / 0.6629.

C2's LODO AUC exceeded the prior locked `215+253+254` fusion by 0.0228, paired bootstrap 95% CI [0.0031, 0.0426]. Its BAcc difference remained inconclusive. This is an internal candidate, not an external breakthrough.

## Why the Highest Internal Fusion Was Not Locked

An exhaustive three-member search favored `207 + 269 + 347`, and an expanded five-member search produced superficially high internal results. Meta-level five-fold cross-selection chose different companion recipes across held-out folds and substantially reduced LODO performance. The new six-view run 347 appeared in every held-out winner, but the companion models were unstable.

The lock therefore keeps C1 and the simple C2 pair. It rejects a more complex recipe selected from thousands of internal combinations.

## Important Remaining Failure

The B1/B2 boundary remains weak:

- C1 LODO B1/B2 risk accuracy: 0.4194 / 0.6292.
- C2 LODO B1/B2 risk accuracy: 0.5000 / 0.6629.

Light boundary auxiliary loss, a conservative boundary expert, subtype auxiliary loss, subtype-aware contrastive learning, and B1/B2 hard negatives all shifted the error tradeoff without improving both boundary classes together.

Doctor-concept analysis is explanatory only. After Fisher tests and within-comparison BH correction:

- C1 OOF low-risk false positives enriched `texture_soft` (q=0.0012).
- C1 LODO high-risk false negatives enriched `nodular_lobulated` (q=0.0498).

These associations require physician image review and new data. They are not established biological mechanisms and were not used as model inputs.

## Broad Negative Evidence

The completed wave includes negative or insufficient results from:

- DINOv2, DINOv3-B/L/qkvb, EVA02, ConvNeXt DINOv3, ViTamin-L, SigLIP-B/L/SO400M, and pooled-token controls.
- Conservative and expanded LoRA on AIMv2 and SigLIP at 384/512.
- VICReg initialization, supervised contrastive learning, cross-source positives, subtype positives, and B1/B2 hard negatives.
- DANN, GroupDRO, REx, class-conditional alignment, MixStyle variants, and SAM optimization.
- Mean, view-gated, statistics, and fixed spatial-pyramid aggregation.
- Six-class, ordinal, concept, prototype, boundary-expert, high-risk sentinel, and MoE variants.
- Automatic contrast/white-balance/unsharp preprocessing and multiple ROI policies.

The report gives exact metrics and failure modes. Do not propose a low-level repeat without explaining what structural limitation changes.

## Next Valid Step

Run C1, C2, and the historical locked comparator once on a new label-blinded external cohort under `FRESH_EXTERNAL_BLIND_TEST_PROTOCOL_20260711.md`.

The current evidence supports a stronger internal representation and a cleaner candidate lock. It does not support the claim that cross-hospital generalization is solved.
