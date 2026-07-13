# H2 Canonical-Coordinate Spatial-Relational Readout Preregistration

Date locked: 2026-07-13

## Scientific Scope

This is the final bounded experiment on the current 591 selected primary photographs. It tests one unresolved assumption only: useful cross-source morphology may be present in the locked C1 dense patch tokens but lost by permutation-invariant gated pooling.

The experiment does not add a backbone, external labels, clinical metadata, physician concepts, confidence routing, rejection, subtype supervision, threshold selection, or probability fusion. Failure closes further optimization of the current single-photograph input.

The three development sources are acquisition batches from the current internal project, not independent external hospitals. They are used as batch-shift stress tests only.

## Locked Data

- Internal cases: 591.
- Sources: batch1 117, batch2 168, third_batch 306.
- Subtypes: A 44, AB 262, B1 62, B2 89, B3 24, TC 110.
- Binary target: A/AB/B1 low risk; B2/B3/TC high risk.
- Frozen feature bank: SigLIP-L at 512 px, six views, 1,024 patch tokens per view, 1,024 dimensions per token, float16.
- Views: whole, specimen crop, and four overlapping 70% specimen-crop quadrants.
- Historical external 108 and newer external 162 are prohibited from loading or enumeration.

The expected feature array shape is exactly `[591, 6, 1024, 1024]`. Asset preparation must assert exact case, source, subtype, split, shape, and one-to-one alignment requirements before training.

## Immutable Asset and Coordinate Manifest

Before any fold is trained, the server-only preparation step must:

1. SHA-256 hash the dense feature array, feature configuration, metadata, locked split CSV, C1/C2 OOF and LODO predictions, experiment scripts, and environment lock.
2. Assert 591 unique case IDs, one row per original case, three expected canonical sources, exact subtype totals, and complete master folds.
3. Explicitly align C1 and C2 predictions with metadata under both protocols.
4. Recompute specimen bounding boxes from the original server-side images using the exact extraction code.
5. Record normalized original-image bounds for all six views and hash the resulting coordinate manifest.

No image, coordinate manifest, case-level prediction, feature array, or model weight may be copied to the local machine or GitHub.

## Locked Variants

Exactly three primary variants are allowed:

1. `matched_gated`: a parameter-light gated token-pooling head using the same loader, source-by-risk sampler, seed, validation rule, and optimizer as the relational variants.
2. `relational_permuted`: the full relational architecture, but patch content is independently and deterministically permuted for every case and view. The permutation is keyed by case ID, view, and the locked permutation seed. Coordinates remain attached to grid locations. This destroys shared anatomical adjacency while preventing a globally fixed inverse permutation from being learned.
3. `relational`: the full architecture with the true patch grid and true coordinates.

The locked historical C1 and C2 predictions remain the principal baselines. The matched and permuted variants are mechanism controls.

## Locked Relational Architecture

- LayerNorm and linear projection: 1,024 to 128 dimensions.
- Six independent 32 by 32 patch grids.
- Two local pre-norm transformer blocks shared across all views and windows.
- Non-overlapping 4 by 4 windows.
- Four attention heads and learned 2D relative-position bias inside each window.
- Mean pool each window, producing an 8 by 8 grid per view.
- Concatenate 384 window tokens across six views.
- Add fixed sinusoidal embeddings for within-view and normalized original-image coordinates.
- Add learned six-view identity embeddings and a learned linear embedding of log view scale.
- Prepend one learned case token.
- Two global pre-norm transformer blocks, four heads, MLP ratio 2.
- LayerNorm, dropout 0.10, and a two-class linear classifier on the case token.

No alternative depth, width, window size, overlap, coordinate encoding, dropout, or attention head count is allowed after results are observed.

## Locked Training

- Frozen feature bank; no encoder fine-tuning.
- Cross-entropy only.
- Source-by-risk inverse-frequency WeightedRandomSampler on the outer training subset.
- No additional class weights.
- AdamW, learning rate 0.0003, weight decay 0.0001.
- Maximum 80 epochs; patience 12.
- Gradient clipping 5.0.
- Batch size 4, no feature-bank RAM copy, automatic mixed precision on CUDA.
- Primary seed 20260713.
- Fixed decision threshold 0.5.
- Early stopping by validation BAcc at threshold 0.5.

A confirmation relational seed 20260714 is allowed only if every primary gate passes. Controls are not repeated for the confirmation seed.

## Outer Evaluation

### Five-fold OOF

- Test fold `k`.
- Validation fold `k+1` cyclically.
- Remaining three folds for training.

### Source LODO

- Hold one complete canonical source out as test.
- Use the predefined next master fold within the remaining sources for validation.
- Train on all remaining cases.

The held-out subset is never used for fitting, stopping, sampling, architecture decisions, or threshold selection.

## Required Reporting

At fixed threshold 0.5 report:

- Acc, BAcc, AUC, high-risk sensitivity, and low-risk specificity;
- TN, FP, FN, and TP;
- fold and source metrics;
- minimum-source BAcc;
- all six subtype risk accuracies;
- B1, B2, and their mean accuracy;
- paired source-by-risk-stratified bootstrap differences versus C2;
- paired comparison between relational and relational-permuted controls.

## Primary GO/NO-GO Gates

Every gate must pass:

1. Five-fold OOF BAcc at least 0.7664.
2. Source-LODO BAcc at least 0.7641.
3. Source-LODO sensitivity at least 0.7354 and specificity at least 0.7527.
4. Paired LODO delta BAcc versus C2 has a 95% bootstrap lower bound greater than 0.
5. Paired LODO sensitivity delta versus C2 has a 95% lower bound greater than -0.02.
6. At least two of three sources improve in BAcc; no source declines by more than 0.02; minimum-source BAcc is at least 0.7083.
7. LODO B1 accuracy is at least 0.5000 and B2 accuracy at least 0.6629.
8. Mean LODO B1/B2 accuracy is at least 0.6115, with neither subtype below its C2 baseline.
9. Relational LODO BAcc exceeds relational-permuted LODO BAcc by at least 0.01.

If any gate fails, the decision is `NO-GO` and no architecture, loss, sampler, seed, threshold, or fusion follow-up is allowed.

If all primary gates pass, the decision is `PROVISIONAL-GO: CONFIRMATION REQUIRED`. The confirmation seed must retain a positive LODO delta versus C2, and the two-seed mean must retain the point-estimate gates before final advancement.

## Interpretation

A positive result would support the claim that spatial relationships already present in frozen C1 tokens were discarded by the previous pooling heads. A negative result would not prove gross photographs are universally insufficient, but it would close further optimization on this repeatedly reused single-photograph cohort and make standardized new multi-view data the sole model-development mainline.
