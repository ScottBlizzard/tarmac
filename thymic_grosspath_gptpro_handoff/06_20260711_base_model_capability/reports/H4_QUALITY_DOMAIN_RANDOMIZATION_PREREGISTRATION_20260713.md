# H4 Quality-Domain Randomization Preregistration (2026-07-13)

## Hypothesis

The frozen PE-Spatial representation contains a stronger risk signal, but its
source-LODO high-risk errors are enriched for low local contrast and low edge
content. Training the same dense-token head to classify both clean and
pathology-preserving degraded features, while enforcing clean/augmented
agreement, may improve high-risk and B2 transfer without using rejection,
quality routing, or test-source calibration.

This is a new internal-development hypothesis derived after H3. It is not an
independent confirmation study.

## Immutable cohort and evaluation

- The same 591 cases and master split file used by H3.
- Low risk: A/AB/B1; high risk: B2/B3/TC.
- Five-fold and source-LODO partitions remain unchanged.
- Threshold 0.5 and 100% coverage.
- Validation and test inference use clean images only.
- Primary seed 20260713.
- Confirmation seed 20260714 is allowed only if every primary gate except the
  confirmation gate passes.

## Frozen representation and views

- Encoder: `facebook/PE-Spatial-L14-448`.
- Official checkpoint and preprocessing from H3.
- Six views: whole, foreground crop, and four crop quadrants.
- Native-aspect grid with up to 1,024 spatial tokens per view.
- Encoder frozen for all H4 runs.
- The clean dense bank must reproduce the H3 bank configuration and hashable
  geometry before training.

## Single allowed augmentation profile

`quality_domain_randomization_v1` is applied independently after each clean H3
view is constructed. It preserves image dimensions and view geometry.
Randomness is deterministic from primary seed, case ID, and view name.

1. Downsample then bilinear upsample with scale sampled uniformly from
   [0.60, 0.90].
2. Gaussian blur with probability 0.75 and radius sampled uniformly from
   [0.20, 1.00].
3. JPEG encode/decode with quality sampled uniformly from integer [55, 90].
4. Brightness, contrast, and saturation factors sampled independently from
   [0.85, 1.15].
5. Per-channel white-balance gains sampled from [0.92, 1.08] and normalized to
   mean gain 1.0.

No rotation, label-dependent transform, specimen erasure, synthetic lesion,
background replacement, or crop-policy change is allowed. There is no severity
search or second augmentation profile.

## Locked model and objective

The classifier, source-risk sampler, optimizer, epoch limit, patience, batch
size, and regularization match the H3B PE-Spatial head.

For each training batch:

`loss = 0.5 * (CE_clean + CE_augmented) + 0.10 * JS(clean, augmented)`

where JS is the symmetric Jensen-Shannon divergence between two-class softmax
probabilities. Validation selection uses clean validation BAcc only. Augmented
validation and test predictions may be generated for robustness diagnostics but
cannot select a checkpoint or alter the clean prediction.

Only one primary candidate is allowed:

`pe_spatial_quality_dr_consistency_v1`

## References

- C2 locked reference: OOF BAcc 0.7514; source-LODO BAcc 0.7441,
  sensitivity 0.7354, specificity 0.7527; B1 0.5000; B2 0.6629.
- H3 PE-Spatial reference: OOF BAcc 0.8003; source-LODO BAcc 0.7539,
  sensitivity 0.6816, specificity 0.8261; B1 0.6452; B2 0.5843.
- C2 source-LODO BAcc: batch1 0.7890, batch2 0.7083, third_batch 0.7337.
- H3 PE source-LODO BAcc: batch1 0.7434, batch2 0.7381,
  third_batch 0.7440.

## Primary success gates

All conditions are required:

1. Five-fold OOF BAcc >= 0.7903, no more than 0.01 below H3 PE.
2. Five-fold sensitivity >= 0.7772 and specificity >= 0.7635, each no more
   than 0.03 below H3 PE.
3. Source-LODO BAcc >= 0.7641.
4. Source-LODO sensitivity >= 0.7354.
5. Source-LODO specificity >= 0.7527.
6. Source-LODO B1 accuracy >= 0.6000.
7. Source-LODO B2 accuracy >= 0.6629.
8. At least two held-out sources improve versus C2.
9. No held-out source declines by more than 0.02 versus C2.
10. Source-LODO BAcc and sensitivity are both higher than H3 PE.
11. Paired source/risk-stratified bootstrap versus C2 has a 95% CI lower bound
    above zero for source-LODO BAcc.
12. The confirmation seed remains directionally positive for OOF BAcc,
    source-LODO BAcc, source-LODO sensitivity, and B2 accuracy.
13. Threshold remains 0.5 with 100% coverage.

## Stopping rule

If the primary candidate fails any deterministic gate, do not run a second
augmentation severity, change consistency weight, add GroupDRO, tune a
threshold, or invoke the confirmation seed. Close H4 and move to a hypothesis
that changes available image information, principally all-image case bags,
standardized prospective multi-view acquisition, or new multicenter data.

## Data and storage

Images, dense banks, predictions, and checkpoints remain on the server. One
clean and one augmented PE dense bank may coexist during H4. Both are
regenerable and may be deleted after hashes, configurations, and aggregate
results are recorded.
