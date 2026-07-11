# Fresh External Blind-Test Protocol for Task7

Date locked: 2026-07-11

## Purpose

Confirm whether the new full-coverage image-only Task7 candidates generalize beyond all cohorts already used for development or inspected evaluation. The historical strict external cohort (108 cases) and newer external cohort (162 cases) are consumed audit sets. They must not be used again for model selection, threshold tuning, calibration, or fusion design.

## Locked Candidates

1. `C1_siglipl512_localpyramid6_gated`
   - SigLIP-L at 512 px.
   - Six deterministic views from one primary image: whole, foreground crop, and four crop quadrants.
   - Dense-token gated pooling.
   - Fixed decision threshold: 0.5.

2. `C2_aimmixstyle_plus_siglipl512_localpyramid6`
   - Equal probability average of locked AIMv2 MixStyle run 253 and C1.
   - Fixed equal weights and fixed decision threshold: 0.5.

3. Historical comparator
   - The previously locked `215 + 253 + 254` equal-weight fusion.
   - Comparator only; it cannot be modified after seeing fresh outcomes.

The machine-readable lock is `scripts/phase2_fresh_external_candidate_lock_20260711.csv`. Prediction and configuration hashes are part of the audit trail.

## Cohort Requirements

- Cases must be absent from old_data, third_batch, strict_external108, and new_external162.
- Prefer consecutive cases from at least two hospitals or acquisition systems. A single new hospital is useful but cannot establish broad multi-center generalization.
- Preserve all eligible A, AB, B1, B2, B3, and thymic carcinoma cases; do not enrich only easy or correctly photographed cases after prediction review.
- Use pathology diagnosis as the reference standard and record mixed/borderline histology explicitly.
- Keep labels unavailable to the modeling team until prediction files and SHA-256 hashes are committed.
- Target at least 200 cases overall, with enough low- and high-risk cases for stable sensitivity and specificity intervals. A stronger paper should target at least 300 cases across centers and meaningful B1/B2 representation.

## Image Rule

- Primary analysis uses one predeclared primary gross image per case because the current development cohorts are overwhelmingly single-image.
- The primary-image selection rule must be fixed before inference, preferably the first adequate cut-surface image under a written acquisition rule.
- If standardized multi-view images are available, retain them for a separately labeled secondary analysis. Do not silently select the image that gives the most favorable prediction.
- Record hospital, camera/acquisition system, fixation state, background, view type, image count, and image adequacy where available.

## Blind Execution

1. Build an image-only registry without risk or subtype labels.
2. Audit duplicate files, duplicate case IDs, and overlap with all existing cohorts.
3. Freeze the registry checksum and candidate configuration checksums.
4. Run C1, C2, and the historical comparator without labels.
5. Write case-level probabilities and fixed-threshold predictions.
6. Commit prediction CSVs and SHA-256 hashes before labels are released.
7. Join labels once, run the frozen metric script once, and retain all cases.

## Primary Reporting

Report 100% coverage forced classification separately for each center and pooled:

- N, prevalence, accuracy, balanced accuracy, ROC AUC.
- High-risk sensitivity, low-risk specificity, FN, FP, and confusion matrix.
- Source-and-label-stratified bootstrap 95% confidence intervals.
- Paired bootstrap differences between C2, C1, and the historical comparator.
- A/AB/B1/B2/B3/TC risk accuracy, with B1 and B2 highlighted.
- Missing-image and exclusion accounting.

Rejection, review, or workflow results are secondary analyses and must never replace the full-coverage table.

## Interpretation Gates

- Internal improvement alone is not evidence of external improvement.
- A credible positive signal requires balanced accuracy at least 0.70 without either sensitivity or specificity below 0.65, plus no center-level collapse.
- A strong manuscript claim should require balanced accuracy at least 0.75 with sensitivity and specificity each at least 0.70, reproduced across centers or in a second fresh cohort.
- Candidate differences whose paired 95% confidence interval crosses zero must be described as inconclusive.
- If this cohort is inspected and then used to modify the model, it becomes a development cohort. Any revised model requires another untouched external cohort.
