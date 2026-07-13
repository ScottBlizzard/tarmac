# H4 Quality-Domain Randomization Results (2026-07-13)

## Decision

H4 is a **NO-GO** under the preregistered cross-domain gates.

Training the frozen PE-Spatial dense-token head on paired clean and realistically
degraded views retained the strong mixed-source result, but it reduced
cross-batch transfer. Five-fold OOF balanced accuracy was 0.8009, whereas
source-LODO balanced accuracy fell to 0.7254. The source-LODO result was 0.0187
below C2 and 0.0285 below the H3 PE model. The paired H4-minus-H3 LODO
difference was statistically negative: 95% CI [-0.0510, -0.0064].

The conditional confirmation seed was not run. No augmentation-severity,
consistency-weight, threshold, routing, or source-calibration search is allowed
after this result.

## Locked setup

- Cohort: the same 591 internal-development cases.
- Endpoint: A/AB/B1 low risk versus B2/B3/TC high risk.
- Encoder: frozen `facebook/PE-Spatial-L14-448`.
- Views: whole image, foreground crop, and four crop quadrants.
- Dense representation: up to 1,024 valid spatial tokens per view.
- Main inference: clean images only.
- Threshold: 0.5 at 100% coverage.
- Primary seed: 20260713.
- Evaluation: unchanged five-fold OOF and acquisition-batch source-LODO.

The one allowed training-only degradation profile combined downsample/upsample,
light blur, JPEG compression, brightness/contrast/saturation variation, and
small per-channel white-balance gains. The objective was:

`0.5 * (CE_clean + CE_augmented) + 0.10 * JS(clean, augmented)`

Checkpoint selection used clean validation BAcc only. Degraded validation and
test predictions were diagnostics and could not select a checkpoint.

## Overall results

| Model | OOF BAcc | OOF AUC | OOF Sens | OOF Spec | LODO BAcc | LODO AUC | LODO Sens | LODO Spec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 | 0.7514 | 0.8377 | 0.7175 | 0.7853 | 0.7441 | 0.8108 | 0.7354 | 0.7527 |
| H3 PE-Spatial | 0.8003 | 0.8700 | 0.8072 | 0.7935 | 0.7539 | 0.7984 | 0.6816 | 0.8261 |
| H4 quality consistency | **0.8009** | **0.8850** | 0.7758 | **0.8261** | 0.7254 | 0.8089 | 0.6682 | 0.7826 |

H4 improved mixed-source ranking but did not improve the fixed-threshold
operating point in a transferable way. Versus C2 under source-LODO, sensitivity
fell by 0.0673, with 95% CI [-0.1300, -0.0045]. The small specificity increase
was uncertain.

## Held-source results

| Held-out source | C2 BAcc | H3 PE BAcc | H4 BAcc | H4 - C2 | H4 - H3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| batch1 | 0.7890 | 0.7434 | 0.7276 | -0.0614 | -0.0158 |
| batch2 | 0.7083 | 0.7381 | 0.7500 | +0.0417 | +0.0119 |
| third_batch | 0.7337 | 0.7440 | 0.6823 | -0.0514 | -0.0617 |

The intervention helped batch2 but materially harmed batch1 and third_batch.
This is the opposite of the required source-stability pattern.

## Subtype results

| Subtype | C2 risk accuracy | H3 PE risk accuracy | H4 risk accuracy |
| --- | ---: | ---: | ---: |
| A | 0.6591 | 0.7045 | 0.7045 |
| AB | 0.8282 | 0.8893 | 0.8282 |
| B1 | 0.5000 | 0.6452 | 0.6452 |
| B2 | 0.6629 | 0.5843 | **0.5281** |
| B3 | 0.7083 | 0.7083 | 0.7500 |
| TC | 0.8000 | 0.7545 | 0.7636 |

The central failure remained B2. Quality consistency preserved H3's B1 gain
but moved another five B2 cases to the wrong low-risk side.

## Paired evidence

- H4 minus C2 source-LODO BAcc: -0.0187, 95% CI
  [-0.0571, +0.0196], probability of a positive delta 0.1687.
- H4 minus H3 PE source-LODO BAcc: -0.0285, 95% CI
  [-0.0510, -0.0064], probability of a positive delta 0.0056.
- H4 versus C2 correctness transitions: 59 rescues, 63 harms, 91 persistent
  errors, and 378 stable correct cases.
- H4 versus H3 PE correctness transitions: 14 rescues, 33 harms, 121
  persistent errors, and 423 stable correct cases.

The degraded-view diagnostic did not reveal hidden robustness. Aggregating fold
confusion counts, degraded-view BAcc was approximately 0.7953 in five-fold and
0.7304 in source-LODO. It did not change the clean-image decision.

## Gate audit

| # | Requirement | Observed | Status |
| ---: | --- | --- | --- |
| 1 | OOF BAcc >= 0.7903 | 0.8009 | Pass |
| 2 | OOF sensitivity >= 0.7772 and specificity >= 0.7635 | 0.7758 / 0.8261 | Fail |
| 3 | LODO BAcc >= 0.7641 | 0.7254 | Fail |
| 4 | LODO sensitivity >= 0.7354 | 0.6682 | Fail |
| 5 | LODO specificity >= 0.7527 | 0.7826 | Pass |
| 6 | LODO B1 >= 0.6000 | 0.6452 | Pass |
| 7 | LODO B2 >= 0.6629 | 0.5281 | Fail |
| 8 | At least two held-out sources improve versus C2 | 1/3 | Fail |
| 9 | No held-out source declines by more than 0.02 | minimum -0.0614 | Fail |
| 10 | LODO BAcc and sensitivity both exceed H3 PE | both lower | Fail |
| 11 | LODO BAcc delta versus C2 CI lower bound > 0 | -0.0571 | Fail |
| 12 | Confirmation seed directionally positive | not eligible | Not evaluated |
| 13 | Threshold 0.5 and coverage 100% | satisfied | Pass |

## Interpretation

1. The quality perturbations were real and the model learned to process them,
   but this did not identify a stable cross-source decision boundary.
2. Better OOF AUC and specificity again coexisted with worse held-source
   sensitivity and B2 accuracy. Ranking improvement is not sufficient evidence
   of stronger clinical classification.
3. The result closes further searches over perturbation severity, consistency
   weight, source-specific calibration, confirmation seeds, and quality-based
   routing on this cohort.
4. The next user-directed exploratory experiment must read visual evidence
   directly and test a materially different representation assumption. The
   lowest-cost untested direction from the research report is true low-rank
   second-order covariance texture modeling on the frozen PE tokens. It must be
   preregistered as a new H5 experiment and cannot be described as confirmation
   of H3 or H4.

## Reproducibility and storage

Server aggregate outputs:

- `/workspace/thymic_project/experiments/h4_quality_domain_randomization_20260713/primary_seed20260713`
- `/workspace/thymic_project/experiments/h4_quality_domain_randomization_20260713/gate_aggregate`
- `/workspace/thymic_project/experiments/h4_quality_domain_randomization_20260713/bank_provenance`

Clean PE bank hashes:

- dense features: `e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34`
- valid mask: `af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c`
- spatial shapes: `14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f`
- H4 clean config: `506e01cd6d279540f90099704e8582e76e8751e5bbdb3a5b3c0b4f8f36b826da`

Quality-randomized bank hashes before deletion:

- dense features: `202f51f2ef18e783f237a5c52b7bc25767545308c080867ca360e63d6561788a`
- valid mask: `af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c`
- spatial shapes: `14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f`
- config: `d26f2ca575e1f4107a7dc2edeea1590d933471cf258efd3a72327e50ed0d54d6`

The quality-randomized dense bank was deleted after hashing. The exact clean PE
bank is temporarily retained for the preregistered H5 texture experiment. No
patient images, case-level predictions, feature arrays, or weights were copied
to the local repository.
