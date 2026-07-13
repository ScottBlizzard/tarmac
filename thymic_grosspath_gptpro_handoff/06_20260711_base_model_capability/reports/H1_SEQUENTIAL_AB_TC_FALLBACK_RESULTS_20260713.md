# H1 Sequential AB/TC Experts With Six-Subtype-Covered Binary Fallback

Date: 2026-07-13
Decision: **NO-GO**

## Purpose and Scope

This report is self-contained and assumes no prior knowledge of the project. It records one preregistered experiment on gross surgical photographs of thymic epithelial tumors. The decision is a NO-GO for this exact sequential AB-expert, TC-expert, and binary-fallback implementation. It does not prove that every possible image-grounded hierarchy is impossible.

The primary scientific objective is a stronger image-only low-risk versus high-risk classifier that predicts every case. The current input is one selected primary gross photograph per case. Rejection, physician review, source-specific calibration, confidence correction, and output stacking are outside this base-capability objective.

The six pathological subtypes are grouped as follows:

- low risk: A, AB, and B1;
- high risk: B2, B3, and thymic carcinoma (TC).

## Data and Evaluation

The internal development cohort contains 591 cases from three acquisition batches:

| Source | Cases |
| --- | ---: |
| batch1 | 117 |
| batch2 | 168 |
| third_batch | 306 |

Subtype distribution:

| A | AB | B1 | B2 | B3 | TC | Low | High |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 44 | 262 | 62 | 89 | 24 | 110 | 368 | 223 |

Two case-level outer evaluation protocols were mandatory:

1. Five-fold out-of-fold evaluation (five-fold OOF), with sources mixed across folds.
2. Source leave-one-domain-out evaluation (source-LODO), holding out batch1, batch2, and third_batch in turn. This is the principal internal domain-shift test.

All model fitting, early stopping, cross-fitted miss mining, and route-threshold selection occurred on the outer training side. Held-out cases were used only for evaluation. Balanced accuracy (BAcc) is the mean of high-risk sensitivity and low-risk specificity.

The previously inspected 108-case historical external cohort and 162-case newer external cohort were not read in this experiment. Both are consumed audit sets and cannot support a new external-generalization claim.

## Comparison Baselines

**C1** is the strongest locked single visual model: SigLIP-L at 512 px, six deterministic views derived from the same photograph (whole image, foreground crop, and four crop quadrants), all dense patch tokens, and a low-capacity gated pooling classifier.

**C2** is a fixed equal-probability ensemble of C1 and a separately trained AIMv2 MixStyle image model. Both members read images; C2 is not a confidence-only or behavior-level corrector.

The historical approximately 92% workflow is not a valid visual baseline. It descended from output-stacking/meta-correction experiments, included label-derived leakage in one winning route, consumed a nominal holdout in a later descendant, and failed to transfer. This experiment is compared only with the leakage-clean C1/C2 visual baselines.

## Locked Design

All three heads consumed the frozen C1 six-view dense-token representation defined above, with separately trained gated pooling heads.

1. An AB-versus-rest visual expert was trained first.
2. A TC expert was trained against A/B1/B2/B3 plus AB cases missed by training-side cross-fitted AB experts.
3. A binary fallback saw all A/B1/B2/B3 and dynamically sampled 40 AB plus 40 TC cases per epoch, half enriched for training-side expert misses.
4. AB-only routing emitted low risk, TC-only routing emitted high risk, and conflicts or non-routes used the fallback.
5. Route thresholds were selected from outer-training-side validation predictions only, targeting 98% AB-route low-risk purity and 95% TC-route high-risk purity. No held-out fold was used for threshold selection.
6. Evaluation used canonical five-fold OOF and three-source LODO. No external cohort was read.

## Main Results

| Protocol | Model | Acc | BAcc | AUC | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Five-fold OOF | C1 | 0.7496 | 0.7477 | 0.8240 | 0.7399 | 0.7554 |
| Five-fold OOF | C2 | 0.7597 | 0.7514 | 0.8377 | 0.7175 | 0.7853 |
| Five-fold OOF | Sequential | 0.7343 | 0.7195 | 0.8118 | 0.6592 | 0.7799 |
| Source-LODO | C1 | 0.7462 | 0.7397 | 0.8072 | 0.7130 | 0.7663 |
| Source-LODO | C2 | 0.7462 | 0.7441 | 0.8108 | 0.7354 | 0.7527 |
| Source-LODO | Sequential | 0.7479 | 0.7278 | 0.7895 | 0.6457 | 0.8098 |

Paired bootstrap versus C2:

- Five-fold delta BAcc: -0.0317, 95% CI [-0.0703, 0.0065].
- LODO delta BAcc: -0.0154, 95% CI [-0.0474, 0.0175].
- LODO delta AUC: -0.0208, 95% CI [-0.0438, 0.0019].
- LODO delta sensitivity: -0.0882, 95% CI [-0.1435, -0.0359].
- LODO delta specificity: +0.0574, 95% CI [0.0217, 0.0951].

The system shifted decisions toward low risk rather than learning a stronger boundary.

## Subtype Failure

Source-LODO risk accuracy, Sequential versus C2:

- A: 0.7273 versus 0.6591.
- AB: 0.8626 versus 0.8282.
- B1: 0.6452 versus 0.5000.
- B2: 0.5281 versus 0.6629.
- B3: 0.6250 versus 0.7083.
- TC: 0.7455 versus 0.8000.

The gain on A/AB/B1 was paid for by worse B2/B3/TC performance.

Source-LODO BAcc by held-out acquisition batch:

| Held-out source | C1 | C2 | Sequential |
| --- | ---: | ---: | ---: |
| batch1 | 0.7145 | 0.7890 | 0.7189 |
| batch2 | 0.7024 | 0.7083 | 0.7024 |
| third_batch | 0.7329 | 0.7337 | 0.7560 |

Only third_batch improved. The sequential model did not produce a consistent cross-source gain.

## Mechanism Audit

Standalone subtype-expert performance at threshold 0.5:

| Expert | Five-fold BAcc/AUC | Source-LODO BAcc/AUC |
| --- | ---: | ---: |
| AB versus rest | 0.7811 / 0.8505 | 0.6178 / 0.6280 |
| TC versus rest | 0.7939 / 0.8691 | 0.7473 / 0.8450 |

Five-fold routing:

- AB: 116 exclusive routes, 97.4% low-risk purity, 40.1% AB recall, 3 high-risk cases absorbed.
- TC: 19 exclusive routes, 89.5% high-risk purity, 13.6% TC recall, 2 low-risk cases absorbed.

Source-LODO routing:

- AB: 29 exclusive routes, 96.6% low-risk purity, 8.0% AB recall, 1 high-risk case absorbed.
- TC: zero routes because the validation purity gate disabled it in all LODO folds.

Routing changed only nine five-fold decisions and rescued eight while harming one. Under LODO it changed three decisions and rescued all three. Routing was not the main failure.

The fallback was the bottleneck:

| Protocol | Fallback all-case BAcc | Fallback A/B1-vs-B2/B3 BAcc | C2 same core BAcc |
| --- | ---: | ---: | ---: |
| Five-fold | 0.7065 | 0.5765 | 0.6072 |
| Source-LODO | 0.7237 | 0.5998 | 0.6193 |

Balanced residual training did not create transferable visual knowledge. It reproduced the prior B1/B2 tradeoff by improving B1 and harming B2.

## Interpretation Boundary

Verified by this experiment:

- the AB expert learned a mixed-source OOF signal that collapsed under source shift;
- the TC expert retained some cross-source ranking signal but could not support the preregistered high-purity route;
- the few actual specialist routes were net positive and were not the dominant cause of failure;
- the fallback did not improve the core A/B1-versus-B2/B3 boundary;
- the final system moved predictions toward low risk and significantly reduced high-risk sensitivity.

Not established by this experiment:

- that no subtype-specific visual signal exists;
- that all soft, joint, or end-to-end hierarchies must fail;
- that the B1/B2 boundary is intrinsically impossible from gross images;
- that more data or standardized additional views would not help.

The correct inference is narrower: this hard sequential routing design, trained on the current single-photograph information and frozen representation, did not create transferable complementary visual knowledge.

## Locked Decision

| Preregistered gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Five-fold BAcc | Greater than C2 0.7514 | 0.7195 | FAIL |
| Source-LODO BAcc | At least 0.7640 | 0.7278 | FAIL |
| Source-LODO sensitivity | At least 0.7354 | 0.6457 | FAIL |
| Minimum held-source BAcc | At least 0.7083 | 0.7024 | FAIL |
| Source-LODO B1 accuracy | At least 0.5000 | 0.6452 | PASS |
| Source-LODO B2 accuracy | At least 0.6630 | 0.5281 | FAIL |

Only the B1 gate passed. The two confirmation seeds, anchor-count variants, and threshold searches are therefore prohibited by the preregistered stopping rule.

The next decision is whether any remaining experiment changes the visual information or learning assumption enough to justify another internal run. If not, the scientifically rational step is standardized additional gross views and a genuinely untouched multicenter cohort rather than further tuning on the same 591 primary photographs.

Server-only results are under:

`/workspace/thymic_project/experiments/sequential_ab_tc_fallback_20260713`

No patient images, case-level predictions, feature arrays, or weights are included in this repository.
