# H1 Sequential AB/TC Experts With Six-Subtype-Covered Binary Fallback

Date: 2026-07-13
Decision: **NO-GO**

## Locked Design

The experiment used 591 internal cases only. Low risk was A/AB/B1 and high risk was B2/B3/TC. All three heads consumed the locked C1 SigLIP-L@512 six-view dense-token representation.

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

## Locked Decision

Only the B1 gate passed. OOF BAcc, LODO BAcc, LODO sensitivity, minimum-source BAcc, and B2 accuracy all failed. The two confirmation seeds, anchor-count variants, and threshold searches are therefore prohibited by the preregistered stopping rule.

Server-only results are under:

`/workspace/thymic_project/experiments/sequential_ab_tc_fallback_20260713`

No patient images, case-level predictions, feature arrays, or weights are included in this repository.
