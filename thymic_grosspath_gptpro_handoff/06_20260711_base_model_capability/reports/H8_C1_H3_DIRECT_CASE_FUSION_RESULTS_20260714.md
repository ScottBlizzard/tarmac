# H8 C1-H3 Direct Case-Fusion Results

## Final decision

`STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT`

H8 did not pass the preregistered source-LODO primary gate. Fivefold screening
and confirmation seeds were therefore not run.

This is a negative capability result, not an execution failure. Correctly
paired C1 and H3 embeddings contain real complementary information, but the
gain is too small relative to the locked H3 branch and it harms the clinically
important B1/B2 boundary cases.

## Locked execution

- Experiment: `H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714`
- Split: three-source leave-one-domain-out (source-LODO)
- Coverage: 591/591 cases at the fixed threshold 0.5
- Seed: `20260714`
- Trainable parameters: 6,226 for every learned configuration
- Asset-manifest SHA256:
  `e0db639a94d922222e1d88db1b59704d443f6e564bbcfa831d0c4f80f1f0fbc5`
- Code commit restoring the locked H3 batching path: `447e22a`
- C1 locked-probability maximum reproduction error: `1.7881e-7`
- H3 locked-probability maximum reproduction error: `1.1102e-16`
- Learned-head train/evaluation wall time: 6.99 seconds

The PE image encoder was executed one case at a time. The frozen H3 pooler was
executed in the original batch size of eight, separately within every fold and
train/validation/test partition. This was required because one-case pooling
changed 124 locked probabilities by more than `1e-5`; restoring the original
partition batching reproduced H3 to numerical precision before any H8 head
was fitted.

## Overall source-LODO results

| Method | Accuracy | BAcc | AUC | Sensitivity | Specificity | TP | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 only, capacity padded | 0.7394 | 0.7236 | 0.7992 | 0.6592 | 0.7880 | 147 | 290 |
| H3 only, capacity padded | 0.7496 | 0.7415 | 0.7955 | 0.7085 | 0.7745 | 158 | 285 |
| **C1-H3 exact pairing** | **0.7783** | **0.7628** | 0.7885 | 0.6996 | **0.8261** | 156 | 304 |
| C1-H3 same-source deranged | 0.7343 | 0.7178 | 0.7604 | 0.6502 | 0.7853 | 145 | 289 |
| Locked C1 | 0.7462 | 0.7397 | 0.8072 | 0.7130 | 0.7663 | 159 | 282 |
| Locked C2 | 0.7462 | 0.7441 | 0.8108 | **0.7354** | 0.7527 | **164** | 277 |
| Locked H3 | 0.7716 | 0.7539 | 0.7984 | 0.6816 | **0.8261** | 152 | **304** |
| Fixed C1-H3 probability mean | 0.7750 | 0.7592 | **0.8293** | 0.6951 | 0.8234 | 155 | 303 |

The exact fusion improved BAcc by only `0.0090` over locked H3. Its stratified
bootstrap interval crossed zero: 95% CI `[-0.0118, 0.0302]`. It also did not
show a reliable advantage over locked C2: delta `0.0187`, 95% CI
`[-0.0128, 0.0498]`.

## Primary gates

| Gate | Result | Observed value | Requirement |
| --- | --- | ---: | --- |
| P1 | PASS | 591 | 591/591, threshold 0.5 |
| P2 | **FAIL** | 0.7628 | BAcc >= 0.7739 |
| P3 | **FAIL** | 156 | TP >= 164 |
| P4 | PASS | 304 | TN >= 299 |
| P5 | **FAIL** | 36/62 | B1 correct >= 40/62 |
| P6 | **FAIL** | 51/89 | B2 correct >= 59/89 |
| P7 | **FAIL** | 15/29 | Third-batch B2 correct >= 18/29 |
| P8 | PASS | 3/3 sources | Positive H3 BAcc delta in at least 2/3 sources |
| P9 | PASS | minimum 0.7440 | No source delta below -0.0200; minimum BAcc >= 0.7381 |
| P10 | PASS | +0.0213 | Exact >= best capacity-padded branch +0.0100 |
| P11 | PASS | +0.0450 | Exact pairing beats deranged control with positive CI |
| P12 | **FAIL** | +0.0090 | Exact-H3 >=0.0200 with positive CI |
| P13 | **FAIL** | -5 cases | B1+B2 net gain vs H3 >=7 while retaining P5/P6 |

## Source behavior

| Held-out source | Exact BAcc | Locked H3 BAcc | Delta | Exact sensitivity | Exact specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 0.7443 | 0.7434 | +0.0009 | 0.7719 | 0.7167 |
| Batch 2 | 0.7440 | 0.7381 | +0.0060 | 0.6667 | 0.8214 |
| Third batch | 0.7700 | 0.7440 | +0.0260 | 0.6829 | 0.8571 |

There is no source-specific collapse. The problem is that the positive changes
are modest and do not concentrate on the boundary subtypes.

## Subtype behavior

| Subtype | Exact fusion | Locked H3 | Locked C2 | Exact minus H3 correct cases |
| --- | ---: | ---: | ---: | ---: |
| A | 34/44 | 31/44 | 29/44 | +3 |
| AB | 234/262 | 233/262 | 217/262 | +1 |
| B1 | **36/62** | **40/62** | 31/62 | **-4** |
| B2 | **51/89** | **52/89** | **59/89** | **-1** |
| B3 | 17/24 | 17/24 | 17/24 | 0 |
| TC | 88/110 | 83/110 | 88/110 | +5 |

Against locked H3, exact fusion rescued 20 and harmed 16 cases overall, but in
B1+B2 it rescued only 6 and harmed 11, for a net loss of 5. Third-batch B2 was
15/29, above locked H3's 13/29 but below locked C2 and the gate target of
18/29.

## Does fusion learn real case-level information?

Yes, but not enough to qualify as a capability improvement.

- Exact pairing vs same-source derangement: BAcc `+0.0450`, 95% CI
  `[0.0112, 0.0792]`, probability of a positive delta `0.9950`.
- Exact pairing vs capacity-padded H3: BAcc `+0.0213`, 95% CI
  `[0.0003, 0.0421]`.
- Exact pairing vs locked H3: BAcc `+0.0090`, 95% CI
  `[-0.0118, 0.0302]`.

The derangement result rejects the explanation that H8 is merely exploiting
source identity or head capacity. However, most of the usable gain shifts
predictions toward easier A/AB/TC cases and higher specificity. It does not
recover the B1/B2 decision boundary and it reduces TP relative to locked C2.
The lower AUC than both locked C2 and the fixed probability mean also shows
that the small MLP does not create a stronger overall ranking function.

## Consequence

H8 must not be promoted as the new base model. Locked H3 remains the stronger
balanced reference, while locked C2 remains the reference for TP and B2
retention. Under the GPT Pro oracle protocol, further classifier-only tuning on
the same 591-case cohort is stopped. A defensible next capability step must add
new information rather than another confidence gate: additional independent
cases, lesion-focused supervision/ROI evidence, stronger pathology-relevant
pretraining, or a prospectively locked external evaluation design.

All images, paths, identifiers, embeddings, checkpoints, per-case predictions,
and derangement maps remain server-only. No server dataset or model artifact
was downloaded for this report.
