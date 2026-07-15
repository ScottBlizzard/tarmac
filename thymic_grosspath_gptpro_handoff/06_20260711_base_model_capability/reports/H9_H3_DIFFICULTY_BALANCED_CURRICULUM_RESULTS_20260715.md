# H9 H3 Difficulty-Balanced Curriculum Results

Date: 2026-07-15

## Decision

`NO_GO_CURRENT_CURRICULUM`

The current PE-Spatial H3 model did not improve when subtype-tempered sampling
and online easy/medium/hard curriculum were applied. The result is a valid
negative experiment rather than an execution failure.

Source-LODO primary gates failed, so the preregistered confirmation seed and
fivefold stage were not run.

## Integrity

The complete 591-case PE dense-token bank was regenerated on the server. All
three historical byte hashes reproduced:

- dense features:
  `e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34`
- valid-token mask:
  `af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c`
- spatial shapes:
  `14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f`

The unchanged source-risk control reproduced every locked H3 OOF probability
with maximum absolute error `0.0`. Its BAcc, confusion matrix, fold validation
optima, and test results exactly match locked H3. Therefore the observed H9
differences come from the sampling strategy, not feature or environment drift.

Each candidate used the same six-view PE token input, H3 gated pooling model,
151,107 trainable parameters, optimizer, loss, initialization, validation
selection, threshold 0.5, and 591/591 coverage.

## Overall source-LODO results

| Method | Accuracy | BAcc | AUC | Sensitivity | Specificity | TP | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Locked/source-risk H3 control | **0.7716** | **0.7539** | 0.7984 | **0.6816** | 0.8261 | **152** | 304 |
| Static subtype-tempered | 0.7530 | 0.7310 | 0.7862 | 0.6413 | 0.8207 | 143 | 302 |
| Dynamic curriculum | 0.7631 | 0.7356 | **0.8199** | 0.6233 | **0.8478** | 139 | **312** |
| Locked C2 reference | 0.7462 | 0.7441 | 0.8108 | 0.7354 | 0.7527 | 164 | 277 |

Dynamic curriculum was `-0.0183` BAcc below locked H3. Its 20,000-replicate
source-by-risk-stratified bootstrap interval crossed zero and favored H3:
95% CI `[-0.0421, 0.0051]`, probability of a positive curriculum delta
`0.0647`.

Curriculum exceeded static sampling by only `0.0046` BAcc, 95% CI
`[-0.0160, 0.0249]`. Therefore the result does not establish either a
curriculum benefit or a subtype-balancing benefit.

## Source behavior

| Held-out source | H3 BAcc | Static BAcc | Curriculum BAcc | Curriculum minus H3 | Curriculum AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1 | 0.7434 | 0.7175 | 0.7430 | -0.0004 | 0.8228 |
| Batch 2 | 0.7381 | 0.7381 | 0.7619 | +0.0238 | 0.8121 |
| Third batch | 0.7440 | 0.7030 | 0.6814 | **-0.0626** | 0.8352 |

The apparent gain is batch2-specific. Curriculum loses 6.26 BAcc points on the
306-case third batch, where sensitivity falls from 0.6220 to 0.4878. The third
batch accounts for 9 fewer correct cases than H3, overwhelming the 4-case gain
on batch2. This fails the generalization objective.

## Subtype behavior

| Subtype | Locked H3 | Static | Curriculum | Curriculum minus H3 |
| --- | ---: | ---: | ---: | ---: |
| A | 31/44 | 33/44 | 34/44 | +3 |
| AB | 233/262 | 231/262 | 234/262 | +1 |
| B1 | 40/62 | 38/62 | **44/62** | +4 |
| B2 | 52/89 | 41/89 | **44/89** | **-8** |
| B3 | 17/24 | 18/24 | 18/24 | +1 |
| TC | 83/110 | 84/110 | 77/110 | -6 |

The curriculum shifts the boundary toward low risk. It improves A, AB, and B1
but loses 14 high-risk B2/TC cases. Third-batch B2 falls from locked H3's
13/29 to 7/29, far below the 18/29 safety target. B1+B2 rescue/harm is 7/11,
for a net loss of 4 cases.

## Difficulty behavior

The independent diagnostic map from locked C1/C2/H3 OOF predictions contained
375 easy, 77 medium, 59 salvage, and 80 persistent-hard cases.

| Diagnostic group | Locked H3 | Static | Curriculum | Curriculum minus H3 |
| --- | ---: | ---: | ---: | ---: |
| Easy | 375/375 | 359/375 | 361/375 | -14 |
| Medium | 39/77 | 44/77 | 41/77 | +2 |
| Salvage | 42/59 | 37/59 | 41/59 | -1 |
| Persistent hard | 0/80 | 5/80 | 8/80 | +8 |

Curriculum does recover 8 cases that all three locked models missed. However,
it harms 24 H3-correct cases while rescuing 19 overall, for a net loss of 5.
This is the same failure pattern as the historical curriculum: a small hard
subset can be recovered, but the learned boundary moves enough to damage
previously stable cases.

## Training dynamics

| Held-out source | Static best epoch / val BAcc | Curriculum best epoch / val BAcc |
| --- | ---: | ---: |
| Batch 1 | 14 / 0.8254 | 12 / 0.8175 |
| Batch 2 | 19 / 0.8627 | 9 / 0.8694 |
| Third batch | 7 / 0.8073 | 11 / 0.8424 |

All three curriculum checkpoints were selected during the epoch 9-16 boundary
bridge, when easy/medium cases were emphasized and current hard cases were
downweighted. No hard-replay checkpoint improved validation enough to be
selected.

Online training difficulty also collapsed as the model memorized its outer
training cases. For example, fold 1 moved from 30 hard cases after epoch 8 to
no hard cases by epoch 24, even though held-out B2 performance deteriorated.
Training-set true probability is therefore not a reliable proxy for external
difficulty in this small cohort.

The strongest warning appears in fold 3: curriculum validation BAcc reached
0.8424 on batch1/batch2 validation cases, yet held-out third-batch BAcc was only
0.6814. A better internal validation score did not translate to the unseen
domain.

## Exploratory calibration diagnostic

This analysis was performed after the preregistered H9 failure and is not part
of the success claim. For each outer fold, the threshold maximizing validation
BAcc was selected from that fold's validation probabilities and applied once
to its held-out test source.

Curriculum thresholds were approximately 0.771, 0.504, and 0.465 for batch1,
batch2, and third-batch holdouts. The resulting OOF BAcc was only 0.7374, with
TP 138, TN 315, B2 45/89, and third-batch B2 9/29. Thus the fixed 0.5 threshold
is not the explanation for H9's failure. The large threshold disagreement is
itself evidence of source-dependent calibration.

The higher overall curriculum AUC does not establish stronger invariant
classification. Source AUC is nearly unchanged on batch1, improves on batch2,
and decreases from 0.8622 to 0.8352 on the third batch.

## Gate result

Passed: control integrity, TN, B1, and persistent-hard rescue.

Failed: BAcc, TP, B2, third-batch B2, curriculum-over-static improvement,
curriculum-over-H3 improvement, source stability, and B1+B2 net rescue.

## Interpretation and next constraint

The user's premise is supported: more cases, more views, or more exposure to a
subtype does not monotonically improve the clinically relevant boundary. The
third batch adds many AB cases, and merely rebalancing observed subtype counts
does not create a source-invariant B1/B2 representation.

The old easy-to-hard curriculum should not be copied onto H3. A future
curriculum would need difficulty generated by nested OOF teachers inside each
outer training set, fixed before final training, with separate safeguards for
high-risk hard cases. Online in-sample difficulty is too easy to memorize.

However, another schedule selected after seeing these same three LODO test
sources would be exploratory tuning, not new generalization evidence. A
defensible confirmatory test now requires a fresh external cohort or a
prospectively untouched holdout. New data should preferentially add diverse
B1/B2 boundary cases and lesion-focused evidence rather than more AB cases.

All images, paths, identifiers, dense features, checkpoints, per-case
predictions, and difficulty assignments remained server-only. No server data
or model artifact was downloaded.
