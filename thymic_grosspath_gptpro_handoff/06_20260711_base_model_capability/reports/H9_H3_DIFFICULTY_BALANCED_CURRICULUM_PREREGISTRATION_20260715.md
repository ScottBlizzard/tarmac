# H9 H3 Difficulty-Balanced Curriculum Preregistration

Date: 2026-07-15

## Question

Does the current PE-Spatial H3 model improve source-held-out risk
classification when training is organized by case-level subtype and dynamic
difficulty, rather than allowing common/easy subtypes to dominate the sampled
gradient?

This is a training experiment, not a selective-prediction workflow. Every case
receives one low/high prediction at threshold 0.5 and coverage is 591/591.

## Motivation fixed before execution

The 591-case cohort is moderately imbalanced at the binary level (368 low risk,
223 high risk) but severely imbalanced by subtype: A 44, AB 262, B1 62, B2 89,
B3 24, and TC 110. The third batch contains 212 AB, 12 B1, 29 B2, and 53 TC
cases, with no A or B3 cases.

H8 provided direct evidence that a higher overall score can conceal worse
boundary learning. Relative to locked H3, H8 exact fusion gained 3 A, 1 AB,
and 5 TC cases but lost 4 B1 and 1 B2 cases. H9 therefore treats B1, B2, and
third-batch B2 retention as mandatory capability endpoints.

The previous curriculum experiment used 285 batch1/batch2 cases and frozen
DINO features. It improved fivefold BAcc from 0.7370 to 0.7649 but left the
65-case hard-core group at 17/65. H9 does not reuse those historical difficulty
labels and does not permanently remove hard cases.

## Data separation

Primary evaluation is three-source leave-one-domain-out (source-LODO):

- fold 1 holds out batch1;
- fold 2 holds out batch2;
- fold 3 holds out third_batch.

The held-out source is never used to derive training difficulty, sampler
weights, model parameters, validation selection, or thresholds. Dynamic
difficulty is computed only from the current model's predictions on its outer
training cases and their known training labels. The fixed validation subset is
the same locked master-fold subset used by H3.

Batch/source is retained as an evaluation boundary and nuisance-balancing
factor. It is not treated as a pathological class.

## Locked representation and model

The experiment regenerates the original complete PE-Spatial-L14-448 dense
token bank from the cached 591-case registry. The following historical bank
hashes must reproduce before training:

- `dense_features.float16.npy`:
  `e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34`
- `valid_token_mask.uint8.npy`:
  `af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c`
- `spatial_shapes.int16.npy`:
  `14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f`

The PE checkpoint SHA256 remains
`47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1`.

All three training strategies use the original H3B architecture:

- six views: whole, crop, and four crop quadrants;
- at most 1,024 dense PE tokens per view;
- 1,024 to 128 projection;
- gated masked attention pooling with attention width 64;
- dropout 0.10 and a two-class linear classifier;
- exactly 151,107 trainable parameters.

No image, view, feature, loss, architecture, threshold, or label changes
between strategies.

## Fixed strategies

### 1. SOURCE_RISK_CONTROL

This is rerun by the unchanged original H3 training script with seed 20260713.
Its sampler gives equal expected mass to each source-by-risk cell. It must
reproduce the locked H3 OOF probabilities to maximum absolute error at most
`1e-5`; otherwise H9 is invalid and no strategy comparison is interpreted.

### 2. SOURCE_RISK_SUBTYPE_TEMPERED

Each present source-by-risk cell receives equal total sampling mass. Within a
cell, subtype target mass is proportional to the square root of subtype count.
This reduces AB/TC dominance without making a four-case B3 cell equal to a
large subtype cell. Per-case weights are median-clipped to `[0.2, 5.0]` and
renormalized. Sampling is with replacement and draws exactly the number of
outer-training cases per epoch.

### 3. SOURCE_RISK_SUBTYPE_CURRICULUM

This starts from the exact tempered weights above. After each epoch, every
outer-training case is predicted in deterministic metadata order. Its true
class probability defines the next epoch's dynamic difficulty:

- easy: `p(true) >= 0.80`;
- medium: `0.50 <= p(true) < 0.80`;
- hard: `p(true) < 0.50`.

Difficulty factors are fixed as follows:

| Epochs | Stage | Easy | Medium | Hard |
| --- | --- | ---: | ---: | ---: |
| 1-8 | balanced warm-up | 1.00 | 1.00 | 1.00 |
| 9-16 | boundary bridge | 1.25 | 1.50 | 0.75 |
| 17-80 | hard replay | 0.75 | 1.25 | 2.00 |

The factors multiply tempered weights, followed by the same `[0.2, 5.0]`
median clip and normalization. Hard cases are never removed. Easy cases remain
present during hard replay to protect already learned boundaries.

## Locked optimization

- primary seed: 20260713;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- cross-entropy without label smoothing or class-weighted loss;
- batch size 8;
- gradient norm cap 5.0;
- maximum 80 epochs;
- validation BAcc checkpoint selection at threshold 0.5;
- patience 12;
- tempered and curriculum candidates run at least 24 epochs so all curriculum
  stages are exposed before early stopping is allowed;
- identical initialization and sampler generator seed for the two candidates;
- exactly one sampled training-set length per epoch.

The static and curriculum candidates may differ only through the epoch-specific
sampling weights. The original control preserves its historical stopping
behavior to test exact reproduction.

## Primary metrics

All metrics are computed from concatenated source-LODO test predictions:

- binary accuracy, BAcc, AUC, sensitivity, specificity, TP, and TN;
- per-source BAcc, sensitivity, and specificity;
- per-subtype correct count and accuracy;
- source-by-subtype results for B1 and B2;
- rescue/harm relative to locked H3 overall, within B1+B2, and within the
  diagnostic persistent-hard group;
- 20,000 source-by-risk-stratified paired bootstrap replicates.

A separate diagnostic difficulty map is defined only for reporting from the
locked OOF predictions of C1, C2, and H3: three correct = easy, two = medium,
one = salvage, zero = persistent hard. This map is not used for H9 training.

## Gates

Gate 0 is mandatory integrity: regenerated bank hashes pass and
SOURCE_RISK_CONTROL reproduces locked H3 probabilities within `1e-5`.

The curriculum is a capability pass only if all primary gates pass:

1. BAcc at least 0.7739.
2. TP at least 164.
3. TN at least 299.
4. B1 correct at least 40/62.
5. B2 correct at least 59/89.
6. Third-batch B2 correct at least 18/29.
7. Curriculum minus static-tempered BAcc at least 0.0100 and paired bootstrap
   95% CI lower bound above zero.
8. Curriculum minus locked H3 BAcc at least 0.0200 and paired bootstrap 95% CI
   lower bound above zero.
9. Positive locked-H3 BAcc delta in at least two sources, no source delta below
   -0.0200, and minimum source BAcc at least 0.7381.
10. B1+B2 net correct gain over locked H3 at least 7 cases.
11. Diagnostic persistent-hard net correct gain over locked H3 at least 5 cases.

If static-tempered sampling shows signal but curriculum does not beat it, the
result is `BALANCING_SIGNAL_ONLY`, not evidence for curriculum learning. Any
other non-passing result is `NO_GO_CURRENT_CURRICULUM`.

No fivefold or confirmation run is started unless the source-LODO curriculum
passes every gate. No test-derived threshold, blend, strategy selection, or
gate relaxation is permitted.

## Privacy and storage

Images, paths, identifiers, dense features, checkpoints, per-case predictions,
and dynamic difficulty assignments remain server-only. GitHub receives only
code, this preregistration, hashes, aggregate tables, and an aggregate report.
No server dataset or model artifact may be downloaded.
