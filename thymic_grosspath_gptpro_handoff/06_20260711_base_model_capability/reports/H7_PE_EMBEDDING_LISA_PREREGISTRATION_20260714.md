# H7 PE-Embedding LISA Preregistration

Date: 2026-07-14

## Status and scope

`H7_PE_EMBEDDING_LISA_20260714` is a bounded **post-H6-stop exploratory
experiment**. H6 correctly triggered its preregistered fixed-data stopping rule.
The user subsequently requested continued exploration until no executable
image-only direction remained. H7 therefore cannot be presented as a
confirmatory continuation of the H6 queue or as independent external
validation.

The experiment tests the only unexecuted method that GPT Pro's fixed-data
literature review ranked as technically executable: LISA selective
augmentation. It changes the training-pair assumption while retaining the
locked image representation, classifier capacity, endpoint, threshold, and
coverage.

## Question

Can a risk predictor become less dependent on acquisition-specific label
correlations when training interpolates only:

1. cases with the same binary risk from different acquisition sources; or
2. cases with opposite binary risk from the same acquisition source?

This differs from prior generic MixStyle, photometric augmentation,
whole/ROI consistency, DANN, REx, GroupDRO, and hard-pair contrastive losses.
Those methods did not use the exact source/label-selective interpolation rule.

## Immutable data and endpoint

- Development cohort: exactly 591 cases.
- Sources: batch1 117, batch2 168, third batch 306.
- Binary endpoint: low risk `A/AB/B1` versus high risk `B2/B3/TC`.
- Primary protocol: source leave-one-domain-out over the three acquisition
  sources.
- Secondary protocol: locked five-fold case-level split, run only if all
  source-LODO gates pass.
- Threshold: 0.5.
- Coverage: 591/591; no rejection, routing, source calibration, or manual
  review.
- The previously inspected 108- and 162-case stress cohorts are excluded from
  training, pairing, validation, checkpoint selection, and advancement.

## Immutable representation and model

- Frozen `facebook/PE-Spatial-L14-448` checkpoint and source snapshot used by
  H3.
- Six H3 views: whole, specimen crop, and four specimen quadrants.
- Exact dense bank shape: `591 x 6 x 1024 x 1024`, with the original valid-token
  mask.
- Exact H3 head through its 128-dimensional normalized case embedding:
  LayerNorm, 1024-to-128 projection, GELU, dropout 0.10, view embeddings,
  masked gated pooling, and output LayerNorm.
- Exact H3 dropout-plus-linear two-class classifier.
- Parameter count: 151,107, identical to H3.
- No source input, subtype head, concept head, quality feature, nuisance feature,
  external model, or additional trainable module is introduced.

## Locked LISA operation

For every source-risk-balanced sampled anchor, choose one training-only partner:

- 50%: same binary risk, different acquisition source;
- 50%: same acquisition source, opposite binary risk.

The two images are independently encoded by the shared H3 visual head. Their
128-dimensional case embeddings and one-hot targets are interpolated with
`lambda ~ Beta(2, 2)`. Each pair contributes both symmetric directions. The
loss is soft-label cross-entropy on mixed embeddings only, matching the
published LISA training principle. No additional ERM coefficient is added.

Partner pools are constructed separately inside every outer-training split.
Held-out and validation cases cannot be partners. Pairing mode, source/risk
transition, pool size, and realized mode fraction are recorded for every fold.

## Optimization and selection

- Source x binary-risk inverse-frequency sampler, identical to H3.
- AdamW, learning rate `3e-4`, weight decay `1e-4`.
- Batch size 8, maximum 80 epochs, patience 12.
- Global gradient clipping 5.0.
- Primary seed `20260716`; fold seed is primary seed plus `1000 * fold_id`.
- Checkpoint order: validation BAcc, then validation sensitivity, then earlier
  epoch.
- No alpha, pairing ratio, mix layer, loss weight, threshold, source, or seed
  sweep is allowed.

## Source-LODO advancement gates

All gates must pass:

| Gate | Requirement |
|---|---:|
| LODO BAcc | at least 0.7641 |
| Sensitivity | at least 0.7354 |
| Specificity | at least 0.7800 |
| B1 | at least 38/62 |
| B2 | at least 59/89 |
| Minimum held-source BAcc | at least 0.7381 |
| Source direction | at least 2/3 sources improve versus H3 |
| Maximum source harm | no source below H3 by more than 0.015 |
| Paired bootstrap versus H3 | mean delta positive, `P(delta>0) >= 0.80`, lower 95% bound above -0.010 |
| Sensitivity versus C2 | paired lower 95% delta bound above -0.020 |
| Pairing integrity | both modes present; cross-source fraction 0.48-0.52 |
| Coverage | 591/591 at threshold 0.5 |

Failure stops H7 without five-fold training, alpha search, ordinary mixup
control, or a second seed.

## Conditional five-fold and confirmation gates

If source-LODO passes, five-fold must retain:

- BAcc at least 0.7903;
- sensitivity and specificity each at least 0.7800;
- B1 at least 40/62 and B2 at least 60/89;
- minimum fold BAcc at least 0.70.

Only if both protocols pass is seed `20260717` run once. It must independently
pass the same point-estimate gates. A confirmation failure is a NO-GO.

## Interpretation ceiling

A positive result would be an exploratory internal cross-batch engineering
improvement on a consumed development cohort. It would still require a new,
locked multicenter cohort for an external-generalization claim. A negative
result closes the remaining distinction between generic augmentation and
source/label-selective interpolation on the current H3 information.

