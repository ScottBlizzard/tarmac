# PE-Spatial Label-Free Part Stability Audit Preregistration (2026-07-13)

## Role and boundary

This is a label-free representation audit after the H3-H5 no-go decision. It
does not train or nominate a risk classifier. Risk labels are not used to form
parts. They are used only after extraction to compare acquisition-source and
risk associations in aggregate stability metrics.

## Cohort and encoder

- All 591 internal selected photographs.
- Frozen local `facebook/PE-Spatial-L14-448` checkpoint and official cached
  source revision already used by H3.
- Aligned final dense tokens from `forward_features(..., norm=True,
  strip_cls_token=True)`.
- Native-aspect tokenization with at most 1,024 patches.
- No model, code, or image download.

## Locked part discovery

For each clean whole photograph:

1. L2-normalize PE tokens.
2. Apply one fixed seed-20260713 Gaussian random projection from 1,024 to 64
   dimensions and renormalize.
3. Fit three-cluster K-means on that image only (`n_init=10`, `max_iter=100`).

The three clusters are intentionally called discovered parts, not anatomical
regions. No label, physician annotation, or cross-case prototype enters this
step.

## Locked perturbations

Extract aligned PE tokens for three deterministic variants:

1. brightness 1.08, contrast 1.10, and color 0.90;
2. Gaussian blur radius 0.8 followed by JPEG quality 85;
3. horizontal flip, spatially unflipped before comparison.

Assign variant tokens to the clean-image centroids. Report clean-versus-variant
label agreement, adjusted Rand index, aligned token cosine similarity, and
cluster-occupancy total variation.

Also compare clean discovered parts with the existing label-free specimen mask
using normalized mutual information. This tests whether apparent parts mainly
separate specimen from background.

## Locked interpretation

Call the maps stable only if the worst perturbation has median adjusted Rand
index at least 0.70 and median token cosine at least 0.85.

Call the maps nondegenerate only if median normalized three-cluster entropy is
at least 0.60 and median largest-cluster fraction is at most 0.80.

Call them background-dominated if median cluster-versus-specimen-mask NMI is at
least 0.50.

For every metric, compute source partial eta-squared controlling risk and risk
partial eta-squared controlling source. Call stability source-sensitive if
source effects dominate risk effects across the stability metrics and any
source partial eta-squared reaches 0.05.

Only stable, nondegenerate, non-background-dominated, and non-source-sensitive
maps may justify a future blinded physician part-annotation study. No outcome
from this audit authorizes another current-cohort classifier.

## Storage

Tokens are processed in memory and discarded after each case. Case-level audit
metrics remain on the server. GitHub receives code, preregistration, and
aggregate results only.
