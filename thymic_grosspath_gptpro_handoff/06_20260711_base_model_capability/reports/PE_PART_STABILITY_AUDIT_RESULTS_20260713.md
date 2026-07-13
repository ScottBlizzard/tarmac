# PE-Spatial Label-Free Part Stability Audit Results (2026-07-13)

## Conclusion

Per-image three-cluster PE-Spatial part maps were stable, nondegenerate, and not
primarily a specimen-versus-background split. However, their perturbation
stability and mask relationship remained acquisition-source sensitive. The
locked decision is therefore:

- stable: **yes**;
- nondegenerate: **yes**;
- background dominated: **no**;
- source sensitive: **yes**;
- supports a new blinded physician part-annotation exercise: **no**;
- authorizes a classifier: **no**.

This is evidence that PE-Spatial contains coherent local organization, not
evidence that the discovered clusters are diagnostic anatomy or transferable
risk features.

## Locked experiment

- 591 internal selected photographs.
- Frozen local `facebook/PE-Spatial-L14-448` aligned final dense layer.
- Three clean-image clusters after a fixed 1,024-to-64 random projection.
- Fixed brightness/contrast/color, blur/JPEG, and aligned horizontal-flip
  variants.
- Risk labels were not used to form clusters.
- Tokens were discarded after each case; no dense bank was retained.

## Stability results

| Perturbation | Median label agreement | Median adjusted Rand | Median token cosine | Median occupancy TV |
|---|---:|---:|---:|---:|
| Photometric | 0.9911 | 0.9708 | 0.9940 | 0.0021 |
| Blur + JPEG | 0.9882 | 0.9607 | 0.9873 | 0.0030 |
| Horizontal flip | 0.9763 | 0.9191 | 0.9642 | 0.0049 |

The worst median adjusted Rand index exceeded the locked 0.70 threshold, and
the worst median token cosine exceeded the locked 0.85 threshold.

## Degeneracy and background test

- Median normalized three-cluster entropy: 0.8258, above the 0.60 requirement.
- Median largest-cluster fraction: 0.6144, below the 0.80 limit.
- Median discovered-cluster versus specimen-mask NMI: 0.0878, below the 0.50
  background-dominance threshold.

The parts therefore do not collapse into one dominant cluster and are not
explained simply by the existing foreground mask.

## Source sensitivity

The most source-associated audit metrics were:

| Metric | Source partial eta-squared | Risk partial eta-squared |
|---|---:|---:|
| Cluster versus specimen-mask NMI | 0.1454 | 0.0031 |
| Blur/JPEG token cosine | 0.0846 | 0.0010 |
| Horizontal-flip adjusted Rand | 0.0332 | 0.0011 |
| Blur/JPEG adjusted Rand | 0.0299 | 0.0022 |
| Photometric adjusted Rand | 0.0255 | 0.0013 |

Mean blur/JPEG adjusted Rand declined from 0.9590 in batch 1 to 0.9395 in batch
2 and 0.9253 in the third batch. Mean flip adjusted Rand similarly declined
from 0.9207 to 0.8915 and 0.8737.

The locked source-sensitivity rule was met because source effects dominated
risk effects across the stability metrics and at least one source partial
eta-squared exceeded 0.05.

## Interpretation

The representation contains visually coherent local structure, but that
structure is not acquisition invariant. A physician annotation exercise would
now add workload without first resolving whether the discovered parts have the
same meaning across batches. The current result does not justify assigning
anatomical names to clusters, selecting a cluster for high-resolution rereading,
or training a part-based risk classifier.

Future use requires newly standardized multicenter images and blinded semantic
part annotations collected independently of model errors. On that new cohort,
the same audit can test whether source sensitivity falls before a part-based
classifier is considered.

## Reproducibility and privacy

Server-only outputs:

- `/workspace/thymic_project/experiments/pe_part_stability_audit_20260713/pe_part_stability_case_metrics.csv`
- `/workspace/thymic_project/experiments/pe_part_stability_audit_20260713/pe_part_stability_effects.csv`
- `/workspace/thymic_project/experiments/pe_part_stability_audit_20260713/pe_part_stability_summary.json`
- `/workspace/thymic_project/experiments/pe_part_stability_audit_20260713/run.log`

No image, token, case-level metric, prediction, or checkpoint was downloaded.

- Executed script SHA-256:
  `f59fdbc42c4d178cf08d2b22d7f8958f832aefdaa2ea90b3448ad0dddc4421b7`
- Aggregate summary SHA-256:
  `e5b6063bd696f7f5615a27d91b1573adbe69c957a3ab5b06e5812060724ee102`
