# H3 PE-Spatial Error-Shift Audit (2026-07-13)

## Scope

This is a retrospective, hypothesis-generating audit of source-LODO predictions.
It compares the locked C2 reference with the H3 PE-Spatial dense-token model.
The physician concept table and image-quality measurements are not model inputs.

- Cases: 591.
- Physician gross-concept coverage: 589/591.
- Threshold: 0.5.
- Coverage: 100%.
- Image-quality measurements were computed server-side from the selected cached
  image at a maximum analysis size of 256 pixels; no patient image was copied to
  the local repository.

## Prediction transitions

PE-Spatial rescued 61 C2 errors and introduced 46 new errors, for a net gain of
15 correctly classified cases.

| Source | Rescue | Harm | Persistent error | Stable correct |
| --- | ---: | ---: | ---: | ---: |
| batch1 | 9 | 14 | 16 | 78 |
| batch2 | 22 | 17 | 27 | 102 |
| third_batch | 30 | 15 | 46 | 215 |

The negative batch1 balance, 9 rescues versus 14 harms, agrees with the formal
source-LODO BAcc regression of -0.0456 versus C2.

| Subtype | Rescue | Harm | Persistent error | Stable correct |
| --- | ---: | ---: | ---: | ---: |
| A | 7 | 5 | 8 | 24 |
| AB | 23 | 7 | 22 | 210 |
| B1 | 15 | 6 | 16 | 25 |
| B2 | 8 | 15 | 22 | 44 |
| B3 | 2 | 2 | 5 | 15 |
| TC | 6 | 11 | 16 | 77 |

The model's net gains came primarily from AB and B1. B2 and TC shifted in the
wrong direction, matching the loss of high-risk sensitivity.

## Physician concepts

No physician gross-morphology concept survived within-comparison
Benjamini-Hochberg correction at q <= 0.10 for any locked contrast. Nominal
patterns such as lobulation or capsule mentions must not be treated as
established mechanisms and do not justify concept-supervised training on this
evidence alone.

This negative result is useful: the current doctor table explains gross
descriptions, but it does not isolate a single stable morphological subtype of
PE error across batches.

## Image-quality associations

Eleven quality associations survived within-comparison BH correction at
q <= 0.10. The consistent direction was loss of local detail in high-risk
errors.

| Contrast | Feature | Median delta, error/rescue group minus comparator | q |
| --- | --- | ---: | ---: |
| High-risk FN vs TP | Laplacian variance | -0.000602 | 0.0020 |
| High-risk FN vs TP | Mean gradient | -0.001447 | 0.0026 |
| High-risk FN vs TP | Gray standard deviation | -0.014037 | 0.0846 |
| B2 FN vs TP | Laplacian variance | -0.000479 | 0.0919 |
| PE harm vs stable correct | Mean saturation | -0.026236 | 0.0947 |
| High-risk PE harm vs stable correct | Mean saturation | -0.041873 | 0.0843 |
| PE rescue vs persistent C2 error | Gray standard deviation | +0.019963 | 0.0148 |
| PE rescue vs persistent C2 error | Gray entropy | +0.274178 | 0.0204 |
| PE rescue vs persistent C2 error | Mean gradient | +0.001058 | 0.0891 |
| PE rescue vs persistent C2 error | Laplacian variance | +0.000383 | 0.0891 |
| PE rescue vs persistent C2 error | Mean saturation | -0.019794 | 0.0900 |

These are observational associations on the same internal cohort used for
development. They do not prove that blur caused an error. They do support a
bounded next hypothesis: PE-Spatial benefits from information-rich local
structure and may be insufficiently invariant to camera/low-detail degradation.

## Consequence for H4

H4 tests training-time, pathology-preserving camera and low-detail degradation
with clean/augmented consistency. It is deliberately different from the failed
deterministic white-balance, contrast, and unsharp preprocessing experiments:

- degradation is stochastic but reproducibly cached;
- only training indices expose augmented features;
- one shared classifier must agree between clean and degraded views;
- validation and test use clean images only;
- no quality-based routing, threshold search, or source-specific calibration is
  permitted.

Server-only audit outputs:

`/workspace/thymic_project/experiments/h3_representation_renewal_20260713/error_shift_audit`
