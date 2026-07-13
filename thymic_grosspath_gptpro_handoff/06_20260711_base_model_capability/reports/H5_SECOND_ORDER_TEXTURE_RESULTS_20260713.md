# H5 Low-Rank Second-Order Texture Results (2026-07-13)

## Decision

H5 is a **NO-GO** under the preregistered direct-classifier gates.

The low-rank covariance branch learned useful mixed-source B1/B2 information,
but that information did not transfer across acquisition batches. Five-fold
OOF B1/B2 risk accuracy reached 0.6613/0.7079, yet source-LODO B1/B2 fell to
0.5968/0.4719. Source-LODO balanced accuracy was 0.7422, sensitivity was
0.6502, and specificity was 0.8342.

The conditional confirmation seed was not run. No covariance dimension,
normalization, fusion, regularization, sampler, loss, threshold, or seed search
is allowed after this result.

## Locked setup

- Cohort: the same 591 internal-development cases.
- Endpoint: A/AB/B1 low risk versus B2/B3/TC high risk.
- Encoder: frozen `facebook/PE-Spatial-L14-448`.
- Views: whole image, foreground crop, and four crop quadrants.
- Dense representation: up to 1,024 valid 1,024-dimensional tokens per view.
- Threshold: 0.5 at 100% coverage.
- Evaluation: unchanged five-fold OOF and acquisition-batch source-LODO.
- Primary seed: 20260713.
- Trainable parameters: 537,668.

The model combined the H3 first-order masked gated branch with one fixed
second-order branch. The latter projected tokens to 64 dimensions, computed a
masked per-view 64 x 64 covariance, retained the upper triangle, applied
signed-square-root and L2 normalization, and pooled six learned texture tokens.
The two image-derived case vectors were fused before one low/high classifier.

The first engineering smoke exposed FP16 gradient overflow before any
scientific output. As preregistered for engineering validation, numerical
execution was changed to BF16 autocast with FP32 fallback and a stable `1e-6`
square-root clamp. The repeated smoke passed; no scientific structure or
hyperparameter changed.

## Overall results

| Model | OOF BAcc | OOF AUC | OOF Sens | OOF Spec | LODO BAcc | LODO AUC | LODO Sens | LODO Spec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 | 0.7514 | 0.8377 | 0.7175 | 0.7853 | 0.7441 | 0.8108 | 0.7354 | 0.7527 |
| H3 PE-Spatial | **0.8003** | **0.8700** | **0.8072** | 0.7935 | **0.7539** | 0.7984 | 0.6816 | 0.8261 |
| H5 second-order texture | 0.7969 | 0.8395 | 0.7623 | **0.8315** | 0.7422 | **0.8133** | 0.6502 | **0.8342** |

H5 retained a high mixed-source BAcc but reduced AUC and sensitivity relative
to H3 PE. Under source-LODO it was not better than C2 and was less sensitive
than both references.

## Held-source results

| Held-out source | C2 BAcc | H3 PE BAcc | H5 BAcc | H5 - C2 | H5 - H3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| batch1 | 0.7890 | 0.7434 | 0.7355 | -0.0535 | -0.0079 |
| batch2 | 0.7083 | 0.7381 | 0.7143 | +0.0060 | -0.0238 |
| third_batch | 0.7337 | 0.7440 | 0.7115 | -0.0223 | -0.0326 |

Only batch2 improved versus C2, by 0.0060. Batch1 declined by 0.0535 and
third_batch declined by 0.0223, so the source-stability gates failed.

## Subtype results

| Subtype | C2 risk accuracy | H3 PE risk accuracy | H5 risk accuracy |
| --- | ---: | ---: | ---: |
| A | 0.6591 | 0.7045 | 0.7045 |
| AB | 0.8282 | 0.8893 | **0.9122** |
| B1 | 0.5000 | **0.6452** | 0.5968 |
| B2 | **0.6629** | 0.5843 | 0.4719 |
| B3 | 0.7083 | 0.7083 | 0.7083 |
| TC | **0.8000** | 0.7545 | 0.7818 |

The model became more accurate on dominant low-risk AB while severely
under-calling B2. On the third-batch held-out fold, high-risk sensitivity was
only 0.5122 while specificity was 0.9107.

## Paired evidence

- H5 minus C2 source-LODO BAcc: -0.0018, 95% CI
  [-0.0392, +0.0356], probability of a positive delta 0.4587.
- H5 minus C2 source-LODO sensitivity: -0.0852, 95% CI
  [-0.1435, -0.0269].
- H5 minus C2 source-LODO specificity: +0.0815, 95% CI
  [+0.0353, +0.1277].
- H5 minus H3 PE source-LODO BAcc: -0.0116, 95% CI
  [-0.0427, +0.0194].
- H5 versus C2 correctness transitions: 70 rescues, 59 harms, 80 persistent
  errors, and 382 stable correct cases.
- H5 versus H3 PE correctness transitions: 39 rescues, 43 harms, 96 persistent
  errors, and 413 stable correct cases.

The apparently favorable rescue count versus C2 is misleading at the clinical
operating point: rescues were concentrated in A/AB/B1, whereas H5 introduced
23 new B2 errors and rescued only six B2 errors.

## Gate audit

| # | Requirement | Observed | Status |
| ---: | --- | --- | --- |
| 1 | OOF BAcc >= 0.7903 | 0.7969 | Pass |
| 2 | OOF sensitivity >= 0.7772 and specificity >= 0.7635 | 0.7623 / 0.8315 | Fail |
| 3 | LODO BAcc >= 0.7641 | 0.7422 | Fail |
| 4 | LODO sensitivity >= 0.7354 | 0.6502 | Fail |
| 5 | LODO specificity >= 0.7527 | 0.8342 | Pass |
| 6 | LODO B1 >= 0.6000 | 0.5968 | Fail |
| 7 | LODO B2 >= 0.6629 | 0.4719 | Fail |
| 8 | At least two held-out sources improve versus C2 | 1/3 | Fail |
| 9 | No held-out source declines by more than 0.02 | minimum -0.0535 | Fail |
| 10 | LODO BAcc and sensitivity both exceed H3 PE | both lower | Fail |
| 11 | B2 exceeds H3 PE and B1 does not decline | both conditions fail | Fail |
| 12 | LODO BAcc delta versus C2 CI lower bound > 0 | -0.0392 | Fail |
| 13 | Confirmation seed directionally positive | not eligible | Not evaluated |
| 14 | Threshold 0.5 and coverage 100% | satisfied | Pass |

## Interpretation

1. True second-order PE feature co-occurrence can improve the mixed-source
   B1/B2 boundary, so the branch was not merely an implementation failure.
2. The gain was acquisition-dependent. Under source shift, the model primarily
   learned a conservative AB-favoring decision rule and lost B2 sensitivity.
3. The same selected photograph does not provide enough stable information for
   another covariance, head-capacity, or loss sweep to be justified.
4. H3 PE remains the strongest mixed-source direct model, but it also failed
   the cross-domain gates. C2 remains the more balanced locked internal
   reference; neither result establishes external generalization.
5. The next valid model experiment requires additional image information or a
   genuinely new cohort, not another confidence gate or current-feature
   reweighting.

## Multi-image feasibility

A corrected server-side aggregate audit found that all three raw dataset roots,
all 591 selected cached images, and all 608 internal image paths in the existing
case-bag registry are currently accessible. The earlier inaccessible-folder
result treated relative source-directory fields as absolute paths. The internal
case bag has 17 two-image old-data cases and 574 one-image cases; the third batch
has one image per case. Multi-image analysis is therefore feasible without a
remount, but its additional-view evidence is limited to 17 old-data cases. See
`reports/MULTI_IMAGE_AVAILABILITY_AUDIT_20260713.md`.

## Reproducibility and storage

Server aggregate outputs:

- `/workspace/thymic_project/experiments/h5_second_order_texture_20260713/primary_seed20260713`
- `/workspace/thymic_project/experiments/h5_second_order_texture_20260713/gate_aggregate`
- `/workspace/thymic_project/experiments/h5_second_order_texture_20260713/data_availability_audit`

The exact PE bank used by H5 has feature SHA-256:

`e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34`

Server execution hashes:

- H5 trainer: `6c96629a5029bff9c37f5f782a4f9faf41372d0ed9c218087a2f689a22a606ab`
- H5 gate summarizer: `aec56d751c8076ef08010923ea5790e5417d67e0a26bc5817a297e714596575a`
- multi-image availability audit: `71d30130decee25d6f029caacab189b9e2d79c2f2fb3146323806479ac98ff1b`
- five-fold run config: `28e3240c074cc63e05ddd0c6d6577c5b353321af0be12742f6ff7ee7809469a9`
- source-LODO run config: `130555244fc18d1941f985e3ecae468e20ccf44b3aaa4f914aa21e4fde558bbb`
- gate summary: `6cee2db0a11c2172fa2e3846244428bcf136f1f3e924490aed69814b151683b4`

All patient images, feature arrays, case-level predictions, and weights stayed
on the server. The regenerable PE bank may be deleted after the H5 code and
aggregate results are recorded.
