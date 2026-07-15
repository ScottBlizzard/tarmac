# H10 Internal Phenotype-Difficulty Redesign Results

Experiment ID: `H10_INTERNAL_PHENOTYPE_DIFFICULTY_REDESIGN_20260715`

Date completed: 2026-07-15

## Decision

`NO_CLEAR_INTERNAL_SAMPLER_WINNER`

Treating all 591 same-institution development cases as one internal cohort was the
correct way to answer the internal-capability question. It did not, however, produce a
90% forced classifier. The descriptively best source-free configuration was binary
risk-balanced sampling at 0.7888 balanced accuracy and 0.7986 ordinary accuracy.

Its balanced-accuracy gain over natural sampling was +0.0229, but the prespecified
20,000-replicate paired bootstrap interval crossed zero: 95% CI
[-0.0059, +0.0527]. It therefore cannot be called a clear sampling-method win.

## Integrity and evaluation

- Cohort: 591 unique cases and 591 unique original pathology IDs.
- Fold construction: subtype-only stratified fivefold with seed 20260715.
- Fold sizes: 119/118/118/118/118.
- Every subtype count differed by at most one across folds.
- Batch/source was not read by the split algorithm and was absent from all three
  sampling-weight definitions.
- Encoder, checkpoint, views, dense-token arrays, head, and optimizer matched locked
  H3.
- Regenerated dense arrays reproduced all historical hashes exactly:
  - features: `e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34`;
  - mask: `af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c`;
  - spatial shapes: `14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f`.
- Threshold was fixed at 0.5 with 100% case coverage.
- The exposed strict external set was not accessed.

## Overall results

| Internal fivefold method | Accuracy | BAcc | AUC | Sensitivity | Specificity | TP | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Risk-balanced | **0.7986** | **0.7888** | 0.8430 | **0.7489** | **0.8288** | **167** | **305** |
| Natural, once per epoch | 0.7800 | 0.7659 | 0.8471 | 0.7085 | 0.8234 | 158 | 303 |
| Subtype-tempered | 0.7733 | 0.7632 | **0.8527** | 0.7220 | 0.8043 | 161 | 296 |

Paired bootstrap comparisons:

| Comparison | Delta BAcc | 95% CI | P(delta > 0) |
| --- | ---: | ---: | ---: |
| Risk-balanced minus natural | +0.0229 | [-0.0059, +0.0527] | 0.9415 |
| Risk-balanced minus subtype-tempered | +0.0257 | [-0.0045, +0.0563] | 0.9514 |
| Natural minus subtype-tempered | +0.0028 | [-0.0251, +0.0307] | 0.5730 |

Risk balancing is therefore the most defensible Stage-2 baseline, but not a confirmed
winner.

## Subtype results

| Method | A | AB | B1 | B2 | B3 | TC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Natural | 37/44 | 232/262 | 34/62 | 54/89 | 19/24 | 85/110 |
| Risk-balanced | 34/44 | **234/262** | 37/62 | **54/89** | **19/24** | **94/110** |
| Subtype-tempered | 32/44 | 226/262 | **38/62** | 52/89 | 16/24 | 93/110 |

The risk-balanced gain came mainly from TC (+9 correct versus natural), AB (+2), and
B1 (+3). It did not improve B2 at all. Subtype tempering gained four B1 cases versus
natural but lost two B2, five A, six AB, three B3, and four total cases. This is direct
evidence for the user's concern that more exposure to underrepresented subtypes is not
monotonically beneficial.

The best source-free H10 result is also below locked historical H3 mixed-source
fivefold BAcc 0.8003 (B1 42/62, B2 62/89). That historical run used the old master
folds and a source-risk sampler, so it is not a like-for-like source-free estimate.
H10 shows that simply declaring the three batches one domain does not create a better
boundary.

## Source audit only

Risk-balanced BAcc after training without source labels:

| Historical source label | BAcc | Sensitivity | Specificity |
| --- | ---: | ---: | ---: |
| batch1 | 0.7868 | 0.8070 | 0.7667 |
| batch2 | 0.7619 | 0.7381 | 0.7857 |
| third batch | 0.7906 | 0.7195 | 0.8616 |

These are audits, not separate domains in the H10 primary analysis.

## Why the old difficulty definition was inadequate

Physician gross findings matched 589/591 cases, but a scalar high/low gross-concept
score had only AUC 0.6621 for the binary target. Of the 589 matched cases, 359 contained
both nominally low- and high-risk concepts. Gross morphology is therefore a phenotype
description, not a deterministic risk label.

The three source-free sampling variants produced the following cross-fitted diagnostic
roles:

| Diagnostic role | Cases | Meaning |
| --- | ---: | --- |
| `canonical_anchor` | 109 | Physician pattern is canonical and all three models are correct |
| `stable_noncanonical` | 299 | All three models are correct despite mixed/discordant/sparse concepts |
| `learnable_boundary` | 107 | One or two models are correct |
| `persistent_canonical_failure` | 12 | Canonical physician pattern, but all three image models fail |
| `persistent_mimic_failure` | 63 | Mixed/discordant phenotype and all three models fail |
| `persistent_sparse_or_missing` | 1 | Sparse/missing concepts and all three models fail |

Thus 408/591 cases were stable across all three internal models, 107 were genuinely
model-dependent boundary cases, and 76 were persistent failures.

B1/B2 accounted for 41/76 persistent failures (53.9%) although they represented only
151/591 cases (25.5%). The enrichment odds ratio was 4.31 (Fisher exact p=1.21e-8).
Within B1/B2 specifically:

- 68 were stable under all three models;
- 42 were model-dependent boundary cases;
- 33 were persistent mixed/discordant mimics;
- 8 were persistent failures despite a canonical physician pattern.

This is a more useful difficulty decomposition than probability thresholds. The 33
phenotype-mimic failures should not be handled identically to the eight canonical
representation failures.

## Physician-concept error anatomy

Exploratory risk-balanced associations were directionally coherent but not sufficient
as a classifier:

- low-risk false positives had less complete-capsule description and more soft texture,
  unclear boundary, attached lung/pericardium, or hemorrhage mentions;
- high-risk false negatives more often had soft texture and less often had attached
  pericardium/fat/lung mentions;
- simple concept-score or test-time lookup is therefore inappropriate.

These associations may define phenotype-matched B1/B2 pairs or auxiliary targets, but
they must be generated from outer-training cases only.

## Post-hoc ensemble diagnostic

Equal averaging was not used to choose H10:

- natural+risk average: BAcc 0.7897, AUC 0.8607, TP/TN 171/299;
- all-three average: BAcc 0.7875, AUC 0.8680, TP/TN 167/304.

The higher AUC did not yield a material fixed-threshold accuracy gain. This repeats the
project-wide finding that correlated models mostly exchange sensitivity and
specificity rather than solve B1/B2.

## Consequence for the next experiment

The next curriculum must not reuse the 591-case OOF roles above as training labels.
For each outer fold it must:

1. generate natural/risk/subtype-tempered predictions using nested OOF restricted to
   the outer training cases;
2. combine those model-stability labels with physician phenotype categories;
3. keep canonical anchors, model-dependent boundary cases, phenotype mimics, and
   canonical representation failures separate;
4. retain all cases with bounded positive weights rather than deleting all hard cases;
5. compare against locked risk-balanced H10 on exactly the same subtype-only folds;
6. require gains in B1 and B2, not only TC/AB or AUC.

All case IDs, predictions, concepts, checkpoints, and derived training roles remain
server-only. Only aggregate results and reproducible code enter the repository.
