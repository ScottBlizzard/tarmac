# H6 Nuisance-Anchored CSD Results

Date: 2026-07-14

## Decision

`H6_NUISANCE_ANCHORED_CSD_20260714` is a **NO-GO**.

- Primary source-LODO gates: **FAIL**.
- Secondary five-fold retention gates: **FAIL**.
- Confirmation seed `20260715`: **not run**.
- Conditional Fishr backup: **not triggered**.
- Locked next action: `STOP_CURRENT_DATA_VISUAL_DEVELOPMENT`.

The result does not support replacing H3. It closes this fixed-representation,
fixed-data common/specific decomposition branch without a rank, coefficient,
environment-count, seed, threshold, fusion, or follow-up method search.

## Locked Scope and Integrity

H6 used the exact 591-case H3 development cohort: batch1 117, batch2 168, and
third batch 306. The 108- and 162-case previously inspected stress cohorts were
excluded from extraction, environment construction, fitting, checkpoint
selection, thresholding, and advancement.

The frozen PE-Spatial-L14-448 representation, six H3 views, masked gated head,
classifier dropout 0.10, threshold 0.5, and 100% coverage were retained. H6
replaced only the final linear readout with rank-1 common/specific weights. The
deployed and evaluated prediction always used the common head without an
environment ID.

Integrity manifest SHA-256:

`0413c895cc2dea2997b2f9b90cf9c59d799151f32add223196d7d920182efe04`

The regenerated PE arrays matched the original H3 byte-level hashes before
training. The authoritative 591-case registry, split, frequency matrix, H3/C2
case predictions, PE checkpoint, training scripts, and 48-file PE source tree
were also locked. The 861-row registry was explicitly rejected because it
contains the 270 consumed stress cases.

## Main Results

| Protocol | Model | BAcc | AUC | Sensitivity | Specificity | B1 | B2 |
|---|---|---:|---:|---:|---:|---:|---:|
| Source-LODO | H3 | 0.7539 | 0.7984 | 0.6816 | 0.8261 | 0.6452 | 0.5843 |
| Source-LODO | C2 | 0.7441 | 0.8108 | 0.7354 | 0.7527 | 0.5000 | 0.6629 |
| Source-LODO | Source-only CSD | 0.7164 | 0.8110 | 0.6502 | 0.7826 | 0.6129 | 0.5393 |
| Source-LODO | NA-CSD | 0.7454 | 0.8226 | 0.6592 | 0.8315 | 0.6774 | 0.4944 |
| Five-fold | H3 | 0.8003 | 0.8700 | 0.8072 | 0.7935 | 0.6774 | 0.6966 |
| Five-fold | C2 | 0.7514 | 0.8377 | 0.7175 | 0.7853 | 0.4516 | 0.5955 |
| Five-fold | Source-only CSD | 0.7689 | 0.8598 | 0.7578 | 0.7799 | 0.5806 | 0.6629 |
| Five-fold | NA-CSD | 0.7687 | 0.8524 | 0.7848 | 0.7527 | 0.5323 | 0.6629 |

Against H3, NA-CSD changed source-LODO BAcc by -0.0085 and five-fold BAcc by
-0.0316. It therefore improved neither the primary cross-batch boundary nor the
strongest mixed-source direct capability.

## Source-Held Failure Localization

| Held source | H3 BAcc | NA-CSD BAcc | Delta |
|---|---:|---:|---:|
| batch1 | 0.7434 | 0.7421 | -0.0013 |
| batch2 | 0.7381 | 0.7679 | +0.0298 |
| third batch | 0.7440 | 0.7030 | -0.0410 |

NA-CSD helped batch2 but harmed the third batch enough to dominate the overall
result. On the held-out third batch, sensitivity was 0.5488 while specificity
was 0.8571. This is a high-risk miss problem, not a general accuracy-only loss.

The most important subtype movement was B2:

| Source-LODO subtype | H3 correct | NA-CSD correct | Change |
|---|---:|---:|---:|
| A | 30/44 | 37/44 | +7 |
| AB | 233/262 | 227/262 | -6 |
| B1 | 40/62 | 42/62 | +2 |
| B2 | 52/89 | 44/89 | -8 |
| B3 | 17/24 | 18/24 | +1 |
| TC | 83/110 | 85/110 | +2 |

The apparent gains in A, B1, B3, and TC do not compensate for the eight-case B2
loss and the third-batch sensitivity collapse.

## Paired Bootstrap

All intervals used 20,000 paired resamples stratified by source and binary risk.

| Comparison | Protocol | Metric | Mean delta | 95% CI | P(delta > 0) |
|---|---|---|---:|---:|---:|
| NA-CSD - H3 | Source-LODO | BAcc | -0.0084 | [-0.0351, 0.0181] | 0.2673 |
| NA-CSD - H3 | Source-LODO | B2 accuracy | -0.0895 | [-0.1724, -0.0114] | 0.0090 |
| NA-CSD - C2 | Source-LODO | Sensitivity | -0.0758 | [-0.1390, -0.0135] | 0.0063 |
| NA-CSD - source-only CSD | Source-LODO | BAcc | +0.0289 | [0.0055, 0.0528] | 0.9921 |
| NA-CSD - H3 | Five-fold | BAcc | -0.0316 | [-0.0641, 0.0003] | 0.0256 |
| NA-CSD - source-only CSD | Five-fold | BAcc | -0.0001 | [-0.0262, 0.0260] | 0.4928 |

Nuisance anchoring was identifiable relative to source-only CSD under LODO, but
that improvement only raised a weak CSD baseline and did not exceed H3. Under
five-fold evaluation, nuisance anchoring and source-only CSD were equivalent.

## Mechanism and Fishr Decision

The specific direction was non-trivial: median gamma-scaled specific/common
classifier-norm ratios were 0.879-0.930 across the three LODO folds. However,
the diagnostic specific head was worse than the common head on every LODO
validation fold by 0.0086 to 0.0365 BAcc. Zero folds met the preregistered
requirement for a specific-head
advantage of at least 0.03.

The conditional Fishr trigger therefore failed even though the common-head
sensitivity/B2 floor also failed. Fishr was correctly not run. Running it, a
second seed, or another CSD hyperparameter would violate the locked stop rule.

## Gate Summary

Primary failures included:

- LODO BAcc 0.7454 versus required 0.7641;
- sensitivity 0.6592 versus required 0.7354;
- B2 44/89 versus required 59/89;
- minimum held-source BAcc 0.7030 versus required 0.7381;
- only one of three sources improved versus H3;
- maximum source harm -0.0410 versus allowed -0.015;
- bootstrap superiority/non-inferiority versus H3 failed;
- sensitivity non-inferiority versus C2 failed.

Secondary failures included:

- five-fold BAcc 0.7687 versus required 0.7903;
- specificity 0.7527 versus required 0.7800;
- B1 33/62 versus required 40/62;
- B2 59/89 versus required 60/89.

## Interpretation

The result rejects the hypothesis that a low-capacity common/specific readout
on the existing frozen PE features can recover a stable low/high-risk boundary.
It does not show that domain generalization is impossible. It shows that the
remaining error is not resolved by another environment-conditioned classifier
decomposition on the same information.

The dominant unresolved problem is a source-dependent high-risk boundary,
especially B2 and held-out third-batch cases. Continuing fixed-data visual
coefficient or architecture search would be post hoc tuning after an explicit
NO-GO. A scientifically defensible next phase must add genuinely new
information, such as a fresh cohort, prospectively specified annotations/ROIs,
or another independently motivated measurement modality. Existing physician
descriptions remain valid for error analysis but must not silently become
training labels unless a new annotation protocol and evaluation design are
declared in advance.

Manuscript-safe wording:

> Explicit separation of training-environment-specific classifier directions
> did not improve the source-held low/high-risk boundary. Together with the
> completed representation, augmentation, alignment, texture, expert and
> spatial-model experiments, this result shows that no tested method established
> a reproducible improvement in cross-batch visual capability on the fixed
> cohort.

## Server Artifacts

Root:

`/workspace/thymic_project/experiments/h6_nuisance_anchored_csd_20260714`

Key immutable outputs:

- source-LODO NA-CSD predictions SHA-256:
  `803884b9b5c356f223f1d112411444a5a77841c40de7e30dfc1c31cfa5ad3ea7`;
- five-fold NA-CSD predictions SHA-256:
  `bf8149792b29b9237e110bb4a9208ac69bbb2bc47238d3098e3d1dd6b5977ef3`;
- decision JSON SHA-256:
  `4f560479f51cd163f98551be6fa72cee0a3358a9fe2a61d722ccb9e9a84e4090`;
- aggregate results SHA-256:
  `18f1ea15ee9bbad7218bfd7181bab1b713f7872dae4f1d3655e26c5409fb70ba`.

All checkpoints, predictions, histories, fold diagnostics, logs, manifests, and
aggregate tables remain on the server. After hashes and queue completion were
verified, the reproducible 7.0 GiB PE memmap cache under `/root` was deleted;
server free space returned from approximately 33 GiB to 40 GiB. No source image,
feature array, prediction, or model was downloaded to the local machine.
