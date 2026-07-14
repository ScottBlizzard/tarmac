# H7 PE-Embedding LISA Results

Date: 2026-07-14

## Decision

`H7_PE_EMBEDDING_LISA_20260714` is a **NO-GO**.

- Primary source-LODO gates: **FAIL**.
- Five-fold secondary training: **not run**.
- Confirmation seed `20260717`: **not run**.
- Locked next action: `STOP_H7_SOURCE_LODO_NO_GO`.

H7 was explicitly registered as a post-H6-stop exploratory experiment after
the user requested continued fixed-data work. It cannot override H6's original
stopping decision or provide independent external validation.

## Locked experiment

H7 retained the exact H3 PE-Spatial-L14-448 dense bank, six deterministic
views, masked gated 128-dimensional case embedding, 151,107-parameter head,
source-risk sampler, optimizer, threshold 0.5, and 100% coverage. It changed
only the training operation on the case embedding:

- 50% same-risk pairs from different acquisition sources;
- 50% opposite-risk pairs from the same acquisition source;
- symmetric embedding and one-hot target interpolation;
- `lambda ~ Beta(2, 2)`;
- soft-label cross-entropy with no added ERM loss.

All partners came only from each outer-training partition. Realized
cross-source pairing fractions were 0.5000, 0.5001, and 0.5000 in the three
folds. Pairing integrity passed.

The 108- and 162-case previously inspected stress cohorts were not read,
paired, trained, validated, or evaluated.

## Main source-LODO result

| Model | BAcc | AUC | Sensitivity | Specificity | B1 | B2 |
|---|---:|---:|---:|---:|---:|---:|
| H3 PE-Spatial | 0.7539 | 0.7984 | 0.6816 | 0.8261 | 40/62 | 52/89 |
| C2 reference | 0.7441 | 0.8108 | 0.7354 | 0.7527 | 31/62 | 59/89 |
| H7 PE-embedding LISA | **0.7382** | **0.8352** | **0.6368** | **0.8397** | **43/62** | **43/89** |

H7 raised AUC and specificity but reduced the fixed-threshold balanced
accuracy and high-risk sensitivity. This is another ranking-versus-decision
tradeoff, not an improvement in the required forced classifier.

## Held-source behavior

| Held source | H3 BAcc | H7 BAcc | Delta |
|---|---:|---:|---:|
| batch1 | 0.7434 | 0.7684 | +0.0250 |
| batch2 | 0.7381 | 0.7500 | +0.0119 |
| third batch | 0.7440 | 0.6869 | -0.0571 |

Two sources improved, but the third-batch decline was almost four times the
maximum allowed harm. On held-out third-batch cases, sensitivity was 0.5122
and specificity was 0.8616. The mean H7 probability shift versus H3 was
-0.0615 in the third batch, showing a systematic move toward low-risk
predictions rather than random fold noise.

## Subtype movement

| Subtype | N | H3 correct | H7 correct | Rescued by H7 | Harmed by H7 | Mean probability shift |
|---|---:|---:|---:|---:|---:|---:|
| A | 44 | 31 | 36 | 6 | 1 | -0.0253 |
| AB | 262 | 233 | 230 | 7 | 10 | -0.0321 |
| B1 | 62 | 40 | 43 | 5 | 2 | -0.0368 |
| B2 | 89 | 52 | 43 | 4 | 13 | -0.0547 |
| B3 | 24 | 17 | 19 | 3 | 1 | +0.0156 |
| TC | 110 | 83 | 80 | 4 | 7 | -0.0207 |

The central tradeoff is explicit: H7 gained three net B1 cases and lost nine
net B2 cases. In the held-out third batch alone, B2 fell from 13/29 to 8/29 and
TC fell from 38/53 to 34/53. Selective interpolation did not create a stronger
B1/B2 visual boundary; it made the classifier more conservative on the source
where high-risk transfer was already weakest.

## Paired bootstrap

All intervals used 20,000 paired resamples stratified by source and binary
risk.

| Comparison | Metric | Mean delta | 95% CI | P(delta > 0) |
|---|---|---:|---:|---:|
| H7 - H3 | BAcc | -0.0156 | [-0.0444, 0.0127] | 0.1439 |
| H7 - H3 | Sensitivity | -0.0447 | [-0.0942, 0.0045] | 0.0289 |
| H7 - H3 | Specificity | +0.0136 | [-0.0163, 0.0435] | 0.7885 |
| H7 - H3 | B1 accuracy | +0.0485 | [-0.0333, 0.1346] | 0.8318 |
| H7 - H3 | B2 accuracy | **-0.1008** | **[-0.1910, -0.0119]** | **0.0081** |
| H7 - C2 | Sensitivity | -0.0986 | [-0.1614, -0.0404] | 0.0006 |

The B2 harm versus H3 and sensitivity harm versus C2 were not compatible with
the advancement hypothesis.

## Gate result

H7 passed only specificity, B1, two-of-three source direction, pairing
integrity, and coverage. It failed:

- BAcc 0.7382 versus required 0.7641;
- sensitivity 0.6368 versus required 0.7354;
- B2 43/89 versus required 59/89;
- minimum source BAcc 0.6869 versus required 0.7381;
- maximum source harm -0.0571 versus allowed -0.015;
- bootstrap superiority/noninferiority versus H3;
- sensitivity noninferiority versus C2.

The queue therefore correctly blocked five-fold training and the confirmation
seed. No alpha, pairing ratio, interpolation layer, threshold, ordinary mixup,
or seed search followed.

## Fixed-data conclusion

H7 closes the remaining distinction between previously failed generic
augmentation and source/label-selective interpolation on the H3 information.
Together with the completed backbone, dense-token, ROI, multi-view, texture,
frequency, concept, subtype, ordinal, contrastive, expert, domain-alignment,
common/specific, and optimization experiments, there is no remaining
project-specific image-only method with both:

1. a materially different, evidence-supported mechanism; and
2. a valid fixed-data evaluation path that has not already been consumed.

This is a statement about exhausted **tested and defensible directions on the
current 591 selected photographs**, not proof that the medical problem is
intrinsically unsolvable. H3 remains the strongest mixed-source direct model
at 0.8003 five-fold BAcc, while no tested candidate established a reproducible
source-held improvement.

## Reproducibility and storage

Server root:

`/workspace/thymic_project/experiments/h7_pe_embedding_lisa_20260714`

Key SHA-256 values:

- source-LODO predictions:
  `52b29a23b1eeb8cdab9c5735d41a20452a2df1e2cf50f3fb4931b16166c24138`;
- decision JSON:
  `c95348f612073e41e3138016d456d5e64d227e52dfc19fb6c37d86b877b46a8a`;
- model metrics:
  `9c1d4f4c75eaa6990a1a4f8e2feee2eeb9efe172b34a89996bc3cabe49b6ea40`;
- paired bootstrap:
  `51a172c5abb70ce7606099982b02f8708c6b497a7d12cfec728cabd9da33bfdf`;
- H7 code lock:
  `6364124cfef5af7bb46baf816c5aeb61cc87894b198e454c45caa2d61b3da298`.

All predictions, fold checkpoints, histories, pairing diagnostics, logs, and
aggregate tables remain on the server. The reproducible 7.0 GiB PE memmap
cache was removed after queue completion and hash verification; free server
space returned to approximately 40 GiB. No patient image, feature array,
prediction table, or checkpoint was downloaded to the local machine.
