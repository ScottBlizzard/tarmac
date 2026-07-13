# H2 Canonical-Coordinate Spatial-Relational Results

Date completed: 2026-07-13

## Decision

**NO-GO. All nine preregistered gates failed.**

The conditional confirmation seed was not run. Under the locked protocol, this result closes further model optimization on the repeatedly reused 591 selected single photographs. It does not close standardized multi-view acquisition or independent multicenter validation.

## Provenance and Scope

- Code commit: `bb28e8f`.
- Integrity manifest SHA-256: `00569f2ddf01f8cc51cdbc4b822ac820587dfa88eef5dd9ce5970ce615800013`.
- Cases: 591 internal cases from three acquisition batches.
- Subtypes: A 44, AB 262, B1 62, B2 89, B3 24, and TC 110.
- Frozen input: SigLIP-L at 512 px, six views, 1,024 patch tokens per view, 1,024 dimensions per token.
- Threshold: 0.5, fixed before training.
- Primary seed: 20260713.
- Evaluation: five-fold OOF and internal acquisition-batch LODO.
- Bootstrap: 5,000 paired source-by-risk-stratified repetitions.

The three sources are not three hospitals. LODO is an internal batch-shift stress test, not multicenter external validation. Historical external sets of 108 and 162 cases were not loaded or enumerated.

## Tested Hypothesis

The experiment tested whether cross-source morphology was already present in the frozen C1 patch tokens but discarded by permutation-invariant pooling. Three heads used the same splits, sampler, optimizer, stopping rule, and threshold:

1. `matched_gated`: lightweight gated pooling.
2. `relational_permuted`: the full relational model with a deterministic independent patch permutation for every case and view while coordinates remained fixed.
3. `relational`: the full model with the true patch grids and canonical coordinates.

The per-case permutation prevented the control from learning one shared inverse permutation. It also retained coordinate and view-scale information, so a relational advantage could not be attributed only to crop geometry.

## Overall Results

| Protocol | Model | BAcc | AUC | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: |
| OOF | C1 | 0.7477 | 0.8240 | 0.7399 | 0.7554 |
| OOF | C2 | 0.7514 | 0.8377 | 0.7175 | 0.7853 |
| OOF | Matched gated | 0.7152 | 0.8053 | 0.7130 | 0.7174 |
| OOF | Relational, permuted | 0.7111 | 0.7698 | 0.7130 | 0.7092 |
| OOF | Relational, true grid | 0.7278 | 0.7715 | 0.7354 | 0.7201 |
| LODO | C1 | 0.7397 | 0.8072 | 0.7130 | 0.7663 |
| LODO | C2 | 0.7441 | 0.8108 | 0.7354 | 0.7527 |
| LODO | Matched gated | 0.7479 | 0.8002 | 0.6861 | 0.8098 |
| LODO | Relational, permuted | 0.7181 | 0.7684 | 0.6726 | 0.7636 |
| LODO | Relational, true grid | 0.6970 | 0.7661 | 0.6413 | 0.7527 |

The true-grid relational model was worse than C2 in both protocols. Its LODO BAcc was 0.0471 lower than C2, and its high-risk sensitivity was 0.0942 lower.

## Paired Bootstrap

| Comparison | Protocol | Metric | Delta | 95% CI |
| --- | --- | --- | ---: | --- |
| Relational - C2 | OOF | BAcc | -0.0236 | [-0.0655, 0.0173] |
| Relational - C2 | OOF | AUC | -0.0661 | [-0.1021, -0.0302] |
| Relational - C2 | LODO | BAcc | -0.0471 | [-0.0866, -0.0062] |
| Relational - C2 | LODO | Sensitivity | -0.0942 | [-0.1614, -0.0269] |
| Relational - permuted | OOF | BAcc | 0.0166 | [-0.0217, 0.0552] |
| Relational - permuted | LODO | BAcc | -0.0211 | [-0.0584, 0.0166] |

There is no evidence that preserving the true patch grid improved cross-batch performance. Under LODO, the point estimate favored the permuted control.

## LODO Source Results

| Source | C2 BAcc | Relational BAcc | Delta | Relational sensitivity | Relational specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| batch1 | 0.7890 | 0.7140 | -0.0750 | 0.5614 | 0.8667 |
| batch2 | 0.7083 | 0.6071 | -0.1012 | 0.6905 | 0.5238 |
| third_batch | 0.7337 | 0.7272 | -0.0065 | 0.6463 | 0.8080 |

No source improved. The largest decline occurred on batch2, while batch1 lost substantial high-risk sensitivity.

## LODO Subtype Results

| Subtype | N | C2 risk accuracy | Relational risk accuracy | Delta |
| --- | ---: | ---: | ---: | ---: |
| A | 44 | 0.6591 | 0.6136 | -0.0455 |
| AB | 262 | 0.8282 | 0.8511 | 0.0229 |
| B1 | 62 | 0.5000 | 0.4355 | -0.0645 |
| B2 | 89 | 0.6629 | 0.6180 | -0.0449 |
| B3 | 24 | 0.7083 | 0.5833 | -0.1250 |
| TC | 110 | 0.8000 | 0.6727 | -0.1273 |

AB improved, but every other subtype declined. The prespecified B1/B2 mean was 0.5267 versus the required 0.6115.

## Gate Audit

| Gate | Requirement | Observed | Pass |
| --- | --- | --- | --- |
| 1 | OOF BAcc >= 0.7664 | 0.7278 | No |
| 2 | LODO BAcc >= 0.7641 | 0.6970 | No |
| 3 | LODO sensitivity >= 0.7354 and specificity >= 0.7527 | 0.6413 / 0.7527 | No |
| 4 | LODO BAcc delta vs C2 CI lower bound > 0 | -0.0866 | No |
| 5 | LODO sensitivity delta vs C2 CI lower bound > -0.02 | -0.1614 | No |
| 6 | Source stability gate | 0/3 improved; minimum BAcc 0.6071 | No |
| 7 | B1 >= 0.5000 and B2 >= 0.6629 | 0.4355 / 0.6180 | No |
| 8 | B1/B2 mean >= 0.6115 and neither below C2 | 0.5267; both below C2 | No |
| 9 | Relational LODO BAcc >= permuted + 0.01 | -0.0211 delta | No |

## Interpretation

The experiment does not support the hypothesis that explicit local adjacency and canonical-coordinate relations recover a robust signal discarded by prior pooling heads. The relational architecture increased complexity but did not improve the direct classifier; under batch shift it reduced sensitivity and was significantly worse than C2.

This is evidence against another architecture or pooling sweep on the current selected primary photographs. It is not evidence that gross pathology is intrinsically unlearnable. The current input is one retrospectively selected photograph per case, acquisition is batch-confounded, and the dataset remains small and subtype-imbalanced.

## Locked Next Mainline

1. Do not tune this H2 model, threshold, sampler, loss, seed, or fusion.
2. Do not run a second current-image architecture experiment.
3. Start prospective standardized multi-view acquisition with fixed views, scale reference, lighting, camera metadata, and complete case inclusion.
4. Add independent hospitals and reserve at least one institution as a never-tuned external test set.
5. Use the existing physician morphology table for retrospective error analysis and acquisition design, not as a hidden model input or substitute label.
6. Reopen model development only on the new multi-view, multicenter cohort with patient-level splits and a direct full-coverage classifier as the primary endpoint.

## Privacy and Storage

All images, feature arrays, coordinate manifests, case-level predictions, and weights remain on the server. This report contains aggregate metrics only. The experiment directory occupies approximately 53 MB; the frozen 7.4 GB feature bank was retained and was not copied locally.
