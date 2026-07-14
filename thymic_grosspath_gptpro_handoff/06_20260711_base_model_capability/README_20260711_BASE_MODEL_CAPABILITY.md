# 2026-07-11 Task7 Base-Model Capability Update

Interpretation updated: 2026-07-13

Repository: https://github.com/ScottBlizzard/tarmac

This folder is the latest handoff for the project's primary scientific objective: improving the 100%-coverage, image-only forced classifier for thymic gross pathology Task7.

Task7 definition:

- Low risk: A / AB / B1.
- High risk: B2 / B3 / thymic carcinoma.

Selective release, rejection, and physician-review workflows remain downstream safety layers. They are not substitutes for a stronger base classifier and are not the optimization target in this update.

## Current Governing Outcome

The preregistered H5 low-rank second-order texture experiment is complete and
is a `NO-GO`. It directly classified frozen PE-Spatial tokens using first-order
gated evidence plus per-view 64 x 64 covariance texture tokens.

- H5 five-fold OOF BAcc/AUC: 0.7969/0.8395;
- H5 source-LODO BAcc/AUC: 0.7422/0.8133;
- H5 source-LODO sensitivity/specificity: 0.6502/0.8342;
- H5 source-LODO B1/B2 accuracy: 0.5968/0.4719;
- H5 minus C2 LODO BAcc: -0.0018, 95% CI [-0.0392, +0.0356];
- H5 minus C2 LODO sensitivity: -0.0852, 95% CI [-0.1435, -0.0269];
- held-source BAcc deltas versus C2: batch1 -0.0535, batch2 +0.0060,
  third_batch -0.0223.

H5 improved mixed-source B1/B2 to 0.6613/0.7079, but B2 collapsed to 0.4719
under source-LODO. H4 quality consistency similarly retained a high OOF result
while reducing LODO BAcc to 0.7254. These experiments confirm that stronger
mixed-source ranking, texture modeling, or specificity does not establish a
stronger transferable classifier.

Do not search covariance dimensions, texture normalizations, perturbation
severity, consistency weight, seed, threshold, quality routing, or source
calibration. H3 PE remains the strongest mixed-source direct model at 0.8003
BAcc, but H3, H4, and H5 all failed the cross-domain advancement gates.

The next valid model experiment requires genuinely new image information or a
new cohort. A corrected root-aware audit found all 591 cached selected images
and all 608 internal paths in the existing case-bag registry accessible. Only
17 old-data cases have a second image, while the remaining 574 cases have one;
the current data therefore provide only a narrow paired-view sensitivity test,
not broad multi-view training evidence.

GPT Pro independently reviewed the completed H3-H5 evidence and recommended
freezing model nomination on the repeatedly inspected 591-case cohort. The two
permitted diagnostic audits have now been completed:

- Fixed Haar frequency features predicted acquisition source after controlling
  risk (BAcc 0.4388 versus chance 0.3333; permutation p=0.0099), while risk after
  controlling source was weak (BAcc 0.5381; p=0.0792). All 156 frequency
  features had larger source than risk partial effects.
- PE-Spatial label-free parts were stable, nondegenerate, and not merely
  foreground/background clusters, but their stability was source-sensitive.
  Blur/JPEG token-cosine source partial eta-squared was 0.0846 versus risk
  partial eta-squared 0.0010.

These audits do not authorize another classifier. Current-cohort model
nomination is frozen; the mainline is now the standardized four-view,
multicenter acquisition and sealed fresh-external protocol.

### 2026-07-14 fixed-data clarification

No additional patient, hospital, image view, label, or unlabeled image pool can
be acquired for this project. The freeze is therefore narrowed to confirmatory
generalization claims and unconstrained model/seed/threshold/fusion searches.
One literature-justified exploratory visual-capability experiment may reopen
under a fixed source-LODO-first protocol. This can improve engineering
performance on the available data, but it cannot create fresh multicenter
confirmation. See
`reports/FIXED_DATA_EXPLORATORY_REOPEN_DECISION_20260714.md` and
`GPTPRO_PROMPT_20260714_FIXED_DATA_LITERATURE_SEARCH.md`.

Current governing files:

- `GPTPRO_PROMPT_20260713_AFTER_H3_H5_NO_GO.md`;
- `GPTPRO_RESPONSE_20260713_AFTER_H3_H5_NO_GO_BLANK.md`;
- `reports/H5_SECOND_ORDER_TEXTURE_RESULTS_20260713.md`;
- `reports/H5_SECOND_ORDER_TEXTURE_PREREGISTRATION_20260713.md`;
- `reports/MULTI_IMAGE_AVAILABILITY_AUDIT_20260713.md`;
- `reports/CURRENT_COHORT_MODEL_FREEZE_DECISION_20260713.md`;
- `reports/FREQUENCY_SOURCE_VS_RISK_AUDIT_PREREGISTRATION_20260713.md`;
- `reports/FREQUENCY_SOURCE_VS_RISK_AUDIT_RESULTS_20260713.md`;
- `reports/PE_PART_STABILITY_AUDIT_PREREGISTRATION_20260713.md`;
- `reports/PE_PART_STABILITY_AUDIT_RESULTS_20260713.md`;
- `reports/LOCAL_ASSET_AUDIT_AFTER_GPTPRO_20260713.md`;
- `reports/FOUR_VIEW_MULTICENTER_ACQUISITION_PROTOCOL_20260713.md`;
- `reports/FIXED_DATA_EXPLORATORY_REOPEN_DECISION_20260714.md`;
- `GPTPRO_PROMPT_20260714_FIXED_DATA_LITERATURE_SEARCH.md`;
- `GPTPRO_RESPONSE_20260714_FIXED_DATA_LITERATURE_SEARCH_BLANK.md`;
- `scripts/audit_task7_frequency_source_vs_risk_20260713.py`;
- `scripts/audit_task7_pe_part_stability_20260713.py`;
- `scripts/run_task7_h5_second_order_texture_20260713.py`;
- `scripts/summarize_task7_h5_gate_20260713.py`;
- `scripts/audit_task7_case_image_availability_20260713.py`;
- `reports/H4_QUALITY_DOMAIN_RANDOMIZATION_RESULTS_20260713.md`;
- `reports/H4_QUALITY_DOMAIN_RANDOMIZATION_PREREGISTRATION_20260713.md`;
- `scripts/extract_task7_h4_quality_augmented_dense_bank_20260713.py`;
- `scripts/run_task7_h4_quality_consistency_20260713.py`;
- `scripts/summarize_task7_h4_gate_20260713.py`;
- `reports/H3_REPRESENTATION_RENEWAL_RESULTS_20260713.md`;
- `reports/H3_REPRESENTATION_RENEWAL_PREREGISTRATION_20260713.md`;
- `scripts/extract_task7_h3_representation_bank_20260713.py`;
- `scripts/extract_task7_h3_dense_bank_20260713.py`;
- `scripts/run_task7_h3_summary_gated_20260713.py`;
- `scripts/run_task7_h3b_masked_gated_20260713.py`;
- `scripts/summarize_task7_h3_representation_screen_20260713.py`;
- `scripts/summarize_task7_h3b_gate_20260713.py`;
- `reports/H2_CANONICAL_SPATIAL_RELATIONAL_RESULTS_20260713.md`;
- `reports/H2_CANONICAL_SPATIAL_RELATIONAL_PREREGISTRATION_20260713.md`;
- `scripts/prepare_task7_spatial_relational_assets_20260713.py`;
- `scripts/run_task7_spatial_relational_20260713.py`;
- `scripts/analyze_task7_spatial_relational_20260713.py`;
- `scripts/run_task7_spatial_relational_queue_20260713.sh`;
- `reports/AI Pathology Model Improvement.md`, section 9.

## Pre-H2 Historical Packet

1. `GPTPRO_PROMPT_20260713_AFTER_SEQUENTIAL_NO_GO.md`
   - Current self-contained English briefing for an independent expert with zero prior project context.
   - Defines the clinical task, data, evaluation, historical 92% correction, leakage-clean baselines, negative evidence, sequential AB/TC result, and exact next decision request.

2. `reports/H1_SEQUENTIAL_AB_TC_FALLBACK_RESULTS_20260713.md`
   - Formal five-fold, source-LODO, bootstrap, subtype, source, routing, and fallback results for the sequential AB/TC expert system.

3. `reports/Task7_WaveC_D1_E1_F1_F2_Results_20260712.md`
   - Exact metrics and predeclared decisions for the preceding direct-image and fixed visual-ensemble experiments.

4. `reports/Task7_Base_Model_Capability_Experiments_20260711.md`
   - Full internal experiment ledger from runs 206-369, including positive, negative, fusion, bootstrap, source-LODO, and error-analysis evidence.

5. `reports/Task7_GPTPro_Plan_Execution_A1_B1_Results_20260712.md`
   - Completed A1 native-detail direct-model and B1 fixed-route M0-M4 results.
   - Both fixed-grid families failed their predeclared OOF/source-LODO advancement gates.

6. `reports/Task7_Visual_Capability_and_Genuine_Coarse_to_Fine_Reframing_20260712.md`
   - Distinguishes visual capability, genuine image-grounded ensembles, behavior-level meta-correction, and selective workflows.

7. `reports/GPTPro_Response_Visual_Cascade_Audit_20260712.md`
   - GPT Pro's returned independent code/provenance audit and exact A1/B1 plan.

8. `reports/GPTPro_Response_Verification_and_Experiment_Decision_20260712.md`
   - Local line-by-line verification of the GPT Pro findings.
   - Records the direct `difficulty` leakage, Candidate 41 output stacking, consumed `holdout234`, original-resolution audit, and the resulting experiment decision.

9. `scripts/run_task7_sequential_ab_tc_fallback_20260713.py`
   - Reproducible implementation of nested training-side AB misses, sequential AB/TC visual experts, dynamic six-subtype fallback sampling, and fixed image-only routing.

10. `scripts/analyze_task7_sequential_ab_tc_fallback_20260713.py`
    - Locked aggregate comparison, paired bootstrap, route accounting, fallback diagnosis, and advancement-gate decision.

11. `GPTPRO_PROMPT_20260712_POST_F1_F2.md`
    - Superseded post-F1/F2 prompt retained for provenance.

12. `reports/PHYSICIAN_ROI_ANNOTATION_LOCK_WORKFLOW_20260712.md`
    - Optional future localization-oracle workflow. It is not a current prerequisite and should not create a new annotation burden without a specific changed-information hypothesis.

13. `reports/G1_PHYSICIAN_ROI_ORACLE_PREREGISTRATION_20260712.md`
    - Optional fixed two-reader manual-ROI oracle retained for provenance.

14. `scripts/phase2_fresh_external_candidate_lock_20260711.csv`
   - Machine-readable lock for the two candidates allowed into a new independent external blind test.
   - Includes fixed metrics, thresholds, member definitions, and prediction hashes.

15. `reports/FRESH_EXTERNAL_BLIND_TEST_PROTOCOL_20260711.md`
   - Exact cohort, image-selection, blinding, hashing, reporting, and interpretation rules for the next external test.

16. `reports/AI Pathology Model Improvement.md`
   - The broad research-lead plan that motivated this experiment wave.

17. `scripts/`
   - Complete implementation set: registries, dense-token extraction, LoRA, contrastive learning, structured pooling, SAM optimization, fusion search, sequential specialists, bootstrap, nested thresholds, error analysis, and queue/recovery scripts.

## Data Boundary

- Internal development: 591 cases.
  - batch1: 117.
  - batch2: 168.
  - third_batch: 306.
- Historical strict external: 108 cases.
- Newer external: 162 deduplicated cases.

The 108-case and 162-case external cohorts were both inspected in Phase 1. They are now consumed audit sets. Phase 2 did not use them for training, selection, thresholding, fusion, or evaluation. Any Phase-2 generalization claim requires a genuinely fresh external cohort.

## Historical Capability Correction

The old-data 92% result is not evidence that a visual classifier reached 92%. The stronger post-response audit also found direct label-derived leakage:

- Early image-grounded visual baseline: approximately 0.765 Acc/BAcc on 285 old-data cases.
- Candidate 41 selected corrected-system output stack: 0.8351 Acc and 0.8348 BAcc.
- No.64 exploratory behavior reviewer with a label-derived router feature: 0.9263 Acc/BAcc.
- Later `base162`, which consumed the nominal third-batch holdout during meta-stacking: 0.9228 Acc and 0.9227 BAcc on old data.

Code audit showed that the winning No.64 corrector used `feature_set=model`, not DINO/image features. Its winning `model_visible` router also included `difficulty`, which was constructed from model correctness and true-class probabilities and therefore requires the true label. Candidate 41 itself consumes predictions, probabilities, confidence, and disagreement from four previously selected corrected systems rather than rereading images. `base162` inherited this lineage and reassigned folds after combining adapt72 with the nominal holdout234. These systems emit full-coverage outputs, but their magnitudes are not leakage-clean prospective estimates and their incremental gain is not demonstrated fine visual morphology.

This distinction matters because the gain did not transfer: `base162` reached only 0.7300 BAcc and 0.5263 high-risk recall on the third-batch strict holdout, and 0.6220 BAcc on the historical strict external evaluable cohort.

The project therefore permits multi-model systems only when visual models learn complementary image evidence. Confidence and disagreement may route a case to a specialist, but they are not credited as image-grounded capability.

## 2026-07-12 A1/B1 Execution Result

GPT Pro's first image-grounded plans were implemented and run end to end. A1 used original-resolution 50%/25% specimen tiles with frozen SigLIP-L dense tokens. Tiles below 60% tissue coverage were masked. B1 used locked C1, a fixed outer-training 40th-percentile margin router, image-only M2, behavior-only M1, inner-crossfit M3, and 1,000 matched-random routes.

| Fixed family | Five-fold BAcc | Source-LODO BAcc | Decision |
| --- | ---: | ---: | --- |
| Locked C1 | 0.7477 | 0.7397 | Reference |
| A1 native hierarchical MIL | 0.7435 | 0.6897 | No-go |
| A1 native cross-attention | 0.6899 | 0.6755 | No-go |
| B1 image-only M2 cascade | 0.7537 | 0.6993 | No-go |
| B1 nested M3 cascade | 0.7599 | 0.7187 | No-go |

On the same routed cases, M2 minus C1 was +0.0165 BAcc in five-fold OOF, 95% CI [-0.0658, 0.1002], but -0.0928 under source LODO, 95% CI [-0.1686, -0.0182]. Actual confidence routing underperformed matched random routing. M2 improved B1 while harming B2, including a -0.1236 B2 accuracy change under LODO. These families are closed; do not continue tile-count, loss, route-percentage, threshold, or fusion-weight searches on them.

An optional blinded manual-ROI oracle was engineered for a future localization-specific question. It is not the current mainline and should not be interpreted as a request for physicians to relabel cases that they have already characterized. No patient images, mappings, or annotations are stored in this repository.

## 2026-07-12 Wave C, D1, E1, F1, and F2

The follow-up wave tested ordinary stronger backbones, label-trained localization, fixed label-free anatomy localization, multi-random-bag consistency, and one fixed image-only ensemble. Every model retained 100% coverage and threshold 0.5.

| Family | Five-fold BAcc/AUC | Source-LODO BAcc/AUC | Decision |
| --- | ---: | ---: | --- |
| SwinV2-L six-view | 0.6909/0.7533 | 0.6098/0.6594 | No-go |
| ConvNeXtV2-L six-view | 0.7384/0.8184 | 0.6831/0.7520 | No-go |
| SigLIP-SO400M six-view | 0.7433/0.8227 | 0.7070/0.7593 | No-go |
| D1 label-trained attention ROI | 0.7655/0.8428 | 0.6788/0.7679 | No-go |
| E1 fixed label-free anatomy ROI | 0.7302/0.7800 | 0.7051/0.7759 | No-go |
| F1 three random bags plus consistency | 0.7450/0.8173 | 0.7357/0.8098 | No-go |
| F2 fixed 0.5 C1 + 0.5 F1 | 0.7401/0.8369 | 0.7443/0.8326 | No-go |

D1's apparent internal gain reversed under source LODO. E1 anatomy localization lost to its matched-random control. F1 improved LODO BAcc by 0.0232 and AUC by 0.0460 versus its matched base, but reduced LODO B1 accuracy by 0.1452. F2 significantly improved ranking versus C1 but did not improve fixed-threshold BAcc; it reduced third-batch BAcc by 0.0480 and harmed B2/B3.

These findings close further searches over backbone size, automatic ROI scores, random-bag count, consistency weight, fusion weight, and threshold on the same 591 single photographs. The recurring gain is ranking/AUC, while fixed-threshold decisions move errors among sources and B1/B2/B3.

## 2026-07-13 Sequential AB/TC Expert Result

The preregistered coarse-to-fine system used independent image-reading heads on the locked C1 six-view dense-token representation:

1. an AB-versus-rest expert;
2. a TC-versus-non-TC expert trained after removing AB, with training-side cross-fitted AB misses reintroduced;
3. a low-versus-high fallback trained on all A/B1/B2/B3 plus dynamically sampled AB/TC coverage cases;
4. fixed exclusive routing, with conflicts and neither-positive cases sent to the fallback.

| Model | Five-fold BAcc/AUC | Source-LODO BAcc/AUC | Sensitivity/Specificity (LODO) |
| --- | ---: | ---: | ---: |
| C1 | 0.7477/0.8240 | 0.7397/0.8072 | 0.7130/0.7663 |
| C2 | 0.7514/0.8377 | 0.7441/0.8108 | 0.7354/0.7527 |
| Sequential AB/TC/fallback | 0.7195/0.8118 | 0.7278/0.7895 | 0.6457/0.8098 |

The system is a formal no-go. Versus C2, LODO sensitivity fell by 0.0897 while specificity rose by 0.0571. The paired bootstrap confirmed significant sensitivity harm: delta -0.0882, 95% CI [-0.1435, -0.0359]. The routing component itself was conservative and slightly helpful, but too small: only 29 AB-exclusive and zero TC-exclusive LODO cases were routed. The fallback caused the main loss and again shifted errors from B1 toward B2/B3/TC.

Standalone AB discrimination collapsed from five-fold BAcc/AUC 0.7811/0.8505 to source-LODO 0.6178/0.6280. TC ranking remained useful at 0.8450 LODO AUC, but no held-source fold met the locked TC route-purity requirement. This closes seed repeats, 30/40/50 coverage-count sweeps, threshold searches, and confidence-gate tuning for this design.

## Optional Physician ROI Infrastructure

The prepared 120-case G1 manual-ROI oracle has an end-to-end executable firewall, but it is archived as optional infrastructure rather than an active physician request:

- two separate 120-row reader forms;
- readers remain blinded to labels, subtype, C1 probability, and C1 correctness;
- strict categorical, coordinate, conditional-field, and completeness validation;
- SHA256 annotation locking before secure-label-key access;
- top-ROI IoU agreement and blinded physician risk-judgment audit;
- each reader's ROI1 at exact and 1.5x context scales;
- matched-random boxes with equal scale and tissue coverage;
- five-fold and source-LODO direct-image heads;
- same-case C1, base-only, random-ROI, B1/B2, source, and paired-bootstrap comparisons.

The full synthetic engineering smoke passed. Synthetic annotations were deleted and are not scientific evidence. Real G1 should start only if a later research question specifically requires a blinded localization oracle and the expected information gain justifies new physician work.

## Main Positive Result

The only clear representation-level gain in this broad wave was:

`SigLIP-L @ 512 px + six deterministic views + all dense patch tokens + low-capacity gated pooling`

The six views are whole image, foreground crop, and four crop quadrants from the same primary image.

Locked single-model candidate C1:

- Five-fold OOF: BAcc 0.7477, AUC 0.8240, sensitivity 0.7399, specificity 0.7554.
- Source-LODO: BAcc 0.7397, AUC 0.8072, sensitivity 0.7130, specificity 0.7663.
- LODO source BAcc: 0.7145 / 0.7024 / 0.7329.
- Versus the prior two-view SigLIP-L@512 model, OOF BAcc improved by 0.0460; paired bootstrap 95% CI [0.0096, 0.0817]. LODO BAcc was unchanged within uncertainty.

Locked two-model candidate C2:

`AIMv2 MixStyle run 253 + C1, equal probability average`

- Five-fold OOF: BAcc 0.7514, AUC 0.8377.
- Source-LODO: BAcc 0.7441, AUC 0.8108.
- LODO minimum-source BAcc: 0.7083.
- LODO B1/B2 risk accuracy: 0.5000 / 0.6629.

C2's LODO AUC exceeded the prior locked `215+253+254` fusion by 0.0228, paired bootstrap 95% CI [0.0031, 0.0426]. Its BAcc difference remained inconclusive. This is an internal candidate, not an external breakthrough.

## Why the Highest Internal Fusion Was Not Locked

An exhaustive three-member search favored `207 + 269 + 347`, and an expanded five-member search produced superficially high internal results. Meta-level five-fold cross-selection chose different companion recipes across held-out folds and substantially reduced LODO performance. The new six-view run 347 appeared in every held-out winner, but the companion models were unstable.

The lock therefore keeps C1 and the simple C2 pair. It rejects a more complex recipe selected from thousands of internal combinations.

## Important Remaining Failure

The B1/B2 boundary remains weak:

- C1 LODO B1/B2 risk accuracy: 0.4194 / 0.6292.
- C2 LODO B1/B2 risk accuracy: 0.5000 / 0.6629.

Light boundary auxiliary loss, a conservative boundary expert, subtype auxiliary loss, subtype-aware contrastive learning, and B1/B2 hard negatives all shifted the error tradeoff without improving both boundary classes together.

Doctor-concept analysis is explanatory only. After Fisher tests and within-comparison BH correction:

- C1 OOF low-risk false positives enriched `texture_soft` (q=0.0012).
- C1 LODO high-risk false negatives enriched `nodular_lobulated` (q=0.0498).

These associations are explanatory hypotheses, not established biological mechanisms, and were not used as model inputs. The existing physician feature table covers 589 of 591 internal cases and is sufficient for the current retrospective error audit; no new 120-case physician exercise is required for that audit.

## Broad Negative Evidence

The completed wave includes negative or insufficient results from:

- DINOv2, DINOv3-B/L/qkvb, EVA02, ConvNeXt DINOv3, ViTamin-L, SigLIP-B/L/SO400M, and pooled-token controls.
- Conservative and expanded LoRA on AIMv2 and SigLIP at 384/512.
- VICReg initialization, supervised contrastive learning, cross-source positives, subtype positives, and B1/B2 hard negatives.
- DANN, GroupDRO, REx, class-conditional alignment, MixStyle variants, and SAM optimization.
- Mean, view-gated, statistics, and fixed spatial-pyramid aggregation.
- Six-class, ordinal, concept, prototype, boundary-expert, high-risk sentinel, and MoE variants.
- Automatic contrast/white-balance/unsharp preprocessing and multiple ROI policies.

The report gives exact metrics and failure modes. Do not propose a low-level repeat without explaining what structural limitation changes.

## Next Valid Steps

1. Freeze new classifier nomination on the current 591 cases and preserve the
   completed experiment ledger.
2. Start the four-slot prospective capture protocol at the development
   hospitals: whole specimen, complete cut surface, geometric-rule solid-region
   close-up, and capsule/interface close-up.
3. Keep at least two external-test hospitals fully sealed from supervised
   training, SSL, normalization fitting, model selection, and thresholding.
4. Monitor capture completeness and acquisition metadata without opening model
   selection before the protocol gates are met.
5. Reopen model development only on the new multicenter package, with a paired
   primary-single-view versus all-four-view comparison.

The current evidence supports a stronger mixed-source base representation and a
cleaner candidate lock. It does not support a 92% base-visual claim, robust
cross-batch generalization, or the claim that cross-hospital generalization is
solved.
