# 2026-07-11 Task7 Base-Model Capability Update

Interpretation corrected: 2026-07-12

Repository: https://github.com/ScottBlizzard/tarmac

This folder is the latest handoff for the project's primary scientific objective: improving the 100%-coverage, image-only forced classifier for thymic gross pathology Task7.

Task7 definition:

- Low risk: A / AB / B1.
- High risk: B2 / B3 / thymic carcinoma.

Selective release, rejection, and physician-review workflows remain downstream safety layers. They are not substitutes for a stronger base classifier and are not the optimization target in this update.

## Read First

1. `GPTPRO_PROMPT_20260712_POST_F1_F2.md`
   - Current English follow-up prompt with the repository URL, all Wave C/D1/E1/F1/F2 results, and the physician-ROI decision branch.

2. `reports/Task7_WaveC_D1_E1_F1_F2_Results_20260712.md`
   - Exact metrics and predeclared decisions for the latest direct-image and fixed visual-ensemble experiments.

3. `reports/PHYSICIAN_ROI_ANNOTATION_LOCK_WORKFLOW_20260712.md`
   - Independent reader forms, no-label validation, cryptographic annotation lock, and G1 execution boundary.

4. `reports/G1_PHYSICIAN_ROI_ORACLE_PREREGISTRATION_20260712.md`
   - Fixed two-reader manual-ROI model, matched-random control, agreement rule, and advancement gate.

5. `reports/Task7_GPTPro_Plan_Execution_A1_B1_Results_20260712.md`
   - Completed A1 native-detail direct-model and B1 fixed-route M0-M4 results.
   - Both fixed-grid families failed their predeclared OOF/source-LODO advancement gates.

6. `reports/PHYSICIAN_BLINDED_ROI_ORACLE_PROTOCOL_20260712.md`
   - Next changed-information experiment: a 120-case, two-reader, label/model-blinded manual-ROI oracle.

7. `reports/GPTPro_Response_Visual_Cascade_Audit_20260712.md`
   - GPT Pro's returned independent code/provenance audit and exact A1/B1 plan.

8. `reports/GPTPro_Response_Verification_and_Experiment_Decision_20260712.md`
   - Local line-by-line verification of the GPT Pro findings.
   - Records the direct `difficulty` leakage, Candidate 41 output stacking, consumed `holdout234`, original-resolution audit, and the resulting experiment decision.

9. `GPTPRO_PROMPT_20260712_VISUAL_CASCADE_AUDIT.md`
   - Superseded visual-cascade audit prompt retained for provenance.
   - Audits the historical 92% result and redesigns the mainline around image-grounded capability.

10. `reports/Task7_Visual_Capability_and_Genuine_Coarse_to_Fine_Reframing_20260712.md`
   - Corrects the distinction between a visual model, an image-grounded ensemble, a behavior-level meta-corrector, and a selective workflow.
   - Includes the post-audit leakage and holdout-consumption erratum.
   - Defines the direct-model and genuine coarse-to-fine experimental plan.

11. `scripts/run_task7_native_detail_a1_20260712.py`
   - RAM-only native-resolution A1 implementation.
   - Fixed C1-view aggregation control plus 14-view hierarchical MIL and global-to-local cross-attention families.

12. `GPTPRO_PROMPT_20260711_POST_EXPERIMENT.md`
   - Superseded post-experiment prompt retained for provenance.

13. `reports/Task7_Base_Model_Capability_Experiments_20260711.md`
   - Full internal experiment ledger from runs 206-369.
   - Canonical five-fold OOF and three-source LODO protocols.
   - Positive, negative, fusion, bootstrap, doctor-concept, and resource-cleanup results.
   - Final internal candidate lock and its limitations.

14. `scripts/phase2_fresh_external_candidate_lock_20260711.csv`
   - Machine-readable lock for the two candidates allowed into a new independent external blind test.
   - Includes fixed metrics, thresholds, member definitions, and prediction hashes.

15. `reports/FRESH_EXTERNAL_BLIND_TEST_PROTOCOL_20260711.md`
   - Exact cohort, image-selection, blinding, hashing, reporting, and interpretation rules for the next external test.

16. `reports/AI Pathology Model Improvement.md`
   - The broad research-lead plan that motivated this experiment wave.

17. `scripts/`
   - Complete local 2026-07-11 implementation set: registries, dense-token extraction, LoRA, contrastive learning, structured pooling, SAM optimization, fusion search, bootstrap, nested thresholds, error analysis, and queue/recovery scripts.

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

The next valid localization test is the blinded manual-ROI oracle. The server packet contains 120 neutral-filename images and 240 independent-reader rows, with the secure label/model key physically separated. No patient images, mappings, or annotations are stored in this repository.

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

## Physician ROI Execution Readiness

The 120-case G1 manual-ROI oracle now has an end-to-end executable firewall:

- two separate 120-row reader forms;
- readers remain blinded to labels, subtype, C1 probability, and C1 correctness;
- strict categorical, coordinate, conditional-field, and completeness validation;
- SHA256 annotation locking before secure-label-key access;
- top-ROI IoU agreement and blinded physician risk-judgment audit;
- each reader's ROI1 at exact and 1.5x context scales;
- matched-random boxes with equal scale and tissue coverage;
- five-fold and source-LODO direct-image heads;
- same-case C1, base-only, random-ROI, B1/B2, source, and paired-bootstrap comparisons.

The full synthetic engineering smoke passed. Synthetic annotations were deleted and are not scientific evidence. Real G1 cannot start until both physicians finish and lock their independent forms.

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

These associations require physician image review and new data. They are not established biological mechanisms and were not used as model inputs.

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

1. Complete the two-reader manual-ROI oracle under `PHYSICIAN_BLINDED_ROI_ORACLE_PROTOCOL_20260712.md`. Train an automatic anatomical ROI detector only if the manual oracle passes all predeclared gates.
2. If the manual ROI oracle fails, stop optimization of the current single photograph and collect standardized cut-surface close-ups, capsule/interface views, and genuinely new multicenter cases.
3. If a fresh label-blinded cohort becomes available, run the already locked C1/C2 comparison once under `FRESH_EXTERNAL_BLIND_TEST_PROTOCOL_20260711.md`.

The current evidence supports a stronger internal representation and a cleaner candidate lock. It does not support a 92% base-visual claim or the claim that cross-hospital generalization is solved.
