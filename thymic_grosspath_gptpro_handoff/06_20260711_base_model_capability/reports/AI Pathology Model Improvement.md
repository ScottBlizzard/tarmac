# Historical Research-Lead Plan (Pre-Experiment)

Status note added 2026-07-11: this document was written before the completed runs 206-369. Its broad method analysis motivated the experiment wave, but its external-data status is historical. Both `strict_external` 108 and `new_external_160`/162 have now been inspected and are consumed audit sets. Use `Task7_Base_Model_Capability_Experiments_20260711.md` and `FRESH_EXTERNAL_BLIND_TEST_PROTOCOL_20260711.md` for the current evidence and data boundary.

## 1. Real Bottleneck Diagnosis

My judgment is that the limitation is **not one thing**. It is a compound bottleneck: **domain shift + ROI/view heterogeneity + high-risk boundary ambiguity + insufficient external-domain diversity + underused dense visual information**. Model capacity matters, but the evidence does not support a simple “just train a bigger DINO” conclusion.

The package itself frames the handoff as a text-first method-design package rather than a directly trainable image/model bundle; raw medical images, checkpoints, large ZIPs, and full prediction tables are intentionally excluded where possible. So the analysis below is an audit of the documented evidence, scripts, result summaries, and research trajectory, not a direct visual re-review of the images.

The most important empirical pattern is this: the current system has strong **in-domain or adapted-domain signal**, but the signal does not transfer reliably to frozen external domains. The old-data model reaches about 92.3% accuracy/balanced accuracy; third-batch adaptation reaches about 83.0% accuracy and about 76.8% balanced accuracy but remains weak on high-risk recall; strict external forced classification is only about 64.8% accuracy, 62.8% balanced accuracy, and about 46.8% high-risk recall. The new_external_160 set has been QC-processed but had not yet undergone formal inference at packaging time.

The strict external confusion structure is clinically decisive. On all 108 strict external cases, the forced classifier reports Acc 0.6481, BAcc 0.6275, AUC 0.6083, high-risk recall 0.4681, low-risk specificity 0.7869, and FN/FP = 25/13. On the 105 strict-evaluable subset, the same high-risk recall 0.4681 persists with FN/FP = 25/13. That means the base model is not just “a bit uncalibrated”; it is missing roughly half of high-risk cases in the external domain.

I would diagnose the bottleneck as follows.

**First, domain shift is the dominant external failure driver.** The old and third-batch results show learnable signal, but strict external introduces changes in imaging system, background, specimen scale, visible tumor fraction, cut/outer/mixed presentation, and possibly case composition. The reports explicitly state that the external drop is not a single threshold problem because both high-risk false negatives and low-risk false positives occur.

**Second, ROI/view heterogeneity is a real input bottleneck, but previous ROI attempts were too shallow to close it.** Cut-only, whole-only, whole+crop, full-to-cut refinement, and view specialists produced partial AUC signals but did not stabilize forced classification. That should not be interpreted as “ROI does not matter.” It should be interpreted as “manual/coarse/static cut selection and lightweight routing were insufficient.” The previous view labels covered only a small part of the full development set, and cut/mixed cases both remained difficult.

**Third, the low/high label boundary is biologically and visually ambiguous in exactly the clinically dangerous region.** The hard-case analysis found 151 easy, 35 medium, 34 salvage-hard, and 65 core-hard cases in old_data. After curriculum learning, easy+medium reached 96.2% accuracy and non-core-hard reached 91.4%, but core-hard stayed at only 26.15%. This strongly suggests that the global score is being dragged down by a concentrated set of boundary/underspecified cases rather than uniform model failure.

**Fourth, high-risk recall is the scientific bottleneck, not overall accuracy.** Third_batch has many low-risk cases; after adaptation, accuracy can look acceptable while high-risk recall remains weak. In the third-batch holdout result, accuracy was 0.8718 but high-risk recall was only 0.5263. Strict external is worse: high-risk recall 0.4681. Any plan that increases Acc by protecting low-risk specificity while leaving FN high is not solving the clinical problem.

**Fifth, the current representation may be underusing dense spatial information.** Many prior experiments used frozen global features, CLS-style representations, logistic/MLP probes, or simple whole/crop concatenation. The negative results do not prove that modern foundation models cannot solve the task; they mostly prove that ordinary backbone replacement, shallow probing, and unstable small-data fine-tuning are inadequate. Dense token MIL, segmentation-guided ROI extraction, multi-image bags, concept supervision, and domain-invariant training remain underexplored relative to their importance.

My bottom-line diagnosis: **the base classifier needs to become a domain-generalized, ROI-aware, multi-instance, high-risk-recall-oriented model.** Rejection/review workflows are useful safety wrappers, but they are not a substitute for improving the full-coverage forced decision function.

---

## 2. Historical Experiment Review and Potentially Misleading Conclusions

I agree with the following historical conclusions.

**The data boundaries are now mostly correct.** old_data is internal development/OOF; third_batch has already been used for observation, adaptation, and strategy selection; strict_external is a frozen stress test; new_external_160 should be treated as the next independent frozen external set. The package repeatedly warns that third_batch must not be called strict external and that selective workflow results must not be reported as full automatic diagnosis.

**Forced classification and selective diagnosis must remain separate.** v195/v195+ can reduce automatic errors by sending difficult cases to review/rejection, but those results do not mean the base model is strong at full coverage. For example, v195 achieves excellent automatic-subset metrics on strict external only while auto-releasing 52/108 cases and reviewing/rejecting 56/108 cases; Phase2 increases strict external auto-release to 57/108 without observed automatic errors, but it still does not retrain or improve the classifier itself.

**Simple input manipulation did not solve the external problem.** Whole/crop variants, cut-only variants, background normalization, image statistics, and TTA produced local signals but did not reliably improve full-coverage external generalization. This supports the conclusion that the problem is not merely color balance or crop selection.

**Ordinary backbone swaps were mostly negative.** ConvNeXt, Swin, EfficientNet, BiomedCLIP, PLIP, SigLIP, EVA02, ViTamin, and similar branches were tried in various forms. The evidence supports “ordinary backbone swapping did not solve this,” especially when evaluated by Acc/BAcc/FN/FP rather than only AUC.

**Blindly adding hard cases or weak labels can damage the decision boundary.** The 204 weak+strong label expansion interfered with easy/medium learning, and adding core-hard cases—even small subsets—disturbed fold performance. The historical warning not to dump all hard cases into training is valid.

**Six-class-to-binary merging is not currently a reliable main route.** Direct Task7 binary training has been more stable than first training six WHO classes and then merging, partly because B3 and other boundary classes are underpowered. However, six-class supervision remains useful as an auxiliary/ordinal task.

Several conclusions may be **premature or misleading** if taken too literally.

**“DINO is the only worthwhile backbone” is too narrow.** The evidence says ordinary single-backbone replacement and shallow probing did not beat the DINO baseline. It does not rule out modern dense-token foundation modeling, adapter/LoRA tuning, multi-scale MIL, or ROI-aware use of DINOv2/DINOv3/SigLIP/EVA/SAM-derived features. The failed experiments were often global-feature or small-data-finetune experiments, not full foundation-model redesigns.

**“Cut-only failed” should not close the ROI direction.** The cut specialist had strong fold-level signals but poor 5-fold stability; full-to-cut refinement improved AUC but not classification stability. That argues for better ROI modeling—SAM/MedSAM/YOLO/DETR-assisted foreground, tumor/cut-surface masks, tile MIL, and whole+ROI consistency—not abandoning ROI.

**“View-aware routing failed” does not prove view information is useless.** The v2 diagnostic report notes that view labels covered only 58/591 development cases, which is too sparse for a definitive failure of view-aware modeling. Prior routing/specialist attempts may have failed because labels were incomplete and routing was hard/discrete. A softer conditioning approach may work better.

**“Experience labels did not help” should be narrowed to “current noisy/limited labels did not stably help.”** The 31 high-confidence core concept labels produced a small signal, but it was unstable under finer grid checking. That suggests the concept direction is alive, but needs physician-confirmed, better-distributed, positive/negative paired labels and should be used as auxiliary supervision rather than a fragile late fusion trick.

**“External failure means gross images are insufficient” is not yet proven.** It may be true for some core-hard cases, but the current evidence cannot separate intrinsic visual insufficiency from poor ROI selection, domain shortcuts, missing multi-image context, and underpowered high-risk examples. Physician adjudication is needed before declaring cases visually impossible.

**“Strict external has already been learned from” must be handled carefully.** Even if it was not used for training in the formal reports, repeated human inspection of strict_external results can influence future method design. For the next round, the cleanest confirmatory evidence should come from new_external_160 after a locked protocol.

---

## 3. Broad Method Space: 20+ Candidate Experiments

The goal of this matrix is not to repeat low-level DINO+probe sweeps. Each candidate changes at least one of the following: input representation, ROI construction, case aggregation, domain generalization objective, supervision structure, or evaluation discipline.

|  # | Candidate direction                                              | Core idea                                                                                                                                               | Why it may help external generalization                                                                | Why this is not just a repeat                                                                      | Required data/code                                                              | Main risks                                                                    | Success criteria                                                                                                              |
| -: | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
|  1 | **SAM/MedSAM/YOLO-assisted ROI + whole/ROI fusion**              | Segment specimen/tumor foreground and extract whole image, foreground crop, cut-surface crop, and boundary crop. Fuse with shared classifier.           | Reduces background, board, hand/tool, and scale shortcuts; preserves global context.                   | Prior crops were coarse/static; this uses learned/dense ROI extraction and multi-ROI fusion.       | New ROI mask script; 50–100 physician-QC boxes/masks; existing case registries. | SAM may segment non-tumor tissue; bad masks can amplify errors.               | Old BAcc stays ≥0.90, third high-risk recall improves ≥8–10 pp, strict/new forced BAcc improves without specificity collapse. |
|  2 | **Dense-token MIL over multi-scale patches**                     | Extract patch tokens from DINOv2/DINOv3/SigLIP/EVA/ViT; use attention MIL over 224/384/512/768 crops.                                                   | External domains may shift global appearance while local morphology remains informative.               | Prior probes mostly used global or simple concatenated features.                                   | Token extraction cache; MIL trainer; ROI/foreground boxes optional.             | Overfitting small n; attention may attend to artifacts.                       | Better pseudo-domain leave-out BAcc and high-risk recall than global probe.                                                   |
|  3 | **All-image case-level MIL**                                     | Use every available image per case as a bag; aggregate across images with attention/SetTransformer.                                                     | Solves single selected-image heuristic and captures complementary cut/outer views.                     | Prior multi-image policies used simple image selection/averaging.                                  | Case-bag registry; split-safe multi-image loader.                               | Old/third may have uneven image counts; image availability differs by domain. | Improves cases with multi-image ambiguity and does not degrade single-image cases.                                            |
|  4 | **Domain-randomized training**                                   | Heavy randomized camera/style augmentation: white balance, exposure, blur, JPEG, blue-board/background replacement, scale, crop, rotation constraints.  | Forces invariance to capture setup and background.                                                     | Not one-off normalization or TTA; it is train-time domain simulation with consistency loss.        | Augmentation module in `thymic_baseline`; foreground masks preferred.           | Over-randomization may erase pathology cues.                                  | Better leave-domain-out third performance and strict/new robustness.                                                          |
|  5 | **Whole–ROI consistency learning**                               | Train whole and ROI views to agree under augmentations while allowing view-specific heads.                                                              | Prevents ROI branch from overfitting local artifacts and whole branch from using background shortcuts. | Prior whole+crop fusion did not enforce representation consistency.                                | ROI crops; paired-view dataloader; consistency loss.                            | May force agreement when views genuinely differ.                              | Higher AUC and high-risk recall on pseudo-external folds, stable threshold.                                                   |
|  6 | **Physician concept bottleneck / auxiliary heads**               | Add labels for hemorrhage, necrosis/cystic change, capsule, boundary, lobulation, uniform pale cut, subject adequacy, view type.                        | Teaches model which visual cues are confounders vs risk-relevant.                                      | Prior weak labels were noisy and sparse; this requires physician-confirmed compact labels.         | 80–150 concept-labeled cases; multitask trainer.                                | Concept labels may be subjective; too many concepts dilute signal.            | Concept prediction is learnable and main Task7 improves on hard/third subsets.                                                |
|  7 | **Ordinal + binary multitask risk modeling**                     | Train binary low/high plus ordinal subtype-risk head: A/AB/B1/B2/B3/TC ordered or partially ordered.                                                    | Keeps subtype boundary information without relying on unstable six-class final prediction.             | Not six-class-then-merge; binary remains primary, subtype is auxiliary.                            | Existing subtype labels; ordinal loss module.                                   | Wrong ordering assumptions; B3 sample scarcity.                               | Improves B2/B3/TC recall while old specificity remains acceptable.                                                            |
|  8 | **High-risk FN-specialized branch**                              | Train a one-vs-low high-risk sentinel focused on B2/B3/TC false negatives, then fuse into base model by learned internal validation.                    | Directly attacks clinically dangerous FN pattern.                                                      | Prior guard/review models selected reliability; this branch is part of forced classifier training. | FN lists; class-balanced sampler; asymmetric/focal/Tversky loss.                | May increase low-risk FP too much.                                            | High-risk recall improves ≥10 pp with low-risk specificity not below prespecified floor.                                      |
|  9 | **B2-specific and B3/TC-specialized contrastive learning**       | Supervised contrastive loss comparing B2/B3/TC against B1/AB mimics.                                                                                    | External diagnostics show B2 is especially error-prone.                                                | More targeted than generic class weighting.                                                        | Subtype labels; hard-pair mining.                                               | Small B3 and TC counts; pair bias.                                            | B2 recall improves in old+third pseudo-domains and strict/new.                                                                |
| 10 | **Group DRO / IRM / REx domain-invariant training**              | Define pseudo-domains: batch1, batch2, third adapt, third holdout, view/quality/scale strata; optimize worst-domain performance.                        | Selects models that do not only fit old_data style.                                                    | Prior domain sampling was lighter and not necessarily worst-domain optimized.                      | Domain labels; group-DRO trainer; pseudo-domain evaluator.                      | Pseudo-domains may not match true external shift.                             | Better minimum-domain BAcc and high-risk recall; smaller old→third gap.                                                       |
| 11 | **Adversarial nuisance removal**                                 | Predict domain/background/scale/brightness in adversarial heads; remove nuisance information from representation.                                       | Reduces shortcuts linked to capture conditions.                                                        | Different from background normalization; learns invariance in feature space.                       | Image statistics; domain labels; gradient reversal.                             | Can remove real morphology correlated with risk.                              | Lower domain predictability with maintained Task7 performance.                                                                |
| 12 | **Self-supervised adaptation on unlabeled images**               | Continue MAE/DINO/iBOT-style pretraining on old+third plus unlabeled strict/new images, without labels; then train classifier only on old+third labels. | Learns gross-pathology image distribution and target image statistics without label leakage.           | Prior fine-tuning was supervised small-data; this is unsupervised representation adaptation.       | All unlabeled images; SSL script; locked target-label firewall.                 | Target-label leakage by later selection; catastrophic drift.                  | Improves pseudo-target adaptation; external inference run only after lock.                                                    |
| 13 | **Test-time adaptation pre-registered on pseudo-targets**        | TENT/CoTTA/SHOT-style entropy/BN adaptation using target images only, no labels.                                                                        | Could adapt to camera/style shift at deployment.                                                       | Different from training on external labels; algorithm locked on old↔third simulations.             | TTA script; no-label batches.                                                   | High leakage risk; may hurt calibration and safety.                           | Must pass old→third and third→old simulations before strict/new use.                                                          |
| 14 | **Prototype/retrieval classifier**                               | Build subtype/risk prototypes from ROI tokens; classify by nearest robust prototypes with shrinkage.                                                    | Less threshold-dependent; can expose external cases far from known prototypes.                         | Not logistic/MLP probe; uses retrieval geometry and prototype robustness.                          | Token bank; prototype selection; retrieval visualization.                       | Prototype bias; poor with heterogeneous TC.                                   | Better interpretability and stable high-risk recall on pseudo-domains.                                                        |
| 15 | **Hybrid handcrafted gross morphology + deep tokens**            | Fuse segmented area ratio, color heterogeneity, red tissue ratio, boundary irregularity, foreground scale, blur, JPEG, with VFM tokens.                 | Captures domain/quality/morphology explicitly; may help model avoid shortcuts.                         | Prior image stats were appended lightly; this is structured hybrid with ROI-derived morphology.    | ROI masks; feature extractor; calibrated fusion.                                | Handcrafted features may be domain artifacts.                                 | Improves hard-error subgroup performance without reducing AUC.                                                                |
| 16 | **Quality/view-conditioned single model**                        | Add view/quality embedding tokens to a shared classifier rather than routing to specialists.                                                            | Handles cut/outer/mixed without brittle hard routes.                                                   | Previous specialists/routing failed; conditioning is soft and shares data.                         | View/quality labels; missing-label handling.                                    | Bad labels may add noise.                                                     | Better view-stratified BAcc and less cut/mixed instability.                                                                   |
| 17 | **Foreground-background shortcut audit and erasure**             | Train background-only, border-only, and subject-masked controls; penalize features that classify too well from background.                              | Quantifies and suppresses non-pathology shortcuts.                                                     | Prior background normalization did not establish or penalize shortcuts.                            | Mask generator; shortcut control scripts.                                       | Could overcorrect if background correlates with legitimate site variation.    | Background-only AUC falls while full model improves cross-domain.                                                             |
| 18 | **Synthetic style augmentation / diffusion background transfer** | Generate style-varied versions preserving specimen morphology: backgrounds, lighting, camera noise, scale.                                              | Increases domain diversity without changing labels.                                                    | More targeted than simple color jitter; physician-QC style changes.                                | Diffusion/style pipeline; QC; no pathology alteration.                          | Synthetic artifacts; label-invalid transformations.                           | Improves pseudo-domain robustness; no degradation on old OOF.                                                                 |
| 19 | **High-resolution tile pyramid model**                           | Use 768–1024px ROI/tile pyramid around cut surface; MIL across high-res tiles.                                                                          | Gross risk cues may be local texture/boundary patterns lost at 224–352.                                | Prior 392/whole/crop not true high-res tile pyramid.                                               | High-res image access; tile extractor; MIL.                                     | GPU cost; noisy tile labels.                                                  | Better salvage-hard and external high-risk recall.                                                                            |
| 20 | **External-domain simulation model selection**                   | Select models only by leave-pseudo-domain-out old+third robustness, not average old score.                                                              | Avoids choosing models that overfit old_data and fail external.                                        | Prior selection often optimized old+third point metrics; this formalizes external simulation.      | Standard evaluator; pseudo-domain registry.                                     | Pseudo-domains imperfect.                                                     | Candidate selected by this protocol beats previous selected model on strict/new.                                              |
| 21 | **Active-learning physician loop**                               | Select high-influence, high-disagreement, boundary cases for ROI/concept/view adjudication.                                                             | Adds labels where generalization fails, not just more easy cases.                                      | More targeted than large weak-label expansion.                                                     | Case-selection script; physician review UI/CSV.                                 | Review burden; confirmation bias from model-selected cases.                   | New annotations improve next-cycle pseudo-domain robustness.                                                                  |
| 22 | **Data governance and label-boundary model**                     | Add “borderline/not visually resolvable/task-boundary” labels; train with soft labels or exclude from model-selection target.                           | Separates true visual ambiguity from model failure.                                                    | Prior clean cohort exists for exclusions but not systematic soft-boundary learning.                | Physician adjudication; soft-label trainer.                                     | Could be misused to remove hard cases.                                        | Better calibrated uncertainty and clearer forced/full vs ambiguous analysis.                                                  |
| 23 | **VLM-assisted concept extraction, not diagnosis**               | Use vision-language models to draft structured descriptors; physicians verify; use descriptors as auxiliary labels.                                     | Accelerates concept labeling and may capture gross morphology language.                                | Not asking VLM to predict low/high; uses it for candidate concept annotation.                      | VLM pipeline; physician verification.                                           | VLM hallucination; privacy constraints.                                       | Reduces physician labeling time while preserving label quality.                                                               |
| 24 | **Small-data adapter/LoRA tuning of foundation models**          | Freeze most VFM; train adapters/LoRA with strong regularization, domain randomization, ordinal/concept heads.                                           | Uses capacity without catastrophic full fine-tuning.                                                   | Prior fine-tuning was head/last-block/full; adapter tuning is lower-rank and multi-objective.      | Adapter trainer; token cache; old+third folds.                                  | Still overfits; engineering cost.                                             | Beats frozen probe on pseudo-domain min-BAcc and locked external.                                                             |
| 25 | **Calibrated partial-AUC / recall-constrained optimization**     | Train/select by high-risk recall at low-risk specificity floor, not only global BAcc.                                                                   | Aligns objective with FN risk.                                                                         | Prior thresholds/fusions sometimes chased accuracy; this predefines clinical constraint.           | Standard metric script; internal validation only.                               | Can inflate FP if floor too low.                                              | HR recall gain with specificity floor met on old+third and tested externally.                                                 |
| 26 | **Public/preclinical gross-image pretraining**                   | Pretrain on non-thymic gross pathology/surgical specimen images if rights allow.                                                                        | Learns specimen/background/lighting invariances beyond small thymic set.                               | Not a backbone swap; domain-specific pretraining.                                                  | Data access; SSL pretraining.                                                   | Data rights; domain mismatch.                                                 | Improves all-domain robustness over ImageNet/self-supervised generic models.                                                  |

---

## 4. Highest-Priority 8-12 Experiments

I would run **12 priority experiments**, grouped by near-term value and breakthrough potential.

**1. ROI-guided multi-scale token-MIL.**
This is the highest-upside experiment. Build SAM/MedSAM/foreground masks, extract DINOv2/DINOv3/SigLIP/EVA patch tokens from whole image + foreground + cut-surface candidate crops, and train an attention MIL classifier. This directly addresses the strongest suspected failure mode: external images shift in background, scale, and framing while pathology-relevant regions may remain local.

**2. All-image case-bag MIL.**
The inherited pipeline often materialized one selected image per case, with multi-image cases using a fixed selection heuristic. That is too brittle for gross pathology. A bag model using all available images per case can learn when cut view, outer view, or mixed view is informative. The code audit confirms the historical frozen input used one image per case and batch-prefixed IDs, so the next model should explicitly move from selected-image classification to case-bag classification.

**3. External-domain simulation protocol.**
Before building more models, create a leave-pseudo-domain-out evaluator over old batch1, old batch2, third adapt/holdout, quality strata, scale strata, and view strata. Select models by worst-domain BAcc and high-risk recall, not by old_data average score. This is not glamorous, but it prevents repeating the historical pattern of selecting models that look good internally and fail strict external.

**4. Domain-randomized consistency training.**
Use aggressive but pathology-preserving camera/background/scale/blur/JPEG/white-balance augmentation, with whole/ROI consistency. This is the most practical direct attack on capture-system shift. It differs from previous color normalization/TTA because it trains invariance rather than applying one deterministic correction.

**5. Physician-confirmed concept bottleneck.**
Do not reuse the 204 weak labels as-is. Build a compact physician-confirmed concept set focused on the documented error patterns: hemorrhage/necrosis-like regions causing low→high FP, and pale homogeneous small clear-boundary high-risk cases causing high→low FN. The previous 31 high-confidence labels showed signal but lacked stability; a better-distributed concept set is a rational next step.

**6. Binary + ordinal/subtype multitask model.**
Keep Task7 binary as the main endpoint, but add an ordinal or auxiliary subtype head. This uses WHO information without letting scarce B3 or fine-grained boundaries dominate final prediction. It is not a repeat of failed six-class-then-merge.

**7. High-risk FN branch with constrained fusion.**
Train a high-risk sentinel specifically on B2/B3/TC vs confusable low-risk cases, with asymmetric/focal/Tversky loss and internal specificity constraints. Fuse it into the forced classifier only if it improves high-risk recall on old+third pseudo-domains without unacceptable FP.

**8. Dense-token foundation-model bank with adapter/LoRA tuning.**
Extract dense tokens from DINOv2, DINOv3, SigLIP-family, EVA/ViT, and pathology-oriented models where available. Train only adapters or small MIL heads, not full backbones. Previous ordinary backbone sweeps were mostly global or shallow; this is a representation and pooling redesign.

**9. Group DRO / domain-adversarial nuisance suppression.**
Use pseudo-domains and image-statistics nuisances to penalize domain-specific shortcuts. This is especially important because the strict external drop is compatible with background/scale/capture shortcuts.

**10. High-resolution tile pyramid.**
If raw image resolution is available in the working environment, build ROI tile pyramids at higher resolution. The model may be losing texture, capsule, boundary, and cut-surface details at standard 224–392 resolution.

**11. Self-supervised gross-image adaptation.**
Continue self-supervised pretraining on old+third and unlabeled external images without labels, then train/evaluate under a locked protocol. This is high-risk because of leakage discipline, but high-upside because the dataset is small and gross pathology is far from ImageNet.

**12. Active-learning annotation cycle.**
Use model disagreement, high influence, and pseudo-domain failure to select 50–100 cases for physician ROI/concept/view/boundary annotation. This is not just “get more labels”; it is targeted acquisition for the failure modes that currently cap external generalization.

The first wave should not include another broad ordinary-backbone leaderboard. It should include a **small number of structurally different models** evaluated under the same forced-classification protocol.

---

## 5. Strict Evaluation Protocol

### Dataset roles

Use the datasets as follows.

| Dataset            | Role in next round                         | Allowed use                                                                                                            |
| ------------------ | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `old_data`         | Internal development / OOF                 | Training, internal cross-validation, threshold selection within folds, model development                               |
| `third_batch`      | Adapted development / same-system new data | Training/adaptation/model selection allowed, but must be reported as development/adaptation, not strict external       |
| `strict_external`  | Frozen external stress test                | Final locked evaluation only; no training, threshold tuning, hyperparameter tuning, rule selection, or model selection |
| `new_external_160` | New independent frozen external set        | Best used as the clean confirmatory external test after protocol lock                                                  |

The package already warns that third_batch has participated in observation, adaptation, and strategy selection, while strict_external must not be used for training/tuning/model selection. That boundary should be made explicit in every table.

### Forced-classification metrics

For every full-coverage forced classifier, report:

| Metric                         | Definition                                                     |
| ------------------------------ | -------------------------------------------------------------- |
| Acc                            | `(TP + TN) / N`                                                |
| BAcc                           | `(high-risk recall + low-risk specificity) / 2`                |
| AUC                            | Case-level ROC AUC using high-risk probability                 |
| High-risk recall / sensitivity | `TP / (TP + FN)`                                               |
| Low-risk specificity           | `TN / (TN + FP)`                                               |
| FN                             | High-risk case predicted low-risk                              |
| FP                             | Low-risk case predicted high-risk                              |
| Confusion matrix               | `TN / FP / FN / TP`                                            |
| CI                             | Bootstrap or Wilson intervals, especially for external metrics |

For Task7, **FN is clinically more dangerous than FP**, so model selection should include a high-risk recall floor. A model with higher overall accuracy but persistent high-risk recall below 60% should not be considered a base-model breakthrough.

### Full coverage versus workflow reporting

Every result table must have two separate blocks:

1. **Full-coverage forced classification**: every case receives low/high prediction.
2. **Selective/rejection workflow**: report automatic coverage, review/rejection rate, automatic-subset Acc/BAcc/FN/FP, automatic errors, and Wilson95 upper bound.

Do not mix them. v195/v195+ results are downstream safety workflows, not evidence that the full-coverage base classifier is strong. The reports explicitly warn that “auto error = 0” does not mean all cases are correct, because many cases are reviewed/rejected.

### Patient, case, and multi-image leakage control

Use case/patient-level splits only. All images from the same case must stay in the same fold. If multiple images per case exist, they must be treated as one bag during splitting. No augmented variant, crop, mask, or tile from a case may cross train/val/test boundaries.

Before any experiment:

* compute perceptual hashes or file hashes for all images;
* check duplicate `case_id`, original filename, and source-prefixed ID;
* confirm old/third/strict/new case IDs cannot collide;
* document whether each case has one or multiple images;
* store an immutable registry checksum.

The code audit indicates that the existing baseline infrastructure already has useful duplicate/missing-image checks and case-level evaluation machinery, but the next round needs explicit multi-image bag leakage auditing.

### Threshold leakage control

Thresholds, class weights, fusion weights, calibration methods, model family choices, and stopping criteria must be selected only inside old_data and third_batch development procedures. strict_external and new_external_160 must not influence thresholds.

For external evaluation:

1. Save locked model config.
2. Save locked threshold.
3. Save registry checksum.
4. Run inference and export case-level probabilities.
5. Commit prediction CSV before reading external metrics.

If strict_external is inspected and then used to modify the model, it must be reclassified as “external feedback/development stress set,” and only new_external_160 can remain confirmatory.

### External-label leakage control

For strict_external and new_external_160:

* labels must not be loaded by training scripts;
* labels must not be used to select ROI method, augmentation, threshold, model family, or release rule;
* external subtype labels may be used only after predictions are committed for subgroup analysis;
* if unsupervised target adaptation is used, the algorithm must be pre-registered on old↔third pseudo-target simulations and must not use target labels.

### Model-selection criteria before external testing

Select candidates using old+third only. A reasonable internal selection rule:

* old_data BAcc ≥ 0.90 or no worse than −2 pp from current locked base;
* third_batch BAcc improves over current adapted model or high-risk recall improves by ≥8 pp;
* worst pseudo-domain BAcc improves;
* high-risk recall improves without low-risk specificity collapsing below a prespecified floor;
* no evidence of fold instability or catastrophic core-hard overfitting.

External success should be interpreted in tiers:

| Tier                      | External forced-classification interpretation                                 |
| ------------------------- | ----------------------------------------------------------------------------- |
| Minimal positive signal   | Strict/new BAcc +3–5 pp and high-risk recall +5–10 pp without FP explosion    |
| Meaningful improvement    | Strict/new BAcc ≥0.70 and high-risk recall ≥0.65                              |
| Strong breakthrough       | Strict/new BAcc ≥0.75 and high-risk recall ≥0.75 with specificity ≥0.70       |
| Clinically promising base | Reproduced on both strict_external and new_external_160 under locked protocol |

---

## 6. Three-Week Execution Plan

### Week 1: freeze data, build the new evaluation spine, and create ROI/token assets

**Main goal:** stop doing ad hoc comparisons and build the infrastructure needed for ROI-aware, multi-image, domain-generalized forced classification.

Run or create:

1. `scripts/prepare_task7_case_bag_registry_202607xx.py`
   Build unified registries for old_data, third_batch, strict_external, and new_external_160. Include all images per case, source, subtype, risk label, domain, image count, QC status, original filename, and dedup hash.

2. `scripts/audit_task7_leakage_202607xx.py`
   Check duplicate files, duplicate case IDs, same-patient multi-image leakage, fold membership, and external-label isolation.

3. `scripts/summarize_task7_forced_metrics_202607xx.py`
   Produce one standard forced-classification table with Acc, BAcc, AUC, high-risk recall, specificity, FN, FP, and confusion matrix for all historical baselines.

4. `scripts/build_task7_roi_masks_sam_202607xx.py`
   Produce specimen/foreground/tumor-candidate masks using SAM/MedSAM/SAM2/threshold+contour fallback. Save QC metrics: foreground area ratio, bbox ratio, blur, brightness, red-tissue ratio, border/background ratios.

5. `scripts/extract_task7_vfm_tokens_202607xx.py`
   Extract whole/ROI/multi-scale dense tokens from at least DINOv2, DINOv3, and one non-DINO modern VFM. Cache tokens per image and per ROI.

6. `scripts/build_task7_pseudodomains_202607xx.py`
   Define pseudo-domains from batch/source, third split, view/quality/scale/brightness/foreground strata.

Reuse first:

* the old/third/strict forced-classification table from the 2026-05-25 report;
* v2 diagnostic training lists: safe core, hard errors, and domain-focus cases;
* DINOv2/DINOv3/no64 meta-stack summaries;
* existing image statistics and background/crop preparation scripts;
* v195/v195+ only as downstream workflow baselines, not as base-classifier baselines.

Treat as historical baselines without rerunning unless reproducibility is needed:

* old/third/strict forced classifier from the 2026-05-25 table;
* curriculum Stage2+Stage3 old_data OOF;
* DINOv2 concat frozen probe baselines;
* DINOv3/no64 adaptation summaries;
* mean_core strict-external-sensitive branch;
* v195/v195+ selective workflows.

### Week 2: run structurally new base-model experiments on old+third only

**Main goal:** generate 6–8 credible candidate base classifiers without touching strict_external or new_external_160 labels.

Run these first:

1. `scripts/run_task7_token_mil_cv_202607xx.py`
   Attention MIL over dense tokens: whole, ROI, foreground, and multi-scale crops.

2. `scripts/run_task7_casebag_mil_cv_202607xx.py`
   All-image case-bag model. Compare single-image historical selection versus all-image bags.

3. `scripts/run_task7_domain_randomization_train_202607xx.py`
   Domain-randomized whole/ROI consistency model using camera/background/scale perturbations.

4. `scripts/run_task7_ordinal_multitask_cv_202607xx.py`
   Binary Task7 + ordinal/subtype auxiliary heads.

5. `scripts/run_task7_highrisk_sentinel_cv_202607xx.py`
   High-risk recall branch with constrained internal fusion.

6. `scripts/run_task7_concept_aux_multitask_202607xx.py`
   Start with existing high-confidence concept labels, then add physician-confirmed labels as they arrive.

7. `scripts/run_task7_groupdro_domain_cv_202607xx.py`
   Group DRO / worst-domain objective over pseudo-domains.

8. Optional high-upside: `scripts/run_task7_ssl_adapt_tokens_202607xx.py`
   Self-supervised adaptation on unlabeled old+third and target-domain images without labels, but only if the leakage firewall is implemented.

Model selection should use old+third pseudo-domain metrics only. Candidate lock criteria should include high-risk recall, worst-domain BAcc, and old_data stability. Do not allow a candidate to win simply because it increases old_data accuracy.

During Week 2, physicians should annotate a compact review batch: 50–100 cases enriched for strict-like failure modes, high-risk FN patterns, low-risk FP patterns, B2/B3/TC, and view/ROI uncertainty.

### Week 3: lock candidates, run frozen external inference, and prepare manuscript-grade outputs

**Main goal:** obtain honest external evidence from locked base classifiers.

Steps:

1. Select 2–3 candidate base models using old+third only.
2. Create `scripts/pre_register_task7_experiment_manifest_202607xx.py` output:

   * model config;
   * threshold;
   * training data checksum;
   * registry checksum;
   * random seed;
   * selected checkpoints;
   * metric script version.
3. Run `scripts/run_task7_locked_external_infer_202607xx.py` on strict_external.
4. Commit prediction CSVs before reading metrics.
5. If no model changes are made after strict_external, run the same locked candidates on new_external_160 as confirmatory external evidence.
6. Generate:

   * full-coverage forced-classification table;
   * external subgroup table by subtype, view, quality, foreground ratio, and ROI adequacy;
   * high-risk FN review list;
   * full-coverage versus v195/v195+ workflow comparison table;
   * ablation table from old+third only.

Important: if strict_external results disappoint and the team changes the model, then new_external_160 should be held back as the next frozen test, not used immediately as part of the same selection loop.

---

## 7. Manuscript Strategy

### If base external generalization improves

Frame the manuscript as a **gross-pathology foundation-model generalization study for thymic epithelial tumor risk stratification**. The story should be:

1. Task7 is clinically meaningful: A/AB/B1 versus B2/B3/TC.
2. Naive in-domain models overestimate readiness.
3. A locked ROI-aware, multi-instance, domain-generalized model improves full-coverage forced external classification.
4. Independent external testing confirms that the gain is not only internal adaptation.
5. Selective/review workflows provide a downstream clinical safety layer, but the central contribution is the stronger base classifier.

The strongest technical framing would be:

> “ROI-aware multi-scale token MIL with domain-generalized training and physician-concept auxiliary supervision improves cross-domain gross pathology risk stratification.”

Report old_data, third_batch, strict_external, and new_external_160 separately. Use third_batch only as adapted development. Use strict_external and new_external_160 as external evidence only if the protocol was locked before inference.

### If base improvement remains limited

This can still be a rigorous and publishable paper, but the framing must change.

The manuscript should become a **data-governance and external-generalization challenge paper**:

1. Gross pathology images contain real low/high-risk signal.
2. Internal and same-system adapted performance can be strong.
3. Strict external stress testing reveals a large high-risk recall collapse.
4. Systematic negative experiments show that simple backbone swaps, crop variants, weak labels, stacking, and confidence guards do not solve the problem.
5. A risk-controlled workflow can reduce automatic errors by reviewing/rejecting uncertain cases, but it does not make the base model a full autonomous diagnostic system.
6. The paper contributes a transparent benchmark, failure taxonomy, leakage-safe protocol, and clinical deployment boundary.

That version may be more credible than overstating a weak external classifier. It would be especially valuable if paired with new_external_160 locked evaluation, even if the result is modest.

### Claims that must not be overstated

Do **not** claim:

* “99% diagnostic accuracy” for all cases if the result comes from selective release/rejection.
* “External validation solved” if strict_external or new_external_160 forced classification remains near the current 60–65% range.
* “The model can replace pathologists.”
* “The model diagnoses WHO subtype reliably from gross images.”
* “third_batch is an independent external validation set.”
* “0 automatic errors” without reporting coverage, review/rejection rate, and Wilson confidence bounds.
* “High-risk cases are safely detected” unless high-risk recall is explicitly strong on locked external full coverage.
* “Concept labels prove causal visual reasoning” unless concepts are physician-confirmed and validated.

---

## 8. Most Needed Inputs From Physicians / Collaborators

The most valuable collaborator inputs are not generic “more data.” They are targeted inputs that attack the current failure modes.

1. **ROI and view annotation**

   * tumor/specimen foreground box or mask;
   * cut surface versus outer surface versus mixed;
   * whether the visible image is adequate for gross risk assessment;
   * best image per case when multiple images exist.

2. **High-risk FN adjudication**

   * B2/B3/TC cases that look pale, homogeneous, round, small, or well circumscribed;
   * whether gross appearance truly lacks high-risk cues or the model missed a visible region;
   * whether additional images would change physician confidence.

3. **Low-risk FP adjudication**

   * A/AB/B1 cases with hemorrhage, necrosis-like regions, fragmentation, irregular surface, fat contamination, or lobulation;
   * which of these are genuine warning signs versus processing/photography artifacts.

4. **Compact concept labels**

   * hemorrhage;
   * necrosis/cystic change;
   * capsule present/absent/involved;
   * boundary clear/unclear;
   * nodular/lobulated;
   * uniform pale/gray-white cut surface;
   * subject too small/insufficient information;
   * mixed cut/outer view;
   * fat/background contamination.

5. **Boundary-case governance**

   * B1/B2 borderline;
   * B2/B3 borderline;
   * B3/TC borderline;
   * mixed B1/B2 or B2/B3 cases;
   * thymic carcinoma variants;
   * special histologies that should be excluded or analyzed separately.

6. **External-domain metadata**

   * hospital/source;
   * camera or acquisition protocol if available;
   * time period;
   * whether images are pre-fixation/post-fixation;
   * whether ruler/board/background differs;
   * number of images per case;
   * reason for selected image.

7. **Acceptance thresholds before testing**

   * minimum acceptable high-risk recall;
   * minimum acceptable low-risk specificity;
   * maximum tolerated FN count;
   * whether the clinical use case is screening, triage, second reader, or research stratification.

8. **New data acquisition priorities**

   * prioritize B2, B3, and TC;
   * prioritize visually confusable low-risk cases;
   * prioritize multiple hospitals/cameras/backgrounds;
   * prioritize cases with multiple gross views;
   * avoid expanding mainly with easy AB/B1 cases, because that will inflate accuracy without solving external high-risk recall.

The next scientific step is to make the model less dependent on the historical selected image and historical domain style. The practical path is: **case-bag registry → ROI/token MIL → domain simulation → concept/ordinal/high-risk supervision → locked strict/new external evaluation**.

---

## 9. 2026-07-13 Final Current-Image Closure

The prior next-step sentence above is now superseded for the existing 591 selected primary photographs.

The preregistered H2 canonical-coordinate spatial-relational experiment directly tested the last unresolved current-image hypothesis: whether useful transferable morphology was already present in the frozen C1 dense patch tokens but discarded by prior pooling heads. It compared a matched gated head, a full per-case/per-view patch-permuted relational control, and the true-grid relational model under identical locked splits and training.

The result was a complete `NO-GO`:

- true-grid relational OOF BAcc: 0.7278 versus C2 0.7514;
- true-grid relational LODO BAcc: 0.6970 versus C2 0.7441;
- paired LODO delta BAcc versus C2: -0.0471, 95% CI [-0.0866, -0.0062];
- true-grid relational LODO BAcc was 0.0211 below the permuted control;
- all three acquisition batches declined versus C2;
- B1/B2 LODO mean accuracy was 0.5267 versus the required 0.6115;
- all nine preregistered advancement gates failed.

Do not continue architecture, pooling, loss, sampler, threshold, seed, or fusion optimization on this current single-photograph cohort. The direct full-coverage base model remains the primary endpoint, but model development should reopen only after standardized prospective multi-view acquisition and independent multicenter data are available.

Current governing documents:

- `2026-07-13_空间关系模型最终实验结果与主线结论.md`;
- `2026-07-13_胸腺大体图像标准化多视图多中心采集方案.md`;
- repository report `H2_CANONICAL_SPATIAL_RELATIONAL_RESULTS_20260713.md` at commit `bb28e8f` plus the subsequent aggregate-results commit.
