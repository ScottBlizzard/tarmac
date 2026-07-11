# Prompt for GPTPro

You are a top-tier research lead in medical AI, computational pathology, and machine learning. Please thoroughly inspect the GitHub handoff package:

Repository: https://github.com/ScottBlizzard/tarmac

`thymic_grosspath_gptpro_handoff/`

Latest completed update, 2026-07-11:

- Read `06_20260711_base_model_capability/README_20260711_BASE_MODEL_CAPABILITY.md` first.
- For the current post-experiment question, follow `06_20260711_base_model_capability/GPTPRO_PROMPT_20260711_POST_EXPERIMENT.md`.
- Then read `06_20260711_base_model_capability/reports/Task7_Base_Model_Capability_Experiments_20260711.md`, the machine-readable candidate lock, and the fresh-external blind-test protocol.
- Phase 2 ran a broad leakage-safe internal wave through runs 206-369 using 591 internal cases: batch1 117, batch2 168, and third_batch 306. Model selection used locked five-fold OOF and canonical three-source leave-one-domain-out evaluation.
- The only clear representation-level gain was SigLIP-L at 512 px with six deterministic views from one primary image and gated pooling over all dense patch tokens. Locked C1 reached OOF BAcc/AUC 0.7477/0.8240 and LODO 0.7397/0.8072.
- Locked C2 is an equal average of AIMv2 MixStyle run 253 and C1. It reached OOF 0.7514/0.8377 and LODO 0.7441/0.8108, with LODO B1/B2 risk accuracy 0.5000/0.6629.
- C2's LODO AUC exceeded the prior locked `215+253+254` fusion by 0.0228, paired bootstrap 95% CI [0.0031, 0.0426], but its BAcc difference remained inconclusive. Do not call this an external breakthrough.
- Meta-level cross-selection rejected the superficially best three- and five-model internal recipes as selection-unstable. C1 appeared in every held-out winner, while its companion recipe changed.
- The B1/B2 boundary remains unresolved. Boundary experts, subtype/ordinal heads, contrastive hard negatives, LoRA, domain-generalization losses, quality preprocessing, structured pooling, and SAM optimization did not produce a stable improvement.
- The historical strict external 108 and newer external 162 cohorts have both been inspected. They are consumed audit sets and must not be reused for Phase-2 selection or confirmation. C1/C2 require a genuinely fresh label-blinded external cohort.
- Your analysis should now focus on whether the candidate lock and blind-test protocol are defensible, what new multi-center data and physician review are needed, and which genuinely structural experiments remain. Do not recommend another low-level repeat of the completed sweep without identifying the changed assumption.

Historical 2026-07-06 update:

- Read `05_20260706_base_model_expansion/README_20260706_BASE_MODEL_EXPANSION.md` before the older files.
- New external inference has now been run. `new_external_160` is no longer only QC-completed; qkvb TTA/fusion results are available under `05_20260706_base_model_expansion/server_metrics/`.
- Strict external qkvb-family models now show a dominant low-risk false-positive pattern, especially B1/AB/A false-high cases.
- A new qkvb subtype-auxiliary experiment has completed internal OOF training: `scripts/run_task7_dinov3_multitask_subtype_aux_20260706.py`, with a binary Task7 head plus six-class subtype auxiliary head and low-risk/B1 loss weighting. Its OOF result is not a base-model improvement: Acc 0.7005, BAcc 0.6791, AUC 0.7654, high-risk recall 0.5919, low-risk specificity 0.7663. Its strict_external TTA4 result is also not a breakthrough: Acc 0.5741, BAcc 0.5937, AUC 0.6442, high-risk recall 0.7447, low-risk specificity 0.4426. Its new_external_160 TTA4 result is Acc 0.6790, BAcc 0.6788, AUC 0.7239, high-risk recall 0.6824, low-risk specificity 0.6753.
- A follow-up queue script has been prepared: `scripts/run_next_subtype_aux_ablation_queue_20260706.sh`. It tests cleaner subtype-aux variants without heavy low-risk sample weighting.
- A domain-consistency script has also been prepared: `scripts/run_task7_dinov3_domain_consistency_20260706.py`, queued by `scripts/run_domain_consistency_after_subtype_queue_20260706.sh`. It tests two-view style/domain consistency rather than another plain CE backbone swap.
- The extra candidate external sweep is complete. Its strict best is `123_qkvb_nocw` with BAcc 0.6182, high-risk recall 0.7447, low-risk specificity 0.4918, which still does not materially beat the prior strict best 154 qkvb full very-low-lr BAcc 0.6198. Its new_external_160 best is `168_qkvb_seed20260525` with BAcc 0.6841, still below the existing fixed qkvb fusion BAcc 0.6947.
- Continue to treat rejection/review workflows as downstream safety wrappers, not as the main optimization target.

Your task is not to produce a conservative summary. Your task is to rethink, from a broad and exploratory perspective, how to improve the cross-domain generalization of the base model for low-risk vs high-risk thymic gross pathology image classification.

Please answer in English.

## Project Background

This is a thymic gross pathology image AI project. The clinically central task is Task7:

- Low risk: A / AB / B1
- High risk: B2 / B3 / thymic carcinoma

The project has already attempted many directions: SuRImage/CNN baselines, DINOv2/DINOv3 frozen features, logistic/MLP probes, cut-only models, whole/crop variants, multi-image policies, view-aware routing, curriculum learning, expert-derived auxiliary labels, third-batch adaptation, strict external testing, risk-controlled selective workflows, and v195/v195+ rejection/review workflows.

For this round, please do **not** keep the main focus on increasing automatic release ratio or tuning rejection rules. Those workflows may still be useful as a downstream safety layer, but the current scientific bottleneck is:

**The full-coverage base forced-classification model must become stronger and must now be confirmed on a genuinely fresh external domain.**

## Dataset Boundary

Please keep the following dataset roles separate:

1. `old_data`
   - 285 cases/images
   - Low risk 144, high risk 141
   - Main internal development / OOF cohort

2. `third_batch`
   - 306 cases/images
   - Low risk 224, high risk 82
   - Same-system new data
   - Already used for observation, adaptation, and strategy selection
   - It must not be treated as strict external validation anymore

3. `strict_external`
   - 108 cases/images
   - Low risk 61, high risk 47
   - Historical external stress test; predictions and labels have been inspected
   - Consumed audit set: never use for Phase-2 training, tuning, selection, or confirmation

4. `new_external_160`
   - 162 deduplicated case-level samples
   - Low risk 77, high risk 85
   - Class counts: A 22, AB 29, B1 26, B2 28, B3 29, TC 28
   - Processing/QC has been completed and formal qkvb-family inference has now been run
   - Also a consumed audit set as of 2026-07-11; not a fresh confirmatory cohort for C1/C2

## Known Metric Situation

Please verify details from the handoff package. As orientation:

- The strongest internal-domain model/workflow reaches about 92.3% Acc/BAcc on old data.
- After adaptation, third-batch performance is about 83.0% Acc and about 76.8% BAcc, but high-risk recall remains weak.
- Strict external full-coverage forced classification is about 64.8% Acc, about 62.8% BAcc, with high-risk recall around 46.8%.
- New external qkvb TTA/fusion now reaches roughly BAcc 0.69 at full coverage, but this is still not enough for a strong medical-engineering claim.
- Complete qkvb/WPC/domain-robust/ConvNeXt external sweeps do not solve strict external generalization; the best strict external candidates remain around BAcc 0.62.
- v195/v195+ workflows can control automatic errors through release/review/rejection, but that is not equivalent to a strong full-coverage base classifier.
- Phase-2 locked C1/C2 improve internal OOF/LODO evidence to roughly BAcc 0.74, but they have no untouched external result yet.

## Files to Read First

Please start with:

0. `06_20260711_base_model_capability/README_20260711_BASE_MODEL_CAPABILITY.md`
0. `06_20260711_base_model_capability/reports/Task7_Base_Model_Capability_Experiments_20260711.md`
0. `06_20260711_base_model_capability/scripts/phase2_fresh_external_candidate_lock_20260711.csv`
0. `06_20260711_base_model_capability/reports/FRESH_EXTERNAL_BLIND_TEST_PROTOCOL_20260711.md`
0. `05_20260706_base_model_expansion/README_20260706_BASE_MODEL_EXPANSION.md`
0. `05_20260706_base_model_expansion/reports/Task7_Base_Model_Expansion_Ledger_20260706.md`
0. `05_20260706_base_model_expansion/reports/AI Pathology Model Improvement.md`
0. `05_20260706_base_model_expansion/server_metrics/experiments/base_model_expansion_20260706/outputs/`
0. `05_20260706_base_model_expansion/scripts/run_task7_dinov3_multitask_subtype_aux_20260706.py`
1. `README.md`
2. `01_local_reports_md_csv/项目阶段性工作详报.md`
3. `01_local_reports_md_csv/Task7项目完成度评估.md`
4. `01_local_reports_md_csv/三批数据集外部泛化拒识增强方案报告_.md`
5. `01_local_reports_md_csv/Task7模型升级负结果阶段汇总_2026-05-16.md`
6. `01_local_reports_md_csv/2026-05-18_Task7课程学习与经验标签阶段汇报_医生版.md`
7. `01_local_reports_md_csv/2026-05-25_Task7第三批与严格外部集后续尝试阶段汇报_医生版.md`
8. `01_local_reports_md_csv/2026-05-26_GrossPath-RC_v2域泛化诊断与训练清单.md`
9. `04_server_snapshot/task_plan.md`
10. `04_server_snapshot/progress.md`
11. `04_server_snapshot/findings.md`
12. `04_server_snapshot/experiments/risk_control_rejection_20260621/reports/`
13. `04_server_snapshot/experiments/risk_control_rejection_20260621/v195_plus_sidecar/`
14. Summary/metrics files under `04_server_snapshot/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/`

Then inspect as needed:

- `02_local_scripts_py/`
- `04_server_snapshot/scripts/`
- `03_local_result_summaries_flat/`

## Previously Attempted Directions

The following directions have already been tried and were mostly unstable or insufficient for external generalization. You are allowed to propose advanced versions of them, but please explicitly explain why your proposal is not merely a low-level repeat of previous failed experiments.

Previously attempted directions include:

- SuRImage / SE-ResNeXt / CNN / CAM crop
- WHO six-class classification followed by binary merging
- DINOv2 frozen features with logistic regression / MLP probes
- DINOv2/DINOv3 fine-tuning variants, including last-block, head-only, full fine-tune, QKVB, and TTA
- Cut-only, whole-only, whole+crop, and full-to-cut refinement
- View-aware routing and cut/outer/mixed specialists
- PLIP, BiomedCLIP, ConvNeXt, Swin, EfficientNet, EVA02, SigLIP, ViTamin, AIMv2, and related backbone swaps or fusions
- Weak expert labels, large-scale hard-sample weighting, and directly adding core-hard samples back into training
- Stacking, guard models, and domain-internal reliability selectors
- Selective release / rejection / auto-review workflows

## What I Want You To Do

Please produce a research-lead-level analysis and plan. Do not stop at high-level suggestions. Complete the following:

1. Diagnose the real bottleneck
   - In your own judgment, is the current limitation mainly data, model capacity, domain shift, label boundary ambiguity, input-view heterogeneity, or a combination?
   - Which historical conclusions do you agree with?
   - Which historical conclusions might be misleading or prematurely accepted?

2. Reopen the method space
   - Do not prematurely narrow to DINO + probe.
   - Think broadly, including but not limited to: stronger vision foundation models, pathology/medical foundation models, SAM/MedSAM/YOLO/DETR/grounding-assisted ROI extraction, multi-instance learning, multi-image case modeling, self-supervised or contrastive learning, unsupervised domain adaptation, source-free adaptation, test-time adaptation, domain-adversarial learning, style randomization, color/background/scale augmentation, synthetic data, diffusion-based augmentation, active learning, physician weak supervision, concept bottlenecks, multitask learning, ordinal/clinical-risk losses, B2/B3/TC-specialized modeling, open-set/OOD decomposition, and external-domain simulation.

3. Build a broad experiment matrix
   - Propose at least 20 candidate directions.
   - For each direction, include:
     - Core idea
     - Why it may help external generalization
     - How it differs from previous attempts
     - Required data/code
     - Main risks
     - Success criteria

4. Prioritize experiments
   - From the 20+ directions, select the 8-12 most worth running first.
   - Do not rank only by safety. Also consider high-risk/high-upside directions that may produce a breakthrough.

5. Design a strict evaluation protocol
   - Keep `old_data`, `third_batch`, `strict_external`, and `new_external_160` separate.
   - `strict_external` and `new_external_160` must not be used for training, tuning, or model selection.
   - Report Acc, BAcc, AUC, high-risk recall, low-risk specificity, FN, and FP.
   - Any workflow/rejection result must be reported separately from full-coverage forced classification.
   - Explain how to avoid patient leakage, multi-image leakage, threshold leakage, and external-label leakage.

6. Create an executable plan
   - What should be run in week 1, week 2, and week 3?
   - Which scripts should be modified or newly created?
   - Which existing files/result tables should be reused first?
   - Which old experiments can be treated as baselines without rerunning?

7. Give manuscript strategy
   - If base external generalization improves to an acceptable level, how should the medical-engineering manuscript be framed?
   - If improvement remains limited, how can this still be written as a rigorous paper about external generalization challenge, risk control, and data governance?
   - Which claims must absolutely not be overstated?

## Output Structure

Please structure your answer exactly as follows:

1. `Real Bottleneck Diagnosis`
2. `Historical Experiment Review and Potentially Misleading Conclusions`
3. `Broad Method Space: 20+ Candidate Experiments`
4. `Highest-Priority 8-12 Experiments`
5. `Strict Evaluation Protocol`
6. `Three-Week Execution Plan`
7. `Manuscript Strategy`
8. `Most Needed Inputs From Physicians / Collaborators`

Please be specific. Do not just say "try model X." If you recommend changing the model backbone or foundation model, specify the candidate model family, input strategy, training strategy, fusion strategy, evaluation strategy, and why it is not simply a repeat of an already failed backbone sweep.
