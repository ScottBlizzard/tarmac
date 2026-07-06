# Prompt for GPTPro

You are a top-tier research lead in medical AI, computational pathology, and machine learning. Please thoroughly inspect the GitHub handoff package:

`thymic_grosspath_gptpro_handoff/`

Your task is not to produce a conservative summary. Your task is to rethink, from a broad and exploratory perspective, how to improve the cross-domain generalization of the base model for low-risk vs high-risk thymic gross pathology image classification.

Please answer in English.

## Project Background

This is a thymic gross pathology image AI project. The clinically central task is Task7:

- Low risk: A / AB / B1
- High risk: B2 / B3 / thymic carcinoma

The project has already attempted many directions: SuRImage/CNN baselines, DINOv2/DINOv3 frozen features, logistic/MLP probes, cut-only models, whole/crop variants, multi-image policies, view-aware routing, curriculum learning, expert-derived auxiliary labels, third-batch adaptation, strict external testing, risk-controlled selective workflows, and v195/v195+ rejection/review workflows.

For this round, please do **not** keep the main focus on increasing automatic release ratio or tuning rejection rules. Those workflows may still be useful as a downstream safety layer, but the current scientific bottleneck is:

**The full-coverage base forced-classification model must become stronger, especially on strict external and new external domains.**

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
   - Frozen external stress test
   - Should not be used for training, hyperparameter tuning, or model selection

4. `new_external_160`
   - 166 raw uploaded files
   - 162 deduplicated case-level samples
   - Low risk 77, high risk 85
   - Class counts: A 22, AB 29, B1 26, B2 28, B3 29, TC 28
   - Processing/QC has been completed, but formal model inference has not yet been run at packaging time

## Known Metric Situation

Please verify details from the handoff package. As orientation:

- The strongest internal-domain model/workflow reaches about 92.3% Acc/BAcc on old data.
- After adaptation, third-batch performance is about 83.0% Acc and about 76.8% BAcc, but high-risk recall remains weak.
- Strict external full-coverage forced classification is about 64.8% Acc, about 62.8% BAcc, with high-risk recall around 46.8%.
- v195/v195+ workflows can control automatic errors through release/review/rejection, but that is not equivalent to a strong full-coverage base classifier.

## Files to Read First

Please start with:

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

