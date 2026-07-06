# Task7 Base Model Expansion Ledger - 2026-07-06

## Main Position

The current main line is full-coverage base-model capability and external generalization for low-risk vs high-risk thymic gross pathology images. Rejection / review workflow is treated as downstream risk control, not the primary contribution.

## Locked Data Boundaries

- old_data: 285 cases, low 144 / high 141.
- third_batch: 306 cases, low 224 / high 82. This is same-system adaptation/development, not strict external.
- strict_external: 108 cases, low 61 / high 47. Frozen stress-test external.
- new_external_160: 162 deduplicated cases, low 77 / high 85. Frozen confirmatory external.

Registry audit output:

- Server: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/registry/`
- Master registry: `task7_four_domain_master_registry.csv`
- Audit found 0 missing images, 0 duplicate case rows, 0 duplicate image paths, and 0 cross-domain original-case collisions.

## Trusted Forced Baseline

Primary clean forced classifier is v135 `main_prob` threshold 0.595 selected on old+third OOF. Workflow / reconstructed outputs are excluded from base-model baselines.

| Domain | N | Acc | BAcc | AUC | High recall | Low specificity |
|---|---:|---:|---:|---:|---:|---:|
| old+third internal | 591 | 0.8748 | 0.8632 | 0.9011 | 0.8161 | 0.9103 |
| old_data | 285 | 0.9228 | 0.9227 | 0.9489 | 0.9149 | 0.9306 |
| third_batch | 306 | 0.8301 | 0.7718 | 0.8137 | 0.6463 | 0.8973 |
| strict_external | 108 | 0.6481 | 0.6275 | 0.6052 | 0.4681 | 0.7869 |

Interpretation: domain-internal performance is strong, but strict external generalization is weak.

## Completed Experiments

### New External 160 QKVB TTA

Output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/new_external_inference/`

Best fixed fusion:

- avg_105_126_149: Acc 0.6975, BAcc 0.6947, AUC 0.7271, high recall 0.7529, low specificity 0.6364.

Single-model fixed 0.5:

- 149 stylelight: BAcc 0.6847.
- 105 qkvb: BAcc 0.6823.
- 126 head-only: BAcc 0.6817.
- 140 convnext: BAcc 0.5861.

OOF threshold transfer did not improve new_external_160.

### Strict External QKVB TTA

Output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/strict_external_inference/`

Single-model fixed 0.5:

| Model | Acc | BAcc | AUC | High recall | Low specificity | TP/FN | TN/FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 105 qkvb | 0.5741 | 0.6034 | 0.6446 | 0.8298 | 0.3770 | 39/8 | 23/38 |
| 149 stylelight | 0.5556 | 0.5846 | 0.6393 | 0.8085 | 0.3607 | 38/9 | 22/39 |
| 126 head-only | 0.5463 | 0.5764 | 0.6516 | 0.8085 | 0.3443 | 38/9 | 21/40 |

Fixed/OOF probability fusion reached only BAcc 0.6107 on strict_external. The dominant failure is low-risk false positives, especially B1: 105 qkvb correctly classified only 1/11 strict-external B1 cases.

### Frozen Feature Bank

Output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/new_external_feature_bank_combined_summary.csv`

Old+third locked frozen-feature models were weak on new_external_160:

- SigLIP-L384: BAcc 0.5977.
- EVA02-L336: BAcc 0.5933.
- SigLIP-SO400M: BAcc 0.5618.

Interpretation: frozen foundation features alone do not solve the external-domain gap.

### ROI / Color / Texture Probe

Output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/roi_stats_probe/`

Hand-crafted ROI/color/texture statistics were weak alone:

- Best new_external_160 BAcc around 0.57.
- Best strict_external BAcc around 0.57.

Interpretation: ROI stats are auxiliary domain/concept signals, not a base classifier.

### QKVB + ROI Meta Fusion

Output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/qkvb_roi_meta_fusion/`

Only old+third OOF labels were used for meta-model selection; external labels were held out.

Best strict_external:

- qkvb_roi__random_forest_balanced: Acc 0.5833, BAcc 0.6165, AUC 0.6721, high recall 0.8723, low specificity 0.3607.

Best new_external_160:

- qkvb_only__random_forest_balanced: Acc 0.6852, BAcc 0.6811, AUC 0.7169.

Interpretation: ROI meta features slightly help strict_external but hurt new_external_160. This is not enough and confirms that representation/input/training must improve.

### Incomplete WPC QKVB Run 128

Run 128 was expected to be a 5-fold qkvb whole_plus_crop model but is incomplete:

- It has only 3 checkpoints and lacks `cv_fold_summary.csv` / `oof_metrics.csv`.
- A fast-crop strict_external inference on available 3 folds gave BAcc 0.5755 and B1 accuracy 0/11.

Interpretation: do not use run 128 as a main candidate. A complete WPC qkvb run needs to be trained or historical complete WPC alternatives must be evaluated.

### Existing Complete Variant External Sweep

Script: `/workspace/thymic_project/scripts/run_existing_variant_external_sweep_20260706.py`

Strict external output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/existing_variant_sweep/existing_variant_sweep_summary.csv`

| Model | Acc | BAcc | AUC | High recall | Low specificity | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---|
| 154 qkvb whole352 full very-low-lr | 0.5926 | 0.6198 | 0.6561 | 0.8298 | 0.4098 | 25/36/8/39 |
| 119 qkvb whole352 last2 | 0.5833 | 0.6116 | 0.6449 | 0.8298 | 0.3934 | 24/37/8/39 |
| 113 qkvb whole448 | 0.5741 | 0.6034 | 0.6725 | 0.8298 | 0.3770 | 23/38/8/39 |
| 92 DINOv3-L WPC | 0.5741 | 0.6034 | 0.6300 | 0.8298 | 0.3770 | 23/38/8/39 |
| 171 qkvb external-mimic | 0.5602 | 0.6025 | 0.6475 | 0.8936 | 0.3115 | 19/42/5/42 |
| 75b DINOv3-B WPC domain-robust | 0.5463 | 0.5706 | 0.5934 | 0.7447 | 0.3967 | 24/37/12/35 |
| 97 ConvNeXt WPC | 0.5093 | 0.5232 | 0.5639 | 0.6170 | 0.4295 | 26/35/18/29 |

New external top-candidate output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/existing_variant_sweep_new_top/existing_variant_sweep_summary.csv`

- 119 qkvb whole352 last2: Acc 0.6852, BAcc 0.6853, AUC 0.7254, high recall 0.6824, low specificity 0.6883.
- 154 qkvb whole352 full very-low-lr: Acc 0.6790, BAcc 0.6782, AUC 0.7218, high recall 0.6941, low specificity 0.6623.
- Neither beats the fixed new-external fusion `avg_105_126_149` BAcc 0.6947.

Subtype diagnostics:

- Strict external low-risk failure remains concentrated in A/AB/B1 false-high calls. For 154, strict subtype correctness was A 3/8, AB 18/39, B1 2/11, B2 20/28, B3 10/10, TC 5/5.
- New external is less distorted: for 119, low-risk subtype correctness was A 13/22, AB 24/29, B1 16/26, while high-risk correctness was B2 18/28, B3 17/29, TC 23/28.

Interpretation: full very-low-lr and last2 tuning slightly improve strict_external, but the dominant strict-external low-risk false-positive failure persists. Complete WPC, ConvNeXt, domain-robust, and external-mimic variants do not solve the base-model gap.

### Task6 / Subtype Historical Recheck

Old Task6 and subtype-cost directions are internally too weak to reuse directly:

- `task7_subtype_weight_runs/01_concat_subtypecost_v1`: old-data OOF BAcc 0.7051.
- `task6_curriculum_vits_vitb_stage3_salvage` folded to Task7: OOF BAcc 0.7048.
- `task6_strict_full_curriculum`: six-class macro F1 0.4349.
- `task6_route_easy_stage1_else_strict_full`: six-class macro F1 0.4334.

Interpretation: six-class/subtype information should not replace direct Task7 binary training. It is still worth using as auxiliary or ordinal supervision on top of the stronger qkvb base.

## Active Experiment

New script: `/workspace/thymic_project/scripts/run_task7_dinov3_multitask_subtype_aux_20260706.py`

Design:

- Shared DINOv3 qkvb ViT-L whole352 encoder.
- Main Task7 binary head remains the deployment output.
- Auxiliary six-class subtype head predicts A/AB/B1/B2/B3/TC during training only.
- Low-risk/B1/AB/A samples receive higher loss weight to attack the strict-external false-high pattern.
- The script saves `best_multitask_model.pt` for reproducibility and a binary-compatible `best_model.pt` for existing TTA evaluation.

Running candidate:

- `201_qkvb_stylelight_lastblock_subtypeaux_lowrisk_20260706`
- Output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/subtype_aux_runs/201_qkvb_stylelight_lastblock_subtypeaux_lowrisk_20260706/`
- Parameters: qkvb ViT-L, whole352, style_light augmentation, last_block tuning, head_lr 2e-4, backbone_lr 1e-6, aux_loss_weight 0.25, low-risk loss weight 1.35, B1 subtype weight 1.60, AB 1.20, A 1.10.
- Smoke test passed. Full 5-fold training is in progress.

## Current Technical Diagnosis

1. Strict external is not failing because the model is too conservative; it is often over-calling low-risk cases as high-risk.
2. B1 low-risk external cases are the clearest failure cluster.
3. Simple thresholds, average fusion, frozen features, and ROI stats are insufficient.
4. The next useful directions are stronger domain-randomized augmentation, subtype/ordinal auxiliary supervision, complete multi-scale/WPC or dense-token MIL training, all-image case-bag MIL, and external-domain simulation that specifically protects low-risk B1/AB specificity without sacrificing B2/B3/TC recall.
