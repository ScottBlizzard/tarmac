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

## Completed Subtype-Auxiliary Experiment 201

New script: `/workspace/thymic_project/scripts/run_task7_dinov3_multitask_subtype_aux_20260706.py`

Design:

- Shared DINOv3 qkvb ViT-L whole352 encoder.
- Main Task7 binary head remains the deployment output.
- Auxiliary six-class subtype head predicts A/AB/B1/B2/B3/TC during training only.
- Low-risk/B1/AB/A samples receive higher loss weight to attack the strict-external false-high pattern.
- The script saves `best_multitask_model.pt` for reproducibility and a binary-compatible `best_model.pt` for existing TTA evaluation.

Completed candidate:

- `201_qkvb_stylelight_lastblock_subtypeaux_lowrisk_20260706`
- Output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/subtype_aux_runs/201_qkvb_stylelight_lastblock_subtypeaux_lowrisk_20260706/`
- Parameters: qkvb ViT-L, whole352, style_light augmentation, last_block tuning, head_lr 2e-4, backbone_lr 1e-6, aux_loss_weight 0.25, low-risk loss weight 1.35, B1 subtype weight 1.60, AB 1.20, A 1.10.
- Smoke test passed. Full 5-fold training completed.

OOF result:

- Acc 0.7005, BAcc 0.6791, AUC 0.7654.
- High-risk recall 0.5919, low-risk specificity 0.7663.
- Confusion: TN 282, FP 86, FN 91, TP 132.

Domain OOF:

| Domain | Acc | BAcc | AUC | High recall | Low specificity |
|---|---:|---:|---:|---:|---:|
| batch1 | 0.7350 | 0.7338 | 0.8035 | 0.6842 | 0.7833 |
| batch2 | 0.6190 | 0.6190 | 0.6806 | 0.6071 | 0.6310 |
| third_batch_adapt72_highfocus | 0.5694 | 0.5828 | 0.7062 | 0.5227 | 0.6429 |
| third_batch_holdout | 0.7821 | 0.6684 | 0.7466 | 0.5000 | 0.8367 |
| old | 0.6667 | 0.6664 | 0.7311 | 0.6383 | 0.6944 |
| third | 0.7320 | 0.6623 | 0.7652 | 0.5122 | 0.8125 |

Fold-level OOF:

- Fold 1: test Acc 0.7805, BAcc 0.7638, AUC 0.8281, high recall 0.6875, low specificity 0.8400.
- Fold 2: test Acc 0.7049, BAcc 0.6766, AUC 0.7563, high recall 0.5532, low specificity 0.8000.
- Fold 3: test Acc 0.6610, BAcc 0.6688, AUC 0.7600, high recall 0.6977, low specificity 0.6400.
- Fold 4: test Acc 0.6552, BAcc 0.6018, AUC 0.7228.
- Fold 5: test Acc 0.6964, BAcc 0.6810, AUC 0.7796.

Interpretation: 201 is not a base-model improvement on old+third OOF. It shifts the tradeoff toward low-risk specificity but sacrifices high-risk recall and is fold/domain unstable. The heavy low-risk/B1 weighting should not be treated as the next main direction unless locked external results show a meaningful and reproducible external benefit. A cleaner aux-only variant without heavy low-risk sample weighting is a better next test.

## Active External Sweep

Post-201 external sweep:

- Script: `/workspace/thymic_project/scripts/run_extra_external_candidates_after_201_20260706.sh`
- Output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/extra_candidate_external_sweep/`
- Candidates: 201 subtype-aux low-risk weighting, 123 qkvb no class-weighting, 137 qkvb class-sampler no-CW, 168 qkvb seed20260525.
- Domains: strict_external and new_external_160, TTA4, summarized by `/workspace/thymic_project/scripts/summarize_external_tta_predictions_20260706.py`.
- Status at this update: completed. Follow-up 202 subtype-aux-only class-weighted training is running.

201 strict_external TTA4:

- Acc 0.5741, BAcc 0.5937, AUC 0.6442.
- High-risk recall 0.7447, low-risk specificity 0.4426.
- Confusion: TN 27, FP 34, FN 12, TP 35.
- Subtype low-risk correctness: A 4/8, AB 19/39, B1 2/11, MNT-assumed-low 2/3.

Interpretation: 201 improves strict low-risk specificity versus 105/149/126-style qkvb runs, but the gain is too small and it loses high-risk recall. It remains weaker than the best strict existing variant 154 (BAcc 0.6198). This supports testing cleaner subtype auxiliary training without heavy low-risk sample weighting.

201 new_external_160 TTA4:

- Acc 0.6790, BAcc 0.6788, AUC 0.7239.
- High-risk recall 0.6824, low-risk specificity 0.6753.
- Confusion: TN 52, FP 25, FN 27, TP 58.
- Subtype correctness: A 12/22, AB 24/29, B1 16/26, B2 18/28, B3 18/29, TC 22/28.

Interpretation: 201 also fails to beat the established new_external_160 fixed fusion `avg_105_126_149` (BAcc 0.6947) or the best single style-light qkvb run 149 (BAcc 0.6847). It is useful as a negative ablation, not as a main model.

Extra candidate external sweep summary:

| Model | Domain | Acc | BAcc | AUC | High recall | Low specificity | TN/FP/FN/TP |
|---|---|---:|---:|---:|---:|---:|---|
| 123 qkvb no-CW | strict_external | 0.6019 | 0.6182 | 0.6432 | 0.7447 | 0.4918 | 30/31/12/35 |
| 137 qkvb class-sampler no-CW | strict_external | 0.5648 | 0.6001 | 0.6223 | 0.8723 | 0.3279 | 20/41/6/41 |
| 201 subtype-aux low-risk | strict_external | 0.5741 | 0.5937 | 0.6442 | 0.7447 | 0.4426 | 27/34/12/35 |
| 168 qkvb seed20260525 | strict_external | 0.5648 | 0.5928 | 0.6536 | 0.8085 | 0.3770 | 23/38/9/38 |
| 168 qkvb seed20260525 | new_external_160 | 0.6852 | 0.6841 | 0.7230 | 0.7059 | 0.6623 | 51/26/25/60 |
| 201 subtype-aux low-risk | new_external_160 | 0.6790 | 0.6788 | 0.7239 | 0.6824 | 0.6753 | 52/25/27/58 |
| 137 qkvb class-sampler no-CW | new_external_160 | 0.6790 | 0.6764 | 0.7328 | 0.7294 | 0.6234 | 48/29/23/62 |
| 123 qkvb no-CW | new_external_160 | 0.6728 | 0.6742 | 0.7290 | 0.6471 | 0.7013 | 54/23/30/55 |

Interpretation:

- None of the extra candidates beats the existing strict_external best 154 qkvb full very-low-lr BAcc 0.6198 by a meaningful margin.
- 123 no-CW is the most informative strict result: it improves low-risk specificity to 0.4918 but loses high-risk recall, so no-CW is a real tradeoff lever rather than a solution.
- 137 class-sampler no-CW restores high-risk recall but collapses specificity, so sampler balancing recreates the false-high failure pattern.
- 168 seed confirms that seed variation is not a breakthrough.
- On new_external_160, none beats the existing fixed qkvb fusion `avg_105_126_149` BAcc 0.6947.

Queued follow-up after this external sweep:

- Script: `/workspace/thymic_project/scripts/run_next_subtype_aux_ablation_queue_20260706.sh`
- Output root: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/subtype_aux_runs/`
- External output root: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/subtype_aux_ablation_external_sweep/`
- 202: qkvb style_light last_block subtype-aux only, class weighting enabled, no sample loss weighting.
- 203: qkvb style_light last_block subtype-aux only, no class weighting, no sample loss weighting.

Additional queued follow-up after 202/203:

- Script: `/workspace/thymic_project/scripts/run_domain_consistency_after_subtype_queue_20260706.sh`
- Training script: `/workspace/thymic_project/scripts/run_task7_dinov3_domain_consistency_20260706.py`
- Output root: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/domain_consistency_runs/`
- External output root: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/domain_consistency_external_sweep/`
- 204: qkvb style_light + domain_robust two-view consistency, class weighting enabled.
- 205: qkvb style_light + external_mimic two-view consistency, no class weighting.

## Case-Bag / Multi-Image Registry

Script: `/workspace/thymic_project/scripts/build_task7_case_bag_registry_20260706.py`

Output: `/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/case_bag_registry/`

Audit:

- Total: 861 cases, 878 image rows, 0 missing images.
- old_data: 285 cases, 302 image rows, 17 multi-image cases, max 2 images/case.
- third_batch: 306 cases, 306 image rows, 0 multi-image cases.
- strict_external: 108 cases, 108 image rows, 0 multi-image cases.
- new_external_160: 162 cases, 162 image rows, 0 multi-image cases.

Interpretation: all-image case-bag MIL is now feasible and leakage-audited, but the multi-image signal is small and only exists in old_data. It is worth implementing as an infrastructure/ablation direction, not as the sole external-generalization bet.

## Current Technical Diagnosis

1. Strict external is not failing because the model is too conservative; it is often over-calling low-risk cases as high-risk.
2. B1 low-risk external cases are the clearest failure cluster.
3. Simple thresholds, average fusion, frozen features, and ROI stats are insufficient.
4. The next useful directions are stronger domain-randomized augmentation, subtype/ordinal auxiliary supervision, complete multi-scale/WPC or dense-token MIL training, all-image case-bag MIL, and external-domain simulation that specifically protects low-risk B1/AB specificity without sacrificing B2/B3/TC recall.
