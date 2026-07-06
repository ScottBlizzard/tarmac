# 2026-07-06 Base Model Expansion Update

This folder contains the newest evidence and scripts for the current main line: improving the full-coverage base forced-classification model for Task7 low-risk vs high-risk thymic gross pathology images.

The focus is no longer rejection/review workflow optimization. Selective workflows remain a downstream safety layer, but the scientific bottleneck is external generalization of the base classifier.

## Read First

1. `reports/AI Pathology Model Improvement.md`
   - Broad research-lead analysis of bottlenecks, misleading historical conclusions, and 20+ candidate method directions.

2. `reports/Task7_Base_Model_Expansion_Ledger_20260706.md`
   - Current four-domain counts, trusted forced baseline, completed 2026-07-06 external sweeps, and the completed subtype-auxiliary qkvb experiment 201.

3. `server_metrics/experiments/base_model_expansion_20260706/outputs/new_external_inference/new_external_160_tta4_summary.csv`
   - New external inference has now been run. Best single qkvb-style models are around BAcc 0.68, and fixed qkvb fusion reaches BAcc 0.6947.

4. `server_metrics/experiments/base_model_expansion_20260706/outputs/strict_external_inference/strict_external_tta4_summary.csv`
   - Strict external qkvb TTA still fails mainly through low-risk false positives, especially B1/AB/A.

5. `server_metrics/experiments/base_model_expansion_20260706/outputs/existing_variant_sweep/existing_variant_sweep_summary.csv`
   - Complete historical qkvb/WPC/domain-robust/ConvNeXt variants do not solve strict external generalization.

6. `scripts/run_task7_dinov3_multitask_subtype_aux_20260706.py`
   - New experiment script: qkvb binary Task7 main head plus six-subtype auxiliary head and optional low-risk/B1 loss weighting.

7. `task7_case_bag_audit.json`, `task7_case_bag_case_summary.csv`, and `task7_case_bag_image_registry.csv` in this folder's `server_metrics/`
   - All-image case-bag audit for future MIL experiments.

## Current Most Important Empirical Facts

- `old_data`: 285 cases, low 144 / high 141.
- `third_batch`: 306 cases, low 224 / high 82. Same-system adaptation/development, not strict external.
- `strict_external`: 108 cases, low 61 / high 47. Frozen stress-test external.
- `new_external_160`: 162 deduplicated cases, low 77 / high 85. Frozen confirmatory external.

Trusted forced baseline v135:

- Old+third internal: Acc 0.8748, BAcc 0.8632, AUC 0.9011.
- Old_data: Acc/BAcc 0.9228/0.9227.
- Third_batch: Acc/BAcc 0.8301/0.7718.
- Strict_external: Acc 0.6481, BAcc 0.6275, AUC 0.6052.

New external qkvb TTA:

- Best fixed fusion `avg_105_126_149`: Acc 0.6975, BAcc 0.6947, AUC 0.7271, high recall 0.7529, low specificity 0.6364.
- Best single models are around BAcc 0.68.

Strict external qkvb TTA:

- 105 qkvb: Acc 0.5741, BAcc 0.6034, AUC 0.6446, high recall 0.8298, low specificity 0.3770.
- 154 full very-low-lr: Acc 0.5926, BAcc 0.6198, AUC 0.6561, high recall 0.8298, low specificity 0.4098.
- The dominant strict-external failure is low-risk false positives, especially B1/AB/A.

Historical Task6/subtype routes:

- Old Task6 and subtype-cost models are internally too weak to reuse directly, with Task7 folded BAcc around 0.70.
- Subtype information is still promising as auxiliary/ordinal supervision on top of the stronger qkvb base.

All-image case-bag audit:

- Four-domain total: 861 cases, 878 image rows, 0 missing images.
- Only `old_data` contains multi-image cases: 17/285 cases have 2 images.
- `third_batch`, `strict_external`, and `new_external_160` are single-image per case in the currently available registries.
- This means case-bag MIL is valid to implement, but the available multi-image signal is small and is unlikely to be the sole external-generalization breakthrough.

## Subtype-Auxiliary Experiment 201

`201_qkvb_stylelight_lastblock_subtypeaux_lowrisk_20260706` has completed 5-fold OOF training on the server.

Core design:

- DINOv3 qkvb ViT-L whole352.
- Task7 binary head remains the deployment output.
- Auxiliary six-class subtype head predicts A/AB/B1/B2/B3/TC during training only.
- Loss weights emphasize low-risk and especially B1/AB/A to address strict-external false-high errors.
- `best_model.pt` remains binary-compatible with the existing TTA evaluator; `best_multitask_model.pt` keeps the auxiliary head.

OOF result:

- Acc 0.7005, BAcc 0.6791, AUC 0.7654.
- High-risk recall 0.5919, low-risk specificity 0.7663.
- Confusion: TN 282, FP 86, FN 91, TP 132.

Interpretation:

- 201 is not an internal base-model improvement.
- Heavy low-risk/B1 loss weighting shifted the tradeoff but sacrificed high-risk recall and remained fold/domain unstable.
- A cleaner aux-only variant without heavy sample weighting is a more rational next test than doubling down on this exact weighting.

Strict external TTA4 result:

- Acc 0.5741, BAcc 0.5937, AUC 0.6442.
- High-risk recall 0.7447, low-risk specificity 0.4426.
- Confusion: TN 27, FP 34, FN 12, TP 35.
- This is weaker than the best strict existing variant 154 (BAcc 0.6198), so 201 should not be treated as a successful base-model improvement.

New external 160 TTA4 result:

- Acc 0.6790, BAcc 0.6788, AUC 0.7239.
- High-risk recall 0.6824, low-risk specificity 0.6753.
- Confusion: TN 52, FP 25, FN 27, TP 58.
- This does not beat the fixed fusion `avg_105_126_149` (BAcc 0.6947) or the best single style-light qkvb run 149 (BAcc 0.6847).

Additional post-201 external sweep has completed:

- `201_subtypeaux_lowrisk`
- `123_qkvb_nocw`
- `137_qkvb_classsampler_nocw`
- `168_qkvb_seed20260525`

The script evaluated each candidate on `strict_external` and `new_external_160` using TTA4. Results are under:

`server_metrics/experiments/base_model_expansion_20260706/outputs/extra_candidate_external_sweep/`

Key results:

- Strict best in this extra sweep: `123_qkvb_nocw`, BAcc 0.6182, AUC 0.6432, high-risk recall 0.7447, low-risk specificity 0.4918.
- This still does not materially beat the prior strict best 154 qkvb full very-low-lr, BAcc 0.6198.
- New-external best in this extra sweep: `168_qkvb_seed20260525`, BAcc 0.6841, AUC 0.7230.
- This still does not beat the existing fixed qkvb fusion `avg_105_126_149`, BAcc 0.6947.
- The no-CW direction is informative because it improves low-risk specificity, but it trades away high-risk recall and is not a solution by itself.

Follow-up queue already prepared:

- `scripts/run_next_subtype_aux_ablation_queue_20260706.sh`
- 202: qkvb style_light last_block subtype-aux only, class weighting enabled, no sample loss weighting.
- 203: qkvb style_light last_block subtype-aux only, no class weighting, no sample loss weighting.
- Each run is followed by strict/new external TTA and summary generation.

Additional domain-consistency queue prepared:

- `scripts/run_task7_dinov3_domain_consistency_20260706.py`
- `scripts/run_domain_consistency_after_subtype_queue_20260706.sh`
- 204: qkvb style_light + domain_robust two-view consistency, class weighting enabled.
- 205: qkvb style_light + external_mimic two-view consistency, no class weighting.

## What GPT Pro Should Reconsider

The best next ideas are not more rejection threshold tuning. Prioritize:

- ROI-guided dense-token or multi-scale MIL.
- All-image case-bag MIL rather than one selected image per case.
- Domain-randomized consistency training.
- Binary plus ordinal/subtype auxiliary supervision.
- Group-DRO / pseudo-domain model selection.
- Physician-confirmed concept bottlenecks focused on B1/AB false-high and B2 false-low boundaries.
- Adapter/LoRA or dense-token foundation-model use that is not just another shallow global-feature backbone swap.
