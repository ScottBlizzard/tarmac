# Phase 3 Main-project Gap Audit

## Result

- passed: True
- recommended_strategy: `keep_sidecar_until_minimal_hooks_are_reviewed`
- future_touchpoint_n: 4
- candidate_hook_found_n: 1
- missing_minimal_hook_n: 3
- main_project_write_performed: False
- strict_external_relaxation_allowed: False
- strict_external_labels_used_for_selection: False
- original_project_code_modified: False

## Gap Table

| Change | Status | Candidate Paths | Recommended Next Action |
| --- | --- | --- | --- |
| `stable_runtime_feature_export_hook` | `candidate_hook_found` | run_task7_concat_curriculum_probe.py;scripts/blend_task56_case_runs.py;scripts/blend_task7_curriculum_stage_outputs.py;scripts/blend_task7_groupwise_sources_foldwise_20260520.py;scripts/blend_task7_three_sources_foldwise.py;scripts/derive_task7_from_task6_predictions.py;scripts/evaluate_foldwise_binary_threshold.py;scripts/evaluate_foldwise_weighted_fine_views.py;scripts/evaluate_hierarchical_gate.py;scripts/evaluate_task4_dino_plip_classblend.py;scripts/evaluate_taskB_foldwise_view_router.py;scripts/gate_task56_hierarchy.py;scripts/generate_batch1_batch2_task67_reports.py;scripts/run_biomedclip_frozen_probe.py;scripts/run_dinov2_frozen_probe.py;scripts/run_dinov2_multitask_finetune.py;scripts/run_dinov2_task4_hierprobe.py;scripts/run_plip_caseprobe.py;scripts/run_plip_frozen_probe.py;scripts/run_task4_dino_plip_casefusion.py;scripts/run_task4_dino_plip_caseprobe.py;scripts/run_task4_dino_plip_stackprobe.py;scripts/run_task6_curriculum_probe_20260520.py;scripts/run_task7_67_dinov3_rescue_rule_external_20260523.py;scripts/run_task7_apply_locked_536567_to_external_20260523.py;scripts/run_task7_concat_curriculum_probe.py;scripts/run_task7_curriculum_antishortcut_aux.py;scripts/run_task7_curriculum_gross_distill_20260520.py;scripts/run_task7_deployable_gross_text_switch_20260521.py;scripts/run_task7_dinov3_external_locked_eval_20260523.py;scripts/run_task7_dinov3_finetune_old_third_20260523.py;scripts/run_task7_dinov3_oof_tta_20260524.py;scripts/run_task7_extended_candidate_stacking_20260521.py;scripts/run_task7_external_folder_coral_probe_20260522.py;scripts/run_task7_external_folder_diag_svc_focus_20260522.py;scripts/run_task7_external_folder_diag_svc_focus_fast_20260522.py;scripts/run_task7_external_folder_unsup_domain_sweep_20260522.py;scripts/run_task7_external_third_batch_64style_20260521.py;scripts/run_task7_external_third_batch_rawtop_stack_20260521.py;scripts/run_task7_external_thymoma_carcinoma_folder_20260522.py;scripts/run_task7_gross_auto_router_20260520.py;scripts/run_task7_gross_calib_meta_stack_20260521.py;scripts/run_task7_gross_feature_probe_20260520.py;scripts/run_task7_gross_text_stacking_20260521.py;scripts/run_task7_hardcore_gross_calibrator_20260520.py;scripts/run_task7_large_oof_meta_ensemble_20260520.py;scripts/run_task7_learned_gross_text_switch_20260521.py;scripts/run_task7_learned_review_router_direct_20260521.py;scripts/run_task7_no64_dinov3_dev_locked_external_fusion_20260523.py;scripts/run_task7_no64_guarded_adapt_overlay_20260522.py;scripts/run_task7_no64_guarded_adapt_tuned_overlay_20260522.py;scripts/run_task7_no64_plus_allthird_headonly_rescue_20260523.py;scripts/run_task7_oldtask6_augmented_stacking_20260521.py;scripts/run_task7_oof_stacking_probe_20260520.py;scripts/run_task7_oracle_hard_gross_text_calibrator_20260521.py;scripts/run_task7_review_router_embedding_probe_20260520.py;scripts/run_task7_specialist_concat_logreg.py;scripts/run_task7_third_adapt72_variant_ensemble_20260523.py;scripts/run_task7_viewaware_concat_probe.py;scripts/run_task7_viewspecialist_concat_mlp.py;scripts/run_task7_viewspecialist_concat_probe.py;scripts/summarize_task7_crop_finetune_full_oof_20260522.py;thymic_baseline/aggregate_cv_run.py;thymic_baseline/train.py;thymic_surimage/ablation_train.py;thymic_surimage/evaluate_strict_pipeline.py;thymic_surimage/evaluate_strict_views.py;thymic_surimage/foreground_multicrop.py;thymic_surimage/strict_train.py;thymic_surimage/train.py | review candidate export sites and add a copied/sidecar-safe runtime-feature export hook only after approval |
| `preflight_contract_validation_hook` | `missing_minimal_hook` | - | add a minimal contract-validation call at the export or sidecar boundary after approval |
| `sidecar_invocation_hook` | `missing_minimal_hook` | - | keep using sidecar files; add a single invocation hook only after approval |
| `decision_output_consumer_hook` | `missing_minimal_hook` | - | define the downstream consumer for frozen decision columns before writing main-project code |
| `ci_reproducibility_gate` | `experiment_side_gate_ready` | experiments/risk_control_rejection_20260621/scripts/phase2_reproducibility_runner.py | keep the experiment-side runner as the required pre-merge gate |

## Interpretation

This audit only reads the original project. It identifies where future minimal
main-project integration could attach, but it performs no writes and does not
promote any strict_external relaxation. The current recommendation remains to
keep the risk-control system as a sidecar until these hooks are explicitly
reviewed and approved.