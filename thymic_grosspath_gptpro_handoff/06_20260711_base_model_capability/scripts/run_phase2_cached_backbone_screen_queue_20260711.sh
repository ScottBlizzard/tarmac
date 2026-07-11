#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
extractor="/root/extract_task7_dense_token_bank_20260711.py"
trainer="/root/run_task7_dense_feature_cv_20260711.py"
registry="/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/registry/task7_four_domain_master_registry.csv"
split="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
bank_root="/root/thymic_feature_banks_20260711"
output_root="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_cached_backbone_screen"
manifest_root="$output_root/feature_bank_manifests"
lora_status="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_lora_source_lodo/PHASE2_LORA_SOURCE_LODO.status"
status_file="$output_root/PHASE2_CACHED_BACKBONE_SCREEN.status"

mkdir -p "$output_root" "$manifest_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

write_status() {
  rm -f -- "$status_file"
  printf '%s\n' "$1" > "$status_file"
}

on_exit() {
  rc=$?
  if [[ $rc -eq 0 ]]; then
    write_status complete
  else
    write_status "failed rc=$rc"
  fi
}
trap on_exit EXIT
write_status waiting_for_lora_queue

while true; do
  upstream="$(cat "$lora_status" 2>/dev/null || true)"
  case "$upstream" in
    complete*) break ;;
    failed*)
      printf 'Upstream LoRA queue failed: %s\n' "$upstream" >&2
      exit 3
      ;;
    *) sleep 30 ;;
  esac
done
write_status running

run_cv() {
  local run_name="$1"
  local bank_dir="$2"
  local split_mode="$3"
  local pooling="$4"
  local output_dir="$output_root/$run_name"
  if [[ -s "$output_dir/oof_metrics.csv" ]]; then
    printf 'SKIP complete %s\n' "$run_name"
    return
  fi
  printf 'START-CV %s %s\n' "$run_name" "$(date --iso-8601=seconds)"
  "$python_bin" "$trainer" \
    --feature-bank-dir "$bank_dir" \
    --output-dir "$output_dir" \
    --split-csv "$split" \
    --split-mode "$split_mode" \
    --pooling "$pooling" \
    --class-weighting \
    --epochs 80 \
    --patience 12 \
    --batch-size 24 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --device cuda
  printf 'DONE-CV %s %s\n' "$run_name" "$(date --iso-8601=seconds)"
}

archive_and_remove_bank() {
  local bank_name="$1"
  local bank_dir="$bank_root/$bank_name"
  local resolved
  resolved="$(realpath -e "$bank_dir")"
  case "$resolved" in
    "$bank_root"/phase2_*) ;;
    *)
      printf 'Refusing to remove unexpected path: %s\n' "$resolved" >&2
      exit 4
      ;;
  esac
  mkdir -p "$manifest_root/$bank_name"
  cp "$bank_dir/metadata.csv" "$bank_dir/feature_bank_config.json" "$manifest_root/$bank_name/"
  rm -rf -- "$resolved"
  sync
}

screen_backbone() {
  local bank_name="$1"
  local model_name="$2"
  local image_size="$3"
  local extract_batch="$4"
  local run_gated="$5"
  local run_viewgated="$6"
  local run_lodo_gated="$7"
  local run_lodo_viewgated="$8"
  local bank_dir="$bank_root/$bank_name"

  mkdir -p "$bank_dir"
  printf 'START-BANK %s model=%s %s\n' "$bank_name" "$model_name" "$(date --iso-8601=seconds)"
  "$python_bin" "$extractor" \
    --registry-csv "$registry" \
    --model-name "$model_name" \
    --image-size "$image_size" \
    --views whole,crop \
    --output-dir "$bank_dir" \
    --batch-size "$extract_batch" \
    --num-workers 4
  printf 'DONE-BANK %s %s\n' "$bank_name" "$(date --iso-8601=seconds)"

  run_cv "$run_gated" "$bank_dir" fivefold gated
  run_cv "$run_viewgated" "$bank_dir" fivefold view_gated
  run_cv "$run_lodo_gated" "$bank_dir" source_lodo gated
  run_cv "$run_lodo_viewgated" "$bank_dir" source_lodo view_gated
  archive_and_remove_bank "$bank_name"
}

screen_backbone \
  phase2_dinov3b_whole_crop_256_internal \
  vit_base_patch16_dinov3.lvd1689m 256 4 \
  276_dinov3b_dense_gated_cw_20260711 \
  277_dinov3b_dense_viewgated_cw_20260711 \
  278_dinov3b_dense_gated_source_lodo_cw_20260711 \
  279_dinov3b_dense_viewgated_source_lodo_cw_20260711

screen_backbone \
  phase2_dinov3l_standard_whole_crop_352_internal \
  vit_large_patch16_dinov3.lvd1689m 352 2 \
  280_dinov3l_standard_dense_gated_cw_20260711 \
  281_dinov3l_standard_dense_viewgated_cw_20260711 \
  282_dinov3l_standard_dense_gated_source_lodo_cw_20260711 \
  283_dinov3l_standard_dense_viewgated_source_lodo_cw_20260711

screen_backbone \
  phase2_siglipb_whole_crop_384_internal \
  vit_base_patch16_siglip_384.v2_webli 384 4 \
  284_siglipb_dense_gated_cw_20260711 \
  285_siglipb_dense_viewgated_cw_20260711 \
  286_siglipb_dense_gated_source_lodo_cw_20260711 \
  287_siglipb_dense_viewgated_source_lodo_cw_20260711

screen_backbone \
  phase2_siglip_so400m_whole_crop_378_internal \
  vit_so400m_patch14_siglip_378.v2_webli 378 2 \
  288_siglip_so400m_dense_gated_cw_20260711 \
  289_siglip_so400m_dense_viewgated_cw_20260711 \
  290_siglip_so400m_dense_gated_source_lodo_cw_20260711 \
  291_siglip_so400m_dense_viewgated_source_lodo_cw_20260711

screen_backbone \
  phase2_vitaminl_whole_crop_384_internal \
  local_vitamin_large_384 384 2 \
  292_vitaminl_dense_gated_cw_20260711 \
  293_vitaminl_dense_viewgated_cw_20260711 \
  294_vitaminl_dense_gated_source_lodo_cw_20260711 \
  295_vitaminl_dense_viewgated_source_lodo_cw_20260711

screen_backbone \
  phase2_siglipl_whole_crop_512_internal \
  vit_large_patch16_siglip_512.v2_webli 512 2 \
  296_siglipl512_dense_gated_cw_20260711 \
  297_siglipl512_dense_viewgated_cw_20260711 \
  298_siglipl512_dense_gated_source_lodo_cw_20260711 \
  299_siglipl512_dense_viewgated_source_lodo_cw_20260711

df -h /
