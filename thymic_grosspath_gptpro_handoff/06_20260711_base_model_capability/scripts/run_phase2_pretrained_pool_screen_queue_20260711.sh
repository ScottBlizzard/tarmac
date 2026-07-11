#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
extractor="/root/extract_task7_dense_token_bank_20260711.py"
trainer="/root/run_task7_dense_feature_cv_20260711.py"
registry="/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/registry/task7_four_domain_master_registry.csv"
split="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
bank_root="/root/thymic_feature_banks_20260711"
output_root="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_pretrained_pool_screen"
manifest_root="$output_root/feature_bank_manifests"
upstream_status="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_cached_backbone_screen/PHASE2_CACHED_BACKBONE_SCREEN.status"
status_file="$output_root/PHASE2_PRETRAINED_POOL_SCREEN.status"

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
write_status waiting_for_dense_backbone_screen

while true; do
  upstream="$(cat "$upstream_status" 2>/dev/null || true)"
  case "$upstream" in
    complete*) break ;;
    failed*)
      printf 'Upstream dense-backbone screen failed: %s\n' "$upstream" >&2
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

screen_model() {
  local bank_name="$1"
  local model_name="$2"
  local image_size="$3"
  local extract_batch="$4"
  local run_mean="$5"
  local run_view="$6"
  local run_lodo_mean="$7"
  local run_lodo_view="$8"
  local bank_dir="$bank_root/$bank_name"

  mkdir -p "$bank_dir"
  printf 'START-BANK %s model=%s %s\n' "$bank_name" "$model_name" "$(date --iso-8601=seconds)"
  "$python_bin" "$extractor" \
    --registry-csv "$registry" \
    --model-name "$model_name" \
    --feature-mode pooled \
    --image-size "$image_size" \
    --views whole,crop \
    --output-dir "$bank_dir" \
    --batch-size "$extract_batch" \
    --num-workers 4
  printf 'DONE-BANK %s %s\n' "$bank_name" "$(date --iso-8601=seconds)"

  run_cv "$run_mean" "$bank_dir" fivefold mean
  run_cv "$run_view" "$bank_dir" fivefold view_gated
  run_cv "$run_lodo_mean" "$bank_dir" source_lodo mean
  run_cv "$run_lodo_view" "$bank_dir" source_lodo view_gated

  resolved="$(realpath -e "$bank_dir")"
  case "$resolved" in
    "$bank_root"/phase2_*_pretrained_pool_*) ;;
    *)
      printf 'Refusing to remove unexpected path: %s\n' "$resolved" >&2
      exit 4
      ;;
  esac
  mkdir -p "$manifest_root/$bank_name"
  cp "$bank_dir/metadata.csv" "$bank_dir/feature_bank_config.json" "$manifest_root/$bank_name/"
  rm -rf -- "$resolved"
}

screen_model \
  phase2_siglipl_pretrained_pool_384_internal \
  vit_large_patch16_siglip_384.v2_webli 384 4 \
  309_siglipl_pretrained_pool_mean_cw_20260711 \
  310_siglipl_pretrained_pool_viewgated_cw_20260711 \
  311_siglipl_pretrained_pool_mean_source_lodo_cw_20260711 \
  312_siglipl_pretrained_pool_viewgated_lodo_cw_20260711

screen_model \
  phase2_siglipso400m_pretrained_pool_378_internal \
  vit_so400m_patch14_siglip_378.v2_webli 378 2 \
  313_siglipso400m_pretrained_pool_mean_cw_20260711 \
  314_siglipso400m_pretrained_pool_viewgated_cw_20260711 \
  315_siglipso400m_pretrained_pool_mean_source_lodo_cw_20260711 \
  316_siglipso400m_pretrained_pool_viewgated_lodo_cw_20260711

screen_model \
  phase2_siglipl512_pretrained_pool_512_internal \
  vit_large_patch16_siglip_512.v2_webli 512 2 \
  317_siglipl512_pretrained_pool_mean_cw_20260711 \
  318_siglipl512_pretrained_pool_viewgated_cw_20260711 \
  319_siglipl512_pretrained_pool_mean_source_lodo_cw_20260711 \
  320_siglipl512_pretrained_pool_viewgated_lodo_cw_20260711

sync
df -h /
