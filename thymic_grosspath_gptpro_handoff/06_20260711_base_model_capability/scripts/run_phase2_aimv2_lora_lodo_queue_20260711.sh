#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
runner="/root/run_task7_lora_dense_finetune_20260711.py"
registry="/root/thymic_task7_internal_registry_cached_max2048_20260711.csv"
split="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
output_root="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_lora_source_lodo"
log_root="/root/thymic_feature_banks_20260711/phase2_lora_source_lodo_logs"
status_file="$output_root/PHASE2_LORA_SOURCE_LODO.status"

mkdir -p "$output_root" "$log_root" /root/thymic_crop_cache_phase2_max2048
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export THYMIC_CROP_CACHE_DIR=/root/thymic_crop_cache_phase2_max2048

on_exit() {
  rc=$?
  if [[ $rc -eq 0 ]]; then
    printf 'complete\n' > "$status_file"
  else
    printf 'failed rc=%s\n' "$rc" > "$status_file"
  fi
}
trap on_exit EXIT
printf 'running\n' > "$status_file"

common=(
  --registry-csv "$registry"
  --split-csv "$split"
  --split-mode source_lodo
  --fold all
  --model-name aimv2_large_patch14_336.apple_pt
  --input-variant whole_plus_crop
  --image-size 336
  --batch-size 8
  --num-workers 8
  --device cuda
  --seed 20260711
  --head-lr 2e-4
  --lora-lr 2e-5
  --weight-decay 1e-4
  --dropout 0.25
  --patience 4
  --label-smoothing 0.02
  --selection-metric balanced_accuracy
  --pooling gated
  --hidden-dim 256
  --attention-dim 128
  --lora-rank 8
  --lora-alpha 16
  --lora-last-blocks 2
  --lora-targets qkv,proj
  --train-final-norm
  --amp
  --amp-dtype bfloat16
)

run_experiment() {
  local run_name="$1"
  shift
  local output_dir="$output_root/$run_name"
  local log_file="$log_root/$run_name.log"
  if [[ -s "$output_dir/oof_metrics.csv" ]]; then
    printf 'SKIP complete %s\n' "$run_name"
    return
  fi
  printf 'START %s %s\n' "$run_name" "$(date --iso-8601=seconds)"
  "$python_bin" "$runner" "${common[@]}" --output-dir "$output_dir" "$@" 2>&1 | tee "$log_file"
  printf 'DONE %s %s\n' "$run_name" "$(date --iso-8601=seconds)"
}

run_experiment \
  271_aimv2l_lora_source_lodo_stylelight_baseline_cw_20260711 \
  --epochs 12 --aug-profile style_light --train-sampler none --class-weighting

run_experiment \
  272_aimv2l_lora_source_lodo_stylelight_sourcebalanced_20260711 \
  --epochs 12 --aug-profile style_light --train-sampler source_label_balanced --no-class-weighting

run_experiment \
  273_aimv2l_lora_source_lodo_last4_sourcebalanced_20260711 \
  --epochs 16 --patience 5 --lora-last-blocks 4 --lora-lr 1e-5 \
  --aug-profile style_light --train-sampler source_label_balanced --no-class-weighting

run_experiment \
  274_aimv2l_lora_source_lodo_domainrobust_sourcebalanced_20260711 \
  --epochs 12 --aug-profile domain_robust --train-sampler source_label_balanced --no-class-weighting

run_experiment \
  275_aimv2l_lora_source_lodo_viewgated_sourcebalanced_20260711 \
  --epochs 12 --pooling view_gated --aug-profile style_light \
  --train-sampler source_label_balanced --no-class-weighting
