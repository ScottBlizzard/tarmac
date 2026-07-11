#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
runner="/root/run_task7_lora_dense_finetune_20260711.py"
registry="/root/thymic_task7_internal_registry_cached_max2048_20260711.csv"
split="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
output_root="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_lora_contrastive_source_lodo"
log_root="/root/thymic_feature_banks_20260711/phase2_lora_contrastive_logs"
upstream_status="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl_lora_source_lodo/PHASE2_SIGLIPL_LORA.status"
status_file="$output_root/PHASE2_LORA_CONTRASTIVE.status"

mkdir -p "$output_root" "$log_root" /root/thymic_crop_cache_phase2_max2048
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export THYMIC_CROP_CACHE_DIR=/root/thymic_crop_cache_phase2_max2048

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
write_status waiting_for_siglipl_lora

while true; do
  upstream="$(cat "$upstream_status" 2>/dev/null || true)"
  case "$upstream" in
    complete*) break ;;
    failed*)
      printf 'Upstream SigLIP-L LoRA queue failed: %s\n' "$upstream" >&2
      exit 3
      ;;
    *) sleep 30 ;;
  esac
done
write_status running

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
  --epochs 12
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
  --aug-profile style_light
  --no-class-weighting
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

smoke_dir="$output_root/smoke/contrastive_fold1"
"$python_bin" "$runner" "${common[@]}" \
  --output-dir "$smoke_dir" \
  --fold 1 \
  --epochs 1 \
  --patience 1 \
  --num-workers 0 \
  --train-sampler source_label_balanced \
  --supervised-contrastive-mode cross_source \
  --supervised-contrastive-weight 0.05 \
  --max-train-batches 1 \
  --max-eval-batches 1

run_experiment \
  300_aimv2l_lora_binary_supcon005_source_lodo_20260711 \
  --train-sampler source_label_balanced \
  --supervised-contrastive-mode binary \
  --supervised-contrastive-weight 0.05

run_experiment \
  301_aimv2l_lora_crosssource_supcon005_source_lodo_20260711 \
  --train-sampler source_label_balanced \
  --supervised-contrastive-mode cross_source \
  --supervised-contrastive-weight 0.05

run_experiment \
  302_aimv2l_lora_sourcesubtypebalanced_source_lodo_20260711 \
  --train-sampler source_subtype_balanced \
  --supervised-contrastive-weight 0

run_experiment \
  303_aimv2l_lora_sourcesubtype_crosssource_supcon005_lodo_20260711 \
  --train-sampler source_subtype_balanced \
  --supervised-contrastive-mode cross_source \
  --supervised-contrastive-weight 0.05
