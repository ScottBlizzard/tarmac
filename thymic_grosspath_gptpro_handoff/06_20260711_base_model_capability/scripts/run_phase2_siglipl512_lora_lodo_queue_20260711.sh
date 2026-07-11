#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
runner="/root/run_task7_lora_dense_finetune_20260711.py"
registry="/root/thymic_task7_internal_registry_cached_max2048_20260711.csv"
split="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
experiment_root="/workspace/thymic_project/experiments/base_model_capability_20260711"
output_root="${experiment_root}/phase2_siglipl512_lora_source_lodo"
log_root="/root/thymic_feature_banks_20260711/phase2_siglipl512_lora_logs"
upstream_status="${experiment_root}/phase2_lora_contrastive_source_lodo/PHASE2_LORA_CONTRASTIVE.status"
status_file="${output_root}/PHASE2_SIGLIPL512_LORA.status"

mkdir -p "${output_root}" "${log_root}" /root/thymic_crop_cache_phase2_max2048
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export THYMIC_CROP_CACHE_DIR=/root/thymic_crop_cache_phase2_max2048

write_status() {
  rm -f -- "${status_file}"
  printf '%s\n' "$1" > "${status_file}"
}

on_exit() {
  rc=$?
  if [[ ${rc} -eq 0 ]]; then
    write_status complete
  else
    write_status "failed rc=${rc}"
  fi
}
trap on_exit EXIT
write_status waiting_for_lora_contrastive

while true; do
  upstream="$(cat "${upstream_status}" 2>/dev/null || true)"
  case "${upstream}" in
    complete*) break ;;
    failed*)
      printf 'Upstream contrastive queue failed: %s\n' "${upstream}" >&2
      exit 3
      ;;
    *) sleep 30 ;;
  esac
done
write_status running

common=(
  --registry-csv "${registry}"
  --split-csv "${split}"
  --split-mode source_lodo
  --fold all
  --model-name vit_large_patch16_siglip_512.v2_webli
  --input-variant whole_plus_crop
  --image-size 512
  --batch-size 2
  --num-workers 4
  --device cuda
  --seed 20260711
  --head-lr 2e-4
  --weight-decay 1e-4
  --dropout 0.25
  --patience 4
  --label-smoothing 0.02
  --selection-metric balanced_accuracy
  --pooling gated
  --hidden-dim 256
  --attention-dim 128
  --lora-targets qkv,proj
  --aug-profile style_light
  --amp
  --amp-dtype bfloat16
)

run_experiment() {
  local run_name="$1"
  shift
  local output_dir="${output_root}/${run_name}"
  local log_file="${log_root}/${run_name}.log"
  if [[ -s "${output_dir}/oof_metrics.csv" ]]; then
    printf 'SKIP complete %s\n' "${run_name}"
    return
  fi
  printf 'START %s %s\n' "${run_name}" "$(date --iso-8601=seconds)"
  "${python_bin}" "${runner}" "${common[@]}" --output-dir "${output_dir}" "$@" 2>&1 | tee "${log_file}"
  printf 'DONE %s %s\n' "${run_name}" "$(date --iso-8601=seconds)"
}

smoke_dir="${output_root}/smoke/fold1"
"${python_bin}" "${runner}" "${common[@]}" \
  --output-dir "${smoke_dir}" \
  --fold 1 \
  --epochs 1 \
  --patience 1 \
  --num-workers 0 \
  --lora-rank 4 \
  --lora-alpha 8 \
  --lora-last-blocks 1 \
  --lora-lr 5e-6 \
  --no-train-final-norm \
  --class-weighting \
  --max-train-batches 1 \
  --max-eval-batches 1

run_experiment \
  339_siglipl512_lora_last1_conservative_source_lodo_20260711 \
  --epochs 12 --lora-rank 4 --lora-alpha 8 --lora-last-blocks 1 --lora-lr 5e-6 \
  --no-train-final-norm --train-sampler none --class-weighting

run_experiment \
  340_siglipl512_lora_last2_baseline_source_lodo_20260711 \
  --epochs 12 --lora-rank 8 --lora-alpha 16 --lora-last-blocks 2 --lora-lr 5e-6 \
  --train-final-norm --train-sampler none --class-weighting

run_experiment \
  341_siglipl512_lora_last2_sourcebalanced_lodo_20260711 \
  --epochs 12 --lora-rank 8 --lora-alpha 16 --lora-last-blocks 2 --lora-lr 5e-6 \
  --train-final-norm --train-sampler source_label_balanced --no-class-weighting

run_experiment \
  342_siglipl512_lora_last4_low_lr_sourcebalanced_lodo_20260711 \
  --epochs 16 --patience 5 --lora-rank 8 --lora-alpha 16 --lora-last-blocks 4 --lora-lr 2e-6 \
  --train-final-norm --train-sampler source_label_balanced --no-class-weighting
