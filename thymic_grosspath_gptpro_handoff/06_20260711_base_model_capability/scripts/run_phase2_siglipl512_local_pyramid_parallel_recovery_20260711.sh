#!/usr/bin/env bash
set -euo pipefail

parent_pid="${1:?usage: $0 ORIGINAL_QUEUE_PARENT_PID}"
python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
trainer="/root/run_task7_dense_feature_cv_20260711.py"
split="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
output_root="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen"
bank_dir="/root/thymic_feature_banks_20260711/phase2_siglipl512_whole_crop_quadrants6_512_internal"

resume_parent() {
  kill -CONT "${parent_pid}" 2>/dev/null || true
}
trap resume_parent EXIT

run_cv() {
  local run_id="$1"
  local pooling="$2"
  local split_mode="$3"
  local output_dir="${output_root}/${run_id}_siglipl512_localpyramid6_${pooling}_${split_mode}_cw_20260711"
  if [[ -s "${output_dir}/oof_metrics.csv" ]]; then
    printf '[skip] %s\n' "${output_dir##*/}"
    return 0
  fi
  printf '[run] %s\n' "${output_dir##*/}"
  "${python_bin}" "${trainer}" \
    --feature-bank-dir "${bank_dir}" \
    --output-dir "${output_dir}" \
    --split-csv "${split}" \
    --split-mode "${split_mode}" \
    --pooling "${pooling}" \
    --class-weighting \
    --epochs 80 \
    --patience 12 \
    --batch-size 8 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --device cuda \
    --no-load-features-to-ram
}

run_group() {
  local pids=()
  local specification
  local run_id pooling split_mode
  for specification in "$@"; do
    read -r run_id pooling split_mode <<< "${specification}"
    run_cv "${run_id}" "${pooling}" "${split_mode}" &
    pids+=("$!")
  done
  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  return "${rc}"
}

run_group \
  "348 gated source_lodo" \
  "349 view_gated fivefold" \
  "350 view_gated source_lodo"

run_group \
  "351 gated_stats fivefold" \
  "352 gated_stats source_lodo" \
  "353 view_gated_stats fivefold"

run_group \
  "354 view_gated_stats source_lodo" \
  "355 spatial_pyramid fivefold" \
  "356 spatial_pyramid source_lodo"

printf '[done] parallel recovery workers complete\n'
