#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
trainer="/root/run_task7_dense_feature_cv_20260711.py"
split="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
output_root="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_boundary_aux_screen"
bank_dir="/root/thymic_feature_banks_20260711/phase2_siglipl512_whole_crop_quadrants6_512_internal"
status_file="${output_root}/PHASE2_SIGLIPL512_BOUNDARY_AUX.status"

mkdir -p "${output_root}"

write_status() {
  rm -f -- "${status_file}"
  printf '%s\n' "$1" > "${status_file}"
}

on_exit() {
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    write_status complete
  else
    write_status "failed rc=${rc}"
  fi
}
trap on_exit EXIT
write_status running

run_cv() {
  local run_id="$1"
  local tag="$2"
  local split_mode="$3"
  shift 3
  local output_dir="${output_root}/${run_id}_siglipl512_localpyramid6_${tag}_${split_mode}_cw_20260711"
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
    --pooling gated \
    --class-weighting \
    --epochs 80 \
    --patience 12 \
    --batch-size 8 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --device cuda \
    --no-load-features-to-ram \
    "$@"
}

run_group() {
  local pids=()
  run_cv 364 boundary_aux fivefold \
    --boundary-loss-weight 0.20 \
    --boundary-triplet-weight 0.05 &
  pids+=("$!")
  run_cv 366 boundary_expert_light fivefold \
    --expert-mode boundary \
    --boundary-loss-weight 0.20 \
    --boundary-relevance-loss-weight 0.10 \
    --boundary-fusion-alpha 0.25 \
    --boundary-triplet-weight 0.05 &
  pids+=("$!")
  run_cv 368 subtype_aux_light fivefold \
    --subtype-loss-weight 0.10 &
  pids+=("$!")
  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  return "${rc}"
}

run_lodo_group() {
  local pids=()
  run_cv 365 boundary_aux source_lodo \
    --boundary-loss-weight 0.20 \
    --boundary-triplet-weight 0.05 &
  pids+=("$!")
  run_cv 367 boundary_expert_light source_lodo \
    --expert-mode boundary \
    --boundary-loss-weight 0.20 \
    --boundary-relevance-loss-weight 0.10 \
    --boundary-fusion-alpha 0.25 \
    --boundary-triplet-weight 0.05 &
  pids+=("$!")
  run_cv 369 subtype_aux_light source_lodo \
    --subtype-loss-weight 0.10 &
  pids+=("$!")
  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
  done
  return "${rc}"
}

run_group
run_lodo_group

write_status complete
trap - EXIT
