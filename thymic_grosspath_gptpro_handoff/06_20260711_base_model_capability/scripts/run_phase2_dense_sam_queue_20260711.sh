#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
trainer="/root/run_task7_dense_feature_cv_20260711.py"
experiment_root="/workspace/thymic_project/experiments/base_model_capability_20260711"
output_root="${experiment_root}/phase2_dense_sam_screen"
upstream_status="${experiment_root}/phase2_quality_preprocessing_lodo/PHASE2_QUALITY_PREPROCESSING_LODO.status"
status_file="${output_root}/PHASE2_DENSE_SAM.status"

mkdir -p "${output_root}"

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
write_status waiting_for_quality_preprocessing_lodo

while true; do
  upstream="$(cat "${upstream_status}" 2>/dev/null || true)"
  case "${upstream}" in
    complete*) break ;;
    failed*)
      printf 'Upstream quality-preprocessing queue failed: %s\n' "${upstream}" >&2
      exit 3
      ;;
    *) sleep 30 ;;
  esac
done
write_status running

run_cv() {
  local run_id="$1"
  local backbone="$2"
  local bank="$3"
  local split_mode="$4"
  local output_dir="${output_root}/${run_id}_${backbone}_gated_sam005_${split_mode}_cw_20260711"
  if [[ -s "${output_dir}/oof_metrics.csv" ]]; then
    printf '[skip] %s\n' "${output_dir##*/}"
    return 0
  fi
  "${python_bin}" "${trainer}" \
    --feature-bank-dir "${bank}" \
    --output-dir "${output_dir}" \
    --split-mode "${split_mode}" \
    --pooling gated \
    --class-weighting \
    --sam-rho 0.05 \
    --epochs 80 \
    --patience 12 \
    --batch-size 24 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --device cuda \
    --no-load-features-to-ram
}

AIM_BANK="/root/thymic_feature_banks_20260711/aimv2l_whole_crop_336_internal"
SIGLIP_BANK="/root/thymic_feature_banks_20260711/siglipl_whole_crop_384_internal"
QKVB_BANK="/root/thymic_feature_banks_20260711/qkvb_whole_crop_352_internal"

run_group() {
  local pids=()
  local specification run_id backbone bank split_mode
  for specification in "$@"; do
    read -r run_id backbone bank split_mode <<< "${specification}"
    run_cv "${run_id}" "${backbone}" "${bank}" "${split_mode}" &
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
  "358 aimv2l ${AIM_BANK} fivefold" \
  "360 siglipl384 ${SIGLIP_BANK} fivefold" \
  "362 qkvb ${QKVB_BANK} fivefold"

run_group \
  "359 aimv2l ${AIM_BANK} source_lodo" \
  "361 siglipl384 ${SIGLIP_BANK} source_lodo" \
  "363 qkvb ${QKVB_BANK} source_lodo"

write_status complete
trap - EXIT
