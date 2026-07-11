#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
extractor="/root/extract_task7_dense_token_bank_20260711.py"
trainer="/root/run_task7_dense_feature_cv_20260711.py"
registry="/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/registry/task7_four_domain_master_registry.csv"
split="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
experiment_root="/workspace/thymic_project/experiments/base_model_capability_20260711"
output_root="${experiment_root}/phase2_quality_preprocessing_lodo"
bank_dir="/dev/shm/thymic_feature_banks_20260711/qkvb_crop_enhance3_352_internal"
upstream_status="${experiment_root}/phase2_siglipl512_local_pyramid_screen/PHASE2_SIGLIPL512_LOCAL_PYRAMID.status"
status_file="${output_root}/PHASE2_QUALITY_PREPROCESSING_LODO.status"
run_dir="${output_root}/357_qkvb_crop_enhance3_viewgated_source_lodo_cw_20260711"

mkdir -p "${output_root}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
write_status waiting_for_siglipl512_local_pyramid

while true; do
  upstream="$(cat "${upstream_status}" 2>/dev/null || true)"
  case "${upstream}" in
    complete*) break ;;
    failed*)
      printf 'Upstream local-pyramid queue failed: %s\n' "${upstream}" >&2
      exit 3
      ;;
    *) sleep 30 ;;
  esac
done
write_status running

if [[ ! -s "${bank_dir}/feature_bank_config.json" ]] || ! grep -q '"complete": true' "${bank_dir}/feature_bank_config.json"; then
  mkdir -p "${bank_dir}"
  "${python_bin}" "${extractor}" \
    --registry-csv "${registry}" \
    --model-name vit_large_patch16_dinov3_qkvb.lvd1689m \
    --image-size 352 \
    --views crop,crop_autocontrast,crop_unsharp \
    --output-dir "${bank_dir}" \
    --batch-size 4 \
    --num-workers 4
fi

if [[ ! -s "${run_dir}/oof_metrics.csv" ]]; then
  "${python_bin}" "${trainer}" \
    --feature-bank-dir "${bank_dir}" \
    --output-dir "${run_dir}" \
    --split-csv "${split}" \
    --split-mode source_lodo \
    --pooling view_gated \
    --class-weighting \
    --epochs 80 \
    --patience 12 \
    --batch-size 24 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --device cuda \
    --no-load-features-to-ram
fi

mkdir -p "${output_root}/feature_bank_manifest"
cp "${bank_dir}/metadata.csv" "${bank_dir}/feature_bank_config.json" "${output_root}/feature_bank_manifest/"
resolved="$(realpath -e "${bank_dir}")"
case "${resolved}" in
  /dev/shm/thymic_feature_banks_20260711/qkvb_crop_enhance3_352_internal) ;;
  *)
    printf 'Refusing to remove unexpected feature bank: %s\n' "${resolved}" >&2
    exit 4
    ;;
esac
rm -rf -- "${resolved}"
