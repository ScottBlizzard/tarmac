#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
extractor="/root/extract_task7_dense_token_bank_20260711.py"
trainer="/root/run_task7_dense_feature_cv_20260711.py"
registry="/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/registry/task7_four_domain_master_registry.csv"
split="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
experiment_root="/workspace/thymic_project/experiments/base_model_capability_20260711"
output_root="${experiment_root}/phase2_siglipl512_local_pyramid_screen"
manifest_root="${output_root}/feature_bank_manifests"
bank_root="/root/thymic_feature_banks_20260711"
bank_name="phase2_siglipl512_whole_crop_quadrants6_512_internal"
bank_dir="${bank_root}/${bank_name}"
upstream_status="${experiment_root}/phase2_subtype_boundary_contrastive_lodo/PHASE2_SUBTYPE_BOUNDARY_CONTRASTIVE.status"
status_file="${output_root}/PHASE2_SIGLIPL512_LOCAL_PYRAMID.status"

mkdir -p "${output_root}" "${manifest_root}"
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
write_status waiting_for_subtype_boundary_contrastive

while true; do
  upstream="$(cat "${upstream_status}" 2>/dev/null || true)"
  case "${upstream}" in
    complete*) break ;;
    failed*)
      printf 'Upstream subtype-boundary contrastive queue failed: %s\n' "${upstream}" >&2
      exit 3
      ;;
    *) sleep 30 ;;
  esac
done
write_status running

if [[ ! -s "${bank_dir}/feature_bank_config.json" ]] || ! grep -q '"complete": true' "${bank_dir}/feature_bank_config.json"; then
  smoke_dir="${bank_root}/smoke_siglipl512_local_pyramid"
  rm -rf -- "${smoke_dir}"
  "${python_bin}" "${extractor}" \
    --registry-csv "${registry}" \
    --model-name vit_large_patch16_siglip_512.v2_webli \
    --image-size 512 \
    --views whole,crop,crop_q0,crop_q1,crop_q2,crop_q3 \
    --output-dir "${smoke_dir}" \
    --batch-size 1 \
    --num-workers 0 \
    --max-cases 2
  rm -rf -- "${smoke_dir}"

  mkdir -p "${bank_dir}"
  "${python_bin}" "${extractor}" \
    --registry-csv "${registry}" \
    --model-name vit_large_patch16_siglip_512.v2_webli \
    --image-size 512 \
    --views whole,crop,crop_q0,crop_q1,crop_q2,crop_q3 \
    --output-dir "${bank_dir}" \
    --batch-size 1 \
    --num-workers 4
fi

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

for specification in \
  "347 gated fivefold" \
  "348 gated source_lodo" \
  "349 view_gated fivefold" \
  "350 view_gated source_lodo" \
  "351 gated_stats fivefold" \
  "352 gated_stats source_lodo" \
  "353 view_gated_stats fivefold" \
  "354 view_gated_stats source_lodo" \
  "355 spatial_pyramid fivefold" \
  "356 spatial_pyramid source_lodo"
do
  read -r run_id pooling split_mode <<< "${specification}"
  run_cv "${run_id}" "${pooling}" "${split_mode}"
done

resolved="$(realpath -e "${bank_dir}")"
case "${resolved}" in
  "${bank_root}"/phase2_siglipl512_whole_crop_quadrants6_512_internal) ;;
  *)
    printf 'Refusing to remove unexpected feature bank: %s\n' "${resolved}" >&2
    exit 4
    ;;
esac
mkdir -p "${manifest_root}/${bank_name}"
cp "${bank_dir}/metadata.csv" "${bank_dir}/feature_bank_config.json" "${manifest_root}/${bank_name}/"
rm -rf -- "${resolved}"
sync
df -h /
