#!/usr/bin/env bash
set -euo pipefail

PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"
TRAIN="/root/run_task7_dense_feature_cv_20260711.py"
PROJECT_ROOT="/workspace/thymic_project"
EXP_ROOT="${PROJECT_ROOT}/experiments/base_model_capability_20260711"
OUT_ROOT="${EXP_ROOT}/phase2_structured_pooling_screen"
UPSTREAM_STATUS="${EXP_ROOT}/phase2_siglipl512_lora_source_lodo/PHASE2_SIGLIPL512_LORA.status"
STATUS="${OUT_ROOT}/PHASE2_STRUCTURED_POOLING.status"

mkdir -p "${OUT_ROOT}"

write_status() {
  rm -f -- "${STATUS}"
  printf '%s\n' "$1" > "${STATUS}"
}

write_status waiting_for_siglipl512_lora
while true; do
  upstream="$(cat "${UPSTREAM_STATUS}" 2>/dev/null || true)"
  if [[ "${upstream}" == "complete" ]]; then
    break
  fi
  if [[ "${upstream}" == failed* ]]; then
    printf '[warning] SigLIP-L@512 LoRA failed; continuing frozen-feature screen: %s\n' "${upstream}"
    break
  fi
  sleep 60
done

write_status running

run_cv() {
  local run_id="$1"
  local backbone="$2"
  local bank="$3"
  local pooling="$4"
  local split_mode="$5"
  local output="${OUT_ROOT}/${run_id}_${backbone}_${pooling}_${split_mode}_cw_20260711"
  if [[ -s "${output}/oof_metrics.csv" ]]; then
    printf '[skip] %s\n' "${output##*/}"
    return 0
  fi
  printf '[run] %s\n' "${output##*/}"
  "${PYTHON}" "${TRAIN}" \
    --feature-bank-dir "${bank}" \
    --output-dir "${output}" \
    --split-mode "${split_mode}" \
    --pooling "${pooling}" \
    --class-weighting \
    --epochs 80 \
    --patience 12 \
    --batch-size 24 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --device cuda
}

AIM_BANK="/root/thymic_feature_banks_20260711/aimv2l_whole_crop_336_internal"
SIGLIP_BANK="/root/thymic_feature_banks_20260711/siglipl_whole_crop_384_internal"
QKVB_BANK="/root/thymic_feature_banks_20260711/qkvb_whole_crop_352_internal"

for specification in \
  "321 aimv2l ${AIM_BANK} gated_stats fivefold" \
  "322 aimv2l ${AIM_BANK} gated_stats source_lodo" \
  "323 aimv2l ${AIM_BANK} view_gated_stats fivefold" \
  "324 aimv2l ${AIM_BANK} view_gated_stats source_lodo" \
  "325 aimv2l ${AIM_BANK} spatial_pyramid fivefold" \
  "326 aimv2l ${AIM_BANK} spatial_pyramid source_lodo" \
  "327 siglipl384 ${SIGLIP_BANK} gated_stats fivefold" \
  "328 siglipl384 ${SIGLIP_BANK} gated_stats source_lodo" \
  "329 siglipl384 ${SIGLIP_BANK} view_gated_stats fivefold" \
  "330 siglipl384 ${SIGLIP_BANK} view_gated_stats source_lodo" \
  "331 siglipl384 ${SIGLIP_BANK} spatial_pyramid fivefold" \
  "332 siglipl384 ${SIGLIP_BANK} spatial_pyramid source_lodo" \
  "333 qkvb ${QKVB_BANK} gated_stats fivefold" \
  "334 qkvb ${QKVB_BANK} gated_stats source_lodo" \
  "335 qkvb ${QKVB_BANK} view_gated_stats fivefold" \
  "336 qkvb ${QKVB_BANK} view_gated_stats source_lodo" \
  "337 qkvb ${QKVB_BANK} spatial_pyramid fivefold" \
  "338 qkvb ${QKVB_BANK} spatial_pyramid source_lodo"
do
  read -r run_id backbone bank pooling split_mode <<< "${specification}"
  run_cv "${run_id}" "${backbone}" "${bank}" "${pooling}" "${split_mode}"
done

write_status complete
