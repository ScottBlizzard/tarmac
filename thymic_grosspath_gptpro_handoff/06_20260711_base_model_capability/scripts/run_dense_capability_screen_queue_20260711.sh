#!/usr/bin/env bash
set -u

PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"
SCRIPT="${PROJECT_ROOT}/scripts/run_task7_dense_feature_cv_20260711.py"
EXTRACT="${PROJECT_ROOT}/scripts/extract_task7_dense_token_bank_20260711.py"
OUT_ROOT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_cv_runs"
BANK_ROOT="/root/thymic_feature_banks_20260711"
mkdir -p "${OUT_ROOT}" "${BANK_ROOT}"

run_cv() {
  local tag="$1"
  local bank="$2"
  shift 2
  local out_dir="${OUT_ROOT}/${tag}"
  if [[ -s "${out_dir}/oof_metrics.csv" ]]; then
    echo "[skip-cv] ${tag}"
    return 0
  fi
  mkdir -p "${out_dir}"
  echo "[run-cv] ${tag}"
  if "${PYTHON}" "${SCRIPT}" \
    --feature-bank-dir "${bank}" \
    --output-dir "${out_dir}" \
    --epochs 80 \
    --patience 12 \
    --batch-size 24 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --device cuda \
    "$@"; then
    echo "[done-cv] ${tag}"
  else
    echo "[failed-cv] ${tag}" >&2
  fi
}

extract_bank() {
  local bank_name="$1"
  local model_name="$2"
  local image_size="$3"
  local views="$4"
  local bank_dir="${BANK_ROOT}/${bank_name}"
  if [[ -s "${bank_dir}/feature_bank_config.json" ]] && grep -q '"complete": true' "${bank_dir}/feature_bank_config.json"; then
    echo "[skip-bank] ${bank_name}"
    return 0
  fi
  mkdir -p "${bank_dir}"
  echo "[extract-bank] ${bank_name} model=${model_name}"
  if "${PYTHON}" "${EXTRACT}" \
    --model-name "${model_name}" \
    --image-size "${image_size}" \
    --views "${views}" \
    --output-dir "${bank_dir}" \
    --batch-size 4 \
    --num-workers 4; then
    echo "[done-bank] ${bank_name}"
  else
    echo "[failed-bank] ${bank_name}" >&2
    return 1
  fi
}

QKVB_BANK="${BANK_ROOT}/qkvb_whole_crop_352_internal"

run_cv "206_qkvb_dense_mean_cw_20260711" "${QKVB_BANK}" \
  --pooling mean --class-weighting

run_cv "207_qkvb_dense_gated_cw_20260711" "${QKVB_BANK}" \
  --pooling gated --class-weighting

run_cv "208_qkvb_dense_viewgated_cw_20260711" "${QKVB_BANK}" \
  --pooling view_gated --class-weighting

run_cv "209_qkvb_dense_gated_concept_subtype_proto_20260711" "${QKVB_BANK}" \
  --pooling gated --class-weighting \
  --concept-loss-weight 0.20 \
  --subtype-loss-weight 0.15 \
  --prototype-loss-weight 0.10

run_cv "210_qkvb_dense_boundary_hardpair_20260711" "${QKVB_BANK}" \
  --pooling gated --expert-mode boundary --class-weighting \
  --subtype-loss-weight 0.25 \
  --boundary-loss-weight 0.35 \
  --boundary-triplet-weight 0.10 \
  --subtype-balanced-sampler

run_cv "211_qkvb_dense_viewgated_moe_20260711" "${QKVB_BANK}" \
  --pooling view_gated --expert-mode moe --class-weighting \
  --subtype-loss-weight 0.15 \
  --prototype-loss-weight 0.10 \
  --moe-specialist-weight 0.25 \
  --moe-balance-weight 0.02

run_cv "212_qkvb_dense_gated_rex_20260711" "${QKVB_BANK}" \
  --pooling gated --risk-objective rex --rex-weight 1.0 --class-weighting

run_cv "213_qkvb_dense_gated_groupdro_20260711" "${QKVB_BANK}" \
  --pooling gated --risk-objective group_dro --group-dro-eta 0.05 --class-weighting

extract_bank "siglipl_whole_crop_384_internal" \
  "vit_large_patch16_siglip_384.v2_webli" 384 "whole,crop"
SIGLIP_BANK="${BANK_ROOT}/siglipl_whole_crop_384_internal"
run_cv "214_siglipl_dense_mean_cw_20260711" "${SIGLIP_BANK}" --pooling mean --class-weighting
run_cv "215_siglipl_dense_gated_cw_20260711" "${SIGLIP_BANK}" --pooling gated --class-weighting

extract_bank "eva02l_whole_crop_336_internal" \
  "eva02_large_patch14_clip_336.merged2b" 336 "whole,crop"
EVA_BANK="${BANK_ROOT}/eva02l_whole_crop_336_internal"
run_cv "216_eva02l_dense_mean_cw_20260711" "${EVA_BANK}" --pooling mean --class-weighting
run_cv "217_eva02l_dense_gated_cw_20260711" "${EVA_BANK}" --pooling gated --class-weighting

extract_bank "aimv2l_whole_crop_336_internal" \
  "aimv2_large_patch14_336.apple_pt" 336 "whole,crop"
AIM_BANK="${BANK_ROOT}/aimv2l_whole_crop_336_internal"
run_cv "218_aimv2l_dense_mean_cw_20260711" "${AIM_BANK}" --pooling mean --class-weighting
run_cv "219_aimv2l_dense_gated_cw_20260711" "${AIM_BANK}" --pooling gated --class-weighting

extract_bank "convnextb_dinov3_whole_crop_352_internal" \
  "convnext_base.dinov3_lvd1689m" 352 "whole,crop"
CONVNEXT_BANK="${BANK_ROOT}/convnextb_dinov3_whole_crop_352_internal"
run_cv "220_convnextb_dinov3_dense_mean_cw_20260711" "${CONVNEXT_BANK}" --pooling mean --class-weighting
run_cv "221_convnextb_dinov3_dense_gated_cw_20260711" "${CONVNEXT_BANK}" --pooling gated --class-weighting

extract_bank "qkvb_quality4_352_internal" \
  "vit_large_patch16_dinov3_qkvb.lvd1689m" 352 "whole,crop,grayworld,crop_grayworld"
QUALITY_BANK="${BANK_ROOT}/qkvb_quality4_352_internal"
run_cv "222_qkvb_quality4_viewgated_cw_20260711" "${QUALITY_BANK}" \
  --pooling view_gated --class-weighting

echo "[queue-complete] dense capability screen"
