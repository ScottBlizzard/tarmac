#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"
TRAIN="/root/thymic_queue_scripts_20260711/run_task7_dense_feature_cv_20260711.py"
EXTRACT="/root/thymic_queue_scripts_20260711/extract_task7_dense_token_bank_20260711.py"
SUMMARIZE="${PROJECT_ROOT}/scripts/summarize_dense_capability_screen_20260711.py"
QKVB_BANK="/root/thymic_feature_banks_20260711/qkvb_whole_crop_352_internal"
AIM_BANK="/root/thymic_feature_banks_20260711/aimv2l_whole_crop_336_internal"
DINOV2_BANK="/root/thymic_feature_banks_20260711/dinov2l_whole_crop_336_internal"
ENHANCE_BANK="/dev/shm/thymic_feature_banks_20260711/qkvb_crop_enhance3_352_internal"
CACHE_REGISTRY="/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv"
OUT_ROOT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_cv_runs"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[wait] high-resolution queue pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

run_cv() {
  local tag="$1"
  local feature_bank="$2"
  shift 2
  local out_dir="${OUT_ROOT}/${tag}"
  if [[ -s "${out_dir}/oof_metrics.csv" ]]; then
    echo "[skip] ${tag}"
    return 0
  fi
  echo "[run] ${tag}"
  "${PYTHON}" "${TRAIN}" \
    --feature-bank-dir "${feature_bank}" \
    --output-dir "${out_dir}" \
    --epochs 80 \
    --patience 12 \
    --batch-size 24 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --device cuda \
    "$@"
}

run_cv "259_qkvb_dense_viewgated_viewconsistency_20260711" "${QKVB_BANK}" \
  --pooling view_gated --class-weighting \
  --view-consistency-weight 0.10 \
  --view-supervision-weight 0.10

run_cv "260_qkvb_dense_gated_ordinalrisk_20260711" "${QKVB_BANK}" \
  --pooling gated --class-weighting --subtype-balanced-sampler \
  --subtype-loss-weight 0.25 \
  --ordinal-loss-weight 0.15 \
  --prototype-loss-weight 0.05 \
  --risk-from-subtype-alpha 0.25

run_cv "261_qkvb_dense_viewgated_ordinal_consistency_20260711" "${QKVB_BANK}" \
  --pooling view_gated --class-weighting --subtype-balanced-sampler \
  --subtype-loss-weight 0.20 \
  --ordinal-loss-weight 0.10 \
  --risk-from-subtype-alpha 0.20 \
  --view-consistency-weight 0.08 \
  --view-supervision-weight 0.08

run_cv "262_aimv2l_dense_viewgated_viewconsistency_20260711" "${AIM_BANK}" \
  --pooling view_gated --class-weighting \
  --view-consistency-weight 0.10 \
  --view-supervision-weight 0.10

run_cv "263_aimv2l_dense_gated_ordinal_consistency_20260711" "${AIM_BANK}" \
  --pooling gated --class-weighting --subtype-balanced-sampler \
  --subtype-loss-weight 0.20 \
  --ordinal-loss-weight 0.10 \
  --risk-from-subtype-alpha 0.20 \
  --view-consistency-weight 0.08 \
  --view-supervision-weight 0.08

run_cv "267_aimv2l_dense_gated_seed20260712_20260711" "${AIM_BANK}" \
  --pooling gated --class-weighting \
  --seed 20260712

run_cv "268_aimv2l_dense_gated_seed20260713_20260711" "${AIM_BANK}" \
  --pooling gated --class-weighting \
  --seed 20260713

run_cv "269_aimv2l_dense_gated_focal1_20260711" "${AIM_BANK}" \
  --pooling gated --class-weighting \
  --focal-gamma 1.0

run_cv "270_qkvb_dense_viewgated_focal1_20260711" "${QKVB_BANK}" \
  --pooling view_gated --class-weighting \
  --focal-gamma 1.0

if [[ -s "${DINOV2_BANK}/feature_bank_config.json" ]] && grep -q '"complete": true' "${DINOV2_BANK}/feature_bank_config.json"; then
  echo "[skip-bank] local DINOv2-L whole plus ROI"
else
  mkdir -p "${DINOV2_BANK}"
  echo "[extract-bank] local DINOv2-L whole plus ROI"
  "${PYTHON}" "${EXTRACT}" \
    --registry-csv "${CACHE_REGISTRY}" \
    --model-name local_dinov2_vitl14 \
    --image-size 336 \
    --views whole,crop \
    --output-dir "${DINOV2_BANK}" \
    --batch-size 4 \
    --num-workers 4
fi

run_cv "264_dinov2l_dense_gated_cw_20260711" "${DINOV2_BANK}" \
  --pooling gated --class-weighting

run_cv "265_dinov2l_dense_viewgated_cw_20260711" "${DINOV2_BANK}" \
  --pooling view_gated --class-weighting

if [[ -s "${ENHANCE_BANK}/feature_bank_config.json" ]] && grep -q '"complete": true' "${ENHANCE_BANK}/feature_bank_config.json"; then
  echo "[skip-bank] qkvb crop enhancement views"
else
  mkdir -p "${ENHANCE_BANK}"
  echo "[extract-bank] qkvb crop, autocontrast, and unsharp views"
  "${PYTHON}" "${EXTRACT}" \
    --registry-csv "${CACHE_REGISTRY}" \
    --model-name vit_large_patch16_dinov3_qkvb.lvd1689m \
    --image-size 352 \
    --views crop,crop_autocontrast,crop_unsharp \
    --output-dir "${ENHANCE_BANK}" \
    --batch-size 4 \
    --num-workers 4
fi

run_cv "266_qkvb_crop_enhance3_viewgated_cw_20260711" "${ENHANCE_BANK}" \
  --pooling view_gated --class-weighting

"${PYTHON}" "${SUMMARIZE}" \
  --runs-root "${OUT_ROOT}" \
  --output-csv "${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_capability_internal_summary.csv"

echo "[queue-complete] whole/ROI consistency and ordinal subtype screen"
