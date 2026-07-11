#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"
TRAIN="${PROJECT_ROOT}/scripts/run_task7_dense_feature_cv_20260711.py"
EXTRACT="/root/thymic_queue_scripts_20260711/extract_task7_dense_token_bank_20260711.py"
SUMMARIZE="${PROJECT_ROOT}/scripts/summarize_dense_capability_screen_20260711.py"
BANK="/root/thymic_feature_banks_20260711/qkvb_crop_quadrants5_352_internal"
OUT_ROOT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_cv_runs"
CACHE_REGISTRY="/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[wait] boundary/MoE refinement pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

if [[ -s "${BANK}/feature_bank_config.json" ]] && grep -q '"complete": true' "${BANK}/feature_bank_config.json"; then
  echo "[skip-bank] qkvb crop quadrants"
else
  mkdir -p "${BANK}"
  echo "[extract-bank] qkvb crop plus four quadrants"
  "${PYTHON}" "${EXTRACT}" \
    --registry-csv "${CACHE_REGISTRY}" \
    --model-name vit_large_patch16_dinov3_qkvb.lvd1689m \
    --image-size 352 \
    --views crop,crop_q0,crop_q1,crop_q2,crop_q3 \
    --output-dir "${BANK}" \
    --batch-size 4 \
    --num-workers 4
fi

run_cv() {
  local tag="$1"
  shift
  local out_dir="${OUT_ROOT}/${tag}"
  if [[ -s "${out_dir}/oof_metrics.csv" ]]; then
    echo "[skip] ${tag}"
    return 0
  fi
  echo "[run] ${tag}"
  "${PYTHON}" "${TRAIN}" \
    --feature-bank-dir "${BANK}" \
    --output-dir "${out_dir}" \
    --epochs 80 \
    --patience 12 \
    --batch-size 24 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --device cuda \
    "$@"
}

run_cv "254_qkvb_crop_quadrants5_viewgated_cw_20260711" \
  --pooling view_gated --class-weighting

run_cv "255_qkvb_crop_quadrants5_moe_20260711" \
  --pooling view_gated --expert-mode moe --class-weighting \
  --subtype-loss-weight 0.15 \
  --prototype-loss-weight 0.10 \
  --moe-specialist-weight 0.25 \
  --moe-balance-weight 0.02

"${PYTHON}" "${SUMMARIZE}" \
  --runs-root "${OUT_ROOT}" \
  --output-csv "${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_capability_internal_summary.csv"

echo "[queue-complete] high-resolution tile pyramid"
