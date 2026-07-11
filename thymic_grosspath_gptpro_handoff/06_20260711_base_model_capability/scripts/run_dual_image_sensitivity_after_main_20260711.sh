#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"
SCRIPT_ROOT="/root/thymic_queue_scripts_20260711"
REGISTRY="/root/thymic_task7_dual17_registry_20260711.csv"
BANK="/root/thymic_feature_banks_20260711/aimv2l_dual17_whole_crop_336"
RUN_DIR="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_cv_runs/219_aimv2l_dense_gated_cw_20260711"
OUTPUT_DIR="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dual_image_oof_sensitivity"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[wait] locked external queue pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

"${PYTHON}" "${SCRIPT_ROOT}/prepare_task7_dual_image_sensitivity_registry_20260711.py" \
  --output-csv "${REGISTRY}"

if [[ -s "${BANK}/feature_bank_config.json" ]] && grep -q '"complete": true' "${BANK}/feature_bank_config.json"; then
  echo "[skip-bank] AIMv2-L dual-image sensitivity bank"
else
  mkdir -p "${BANK}"
  "${PYTHON}" "${SCRIPT_ROOT}/extract_task7_dense_token_bank_20260711.py" \
    --registry-csv "${REGISTRY}" \
    --domains old_data \
    --model-name aimv2_large_patch14_336.apple_pt \
    --image-size 336 \
    --views whole,crop \
    --output-dir "${BANK}" \
    --batch-size 4 \
    --num-workers 4
fi

"${PYTHON}" "${SCRIPT_ROOT}/evaluate_task7_dual_image_oof_sensitivity_20260711.py" \
  --run-dir "${RUN_DIR}" \
  --feature-bank-dir "${BANK}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda

echo "[queue-complete] dual-image OOF sensitivity"
