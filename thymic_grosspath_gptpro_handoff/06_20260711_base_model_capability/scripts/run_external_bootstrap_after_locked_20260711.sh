#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[wait] locked external queue pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

"${PYTHON}" /root/thymic_queue_scripts_20260711/bootstrap_locked_external_metrics_20260711.py \
  --dense-output-root "${PROJECT_ROOT}/experiments/base_model_capability_20260711/locked_dense_external" \
  --lora-output-root "${PROJECT_ROOT}/experiments/base_model_capability_20260711/locked_lora_external" \
  --output-dir "${PROJECT_ROOT}/experiments/base_model_capability_20260711/locked_external_bootstrap" \
  --iterations 5000

echo "[queue-complete] locked external bootstrap confidence intervals"
