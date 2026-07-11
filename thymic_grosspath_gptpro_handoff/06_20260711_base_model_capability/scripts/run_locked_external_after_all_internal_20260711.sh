#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[wait] final internal refinement pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

LOCK_MARKER="/root/thymic_feature_banks_20260711/INTERNAL_SELECTION_LOCK.status"
if [[ ! -f "${LOCK_MARKER}" ]] || [[ "$(tr -d '\r\n' < "${LOCK_MARKER}")" != "complete" ]]; then
  echo "[abort] internal selection did not complete successfully: ${LOCK_MARKER}" >&2
  exit 1
fi

df -h /root

"${PYTHON}" "/root/thymic_queue_scripts_20260711/run_locked_dense_external_shortlist_20260711.py" \
  --internal-summary "${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_capability_internal_summary.csv" \
  --runs-root "${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_cv_runs" \
  --external-bank-root "/root/thymic_feature_banks_20260711/locked_external" \
  --output-root "${PROJECT_ROOT}/experiments/base_model_capability_20260711/locked_dense_external" \
  --fusion-recipe "${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_locked_internal_fusion/LOCKED_INTERNAL_DENSE_FUSION_MEMBERS.csv" \
  --max-candidates 3 \
  --min-overall-bacc 0.68 \
  --min-source-bacc 0.60 \
  --device cuda

"${PYTHON}" "${PROJECT_ROOT}/scripts/run_locked_lora_external_candidate_20260711.py" \
  --runs-root "${PROJECT_ROOT}/experiments/base_model_capability_20260711/lora_dense_runs" \
  --output-root "${PROJECT_ROOT}/experiments/base_model_capability_20260711/locked_lora_external" \
  --device cuda

echo "[queue-complete] locked external evaluations"
