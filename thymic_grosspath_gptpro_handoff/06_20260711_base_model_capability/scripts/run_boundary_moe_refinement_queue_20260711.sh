#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"
TRAIN="${PROJECT_ROOT}/scripts/run_task7_dense_feature_cv_20260711.py"
SUMMARIZE="${PROJECT_ROOT}/scripts/summarize_dense_capability_screen_20260711.py"
FUSE="${PROJECT_ROOT}/scripts/evaluate_fixed_dense_oof_fusions_20260711.py"
BANK="/root/thymic_feature_banks_20260711/qkvb_whole_crop_352_internal"
OUT_ROOT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_cv_runs"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[wait] hierarchical queue pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
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

run_cv "239_qkvb_dense_b1b2_relevance_expert_20260711" \
  --pooling gated --expert-mode boundary --class-weighting \
  --boundary-loss-weight 0.35 \
  --boundary-relevance-loss-weight 0.20 \
  --concept-loss-weight 0.10

run_cv "240_qkvb_dense_low_b2_relevance_expert_20260711" \
  --pooling gated --expert-mode low_b2 --class-weighting \
  --boundary-loss-weight 0.35 \
  --boundary-relevance-loss-weight 0.20 \
  --concept-loss-weight 0.15

run_cv "241_qkvb_dense_moe_gate_balanced_20260711" \
  --pooling view_gated --expert-mode moe --class-weighting \
  --concept-loss-weight 0.10 \
  --subtype-loss-weight 0.20 \
  --prototype-loss-weight 0.10 \
  --moe-specialist-weight 0.30 \
  --moe-balance-weight 0.02 \
  --moe-gate-supervision-weight 0.15 \
  --soft-balanced-loss-weight 0.20

run_cv "242_qkvb_dense_moe_gate_groupdro_20260711" \
  --pooling view_gated --expert-mode moe --risk-objective group_dro --group-dro-eta 0.05 --class-weighting \
  --concept-loss-weight 0.10 \
  --subtype-loss-weight 0.20 \
  --prototype-loss-weight 0.10 \
  --moe-specialist-weight 0.30 \
  --moe-balance-weight 0.02 \
  --moe-gate-supervision-weight 0.15 \
  --soft-balanced-loss-weight 0.20

"${PYTHON}" "${SUMMARIZE}" \
  --runs-root "${OUT_ROOT}" \
  --output-csv "${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_capability_internal_summary.csv"

"${PYTHON}" "${FUSE}" \
  --runs-root "${OUT_ROOT}" \
  --output-dir "${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_fixed_fusions"

echo "[queue-complete] boundary and MoE refinements"
