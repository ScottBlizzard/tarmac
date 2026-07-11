#!/usr/bin/env bash
set -u

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"
TRAIN="${PROJECT_ROOT}/scripts/run_task7_dense_feature_cv_20260711.py"
EXTRACT="${PROJECT_ROOT}/scripts/extract_task7_dense_token_bank_20260711.py"
SUMMARIZE="${PROJECT_ROOT}/scripts/summarize_dense_capability_screen_20260711.py"
OUT_ROOT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_cv_runs"
BANK_ROOT="/root/thymic_feature_banks_20260711"
mkdir -p "${OUT_ROOT}" "${BANK_ROOT}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[wait] dense screen pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

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
  if "${PYTHON}" "${TRAIN}" \
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
  local views="$2"
  local bank_dir="${BANK_ROOT}/${bank_name}"
  if [[ -s "${bank_dir}/feature_bank_config.json" ]] && grep -q '"complete": true' "${bank_dir}/feature_bank_config.json"; then
    echo "[skip-bank] ${bank_name}"
    return 0
  fi
  mkdir -p "${bank_dir}"
  echo "[extract-bank] ${bank_name} views=${views}"
  if "${PYTHON}" "${EXTRACT}" \
    --model-name "vit_large_patch16_dinov3_qkvb.lvd1689m" \
    --image-size 352 \
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

extract_bank "qkvb_roi4_352_internal" "whole,crop,crop_tight,masked_gray"
ROI_BANK="${BANK_ROOT}/qkvb_roi4_352_internal"

run_cv "223_qkvb_roi4_viewgated_cw_20260711" "${ROI_BANK}" \
  --pooling view_gated --class-weighting

run_cv "224_qkvb_roi4_gated_cw_20260711" "${ROI_BANK}" \
  --pooling gated --class-weighting

run_cv "225_qkvb_roi4_viewgated_domainadv_20260711" "${ROI_BANK}" \
  --pooling view_gated --class-weighting \
  --domain-adversarial-weight 0.20 \
  --domain-adversarial-lambda 0.50

extract_bank "qkvb_background_only_352_internal" "background_only"
BACKGROUND_BANK="${BANK_ROOT}/qkvb_background_only_352_internal"

run_cv "226_qkvb_background_only_mean_audit_20260711" "${BACKGROUND_BANK}" \
  --pooling mean --class-weighting

run_cv "227_qkvb_dense_gated_domainadv_20260711" "${QKVB_BANK}" \
  --pooling gated --class-weighting \
  --domain-adversarial-weight 0.20 \
  --domain-adversarial-lambda 0.50

run_cv "228_qkvb_dense_concept_boundary_combined_20260711" "${QKVB_BANK}" \
  --pooling gated --expert-mode boundary --class-weighting \
  --concept-loss-weight 0.15 \
  --subtype-loss-weight 0.20 \
  --prototype-loss-weight 0.10 \
  --boundary-loss-weight 0.35 \
  --boundary-triplet-weight 0.10 \
  --subtype-balanced-sampler

run_cv "229_qkvb_dense_gated_nocw_20260711" "${QKVB_BANK}" \
  --pooling gated --no-class-weighting

run_cv "230_qkvb_dense_boundary_nocw_20260711" "${QKVB_BANK}" \
  --pooling gated --expert-mode boundary --no-class-weighting \
  --subtype-loss-weight 0.25 \
  --boundary-loss-weight 0.35 \
  --boundary-triplet-weight 0.10 \
  --subtype-balanced-sampler

run_cv "231_qkvb_roi4_viewgated_groupdro_20260711" "${ROI_BANK}" \
  --pooling view_gated --risk-objective group_dro --group-dro-eta 0.05 --class-weighting

"${PYTHON}" "${SUMMARIZE}" \
  --runs-root "${OUT_ROOT}" \
  --output-csv "${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_capability_internal_summary.csv"

echo "[queue-complete] dense capability follow-up"
