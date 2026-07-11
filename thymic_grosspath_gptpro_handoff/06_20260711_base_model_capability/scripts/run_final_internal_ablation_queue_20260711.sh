#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"
TRAIN="${PROJECT_ROOT}/scripts/run_task7_dense_feature_cv_20260711.py"
SUMMARIZE="${PROJECT_ROOT}/scripts/summarize_dense_capability_screen_20260711.py"
SELECT_FUSION="/root/thymic_queue_scripts_20260711/select_locked_dense_oof_fusion_20260711.py"
RUN_SOURCE_LODO="/root/thymic_queue_scripts_20260711/run_locked_dense_source_lodo_shortlist_20260711.py"
LORA_TRAIN="${PROJECT_ROOT}/scripts/run_task7_lora_dense_finetune_20260711.py"
BANK="/root/thymic_feature_banks_20260711/qkvb_whole_crop_352_internal"
OUT_ROOT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_cv_runs"
LORA_OUT_ROOT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/lora_dense_runs"
CACHE_REGISTRY="/root/thymic_task7_internal_registry_cached_max2048_20260711.csv"
CROP_CACHE_DIR="/root/thymic_task7_crop_cache_max2048_20260711"
SUMMARY="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_capability_internal_summary.csv"
FUSION_OUT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_locked_internal_fusion"
SOURCE_LODO_OUT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/dense_source_lodo_runs"
LOCK_MARKER="/root/thymic_feature_banks_20260711/INTERNAL_SELECTION_LOCK.status"

mkdir -p "$(dirname "${LOCK_MARKER}")"
printf 'in_progress\n' > "${LOCK_MARKER}"
trap 'printf "failed\n" > "${LOCK_MARKER}"' ERR
export THYMIC_CROP_CACHE_DIR="${CROP_CACHE_DIR}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[wait] boundary/MoE refinement pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

run_cv_bank() {
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

run_cv() {
  local tag="$1"
  shift
  run_cv_bank "${tag}" "${BANK}" "$@"
}

run_cv "243_qkvb_dense_gated_conflictsoft10_20260711" \
  --pooling gated --class-weighting \
  --visual-conflict-softening 0.10

run_cv "244_qkvb_dense_gated_conflictsoft20_20260711" \
  --pooling gated --class-weighting \
  --visual-conflict-softening 0.20

run_cv "245_qkvb_dense_moe_conflictsoft10_20260711" \
  --pooling view_gated --expert-mode moe --class-weighting \
  --subtype-loss-weight 0.15 \
  --prototype-loss-weight 0.10 \
  --moe-specialist-weight 0.25 \
  --moe-balance-weight 0.02 \
  --visual-conflict-softening 0.10

run_cv "246_qkvb_dense_gated_classconditionalalign_20260711" \
  --pooling gated --class-weighting \
  --class-conditional-align-weight 0.05

run_cv "247_qkvb_dense_moe_classconditionalalign_20260711" \
  --pooling view_gated --expert-mode moe --class-weighting \
  --subtype-loss-weight 0.15 \
  --prototype-loss-weight 0.10 \
  --moe-specialist-weight 0.25 \
  --moe-balance-weight 0.02 \
  --class-conditional-align-weight 0.03

run_cv "248_qkvb_dense_concept_conflictsoft10_20260711" \
  --pooling gated --class-weighting \
  --concept-loss-weight 0.20 \
  --subtype-loss-weight 0.15 \
  --prototype-loss-weight 0.10 \
  --visual-conflict-softening 0.10

run_cv "249_qkvb_dense_gated_highrisk_sentinel_20260711" \
  --pooling gated --class-weighting \
  --sentinel-fusion-alpha 0.25 \
  --sentinel-loss-weight 0.30 \
  --sentinel-positive-weight 1.50 \
  --sentinel-negative-gamma 2.0

run_cv "250_qkvb_dense_moe_highrisk_sentinel_20260711" \
  --pooling view_gated --expert-mode moe --class-weighting \
  --subtype-loss-weight 0.15 \
  --prototype-loss-weight 0.10 \
  --moe-specialist-weight 0.25 \
  --moe-balance-weight 0.02 \
  --sentinel-fusion-alpha 0.20 \
  --sentinel-loss-weight 0.25 \
  --sentinel-positive-weight 1.50 \
  --sentinel-negative-gamma 2.0

run_cv "252_qkvb_dense_gated_mixstyle_20260711" \
  --pooling gated --class-weighting \
  --mixstyle-probability 0.50 \
  --mixstyle-alpha 0.10

run_cv_bank "253_aimv2l_dense_gated_mixstyle_20260711" \
  "/root/thymic_feature_banks_20260711/aimv2l_whole_crop_336_internal" \
  --pooling gated --class-weighting \
  --mixstyle-probability 0.50 \
  --mixstyle-alpha 0.10

run_cv_bank "256_aimv2l_dense_gated_strongreg_20260711" \
  "/root/thymic_feature_banks_20260711/aimv2l_whole_crop_336_internal" \
  --pooling gated --class-weighting \
  --hidden-dim 192 \
  --attention-dim 96 \
  --dropout 0.40 \
  --weight-decay 0.001

run_cv_bank "257_aimv2l_dense_viewgated_cw_20260711" \
  "/root/thymic_feature_banks_20260711/aimv2l_whole_crop_336_internal" \
  --pooling view_gated --class-weighting

LORA_DOMAIN_OUT="${LORA_OUT_ROOT}/251_qkvb_lora_dense_gated_domainrobust_20260711"
if [[ -s "${LORA_DOMAIN_OUT}/oof_metrics.csv" ]]; then
  echo "[skip-lora] 251_qkvb_lora_dense_gated_domainrobust_20260711"
else
  echo "[run-lora] 251_qkvb_lora_dense_gated_domainrobust_20260711"
  "${PYTHON}" "${LORA_TRAIN}" \
    --output-dir "${LORA_DOMAIN_OUT}" \
    --registry-csv "${CACHE_REGISTRY}" \
    --fold all \
    --model-name vit_large_patch16_dinov3_qkvb.lvd1689m \
    --input-variant whole_plus_crop \
    --image-size 352 \
    --epochs 12 \
    --patience 4 \
    --batch-size 8 \
    --num-workers 8 \
    --head-lr 0.0002 \
    --lora-lr 0.00002 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-last-blocks 2 \
    --lora-targets qkv,proj \
    --pooling gated \
    --class-weighting \
    --aug-profile domain_robust \
    --amp-dtype bfloat16 \
    --device cuda
fi

LORA_AIM_OUT="${LORA_OUT_ROOT}/258_aimv2l_lora_dense_gated_stylelight_20260711"
if [[ -s "${LORA_AIM_OUT}/oof_metrics.csv" ]]; then
  echo "[skip-lora] 258_aimv2l_lora_dense_gated_stylelight_20260711"
else
  echo "[run-lora] 258_aimv2l_lora_dense_gated_stylelight_20260711"
  "${PYTHON}" "${LORA_TRAIN}" \
    --output-dir "${LORA_AIM_OUT}" \
    --registry-csv "${CACHE_REGISTRY}" \
    --fold all \
    --model-name aimv2_large_patch14_336.apple_pt \
    --input-variant whole_plus_crop \
    --image-size 336 \
    --epochs 12 \
    --patience 4 \
    --batch-size 8 \
    --num-workers 8 \
    --head-lr 0.0002 \
    --lora-lr 0.00002 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-last-blocks 2 \
    --lora-targets qkv,proj \
    --pooling gated \
    --class-weighting \
    --aug-profile style_light \
    --amp-dtype bfloat16 \
    --device cuda
fi

"${PYTHON}" "${SUMMARIZE}" \
  --runs-root "${OUT_ROOT}" \
  --output-csv "${SUMMARY}" || exit 1

# Refuse to lock a shortlist unless every predeclared internal experiment completed.
for run_id in {206..231} {236..250} 252 253 254 255 256 257 259 260 261 262 263 264 265 266 267 268 269 270; do
  if ! compgen -G "${OUT_ROOT}/${run_id}_*/oof_metrics.csv" > /dev/null; then
    echo "[abort] missing dense OOF result for run ${run_id}" >&2
    exit 1
  fi
done
for run_id in 232 233 234 235 251 258; do
  if ! compgen -G "${LORA_OUT_ROOT}/${run_id}_*/oof_metrics.csv" > /dev/null; then
    echo "[abort] missing LoRA OOF result for run ${run_id}" >&2
    exit 1
  fi
done
for fold in {1..5}; do
  if [[ ! -s "${PROJECT_ROOT}/experiments/base_model_capability_20260711/vicreg_lora_internal/fold_${fold}/encoder_lora_state.pt" ]]; then
    echo "[abort] missing VICReg encoder state for fold ${fold}" >&2
    exit 1
  fi
done

"${PYTHON}" "${RUN_SOURCE_LODO}" \
  --summary-csv "${SUMMARY}" \
  --runs-root "${OUT_ROOT}" \
  --output-root "${SOURCE_LODO_OUT}" \
  --max-candidates 15 \
  --min-overall-bacc 0.67 \
  --min-source-bacc 0.60 || exit 1

"${PYTHON}" "${SELECT_FUSION}" \
  --runs-root "${OUT_ROOT}" \
  --summary-csv "${SUMMARY}" \
  --output-dir "${FUSION_OUT}" \
  --min-single-bacc 0.67 \
  --min-single-source-bacc 0.60 \
  --max-pool-size 15 \
  --max-members 3 \
  --min-class-recall 0.60 \
  --candidate-manifest "${SOURCE_LODO_OUT}/LOCKED_INTERNAL_SOURCE_LODO_SHORTLIST.csv" \
  --lodo-runs-root "${SOURCE_LODO_OUT}" \
  --lodo-weight 0.45 \
  --min-lodo-class-recall 0.50 || exit 1

printf 'complete\n' > "${LOCK_MARKER}"
trap - ERR

echo "[queue-complete] final internal ablations and locked OOF fusion"
