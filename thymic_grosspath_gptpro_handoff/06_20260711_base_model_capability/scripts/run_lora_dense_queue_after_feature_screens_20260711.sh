#!/usr/bin/env bash
set -u

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"
TRAIN="${PROJECT_ROOT}/scripts/run_task7_lora_dense_finetune_20260711.py"
CACHE_PREP="${PROJECT_ROOT}/scripts/prepare_task7_resized_image_cache_20260711.py"
CACHE_DIR="/root/thymic_task7_internal_image_cache_max2048_20260711"
CACHE_REGISTRY="/root/thymic_task7_internal_registry_cached_max2048_20260711.csv"
SOURCE_REGISTRY="${PROJECT_ROOT}/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/registry.csv"
CROP_CACHE_DIR="/root/thymic_task7_crop_cache_max2048_20260711"
OUT_ROOT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/lora_dense_runs"
mkdir -p "${OUT_ROOT}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[wait] dense follow-up pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

"${PYTHON}" "${CACHE_PREP}" \
  --registry-csv "${SOURCE_REGISTRY}" \
  --cache-dir "${CACHE_DIR}" \
  --output-registry "${CACHE_REGISTRY}" \
  --max-long-side 2048 \
  --jpeg-quality 95 \
  --minimum-free-gib 4.0

export THYMIC_CROP_CACHE_DIR="${CROP_CACHE_DIR}"

run_lora() {
  local tag="$1"
  shift
  local out_dir="${OUT_ROOT}/${tag}"
  if [[ -s "${out_dir}/oof_metrics.csv" ]]; then
    echo "[skip-lora] ${tag}"
    return 0
  fi
  mkdir -p "${out_dir}"
  echo "[run-lora] ${tag}"
  if "${PYTHON}" "${TRAIN}" \
    --output-dir "${out_dir}" \
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
    --aug-profile style_light \
    --amp-dtype bfloat16 \
    --device cuda \
    "$@"; then
    echo "[done-lora] ${tag}"
  else
    echo "[failed-lora] ${tag}" >&2
  fi
}

run_lora "232_qkvb_lora_dense_gated_qkvproj_cw_20260711" \
  --pooling gated --lora-targets qkv,proj --class-weighting

run_lora "233_qkvb_lora_dense_viewgated_qkvproj_nocw_20260711" \
  --pooling view_gated --lora-targets qkv,proj --no-class-weighting

run_lora "234_qkvb_lora_dense_gated_qkvprojmlp_cw_20260711" \
  --pooling gated --lora-targets qkv,proj,fc1,fc2 --class-weighting

echo "[queue-complete] LoRA dense fine-tuning"
