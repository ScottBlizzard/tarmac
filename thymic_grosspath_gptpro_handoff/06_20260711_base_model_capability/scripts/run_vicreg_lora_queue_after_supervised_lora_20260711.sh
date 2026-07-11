#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
PYTHON="/root/miniconda3/envs/thymic_baseline/bin/python"
SSL_SCRIPT="${PROJECT_ROOT}/scripts/run_task7_foldwise_lora_vicreg_20260711.py"
FT_SCRIPT="${PROJECT_ROOT}/scripts/run_task7_lora_dense_finetune_20260711.py"
SSL_OUT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/vicreg_lora_internal"
FT_OUT="${PROJECT_ROOT}/experiments/base_model_capability_20260711/lora_dense_runs/235_qkvb_vicreginit_lora_dense_gated_cw_20260711"
CACHE_REGISTRY="/root/thymic_task7_internal_registry_cached_max2048_20260711.csv"
CROP_CACHE_DIR="/root/thymic_task7_crop_cache_max2048_20260711"

export THYMIC_CROP_CACHE_DIR="${CROP_CACHE_DIR}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[wait] supervised LoRA pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

if [[ ! -s "${SSL_OUT}/fold_5/encoder_lora_state.pt" ]]; then
  echo "[run-ssl] foldwise VICReg LoRA"
  "${PYTHON}" "${SSL_SCRIPT}" \
    --output-dir "${SSL_OUT}" \
    --registry-csv "${CACHE_REGISTRY}" \
    --fold all \
    --model-name vit_large_patch16_dinov3_qkvb.lvd1689m \
    --input-variant whole \
    --image-size 352 \
    --view-a-profile style_light \
    --view-b-profile domain_robust \
    --epochs 15 \
    --patience 4 \
    --batch-size 8 \
    --num-workers 8 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-last-blocks 2 \
    --lora-targets qkv,proj \
    --lr 0.00002 \
    --projector-lr 0.0002 \
    --amp-dtype bfloat16 \
    --device cuda
else
  echo "[skip-ssl] foldwise VICReg LoRA"
fi

if [[ ! -s "${FT_OUT}/oof_metrics.csv" ]]; then
  echo "[run-ft] VICReg-initialized supervised LoRA"
  "${PYTHON}" "${FT_SCRIPT}" \
    --output-dir "${FT_OUT}" \
    --registry-csv "${CACHE_REGISTRY}" \
    --fold all \
    --model-name vit_large_patch16_dinov3_qkvb.lvd1689m \
    --input-variant whole_plus_crop \
    --image-size 352 \
    --pooling gated \
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
    --init-encoder-state-template "${SSL_OUT}/fold_{fold}/encoder_lora_state.pt" \
    --aug-profile style_light \
    --class-weighting \
    --amp-dtype bfloat16 \
    --device cuda
else
  echo "[skip-ft] VICReg-initialized supervised LoRA"
fi

echo "[queue-complete] foldwise VICReg plus supervised LoRA"
