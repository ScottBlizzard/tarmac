#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
runner="/root/run_task7_attention_roi_d1_20260712.py"
analyzer="/root/analyze_task7_attention_roi_d1_20260712.py"
output_dir="/workspace/thymic_project/experiments/base_model_capability_20260711/d1_attention_roi_20260712"
status_file="$output_dir/D1_ATTENTION_ROI.status"

mkdir -p "$output_dir"

write_status() {
  rm -f -- "$status_file"
  printf '%s\n' "$1" > "$status_file"
}

on_exit() {
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    write_status complete
  else
    write_status "failed rc=$rc"
  fi
}
trap on_exit EXIT

write_status running
export MALLOC_ARENA_MAX=2
export TQDM_DISABLE=1

"$python_bin" "$runner" \
  --output-dir "$output_dir" \
  --split-modes fivefold,source_lodo \
  --model-name vit_large_patch16_siglip_512.v2_webli \
  --image-size 512 \
  --extract-batch-size 1 \
  --num-workers 2 \
  --head-batch-size 8 \
  --hidden-dim 256 \
  --attention-dim 128 \
  --dropout 0.25 \
  --selector-epochs 60 \
  --selector-patience 10 \
  --diagnostic-epochs 80 \
  --diagnostic-patience 12 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --roi-centers 2 \
  --roi-scales 0.40,0.25 \
  --minimum-tissue 0.60 \
  --minimum-center-distance 0.20 \
  --heatmap-size 64 \
  --seed 20260712 \
  --device cuda

"$python_bin" "$analyzer" \
  --run-dir "$output_dir" \
  --bootstrap-iterations 5000 \
  --seed 20260712

write_status complete
