#!/usr/bin/env bash
set -euo pipefail

cd /workspace/thymic_project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline

python scripts/run_existing_variant_external_sweep_20260706.py \
  --domains strict_external \
  --batch-size 2 \
  --num-workers 2 \
  --device cuda
