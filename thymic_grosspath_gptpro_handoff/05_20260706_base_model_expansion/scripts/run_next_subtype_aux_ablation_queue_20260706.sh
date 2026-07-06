#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
PROJECT_ROOT="/workspace/thymic_project"
cd "${PROJECT_ROOT}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thymic_baseline

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
fi

export THYMIC_AUG_PROFILE=style_light

OUT_ROOT="experiments/base_model_expansion_20260706/outputs/subtype_aux_runs"
EXT_ROOT="experiments/base_model_expansion_20260706/outputs/subtype_aux_ablation_external_sweep"
STRICT_REG="experiments/base_model_expansion_20260706/outputs/registry/strict_external_registry_for_inference.csv"
NEW_REG="experiments/base_model_expansion_20260706/outputs/registry/new_external_160_registry_for_inference.csv"
mkdir -p "${OUT_ROOT}" "${EXT_ROOT}"

train_if_needed() {
  local tag="$1"
  shift
  local out_dir="${OUT_ROOT}/${tag}"
  mkdir -p "${out_dir}"
  if [[ -s "${out_dir}/oof_metrics.csv" ]]; then
    echo "[skip-train] ${tag} already has oof_metrics.csv"
    return 0
  fi
  echo "[train] ${tag}"
  python scripts/run_task7_dinov3_multitask_subtype_aux_20260706.py \
    --output-dir "${out_dir}" \
    --fold all \
    --model-name vit_large_patch16_dinov3_qkvb.lvd1689m \
    --input-variant whole \
    --image-size 352 \
    --global-pool token \
    --tune-scope last_block \
    --head-lr 0.0002 \
    --backbone-lr 0.000001 \
    --dropout 0.2 \
    --head-type mlp \
    --hidden-dim 512 \
    --epochs 10 \
    --batch-size 4 \
    --num-workers 4 \
    --label-smoothing 0.02 \
    --aux-loss-weight 0.25 \
    --no-sample-loss-weighting \
    --patience 4 \
    --device cuda \
    "$@"
}

eval_external_if_needed() {
  local tag="$1"
  local run_dir="${OUT_ROOT}/${tag}"
  for domain in strict_external new_external_160; do
    local reg="${STRICT_REG}"
    if [[ "${domain}" == "new_external_160" ]]; then
      reg="${NEW_REG}"
    fi
    local out_dir="${EXT_ROOT}/${tag}__${domain}"
    mkdir -p "${out_dir}"
    if [[ ! -s "${out_dir}/external_tta_predictions.csv" ]]; then
      echo "[external-tta] ${tag} ${domain}"
      python scripts/run_task7_dinov3_external_tta_fastcrop_20260706.py \
        --run-dir "${run_dir}" \
        --external-registry-csv "${reg}" \
        --output-dir "${out_dir}" \
        --views orig,hflip,vflip,hvflip \
        --batch-size 4 \
        --num-workers 2 \
        --device cuda
    fi
    python scripts/summarize_external_tta_predictions_20260706.py \
      --pred-csv "${out_dir}/external_tta_predictions.csv" \
      --output-dir "${out_dir}"
  done
}

train_if_needed "202_qkvb_stylelight_lastblock_subtypeaux_auxonly_cw_20260706" --class-weighting
eval_external_if_needed "202_qkvb_stylelight_lastblock_subtypeaux_auxonly_cw_20260706"

train_if_needed "203_qkvb_stylelight_lastblock_subtypeaux_auxonly_nocw_20260706" --no-class-weighting
eval_external_if_needed "203_qkvb_stylelight_lastblock_subtypeaux_auxonly_nocw_20260706"

python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("experiments/base_model_expansion_20260706/outputs/subtype_aux_ablation_external_sweep")
rows = []
for p in sorted(root.glob("*/external_tta_metrics_summary.csv")):
    df = pd.read_csv(p)
    overall = df[df["group"].astype(str).eq("overall")].copy()
    if overall.empty:
        continue
    tag, domain = p.parent.name.split("__", 1)
    overall.insert(0, "model_tag", tag)
    overall.insert(1, "domain", domain)
    rows.append(overall)
if rows:
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(root / "subtype_aux_ablation_external_sweep_summary.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
else:
    print("No summary rows found.")
PY
