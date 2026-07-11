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

OUT_ROOT="experiments/base_model_expansion_20260706/outputs/extra_candidate_external_sweep"
mkdir -p "${OUT_ROOT}"

STRICT_REG="experiments/base_model_expansion_20260706/outputs/registry/strict_external_registry_for_inference.csv"
NEW_REG="experiments/base_model_expansion_20260706/outputs/registry/new_external_160_registry_for_inference.csv"

declare -A RUNS=(
  [201_subtypeaux_lowrisk]="experiments/base_model_expansion_20260706/outputs/subtype_aux_runs/201_qkvb_stylelight_lastblock_subtypeaux_lowrisk_20260706"
  [123_qkvb_nocw]="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/123_dinov3_vitl16_qkvb_task7_whole352_nocw_lastblock_5fold_20260524"
  [137_qkvb_classsampler_nocw]="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/137_dinov3_vitl16_qkvb_task7_whole352_class_sampler_nocw_5fold_20260524"
  [168_qkvb_seed20260525]="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/168_dinov3_vitl16_qkvb_task7_whole352_token_lastblock_lowlr_seed20260525_5fold_20260524"
)

for tag in 201_subtypeaux_lowrisk 123_qkvb_nocw 137_qkvb_classsampler_nocw 168_qkvb_seed20260525; do
  run_dir="${RUNS[$tag]}"
  for domain in strict_external new_external_160; do
    if [[ "${domain}" == "strict_external" ]]; then
      reg="${STRICT_REG}"
    else
      reg="${NEW_REG}"
    fi
    out_dir="${OUT_ROOT}/${tag}__${domain}"
    mkdir -p "${out_dir}"
    if [[ ! -s "${out_dir}/external_tta_predictions.csv" ]]; then
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
done

python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("experiments/base_model_expansion_20260706/outputs/extra_candidate_external_sweep")
rows = []
for p in sorted(root.glob("*__/external_tta_metrics_summary.csv")):
    pass
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
    out.to_csv(root / "extra_candidate_external_sweep_summary.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
else:
    print("No summary rows found.")
PY
