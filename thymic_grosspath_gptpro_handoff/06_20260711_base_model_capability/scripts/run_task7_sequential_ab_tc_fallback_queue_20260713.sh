#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
trainer="/root/run_task7_sequential_ab_tc_fallback_20260713.py"
feature_bank="/root/thymic_feature_banks_20260711/phase2_siglipl512_whole_crop_quadrants6_512_internal"
split_csv="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
output_root="/workspace/thymic_project/experiments/sequential_ab_tc_fallback_20260713"
status_file="${output_root}/SEQUENTIAL_AB_TC_QUEUE.status"

mkdir -p "${output_root}"

write_status() {
  local temporary="${status_file}.tmp"
  printf '%s\n' "$1" > "${temporary}"
  mv -f "${temporary}" "${status_file}"
}

on_exit() {
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    write_status complete
  else
    write_status "failed rc=${rc}"
  fi
}
trap on_exit EXIT

write_status waiting_for_feature_bank
while true; do
  if [[ -s "${feature_bank}/feature_bank_config.json" ]] \
    && grep -q '"complete": true' "${feature_bank}/feature_bank_config.json"; then
    break
  fi
  sleep 30
done

common_args=(
  --feature-bank-dir "${feature_bank}"
  --split-csv "${split_csv}"
  --hidden-dim 256
  --attention-dim 128
  --dropout 0.25
  --epochs 80
  --patience 12
  --batch-size 8
  --num-workers 0
  --lr 0.0003
  --weight-decay 0.0001
  --grad-clip 5.0
  --anchor-sample-count 40
  --ab-risk-purity 0.98
  --tc-risk-purity 0.95
  --min-route-n 10
  --inner-folds 3
  --seed 20260713
  --device cuda
)

smoke_dir="${output_root}/smoke_fold1_two_epochs"
write_status smoke_running
rm -rf -- "${smoke_dir}"
"${python_bin}" "${trainer}" \
  "${common_args[@]}" \
  --output-dir "${smoke_dir}" \
  --split-mode fivefold \
  --fold 1 \
  --max-epochs 2

test -s "${smoke_dir}/fold_1/test_predictions.csv"
test -s "${smoke_dir}/fold_1/fold_summary.json"
"${python_bin}" -c \
  "import json, pandas as pd; p='${smoke_dir}/fold_1/test_predictions.csv'; f=pd.read_csv(p); assert len(f)==123; assert {'prob_ab','prob_tc','prob_fallback','prob_high','route_decision'}.issubset(f.columns); json.load(open('${smoke_dir}/fold_1/fold_summary.json', encoding='utf-8'))"
write_status smoke_complete

fivefold_dir="${output_root}/fivefold_primary_seed20260713"
write_status fivefold_running
rm -rf -- "${fivefold_dir}"
"${python_bin}" "${trainer}" \
  "${common_args[@]}" \
  --output-dir "${fivefold_dir}" \
  --split-mode fivefold \
  --fold all

test -s "${fivefold_dir}/oof_predictions.csv"
test -s "${fivefold_dir}/metrics.csv"

lodo_dir="${output_root}/source_lodo_primary_seed20260713"
write_status source_lodo_running
rm -rf -- "${lodo_dir}"
"${python_bin}" "${trainer}" \
  "${common_args[@]}" \
  --output-dir "${lodo_dir}" \
  --split-mode source_lodo \
  --fold all

test -s "${lodo_dir}/oof_predictions.csv"
test -s "${lodo_dir}/metrics.csv"
write_status analysis_ready
