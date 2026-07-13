#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
prepare_script="/root/prepare_task7_spatial_relational_assets_20260713.py"
trainer_script="/root/run_task7_spatial_relational_20260713.py"
analyzer_script="/root/analyze_task7_spatial_relational_20260713.py"
queue_script="/root/run_task7_spatial_relational_queue_20260713.sh"

feature_bank="/root/thymic_feature_banks_20260711/phase2_siglipl512_whole_crop_quadrants6_512_internal"
split_csv="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
c1_oof="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/347_siglipl512_localpyramid6_gated_fivefold_cw_20260711/oof_predictions.csv"
c1_lodo="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711/oof_predictions.csv"
c2_root="/workspace/thymic_project/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion"
c2_oof="${c2_root}/oof_predictions.csv"
c2_lodo="${c2_root}/lodo_predictions.csv"

output_root="/workspace/thymic_project/experiments/spatial_relational_20260713"
asset_dir="${output_root}/assets"
smoke_root="${output_root}/smoke"
run_root="${output_root}/runs"
analysis_dir="${output_root}/analysis"
status_file="${output_root}/SPATIAL_RELATIONAL_QUEUE.status.json"

mkdir -p "${output_root}" "${asset_dir}" "${smoke_root}" "${run_root}" "${analysis_dir}"
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export MALLOC_ARENA_MAX=2

write_status() {
  local state="$1"
  local detail="$2"
  "${python_bin}" - "${status_file}" "${state}" "${detail}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_suffix(path.suffix + ".tmp")
payload = {
    "state": sys.argv[2],
    "detail": sys.argv[3],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}
temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
temporary.replace(path)
PY
}

completed=0
on_exit() {
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    write_status failed "queue exited with rc=${rc}"
  elif [[ ${completed} -ne 1 ]]; then
    write_status failed "queue exited before a locked decision was produced"
  fi
}
trap on_exit EXIT

for required in \
  "${python_bin}" \
  "${prepare_script}" \
  "${trainer_script}" \
  "${analyzer_script}" \
  "${queue_script}" \
  "${feature_bank}/dense_features.float16.npy" \
  "${feature_bank}/metadata.csv" \
  "${feature_bank}/feature_bank_config.json" \
  "${split_csv}" \
  "${c1_oof}" \
  "${c1_lodo}" \
  "${c2_oof}" \
  "${c2_lodo}"
do
  test -e "${required}"
done

available_kb="$(df --output=avail / | tail -1 | tr -d ' ')"
if [[ "${available_kb}" -lt 5242880 ]]; then
  printf 'At least 5 GiB free space is required; found %s KiB\n' "${available_kb}" >&2
  exit 5
fi

manifest="${asset_dir}/integrity_manifest.json"
if [[ ! -s "${manifest}" ]]; then
  write_status preparing_assets "hashing immutable inputs and rebuilding canonical coordinates"
  "${python_bin}" "${prepare_script}" \
    --feature-bank-dir "${feature_bank}" \
    --split-csv "${split_csv}" \
    --c1-oof "${c1_oof}" \
    --c1-lodo "${c1_lodo}" \
    --c2-oof "${c2_oof}" \
    --c2-lodo "${c2_lodo}" \
    --output-dir "${asset_dir}" \
    --code-file "${prepare_script}" \
    --code-file "${trainer_script}" \
    --code-file "${analyzer_script}" \
    --code-file "${queue_script}"
fi
test -s "${asset_dir}/PREPARATION.status"
test -s "${asset_dir}/view_bounds.float32.npy"
manifest_sha256="$(sha256sum "${manifest}" | awk '{print $1}')"

common_args=(
  --feature-bank-dir "${feature_bank}"
  --coordinate-dir "${asset_dir}"
  --integrity-manifest "${manifest}"
  --expected-integrity-sha256 "${manifest_sha256}"
  --split-csv "${split_csv}"
  --hidden-dim 128
  --attention-dim 64
  --window-size 4
  --local-layers 2
  --global-layers 2
  --attention-heads 4
  --dropout 0.10
  --epochs 80
  --patience 12
  --batch-size 4
  --num-workers 0
  --lr 0.0003
  --weight-decay 0.0001
  --grad-clip 5.0
  --permutation-seed 20260713
  --device cuda
)

run_training() {
  local variant="$1"
  local split_mode="$2"
  local seed="$3"
  local output_dir="$4"
  shift 4
  if [[ -s "${output_dir}/oof_predictions.csv" ]]; then
    printf '[resume-skip] %s %s seed=%s\n' "${variant}" "${split_mode}" "${seed}"
    return 0
  fi
  mkdir -p "${output_dir}"
  "${python_bin}" "${trainer_script}" \
    "${common_args[@]}" \
    --variant "${variant}" \
    --split-mode "${split_mode}" \
    --seed "${seed}" \
    --output-dir "${output_dir}" \
    "$@"
  test -s "${output_dir}/oof_predictions.csv"
  test -s "${output_dir}/overall_metrics.json"
}

smoke_marker="${smoke_root}/SMOKE.status"
if [[ ! -s "${smoke_marker}" ]]; then
  write_status smoke_running "one real epoch for all three locked variants"
  for variant in matched_gated relational_permuted relational
  do
    smoke_dir="${smoke_root}/${variant}_fivefold_fold1"
    if [[ ! -s "${smoke_dir}/fold_1/test_predictions.csv" ]]; then
      "${python_bin}" "${trainer_script}" \
        "${common_args[@]}" \
        --variant "${variant}" \
        --split-mode fivefold \
        --fold 1 \
        --seed 20260713 \
        --output-dir "${smoke_dir}" \
        --max-epochs 1
    fi
    test -s "${smoke_dir}/fold_1/test_predictions.csv"
  done
  "${python_bin}" - "${smoke_root}" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
for variant in ("matched_gated", "relational_permuted", "relational"):
    frame = pd.read_csv(root / f"{variant}_fivefold_fold1/fold_1/test_predictions.csv")
    assert len(frame) > 0
    assert frame["case_id"].is_unique
    assert np.isfinite(frame["prob_high"]).all()
    assert frame["prob_high"].between(0.0, 1.0).all()
(root / "SMOKE.status").write_text("complete\n", encoding="utf-8")
PY
fi

for variant in matched_gated relational_permuted relational
do
  for split_mode in fivefold source_lodo
  do
    write_status training "primary ${variant} ${split_mode} seed 20260713"
    run_training \
      "${variant}" \
      "${split_mode}" \
      20260713 \
      "${run_root}/primary_seed20260713/${variant}/${split_mode}" \
      --fold all
  done
done

analysis_common=(
  --feature-bank-dir "${feature_bank}"
  --split-csv "${split_csv}"
  --integrity-manifest "${manifest}"
  --expected-integrity-sha256 "${manifest_sha256}"
  --c1-oof "${c1_oof}"
  --c1-lodo "${c1_lodo}"
  --c2-oof "${c2_oof}"
  --c2-lodo "${c2_lodo}"
  --matched-oof "${run_root}/primary_seed20260713/matched_gated/fivefold/oof_predictions.csv"
  --matched-lodo "${run_root}/primary_seed20260713/matched_gated/source_lodo/oof_predictions.csv"
  --permuted-oof "${run_root}/primary_seed20260713/relational_permuted/fivefold/oof_predictions.csv"
  --permuted-lodo "${run_root}/primary_seed20260713/relational_permuted/source_lodo/oof_predictions.csv"
  --relational-oof "${run_root}/primary_seed20260713/relational/fivefold/oof_predictions.csv"
  --relational-lodo "${run_root}/primary_seed20260713/relational/source_lodo/oof_predictions.csv"
  --output-dir "${analysis_dir}"
  --bootstrap-repetitions 5000
  --seed 20260713
)

write_status analyzing_primary "locked paired bootstrap and nine primary gates"
"${python_bin}" "${analyzer_script}" "${analysis_common[@]}"
decision="$("${python_bin}" -c "import json; print(json.load(open('${analysis_dir}/decision.json', encoding='utf-8'))['decision'])")"

if [[ "${decision}" == "PROVISIONAL-GO: CONFIRMATION REQUIRED" ]]; then
  for split_mode in fivefold source_lodo
  do
    write_status training_confirmation "relational ${split_mode} seed 20260714"
    run_training \
      relational \
      "${split_mode}" \
      20260714 \
      "${run_root}/confirmation_seed20260714/relational/${split_mode}" \
      --fold all
  done
  write_status analyzing_confirmation "two-seed mean and final locked decision"
  "${python_bin}" "${analyzer_script}" \
    "${analysis_common[@]}" \
    --confirmation-relational-oof "${run_root}/confirmation_seed20260714/relational/fivefold/oof_predictions.csv" \
    --confirmation-relational-lodo "${run_root}/confirmation_seed20260714/relational/source_lodo/oof_predictions.csv"
  decision="$("${python_bin}" -c "import json; print(json.load(open('${analysis_dir}/decision.json', encoding='utf-8'))['decision'])")"
fi

write_status complete "locked decision: ${decision}"
completed=1
printf '[complete] locked decision: %s\n' "${decision}"
