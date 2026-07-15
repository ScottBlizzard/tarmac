#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT=/workspace/thymic_project
PY=/root/miniconda3/envs/thymic_baseline/bin/python3.11
CODE=/root/task7_h9_20260715
OUT=${ROOT}/experiments/h9_h3_difficulty_balanced_curriculum_20260715
LOGS=${CODE}/logs
ARTIFACTS=${CODE}/artifacts
BANK=${CODE}/bank/pe_spatial_l14_448_dense
CONTROL=${ARTIFACTS}/source_lodo/source_risk_control_seed20260713
PRIMARY=${ARTIFACTS}/source_lodo/tempered_curriculum_seed20260713
CONFIRMATION=${ARTIFACTS}/source_lodo/tempered_curriculum_seed20260715

REGISTRY=/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv
SPLIT=${ROOT}/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv
PE=/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt
PE_SOURCE=/root/third_party/perception_models_3e352cca

C1_LODO=${ROOT}/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711/oof_predictions.csv
C2_LODO=${ROOT}/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/lodo_predictions.csv
H3_LODO=${ROOT}/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/source_lodo/oof_predictions.csv

EXTRACTOR=${ROOT}/scripts/extract_task7_h3_dense_bank_20260713.py
CONTROL_TRAINER=${ROOT}/scripts/run_task7_h3b_masked_gated_20260713.py
TRAINER=${CODE}/run_task7_h9_h3_difficulty_curriculum_20260715.py
ANALYZER=${CODE}/analyze_task7_h9_h3_difficulty_curriculum_20260715.py
QUEUE=${CODE}/run_task7_h9_h3_difficulty_curriculum_queue_20260715.sh

write_status() {
  value=$1
  temporary=${OUT}/.RUN.status.$$
  printf '%s\n' "${value}" > "${temporary}"
  mv -f "${temporary}" "${OUT}/RUN.status"
}

mark_failed() {
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    write_status "failed rc=${rc}"
  fi
}

mkdir -p "${OUT}"/source_lodo/{primary,confirmation} "${LOGS}" "${ARTIFACTS}" "${BANK}"
trap mark_failed EXIT
write_status running

available_bytes=$(df --output=avail -B1 /root | tail -1 | tr -d ' ')
if (( available_bytes < 16106127360 )); then
  echo "H9 requires at least 15 GiB free on /root; available=${available_bytes}" >&2
  exit 1
fi

sha256sum "${TRAINER}" "${ANALYZER}" "${QUEUE}" > "${LOGS}/code.sha256.tmp"
sha256sum \
  "${EXTRACTOR}" \
  "${ROOT}/scripts/extract_task7_h3_representation_bank_20260713.py" \
  "${CONTROL_TRAINER}" \
  "${ROOT}/scripts/run_task7_h3_summary_gated_20260713.py" \
  "${ROOT}/scripts/run_task7_spatial_relational_20260713.py" >> "${LOGS}/code.sha256.tmp"
mv -f "${LOGS}/code.sha256.tmp" "${LOGS}/code.sha256"

actual_pe_sha=$(sha256sum "${PE}" | cut -d' ' -f1)
if [[ "${actual_pe_sha}" != "47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1" ]]; then
  echo "PE checkpoint hash changed: ${actual_pe_sha}" >&2
  exit 1
fi

"${PY}" "${EXTRACTOR}" \
  --backend pe \
  --registry-csv "${REGISTRY}" \
  --model-id "${PE}" \
  --canonical-model-id facebook/PE-Spatial-L14-448 \
  --weight-sha256 47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1 \
  --revision master \
  --model-code-dir "${PE_SOURCE}" \
  --code-revision 3e352cca660658d4b5c90f42a7808b11469e4c66 \
  --output-dir "${BANK}" \
  --domains old_data,third_batch \
  --views whole,crop,crop_q0,crop_q1,crop_q2,crop_q3 \
  --max-num-patches 1024 \
  --device cuda \
  --seed 20260713 \
  2>&1 | tee "${LOGS}/dense_bank_extraction.log"

printf '%s  %s\n' \
  e75ddf3e4bee2476e7232c000858730e87f47ff0d0a7779ea2384f1e6873ed34 \
  "${BANK}/dense_features.float16.npy" > "${LOGS}/dense_bank_expected.sha256"
printf '%s  %s\n' \
  af92eb26a78f5b563c50ede322287d5f0342bcefb022ca6f301b9447ab07a48c \
  "${BANK}/valid_token_mask.uint8.npy" >> "${LOGS}/dense_bank_expected.sha256"
printf '%s  %s\n' \
  14ed638d5194353da15b52d16a65f8363869259cb4431680510ee888a3406e9f \
  "${BANK}/spatial_shapes.int16.npy" >> "${LOGS}/dense_bank_expected.sha256"
sha256sum -c "${LOGS}/dense_bank_expected.sha256" 2>&1 | tee "${LOGS}/dense_bank_hash_verification.log"

"${PY}" - "${BANK}/dense_bank_config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "backend": "pe",
    "canonical_model_id": "facebook/PE-Spatial-L14-448",
    "weight_sha256": "47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1",
    "views": ["whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3"],
    "max_num_patches": 1024,
    "feature_dim": 1024,
    "dense_shape": [591, 6, 1024, 1024],
    "complete": True,
    "completed_cases": 591,
}
mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
if mismatch:
    raise SystemExit(f"Dense bank semantic lock failed: {mismatch}")
print(json.dumps({"status": "dense_bank_semantic_lock_passed"}, indent=2))
PY

"${PY}" "${CONTROL_TRAINER}" \
  --feature-bank-dir "${BANK}" \
  --split-csv "${SPLIT}" \
  --output-dir "${CONTROL}" \
  --candidate H9_SOURCE_RISK_CONTROL \
  --split-mode source_lodo \
  --fold all \
  --hidden-dim 128 \
  --attention-dim 64 \
  --dropout 0.10 \
  --epochs 80 \
  --patience 12 \
  --batch-size 8 \
  --num-workers 0 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --grad-clip 5.0 \
  --seed 20260713 \
  --device cuda \
  2>&1 | tee "${LOGS}/source_lodo_control_training.log"

"${PY}" "${TRAINER}" \
  --feature-bank-dir "${BANK}" \
  --split-csv "${SPLIT}" \
  --output-dir "${PRIMARY}" \
  --split-mode source_lodo \
  --hidden-dim 128 \
  --attention-dim 64 \
  --dropout 0.10 \
  --epochs 80 \
  --patience 12 \
  --minimum-epochs 24 \
  --batch-size 8 \
  --num-workers 0 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --grad-clip 5.0 \
  --seed 20260713 \
  --device cuda \
  2>&1 | tee "${LOGS}/source_lodo_primary_training.log"

set +e
"${PY}" "${ANALYZER}" \
  --run-dir "${PRIMARY}" \
  --control-predictions "${CONTROL}/oof_predictions.csv" \
  --c1-predictions "${C1_LODO}" \
  --c2-predictions "${C2_LODO}" \
  --h3-predictions "${H3_LODO}" \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 20260715 \
  --output-dir "${OUT}/source_lodo/primary" \
  --enforce-gates \
  2>&1 | tee "${LOGS}/source_lodo_primary_analysis.log"
analysis_rc=${PIPESTATUS[0]}
set -e
if [[ ${analysis_rc} -ne 0 ]]; then
  if [[ ! -f "${OUT}/source_lodo/primary/gate_decision.json" ]]; then
    exit "${analysis_rc}"
  fi
  write_status stopped_after_source_lodo
  exit 0
fi

write_status source_lodo_primary_passed

"${PY}" "${TRAINER}" \
  --feature-bank-dir "${BANK}" \
  --split-csv "${SPLIT}" \
  --output-dir "${CONFIRMATION}" \
  --split-mode source_lodo \
  --hidden-dim 128 --attention-dim 64 --dropout 0.10 \
  --epochs 80 --patience 12 --minimum-epochs 24 --batch-size 8 --num-workers 0 \
  --lr 0.0003 --weight-decay 0.0001 --grad-clip 5.0 \
  --seed 20260715 --device cuda \
  2>&1 | tee "${LOGS}/source_lodo_confirmation_training.log"

set +e
"${PY}" "${ANALYZER}" \
  --run-dir "${CONFIRMATION}" \
  --control-predictions "${CONTROL}/oof_predictions.csv" \
  --c1-predictions "${C1_LODO}" --c2-predictions "${C2_LODO}" --h3-predictions "${H3_LODO}" \
  --bootstrap-replicates 20000 --bootstrap-seed 20260716 \
  --output-dir "${OUT}/source_lodo/confirmation" --enforce-gates \
  2>&1 | tee "${LOGS}/source_lodo_confirmation_analysis.log"
confirmation_rc=${PIPESTATUS[0]}
set -e
if [[ ${confirmation_rc} -ne 0 ]]; then
  if [[ ! -f "${OUT}/source_lodo/confirmation/gate_decision.json" ]]; then
    exit "${confirmation_rc}"
  fi
  write_status stopped_after_confirmation
  exit 0
fi

write_status source_lodo_confirmed
