#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT=/workspace/thymic_project
PY=/root/miniconda3/envs/thymic_baseline/bin/python3.11
CODE=/root/task7_h10_20260715
OUT=${ROOT}/experiments/h10_internal_phenotype_difficulty_redesign_20260715
LOGS=${CODE}/logs
BANK=${CODE}/bank/pe_spatial_l14_448_dense
SPLIT=${CODE}/internal_subtype_only_split.csv
SPLIT_MANIFEST=${CODE}/internal_subtype_only_split_manifest.json
PE=/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt
PE_SOURCE=/root/third_party/perception_models_3e352cca
REGISTRY=/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv
CONCEPT=${ROOT}/outputs/grosspath_rc_v0_20260526/gross_concepts_v1.csv

EXTRACTOR=${ROOT}/scripts/extract_task7_h3_dense_bank_20260713.py
SPLIT_BUILDER=${CODE}/build_task7_h10_internal_split_20260715.py
TRAINER=${CODE}/run_task7_h10_internal_sampler_20260715.py
ANALYZER=${CODE}/analyze_task7_h10_internal_phenotype_20260715.py
QUEUE=${CODE}/run_task7_h10_internal_phenotype_queue_20260715.sh

NATURAL=${CODE}/artifacts/internal_natural_seed20260713
RISK=${CODE}/artifacts/internal_risk_balanced_seed20260713
TEMPERED=${CODE}/artifacts/internal_subtype_tempered_seed20260713

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

mkdir -p "${OUT}" "${LOGS}" "${CODE}/artifacts" "${BANK}"
trap mark_failed EXIT
write_status running

available_bytes=$(df --output=avail -B1 /root | tail -1 | tr -d ' ')
if (( available_bytes < 16106127360 )); then
  echo "H10 requires at least 15 GiB free on /root; available=${available_bytes}" >&2
  exit 1
fi

sha256sum \
  "${SPLIT_BUILDER}" \
  "${TRAINER}" \
  "${ANALYZER}" \
  "${QUEUE}" \
  "${EXTRACTOR}" \
  "${ROOT}/scripts/run_task7_h3b_masked_gated_20260713.py" \
  "${ROOT}/scripts/run_task7_h3_summary_gated_20260713.py" \
  "${ROOT}/scripts/run_task7_spatial_relational_20260713.py" \
  > "${LOGS}/h10_code.sha256"
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

"${PY}" "${SPLIT_BUILDER}" \
  --metadata "${BANK}/metadata.csv" \
  --output-csv "${SPLIT}" \
  --manifest-json "${SPLIT_MANIFEST}" \
  --seed 20260715 \
  2>&1 | tee "${LOGS}/split_build.log"

run_candidate() {
  sampler=$1
  candidate=$2
  destination=$3
  "${PY}" "${TRAINER}" \
    --h10-sampler "${sampler}" \
    --feature-bank-dir "${BANK}" \
    --split-csv "${SPLIT}" \
    --output-dir "${destination}" \
    --candidate "${candidate}" \
    --split-mode fivefold \
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
    --device cuda
}

run_candidate natural H10_INTERNAL_NATURAL "${NATURAL}" \
  2>&1 | tee "${LOGS}/internal_natural.log"
run_candidate risk_balanced H10_INTERNAL_RISK_BALANCED "${RISK}" \
  2>&1 | tee "${LOGS}/internal_risk_balanced.log"
run_candidate subtype_tempered H10_INTERNAL_SUBTYPE_TEMPERED "${TEMPERED}" \
  2>&1 | tee "${LOGS}/internal_subtype_tempered.log"

"${PY}" "${ANALYZER}" \
  --natural-predictions "${NATURAL}/oof_predictions.csv" \
  --risk-predictions "${RISK}/oof_predictions.csv" \
  --tempered-predictions "${TEMPERED}/oof_predictions.csv" \
  --metadata "${BANK}/metadata.csv" \
  --concept-csv "${CONCEPT}" \
  --split-csv "${SPLIT}" \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 20260715 \
  --output-dir "${OUT}/stage1_internal_baselines" \
  2>&1 | tee "${LOGS}/stage1_analysis.log"

write_status complete
