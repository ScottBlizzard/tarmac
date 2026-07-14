#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT=/workspace/thymic_project
PYTHON=/root/miniconda3/envs/thymic_baseline/bin/python3.11
EXP=${ROOT}/experiments/h6_nuisance_anchored_csd_20260714
SCRIPTS=${ROOT}/scripts
REGISTRY=/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv
SPLIT=${ROOT}/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv
FREQ=${ROOT}/experiments/frequency_source_vs_risk_audit_20260713
# NumPy memmap requires a POSIX filesystem; /workspace is an OSS object mount.
PE_BANK=/root/thymic_h6_nuisance_anchored_csd_20260714/pe_dense_bank
MANIFEST=${EXP}/manifest/integrity.json
CHECKPOINT=/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt
PE_CODE=/root/third_party/perception_models_3e352cca
H3_ROOT=${ROOT}/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448
C2_ROOT=${ROOT}/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion

EXTRACTOR=${SCRIPTS}/extract_task7_h3_dense_bank_20260713.py
TRAINER=${SCRIPTS}/run_task7_h6_nuisance_csd_20260714.py
ANALYZER=${SCRIPTS}/analyze_task7_h6_nuisance_csd_20260714.py
MANIFEST_BUILDER=${SCRIPTS}/build_task7_h6_integrity_manifest_20260714.py

mkdir -p "${EXP}"/{logs,manifest,aggregate}

"${PYTHON}" "${EXTRACTOR}" \
  --backend pe \
  --registry-csv "${REGISTRY}" \
  --model-id "${CHECKPOINT}" \
  --canonical-model-id facebook/PE-Spatial-L14-448 \
  --weight-sha256 47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1 \
  --model-code-dir "${PE_CODE}" \
  --code-revision 3e352cca660658d4b5c90f42a7808b11469e4c66 \
  --output-dir "${PE_BANK}" \
  --domains old_data,third_batch \
  --views whole,crop,crop_q0,crop_q1,crop_q2,crop_q3 \
  --max-num-patches 1024 \
  --device cuda \
  --seed 20260714 \
  --local-files-only \
  2>&1 | tee "${EXP}/logs/extract_pe_bank.log"

"${PYTHON}" "${MANIFEST_BUILDER}" \
  --feature-bank-dir "${PE_BANK}" \
  --registry-csv "${REGISTRY}" \
  --split-csv "${SPLIT}" \
  --frequency-features "${FREQ}/frequency_features.float32.npy" \
  --frequency-metadata "${FREQ}/frequency_feature_metadata.csv" \
  --h3-oof "${H3_ROOT}/fivefold/oof_predictions.csv" \
  --h3-lodo "${H3_ROOT}/source_lodo/oof_predictions.csv" \
  --c2-oof "${C2_ROOT}/oof_predictions.csv" \
  --c2-lodo "${C2_ROOT}/lodo_predictions.csv" \
  --trainer "${TRAINER}" \
  --analyzer "${ANALYZER}" \
  --extractor "${EXTRACTOR}" \
  --model-checkpoint "${CHECKPOINT}" \
  --model-code-dir "${PE_CODE}" \
  --output-dir "${EXP}/manifest" \
  2>&1 | tee "${EXP}/logs/build_integrity_manifest.log"

for VARIANT in nuisance_csd source_csd; do
  "${PYTHON}" "${TRAINER}" \
    --feature-bank-dir "${PE_BANK}" \
    --frequency-features "${FREQ}/frequency_features.float32.npy" \
    --frequency-metadata "${FREQ}/frequency_feature_metadata.csv" \
    --split-csv "${SPLIT}" \
    --integrity-manifest "${MANIFEST}" \
    --variant "${VARIANT}" \
    --split-mode source_lodo \
    --output-dir "${EXP}/source_lodo/${VARIANT}" \
    --seed 20260714 \
    --device cuda \
    2>&1 | tee "${EXP}/logs/source_lodo_${VARIANT}.log"
done

for VARIANT in nuisance_csd source_csd; do
  "${PYTHON}" "${TRAINER}" \
    --feature-bank-dir "${PE_BANK}" \
    --frequency-features "${FREQ}/frequency_features.float32.npy" \
    --frequency-metadata "${FREQ}/frequency_feature_metadata.csv" \
    --split-csv "${SPLIT}" \
    --integrity-manifest "${MANIFEST}" \
    --variant "${VARIANT}" \
    --split-mode fivefold \
    --output-dir "${EXP}/fivefold/${VARIANT}" \
    --seed 20260714 \
    --device cuda \
    2>&1 | tee "${EXP}/logs/fivefold_${VARIANT}.log"
done

ANALYZE_ARGS=(
  --h6-root "${EXP}"
  --h3-oof "${H3_ROOT}/fivefold/oof_predictions.csv"
  --h3-lodo "${H3_ROOT}/source_lodo/oof_predictions.csv"
  --c2-oof "${C2_ROOT}/oof_predictions.csv"
  --c2-lodo "${C2_ROOT}/lodo_predictions.csv"
  --output-dir "${EXP}/aggregate"
  --bootstrap-repetitions 20000
  --seed 20260714
)
"${PYTHON}" "${ANALYZER}" "${ANALYZE_ARGS[@]}" \
  2>&1 | tee "${EXP}/logs/analyze_primary.log"

NEXT_ACTION=$("${PYTHON}" -c "import json; print(json.load(open('${EXP}/aggregate/decision.json'))['next_action'])")
if [[ "${NEXT_ACTION}" == "RUN_CONFIRMATION_SEED_20260715" ]]; then
  for PROTOCOL in source_lodo fivefold; do
    "${PYTHON}" "${TRAINER}" \
      --feature-bank-dir "${PE_BANK}" \
      --frequency-features "${FREQ}/frequency_features.float32.npy" \
      --frequency-metadata "${FREQ}/frequency_feature_metadata.csv" \
      --split-csv "${SPLIT}" \
      --integrity-manifest "${MANIFEST}" \
      --variant nuisance_csd \
      --split-mode "${PROTOCOL}" \
      --output-dir "${EXP}/confirmation_seed20260715/${PROTOCOL}/nuisance_csd" \
      --seed 20260715 \
      --device cuda \
      2>&1 | tee "${EXP}/logs/confirmation_${PROTOCOL}_nuisance_csd.log"
  done
  "${PYTHON}" "${ANALYZER}" "${ANALYZE_ARGS[@]}" \
    --confirmation-root "${EXP}/confirmation_seed20260715" \
    2>&1 | tee "${EXP}/logs/analyze_confirmation.log"
elif [[ "${NEXT_ACTION}" == "RUN_SINGLE_CONDITIONAL_FISHR_BACKUP" ]]; then
  printf '%s\n' "Conditional Fishr trigger passed; implementation audit is required before the single backup run." \
    | tee "${EXP}/FISHR_REQUIRED.status"
else
  printf '%s\n' "${NEXT_ACTION}" | tee "${EXP}/FINAL.status"
fi

printf '%s\n' "queue_complete" > "${EXP}/QUEUE.status"
