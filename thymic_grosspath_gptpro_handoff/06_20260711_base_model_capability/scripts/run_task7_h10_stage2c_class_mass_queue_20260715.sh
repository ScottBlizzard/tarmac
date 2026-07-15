#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export H10_STAGE2_ALLOW_SMOKE=1

ROOT=/workspace/thymic_project
PY=/root/miniconda3/envs/thymic_baseline/bin/python3.11
CODE=/root/task7_h10_20260715
BANK=${CODE}/bank/pe_spatial_l14_448_dense
SPLIT=${CODE}/internal_subtype_only_split.csv
CONCEPT=${ROOT}/outputs/grosspath_rc_v0_20260526/gross_concepts_v1.csv
BASELINE=${CODE}/artifacts/internal_risk_balanced_seed20260713/oof_predictions.csv
ROLE_SOURCE=${CODE}/artifacts/nested_phenotype_curriculum_seed20260713
RUN=${CODE}/artifacts/nested_phenotype_class_mass_seed20260713
OUT=${ROOT}/experiments/h10_internal_phenotype_difficulty_redesign_20260715/stage2c_class_mass_preserving
LOGS=${CODE}/logs_stage2c
TRAINER=${CODE}/run_task7_h10_stage2c_class_mass_preserving_20260715.py
BASE_TRAINER=${CODE}/run_task7_h10_nested_phenotype_curriculum_20260715.py
ANALYZER=${CODE}/analyze_task7_h10_nested_phenotype_curriculum_20260715.py
QUEUE=${CODE}/run_task7_h10_stage2c_class_mass_queue_20260715.sh

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

mkdir -p "${OUT}" "${LOGS}" "${RUN}"
trap mark_failed EXIT
write_status running

sha256sum "${TRAINER}" "${BASE_TRAINER}" "${ANALYZER}" "${QUEUE}" > "${LOGS}/code.sha256"

actual_split_sha=$(sha256sum "${SPLIT}" | cut -d' ' -f1)
if [[ "${actual_split_sha}" != "f13030d7467c907851ed89abe14e385e024fcdddb6ad4be26433bc7994a10beb" ]]; then
  echo "H10 subtype-only split changed: ${actual_split_sha}" >&2
  exit 1
fi

: > "${LOGS}/copied_roles.sha256"
for fold in 1 2 3 4 5; do
  source_file=${ROLE_SOURCE}/fold_${fold}/nested_training_roles_server_only.csv
  destination_dir=${RUN}/fold_${fold}
  destination_file=${destination_dir}/nested_training_roles_server_only.csv
  mkdir -p "${destination_dir}"
  cp -f "${source_file}" "${destination_file}"
  sha=$(sha256sum "${source_file}" | cut -d' ' -f1)
  printf '%s  %s\n' "${sha}" "${destination_file}" >> "${LOGS}/copied_roles.sha256"
done
sha256sum -c "${LOGS}/copied_roles.sha256" | tee "${LOGS}/role_hash_verification.log"

"${PY}" "${TRAINER}" \
  --feature-bank-dir "${BANK}" \
  --split-csv "${SPLIT}" \
  --concept-csv "${CONCEPT}" \
  --output-dir "${RUN}" \
  --fold all \
  --inner-folds 3 \
  --teacher-epochs 12 \
  --epochs 16 \
  --minimum-epochs 1 \
  --patience 16 \
  --batch-size 8 \
  --num-workers 0 \
  --hidden-dim 128 \
  --attention-dim 64 \
  --dropout 0.10 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --grad-clip 5.0 \
  --seed 20260713 \
  --device cuda \
  --max-epochs 16 \
  2>&1 | tee "${LOGS}/class_mass_training.log"

"${PY}" "${ANALYZER}" \
  --candidate-predictions "${RUN}/oof_predictions.csv" \
  --baseline-predictions "${BASELINE}" \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 20260715 \
  --output-dir "${OUT}" \
  2>&1 | tee "${LOGS}/class_mass_analysis.log"

write_status complete
