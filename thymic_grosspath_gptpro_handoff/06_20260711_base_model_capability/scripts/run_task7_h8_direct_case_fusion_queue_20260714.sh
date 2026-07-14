#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT=/workspace/thymic_project
PY=/root/miniconda3/envs/thymic_baseline/bin/python3.11
CODE=/root/task7_h8_20260714
OUT=${ROOT}/experiments/h8_c1_h3_direct_case_fusion_20260714
LOGS=${CODE}/logs
ARTIFACTS=${CODE}/artifacts
SOURCE_EMBEDDINGS=${ARTIFACTS}/source_lodo/embeddings
SOURCE_PRIMARY=${ARTIFACTS}/source_lodo/primary_seed20260714
FIVEFOLD_EMBEDDINGS=${ARTIFACTS}/fivefold/embeddings
FIVEFOLD_PRIMARY=${ARTIFACTS}/fivefold/primary_seed20260714
CONFIRMATION_RUN=${ARTIFACTS}/confirmation/source_lodo_seed20260715

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

REGISTRY=/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv
SPLIT=${ROOT}/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv
C1_BANK=/root/thymic_feature_banks_20260711/phase2_siglipl512_whole_crop_quadrants6_512_internal
PE=/root/model_weights/modelscope/facebook/PE-Spatial-L14-448/PE-Spatial-L14-448.pt
PE_SOURCE=/root/third_party/perception_models_3e352cca

C1_LODO=${ROOT}/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711
C2_LODO=${ROOT}/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/lodo_predictions.csv
H3_LODO=${ROOT}/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/source_lodo

C1_5F=${ROOT}/experiments/base_model_capability_20260711/phase2_siglipl512_local_pyramid_screen/347_siglipl512_localpyramid6_gated_fivefold_cw_20260711
C2_5F=${ROOT}/experiments/base_model_capability_20260711/phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/oof_predictions.csv
H3_5F=${ROOT}/experiments/h3_representation_renewal_20260713/h3b_runs/pe_spatial_l14_448/fivefold

LOCKER=${CODE}/lock_task7_h8_assets_20260714.py
EXTRACTOR=${CODE}/extract_task7_h8_fold_embeddings_20260714.py
TRAINER=${CODE}/run_task7_h8_direct_case_fusion_20260714.py
ANALYZER=${CODE}/analyze_task7_h8_direct_case_fusion_20260714.py
QUEUE=${CODE}/run_task7_h8_direct_case_fusion_queue_20260714.sh

mkdir -p "${OUT}"/{locks,source_lodo,fivefold,confirmation} "${LOGS}" "${ARTIFACTS}"
trap mark_failed EXIT
write_status running
sha256sum "${LOCKER}" "${EXTRACTOR}" "${TRAINER}" "${ANALYZER}" "${QUEUE}" > "${LOGS}/code.sha256.tmp"
sha256sum \
  "${ROOT}/scripts/extract_task7_h3_dense_bank_20260713.py" \
  "${ROOT}/scripts/extract_task7_h3_representation_bank_20260713.py" \
  "${ROOT}/scripts/extract_task7_dense_token_bank_20260711.py" \
  "${ROOT}/scripts/run_task7_dense_feature_cv_20260711.py" \
  "${ROOT}/scripts/run_task7_h3b_masked_gated_20260713.py" \
  "${ROOT}/scripts/run_task7_h3_summary_gated_20260713.py" \
  "${ROOT}/scripts/run_task7_spatial_relational_20260713.py" >> "${LOGS}/code.sha256.tmp"
mv -f "${LOGS}/code.sha256.tmp" "${LOGS}/code.sha256"

"${PY}" "${LOCKER}" \
  --evidence-commit ff81fe4c98fd5f1b9d2bcfd53f4379e5741e38e8 \
  --split-mode source_lodo \
  --registry-csv "${REGISTRY}" \
  --split-csv "${SPLIT}" \
  --c1-feature-bank "${C1_BANK}" \
  --c1-root "${C1_LODO}" \
  --c1-predictions "${C1_LODO}/oof_predictions.csv" \
  --c2-predictions "${C2_LODO}" \
  --h3-root "${H3_LODO}" \
  --h3-predictions "${H3_LODO}/oof_predictions.csv" \
  --pe-checkpoint "${PE}" \
  --expected-pe-sha256 47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1 \
  --pe-source-root "${PE_SOURCE}" \
  --expected-pe-source-revision 3e352cca660658d4b5c90f42a7808b11469e4c66 \
  --output-manifest "${OUT}/locks/source_lodo_assets.json" \
  2>&1 | tee "${LOGS}/source_lodo_asset_lock.log"

"${PY}" "${EXTRACTOR}" \
  --asset-manifest "${OUT}/locks/source_lodo_assets.json" \
  --split-mode source_lodo \
  --views whole,crop,crop_q0,crop_q1,crop_q2,crop_q3 \
  --c1-image-size 512 \
  --h3-image-size 448 \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda \
  --seed 20260714 \
  --output-dir "${SOURCE_EMBEDDINGS}" \
  2>&1 | tee "${LOGS}/source_lodo_embedding_extraction.log"

"${PY}" "${TRAINER}" \
  --embedding-manifest "${SOURCE_EMBEDDINGS}/embedding_manifest.json" \
  --split-csv "${SPLIT}" \
  --split-mode source_lodo \
  --configuration H8_C1_H3_CONCAT_MLP16 \
  --hidden-dim 16 \
  --dropout 0.10 \
  --epochs 80 \
  --patience 12 \
  --batch-size 32 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --grad-clip 5.0 \
  --seed 20260714 \
  --device cuda \
  --output-dir "${SOURCE_PRIMARY}" \
  2>&1 | tee "${LOGS}/source_lodo_primary_training.log"

if ! "${PY}" "${ANALYZER}" \
  --stage source_lodo \
  --run-dir "${SOURCE_PRIMARY}" \
  --c1-predictions "${C1_LODO}/oof_predictions.csv" \
  --c2-predictions "${C2_LODO}" \
  --h3-predictions "${H3_LODO}/oof_predictions.csv" \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 20260714 \
  --enforce-gates \
  --output-dir "${OUT}/source_lodo/aggregate" \
  2>&1 | tee "${LOGS}/source_lodo_analysis.log"; then
  write_status stopped_after_source_lodo
  exit 0
fi

"${PY}" "${LOCKER}" \
  --evidence-commit ff81fe4c98fd5f1b9d2bcfd53f4379e5741e38e8 \
  --split-mode fivefold \
  --registry-csv "${REGISTRY}" \
  --split-csv "${SPLIT}" \
  --c1-feature-bank "${C1_BANK}" \
  --c1-root "${C1_5F}" \
  --c1-predictions "${C1_5F}/oof_predictions.csv" \
  --c2-predictions "${C2_5F}" \
  --h3-root "${H3_5F}" \
  --h3-predictions "${H3_5F}/oof_predictions.csv" \
  --pe-checkpoint "${PE}" \
  --expected-pe-sha256 47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1 \
  --pe-source-root "${PE_SOURCE}" \
  --expected-pe-source-revision 3e352cca660658d4b5c90f42a7808b11469e4c66 \
  --output-manifest "${OUT}/locks/fivefold_assets.json" \
  2>&1 | tee "${LOGS}/fivefold_asset_lock.log"

"${PY}" "${EXTRACTOR}" \
  --asset-manifest "${OUT}/locks/fivefold_assets.json" \
  --split-mode fivefold \
  --views whole,crop,crop_q0,crop_q1,crop_q2,crop_q3 \
  --c1-image-size 512 \
  --h3-image-size 448 \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda \
  --seed 20260714 \
  --output-dir "${FIVEFOLD_EMBEDDINGS}" \
  2>&1 | tee "${LOGS}/fivefold_embedding_extraction.log"

"${PY}" "${TRAINER}" \
  --embedding-manifest "${FIVEFOLD_EMBEDDINGS}/embedding_manifest.json" \
  --split-csv "${SPLIT}" \
  --split-mode fivefold \
  --configuration H8_C1_H3_CONCAT_MLP16 \
  --hidden-dim 16 --dropout 0.10 --epochs 80 --patience 12 --batch-size 32 \
  --lr 0.0003 --weight-decay 0.0001 --grad-clip 5.0 \
  --seed 20260714 --device cuda \
  --output-dir "${FIVEFOLD_PRIMARY}" \
  2>&1 | tee "${LOGS}/fivefold_training.log"

if ! "${PY}" "${ANALYZER}" \
  --stage fivefold \
  --run-dir "${FIVEFOLD_PRIMARY}" \
  --c1-predictions "${C1_5F}/oof_predictions.csv" \
  --c2-predictions "${C2_5F}" \
  --h3-predictions "${H3_5F}/oof_predictions.csv" \
  --bootstrap-replicates 20000 --bootstrap-seed 20260714 --enforce-gates \
  --output-dir "${OUT}/fivefold/aggregate" \
  2>&1 | tee "${LOGS}/fivefold_analysis.log"; then
  write_status stopped_after_fivefold
  exit 0
fi

"${PY}" "${TRAINER}" \
  --embedding-manifest "${SOURCE_EMBEDDINGS}/embedding_manifest.json" \
  --split-csv "${SPLIT}" --split-mode source_lodo \
  --configuration H8_C1_H3_CONCAT_MLP16 \
  --hidden-dim 16 --dropout 0.10 --epochs 80 --patience 12 --batch-size 32 \
  --lr 0.0003 --weight-decay 0.0001 --grad-clip 5.0 \
  --seed 20260715 --device cuda \
  --output-dir "${CONFIRMATION_RUN}" \
  2>&1 | tee "${LOGS}/confirmation_training.log"

if ! "${PY}" "${ANALYZER}" \
  --stage confirmation \
  --run-dir "${CONFIRMATION_RUN}" \
  --primary-run-dir "${SOURCE_PRIMARY}" \
  --c1-predictions "${C1_LODO}/oof_predictions.csv" \
  --c2-predictions "${C2_LODO}" \
  --h3-predictions "${H3_LODO}/oof_predictions.csv" \
  --bootstrap-replicates 20000 --bootstrap-seed 20260715 --enforce-gates \
  --output-dir "${OUT}/confirmation/aggregate" \
  2>&1 | tee "${LOGS}/confirmation_analysis.log"; then
  write_status stopped_after_confirmation
  exit 0
fi

write_status complete_all_gates_pass
