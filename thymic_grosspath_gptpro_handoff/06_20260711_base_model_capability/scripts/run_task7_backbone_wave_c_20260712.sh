#!/usr/bin/env bash
set -euo pipefail

python_bin="/root/miniconda3/envs/thymic_baseline/bin/python"
extractor="/root/extract_task7_dense_token_bank_20260711.py"
trainer="/root/run_task7_dense_feature_cv_20260711.py"
registry="/workspace/thymic_project/experiments/base_model_expansion_20260706/outputs/registry/task7_four_domain_master_registry.csv"
split_csv="/workspace/thymic_project/outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/45_old_third_all_balanced_finetune_inputs_20260523/split.csv"
bank_root="/root/thymic_feature_banks_20260712"
output_root="/workspace/thymic_project/experiments/base_model_capability_20260711/backbone_wave_c_sixview_20260712"
manifest_root="$output_root/feature_bank_manifests"
status_file="$output_root/BACKBONE_WAVE_C.status"
views="whole,crop,crop_q0,crop_q1,crop_q2,crop_q3"
checkpoint_root="/root/model_weights_20260712"

mkdir -p "$bank_root" "$output_root" "$manifest_root"

write_status() {
  rm -f -- "$status_file"
  printf '%s\n' "$1" > "$status_file"
}

on_exit() {
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    write_status complete
  else
    write_status "failed rc=$rc"
  fi
}
trap on_exit EXIT

run_cv() {
  local run_name="$1"
  local bank_dir="$2"
  local split_mode="$3"
  local output_dir="$output_root/$run_name"
  if [[ -s "$output_dir/oof_metrics.csv" && -s "$output_dir/oof_predictions.csv" ]]; then
    printf 'SKIP-CV %s\n' "$run_name"
    return
  fi
  printf 'START-CV %s %s\n' "$run_name" "$(date --iso-8601=seconds)"
  "$python_bin" "$trainer" \
    --feature-bank-dir "$bank_dir" \
    --output-dir "$output_dir" \
    --split-csv "$split_csv" \
    --split-mode "$split_mode" \
    --pooling gated \
    --class-weighting \
    --epochs 80 \
    --patience 12 \
    --batch-size 24 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --seed 20260712 \
    --device cuda
  printf 'DONE-CV %s %s\n' "$run_name" "$(date --iso-8601=seconds)"
}

archive_and_remove_bank() {
  local bank_name="$1"
  local bank_dir="$bank_root/$bank_name"
  local resolved
  resolved="$(realpath -e "$bank_dir")"
  case "$resolved" in
    "$bank_root"/wave_c_*) ;;
    *)
      printf 'Refusing to remove unexpected feature path: %s\n' "$resolved" >&2
      exit 4
      ;;
  esac
  mkdir -p "$manifest_root/$bank_name"
  cp "$bank_dir/metadata.csv" "$bank_dir/feature_bank_config.json" "$manifest_root/$bank_name/"
  sha256sum "$bank_dir/metadata.csv" "$bank_dir/feature_bank_config.json" \
    > "$manifest_root/$bank_name/SHA256SUMS.txt"
  rm -rf -- "$resolved"
  sync
}

screen_backbone() {
  local bank_name="$1"
  local model_name="$2"
  local checkpoint_path="$3"
  local image_size="$4"
  local extract_batch="$5"
  local oof_name="$6"
  local lodo_name="$7"
  local bank_dir="$bank_root/$bank_name"

  if [[ ! -s "$checkpoint_path" ]]; then
    printf 'Missing checkpoint: %s\n' "$checkpoint_path" >&2
    exit 5
  fi

  write_status "running $bank_name"
  if [[ ! -s "$output_root/$oof_name/oof_metrics.csv" || ! -s "$output_root/$lodo_name/oof_metrics.csv" ]]; then
    mkdir -p "$bank_dir"
    printf 'START-BANK %s model=%s %s\n' "$bank_name" "$model_name" "$(date --iso-8601=seconds)"
    "$python_bin" "$extractor" \
      --registry-csv "$registry" \
      --model-name "$model_name" \
      --checkpoint-path "$checkpoint_path" \
      --image-size "$image_size" \
      --views "$views" \
      --output-dir "$bank_dir" \
      --batch-size "$extract_batch" \
      --num-workers 4
    printf 'DONE-BANK %s %s\n' "$bank_name" "$(date --iso-8601=seconds)"
    run_cv "$oof_name" "$bank_dir" fivefold
    run_cv "$lodo_name" "$bank_dir" source_lodo
  fi
  if [[ -d "$bank_dir" ]]; then
    archive_and_remove_bank "$bank_name"
  fi
  df -h /
}

write_status running

# Prespecified family 1: hierarchical local attention at 384 px.
screen_backbone \
  wave_c_swinv2l_sixview_384_internal \
  swinv2_large_window12to24_192to384.ms_in22k_ft_in1k \
  "$checkpoint_root/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k.safetensors" \
  384 1 \
  370_swinv2l_sixview_gated_oof_20260712 \
  371_swinv2l_sixview_gated_lodo_20260712

# Prespecified family 2: convolutional FCMAE representation at 384 px.
screen_backbone \
  wave_c_convnextv2l_sixview_384_internal \
  convnextv2_large.fcmae_ft_in22k_in1k_384 \
  "$checkpoint_root/convnextv2_large.fcmae_ft_in22k_in1k_384.safetensors" \
  384 2 \
  372_convnextv2l_sixview_gated_oof_20260712 \
  373_convnextv2l_sixview_gated_lodo_20260712

# Prespecified family 3: SigLIP capacity-resolution interaction at 512 px.
screen_backbone \
  wave_c_siglipso400m_sixview_512_internal \
  vit_so400m_patch16_siglip_512.v2_webli \
  "$checkpoint_root/vit_so400m_patch16_siglip_512.v2_webli.safetensors" \
  512 1 \
  374_siglipso400m_sixview_gated_oof_20260712 \
  375_siglipso400m_sixview_gated_lodo_20260712

write_status complete
