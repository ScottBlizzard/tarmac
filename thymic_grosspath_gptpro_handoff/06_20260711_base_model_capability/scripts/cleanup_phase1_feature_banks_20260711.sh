#!/usr/bin/env bash
set -euo pipefail

base="/root/thymic_feature_banks_20260711"
experiment_root="/workspace/thymic_project/experiments/base_model_capability_20260711"
archive="$experiment_root/cleaned_feature_bank_manifests_20260711.tar.gz"

banks=(
  dinov2l_whole_crop_336_internal
  eva02l_whole_crop_336_internal
  qkvb_quality4_352_internal
  qkvb_roi4_352_internal
  locked_external
  qkvb_crop_quadrants5_352_internal
  aimv2l_dual17_whole_crop_336
)

for bank in "${banks[@]}"; do
  path="$(realpath -e "$base/$bank")"
  case "$path" in
    "$base"/*) ;;
    *)
      printf 'Refusing path outside feature-bank root: %s\n' "$path" >&2
      exit 2
      ;;
  esac
done

cd "$base"
find "${banks[@]}" -type f \
  \( -name metadata.csv -o -name feature_bank_config.json \) \
  -print0 | tar --null -czf "$archive" --files-from=-

printf 'Archived retained manifests:\n'
ls -lh "$archive"

for bank in "${banks[@]}"; do
  rm -rf -- "$base/$bank"
done

sync
printf '\nRemaining feature banks:\n'
du -sh "$base"/* 2>/dev/null | sort -h
printf '\nRoot filesystem:\n'
df -h /
