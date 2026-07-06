from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_biomedclip_frozen_probe import (  # noqa: E402
    extract_dataset_features as extract_biomedclip_features,
    load_biomedclip_model,
)
from run_plip_caseprobe import (  # noqa: E402
    extract_dataset_features as extract_plip_features,
    load_plip_model,
)
from run_task7_external_thymoma_carcinoma_folder_20260522 import build_external_registry  # noqa: E402
from thymic_baseline.config import get_task  # noqa: E402
from thymic_baseline.registry import expand_registry_to_images, filter_registry_for_task, load_registry, load_split_assignments, merge_registry_with_splits  # noqa: E402
from thymic_baseline.train import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract old/external Task7 features using BiomedCLIP or PLIP.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--encoder", choices=("biomedclip", "plip"), required=True)
    parser.add_argument("--input-variant", default="whole", choices=("whole", "crop", "whole_plus_crop"))
    parser.add_argument(
        "--old-registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--old-split-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_5fold_assignments.csv",
    )
    parser.add_argument("--old-images-root", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/selected_images")
    parser.add_argument("--external-images-root", default="datasets/external_thymoma_carcinoma_20260522")
    parser.add_argument("--old-output-dir", required=True)
    parser.add_argument("--external-output-dir", required=True)
    parser.add_argument("--biomedclip-model-dir", default="third_party/round3/local_models/biomedclip")
    parser.add_argument("--plip-model-dir", default="third_party/round3/local_models/plip")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_old_image_df(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    task = get_task("task7_lowrisk_vs_highrisk_tc")
    registry = load_registry(root / args.old_registry_csv)
    split_df = load_split_assignments(root / args.old_split_csv)
    merged = merge_registry_with_splits(registry, split_df)
    filtered = filter_registry_for_task(merged, task)
    return expand_registry_to_images(filtered, task=task, images_root=root / args.old_images_root)


def load_external_image_df(root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = build_external_registry(root / args.external_images_root)
    image_df = registry[["case_id", "label_idx", "image_path", "image_name"]].copy()
    return registry, image_df


def aggregate_to_cases(
    features: np.ndarray,
    labels: np.ndarray,
    case_ids: list[str],
    image_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    groups: dict[str, dict[str, Any]] = {}
    for idx, case_id in enumerate(case_ids):
        item = groups.setdefault(case_id, {"features": [], "label_idx": int(labels[idx]), "image_names": []})
        item["features"].append(features[idx])
        item["image_names"].append(str(image_names[idx]))
    rows = []
    case_features = []
    for feature_idx, case_id in enumerate(sorted(groups), start=0):
        item = groups[case_id]
        case_features.append(np.mean(np.stack(item["features"], axis=0), axis=0))
        rows.append(
            {
                "case_id": case_id,
                "feature_idx": feature_idx,
                "label_idx": int(item["label_idx"]),
                "image_names": "|".join(item["image_names"]),
            }
        )
    return pd.DataFrame(rows), np.stack(case_features, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    old_out = root / args.old_output_dir
    external_out = root / args.external_output_dir
    old_out.mkdir(parents=True, exist_ok=True)
    external_out.mkdir(parents=True, exist_ok=True)

    old_image_df = load_old_image_df(root, args)
    external_registry, external_image_df = load_external_image_df(root, args)
    device = resolve_device(args.device)

    if args.encoder == "biomedclip":
        model, preprocess = load_biomedclip_model(str(root / args.biomedclip_model_dir), device)
        extractor = extract_biomedclip_features
        extractor_args = (args.input_variant, args.batch_size, args.num_workers, preprocess, model, device)
    else:
        model, processor = load_plip_model(root / args.plip_model_dir, device)
        extractor = extract_plip_features
        extractor_args = (args.input_variant, args.batch_size, args.num_workers, processor, model, device)

    print(f"[extract] encoder={args.encoder} old_images={len(old_image_df)} external_images={len(external_image_df)}", flush=True)
    old_feat, old_labels, old_case_ids, old_image_names = extractor(old_image_df, *extractor_args)
    external_feat, external_labels, external_case_ids, external_image_names = extractor(external_image_df, *extractor_args)

    old_table, old_cases = aggregate_to_cases(old_feat, old_labels, old_case_ids, old_image_names)
    external_table, external_cases = aggregate_to_cases(external_feat, external_labels, external_case_ids, external_image_names)

    old_table.to_csv(old_out / "case_dino_concat_feature_table.csv", index=False, encoding="utf-8-sig")
    np.save(old_out / "case_dino_concat_features.npy", old_cases)
    external_table[["case_id", "feature_idx"]].to_csv(external_out / "third_batch_dino_concat_feature_table.csv", index=False, encoding="utf-8-sig")
    np.save(external_out / "third_batch_dino_concat_features.npy", external_cases)
    external_registry.to_csv(external_out / "external_folder_task7_registry.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "encoder": args.encoder,
        "input_variant": args.input_variant,
        "old_cases": int(len(old_table)),
        "external_cases": int(len(external_table)),
        "feature_dim": int(old_cases.shape[1]),
    }
    (external_out / "feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] old={old_cases.shape} external={external_cases.shape}", flush=True)


if __name__ == "__main__":
    main()
