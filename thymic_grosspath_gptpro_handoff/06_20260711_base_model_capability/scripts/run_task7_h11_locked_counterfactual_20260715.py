from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

_SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = (
    _SCRIPT_PROJECT_ROOT
    if (_SCRIPT_PROJECT_ROOT / "thymic_baseline").exists()
    else Path("/workspace/thymic_project")
)
for item in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from extract_task7_dense_token_bank_20260711 import make_view  # noqa: E402
from extract_task7_h3_dense_bank_20260713 import PeAdapter  # noqa: E402
from run_task7_h3b_masked_gated_20260713 import MaskedDenseGatedHead  # noqa: E402
from thymic_baseline.cropping import (  # noqa: E402
    detect_specimen_bbox,
    detect_specimen_mask,
    expand_bbox,
    extract_specimen_crop,
)

STANDARD_VIEWS = ("whole", "crop", "crop_q0", "crop_q1", "crop_q2", "crop_q3")
DEFAULT_VARIANTS = (
    "original_standard6",
    "tight_standard6",
    "neutral_background_standard6",
    "scale_normalized_standard6",
    "evidence_top4",
    "background_only_standard6",
)
MODEL_NAMES = ("natural", "risk_balanced", "subtype_tempered")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked H11 counterfactual inference without retraining or threshold tuning."
    )
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--natural-root", required=True)
    parser.add_argument("--risk-balanced-root", required=True)
    parser.add_argument("--subtype-tempered-root", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-code-dir", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--max-num-patches", type=int, default=1024)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--reproduction-tolerance", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, records: dict[tuple[str, str], dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(records):
            handle.write(json.dumps(records[key], ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[(str(record["case_id"]), str(record["variant"]))] = record
    return records


def validate_unique(frame: pd.DataFrame, name: str) -> None:
    if frame["case_id"].duplicated().any():
        examples = frame.loc[frame["case_id"].duplicated(), "case_id"].head().tolist()
        raise ValueError(f"{name} contains duplicate case_id values: {examples}")


def load_records(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest = pd.read_csv(args.manifest_csv, dtype={"case_id": str, "original_case_id": str})
    registry = pd.read_csv(args.registry_csv, dtype={"case_id": str, "original_case_id": str})
    split = pd.read_csv(args.split_csv, dtype={"case_id": str})
    for frame, name in ((manifest, "manifest"), (registry, "registry"), (split, "split")):
        validate_unique(frame, name)

    registry = registry.set_index("case_id")
    split = split.set_index("case_id")
    missing_registry = sorted(set(manifest["case_id"]) - set(registry.index))
    missing_split = sorted(set(manifest["case_id"]) - set(split.index))
    if missing_registry or missing_split:
        raise ValueError(
            f"Missing joins: registry={missing_registry[:5]} split={missing_split[:5]}"
        )

    records = manifest.copy()
    records["image_name"] = records["case_id"].map(registry["image_name"])
    records["image_path"] = records["case_id"].map(registry["image_path"])
    records["registry_label"] = records["case_id"].map(registry["label_idx"]).astype(int)
    records["registry_subtype"] = records["case_id"].map(registry["task_l6_label"])
    records["registry_fold"] = records["case_id"].map(registry["master_fold_id"]).astype(int)
    records["split_fold"] = records["case_id"].map(split["master_fold_id"]).astype(int)
    records["label_idx"] = records["label_idx"].astype(int)
    records["fold_id"] = records["fold_id"].astype(int)
    records["model_correct_count"] = records["model_correct_count"].astype(int)
    records["image_name_match"] = records["local_image_name"].fillna("").astype(str).eq(
        records["image_name"].fillna("").astype(str)
    )
    mismatch = records[
        (records["label_idx"] != records["registry_label"])
        | (records["task_l6_label"] != records["registry_subtype"])
        | (records["fold_id"] != records["registry_fold"])
        | (records["fold_id"] != records["split_fold"])
    ]
    if not mismatch.empty:
        raise ValueError(f"Manifest/registry/split mismatch for {len(mismatch)} cases")
    missing_images = [path for path in records["image_path"] if not Path(str(path)).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} cached images")
    records = records.sort_values("case_id").reset_index(drop=True)
    if args.max_cases is not None:
        records = records.head(int(args.max_cases)).copy()

    model_roots = {
        "natural": Path(args.natural_root),
        "risk_balanced": Path(args.risk_balanced_root),
        "subtype_tempered": Path(args.subtype_tempered_root),
    }
    oof_frames: dict[str, pd.DataFrame] = {}
    for name, root in model_roots.items():
        oof_path = root / "oof_predictions.csv"
        if not oof_path.is_file():
            raise FileNotFoundError(oof_path)
        oof = pd.read_csv(oof_path, dtype={"case_id": str}).set_index("case_id")
        missing = sorted(set(records["case_id"]) - set(oof.index))
        if missing:
            raise ValueError(f"{name} OOF is missing {missing[:5]}")
        oof_frames[name] = oof
    return records, oof_frames


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)
    image_area = binary.shape[0] * binary.shape[1]
    candidates: list[tuple[int, int]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < 0.005 * image_area:
            continue
        if width >= 0.98 * binary.shape[1] and height >= 0.98 * binary.shape[0]:
            continue
        candidates.append((area, label))
    if not candidates:
        label = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        label = max(candidates)[1]
    return labels == label


def neutral_background(image: Image.Image) -> tuple[Image.Image, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = detect_specimen_mask(rgb) > 0
    output = np.full_like(rgb, 127)
    output[mask] = rgb[mask]
    return Image.fromarray(output, mode="RGB"), float(mask.mean())


def background_only(image: Image.Image) -> tuple[Image.Image, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    mask = detect_specimen_mask(rgb) > 0
    rgb[mask] = 127
    return Image.fromarray(rgb, mode="RGB"), float(mask.mean())


def scale_normalized(image: Image.Image, occupancy: float = 0.84) -> tuple[Image.Image, dict[str, Any]]:
    tight = extract_specimen_crop(image, margin_ratio=0.02).convert("RGB")
    side = max(1, int(math.ceil(max(tight.size) / occupancy)))
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    offset = ((side - tight.width) // 2, (side - tight.height) // 2)
    canvas.paste(tight, offset)
    return canvas, {
        "tight_width": tight.width,
        "tight_height": tight.height,
        "canvas_side": side,
        "target_long_side_occupancy": occupancy,
    }


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return float(intersection / union) if union else 0.0


def select_nonoverlapping(
    candidates: list[dict[str, Any]], count: int, selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = list(selected)
    for candidate in sorted(candidates, key=lambda item: (-item["score"], item["box"])):
        if all(box_iou(tuple(candidate["box"]), tuple(item["box"])) <= 0.35 for item in output):
            output.append(candidate)
        if len(output) >= len(selected) + count:
            break
    return output[len(selected) :]


def evidence_top4(image: Image.Image) -> tuple[list[Image.Image], dict[str, Any]]:
    tight = extract_specimen_crop(image, margin_ratio=0.02).convert("RGB")
    rgb = np.asarray(tight, dtype=np.uint8)
    height, width = rgb.shape[:2]
    component = largest_component(detect_specimen_mask(rgb))
    gray = rgb.astype(np.float32).mean(axis=2) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(np.diff(gray, axis=1))
    gy[1:, :] = np.abs(np.diff(gray, axis=0))
    gradient = np.minimum(1.0, gx + gy)
    boundary = cv2.morphologyEx(
        component.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)
    ).astype(bool)

    candidates: list[dict[str, Any]] = []
    for scale in (0.44, 0.58):
        patch_height = min(height, max(16, int(round(height * scale))))
        patch_width = min(width, max(16, int(round(width * scale))))
        starts_y = np.unique(np.linspace(0, max(0, height - patch_height), 5).round().astype(int))
        starts_x = np.unique(np.linspace(0, max(0, width - patch_width), 5).round().astype(int))
        for y1 in starts_y:
            for x1 in starts_x:
                y2, x2 = int(y1 + patch_height), int(x1 + patch_width)
                local_mask = component[y1:y2, x1:x2]
                coverage = float(local_mask.mean())
                local_gray = gray[y1:y2, x1:x2]
                pixels = rgb[y1:y2, x1:x2][local_mask]
                if pixels.size:
                    color_std = float(pixels.astype(np.float32).std(axis=0).mean() / 255.0)
                    gray_std = float(local_gray[local_mask].std())
                    gradient_mean = float(gradient[y1:y2, x1:x2][local_mask].mean())
                else:
                    color_std = gray_std = gradient_mean = 0.0
                heterogeneity = 0.45 * gray_std + 0.35 * gradient_mean + 0.20 * color_std
                boundary_density = float(boundary[y1:y2, x1:x2].mean())
                base = {
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "scale": scale,
                    "coverage": coverage,
                    "heterogeneity": heterogeneity,
                    "boundary_density": boundary_density,
                }
                if coverage >= 0.55:
                    candidates.append({**base, "kind": "interior", "score": heterogeneity})
                if 0.10 <= coverage <= 0.90 and boundary_density > 0:
                    candidates.append(
                        {
                            **base,
                            "kind": "boundary",
                            "score": 0.65 * boundary_density + 0.35 * heterogeneity,
                        }
                    )

    interior = select_nonoverlapping(
        [item for item in candidates if item["kind"] == "interior"], 2, []
    )
    boundary_selected = select_nonoverlapping(
        [item for item in candidates if item["kind"] == "boundary"], 2, interior
    )
    selected = interior + boundary_selected
    fallback_index = 0
    while len(selected) < 4:
        selected.append(
            {
                "kind": "fallback_quadrant",
                "score": 0.0,
                "box": None,
                "quadrant": fallback_index,
            }
        )
        fallback_index += 1

    patches: list[Image.Image] = []
    for item in selected[:4]:
        if item["box"] is None:
            patches.append(make_view(tight, f"crop_q{item['quadrant']}"))
        else:
            x1, y1, x2, y2 = item["box"]
            patches.append(Image.fromarray(rgb[y1:y2, x1:x2], mode="RGB"))
    views = [image, tight, *patches]
    metadata = {
        "tight_width": width,
        "tight_height": height,
        "component_fraction": float(component.mean()),
        "selected": selected[:4],
    }
    return views, metadata


def make_variant_views(image: Image.Image, variant: str) -> tuple[list[Image.Image], dict[str, Any]]:
    if variant == "original_standard6":
        return [make_view(image, name) for name in STANDARD_VIEWS], {}
    if variant == "tight_standard6":
        base = extract_specimen_crop(image, margin_ratio=0.02).convert("RGB")
        return [make_view(base, name) for name in STANDARD_VIEWS], {
            "base_width": base.width,
            "base_height": base.height,
        }
    if variant == "neutral_background_standard6":
        base, fraction = neutral_background(image)
        return [make_view(base, name) for name in STANDARD_VIEWS], {
            "specimen_mask_fraction": fraction
        }
    if variant == "scale_normalized_standard6":
        base, metadata = scale_normalized(image)
        return [make_view(base, name) for name in STANDARD_VIEWS], metadata
    if variant == "evidence_top4":
        return evidence_top4(image)
    if variant == "background_only_standard6":
        base, fraction = background_only(image)
        return [make_view(base, name) for name in STANDARD_VIEWS], {
            "removed_mask_fraction": fraction
        }
    raise ValueError(f"Unsupported H11 variant: {variant}")


def extract_case_features(
    adapter: PeAdapter,
    views: list[Image.Image],
    max_num_patches: int,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    if len(views) != len(STANDARD_VIEWS):
        raise ValueError(f"Expected six views, got {len(views)}")
    features = torch.zeros(
        (1, len(views), max_num_patches, feature_dim), dtype=torch.float16
    )
    masks = torch.zeros((1, len(views), max_num_patches), dtype=torch.bool)
    spatial_shapes: list[list[int]] = []
    for view_index, view in enumerate(views):
        result = adapter.extract(view)
        if result.tokens.shape[1] != feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: {result.tokens.shape[1]} != {feature_dim}"
            )
        token_count = min(max_num_patches, int(result.tokens.shape[0]))
        features[0, view_index, :token_count] = result.tokens[:token_count].to(torch.float16)
        masks[0, view_index, :token_count] = result.valid_mask[:token_count].bool()
        spatial_shapes.append([int(result.spatial_shape[0]), int(result.spatial_shape[1])])
    return features, masks, spatial_shapes


def load_heads(
    args: argparse.Namespace, device: torch.device
) -> tuple[dict[str, dict[int, MaskedDenseGatedHead]], dict[str, Path]]:
    roots = {
        "natural": Path(args.natural_root),
        "risk_balanced": Path(args.risk_balanced_root),
        "subtype_tempered": Path(args.subtype_tempered_root),
    }
    heads: dict[str, dict[int, MaskedDenseGatedHead]] = {}
    for name, root in roots.items():
        heads[name] = {}
        for fold_id in range(1, 6):
            path = root / f"fold_{fold_id}" / "best_head.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = MaskedDenseGatedHead(
                feature_dim=args.feature_dim,
                num_views=len(STANDARD_VIEWS),
                hidden_dim=args.hidden_dim,
                attention_dim=args.attention_dim,
                dropout=args.dropout,
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            heads[name][fold_id] = model.eval().to(device)
    return heads, roots


@torch.no_grad()
def predict_heads(
    heads: dict[str, dict[int, MaskedDenseGatedHead]],
    fold_id: int,
    features: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    device_features = features.to(device, non_blocking=True)
    device_masks = masks.to(device, non_blocking=True)
    probabilities: dict[str, float] = {}
    for name in MODEL_NAMES:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = heads[name][fold_id](device_features, device_masks)
        probabilities[name] = float(torch.softmax(logits.float(), dim=-1)[0, 1].cpu())
    return probabilities


def reproduction_gate(
    predictions: pd.DataFrame, expected_rows: int, tolerance: float
) -> dict[str, Any]:
    subset = predictions[predictions["variant"] == "original_standard6"].copy()
    subset["absolute_probability_error"] = (
        subset["prob_high"] - subset["existing_oof_prob"]
    ).abs()
    subset["prediction_match"] = subset["pred_idx"] == subset["existing_oof_pred_idx"]
    by_model: dict[str, Any] = {}
    for name, group in subset.groupby("model_name", sort=True):
        by_model[str(name)] = {
            "rows": int(len(group)),
            "prediction_match_count": int(group["prediction_match"].sum()),
            "max_absolute_probability_error": float(group["absolute_probability_error"].max()),
            "mean_absolute_probability_error": float(group["absolute_probability_error"].mean()),
        }
    complete = len(subset) == expected_rows
    class_match = complete and bool(subset["prediction_match"].all())
    max_error = float(subset["absolute_probability_error"].max()) if len(subset) else math.inf
    passed = complete and class_match and max_error <= tolerance
    return {
        "passed": passed,
        "expected_rows": expected_rows,
        "observed_rows": int(len(subset)),
        "classification_match": class_match,
        "max_absolute_probability_error": max_error,
        "tolerance": tolerance,
        "by_model": by_model,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    if not variants or variants[0] != "original_standard6":
        raise ValueError("original_standard6 must be the first variant")
    unsupported = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unsupported:
        raise ValueError(f"Unsupported variants: {unsupported}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "h11_private_predictions.csv"
    metadata_path = output_dir / "h11_view_metadata.jsonl"
    reproduction_path = output_dir / "h11_reproduction_gate.json"
    records, oof_frames = load_records(args)
    heads, roots = load_heads(args, device)
    adapter_args = argparse.Namespace(
        model_id=args.model_id,
        model_code_dir=args.model_code_dir,
        code_revision=args.code_revision,
        max_num_patches=args.max_num_patches,
    )
    adapter = PeAdapter(adapter_args, device)
    if int(adapter.feature_dim) != args.feature_dim:
        raise ValueError(f"PE feature dimension {adapter.feature_dim} != {args.feature_dim}")

    config = {
        "experiment": "H11_LOCKED_VISUAL_EVIDENCE_DISENTANGLEMENT_20260715",
        "args": vars(args),
        "variants": list(variants),
        "standard_view_slots": list(STANDARD_VIEWS),
        "manifest_sha256": sha256_file(args.manifest_csv),
        "registry_sha256": sha256_file(args.registry_csv),
        "split_sha256": sha256_file(args.split_csv),
        "model_weight_sha256": sha256_file(args.model_id),
        "model_roots": {name: str(path.resolve()) for name, path in roots.items()},
        "case_count": int(len(records)),
        "image_name_match_count": int(records["image_name_match"].sum()),
        "no_retraining": True,
        "selection_threshold": 0.5,
    }
    write_json(output_dir / "h11_run_config.json", config)

    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path, dtype={"case_id": str})
    else:
        predictions = pd.DataFrame()
    metadata_records = load_jsonl(metadata_path)

    common_columns = [
        "case_id",
        "original_case_id",
        "task_l6_label",
        "label_idx",
        "source_dataset",
        "fold_id",
        "model_correct_count",
        "diagnostic_role",
        "difficulty_tier",
        "audit_code",
        "image_evidence_sufficiency",
        "visual_risk_impression",
        "visual_result",
        "post_unblind_attribution",
        "image_name_match",
    ]
    for variant in variants:
        progress = tqdm(
            records.itertuples(index=False),
            total=len(records),
            desc=f"H11 {variant}",
            dynamic_ncols=True,
        )
        for position, row in enumerate(progress, start=1):
            case_id = str(row.case_id)
            if not predictions.empty:
                completed = predictions[
                    (predictions["case_id"] == case_id)
                    & (predictions["variant"] == variant)
                ]
                if set(completed["model_name"]) == set(MODEL_NAMES) and len(completed) == 3:
                    continue
                predictions = predictions[
                    ~(
                        (predictions["case_id"] == case_id)
                        & (predictions["variant"] == variant)
                    )
                ].copy()

            with Image.open(str(row.image_path)) as source:
                image = source.convert("RGB")
            views, view_metadata = make_variant_views(image, variant)
            features, masks, spatial_shapes = extract_case_features(
                adapter, views, args.max_num_patches, args.feature_dim
            )
            probabilities = predict_heads(heads, int(row.fold_id), features, masks, device)
            row_values = {column: getattr(row, column) for column in common_columns}
            new_rows: list[dict[str, Any]] = []
            for model_name, probability in probabilities.items():
                expected = oof_frames[model_name].loc[case_id]
                pred_idx = int(probability >= 0.5)
                label_idx = int(row.label_idx)
                new_rows.append(
                    {
                        **row_values,
                        "variant": variant,
                        "model_name": model_name,
                        "prob_high": probability,
                        "pred_idx": pred_idx,
                        "correct": pred_idx == label_idx,
                        "true_class_probability": probability if label_idx == 1 else 1.0 - probability,
                        "existing_oof_prob": float(expected["prob_high"]),
                        "existing_oof_pred_idx": int(expected["pred_idx"]),
                    }
                )
            predictions = pd.concat([predictions, pd.DataFrame(new_rows)], ignore_index=True)
            metadata_records[(case_id, variant)] = {
                "case_id": case_id,
                "variant": variant,
                "image_name_match": bool(row.image_name_match),
                "original_width": image.width,
                "original_height": image.height,
                "spatial_shapes": spatial_shapes,
                "view_metadata": view_metadata,
            }
            if position % 5 == 0 or position == len(records):
                predictions = predictions.sort_values(
                    ["variant", "case_id", "model_name"]
                ).reset_index(drop=True)
                write_csv(prediction_path, predictions)
                write_jsonl(metadata_path, metadata_records)

        if variant == "original_standard6":
            gate = reproduction_gate(
                predictions, expected_rows=len(records) * len(MODEL_NAMES), tolerance=args.reproduction_tolerance
            )
            write_json(reproduction_path, gate)
            print(json.dumps(gate, indent=2, ensure_ascii=False), flush=True)
            if not gate["passed"]:
                (output_dir / "RUN.status").write_text("REPRODUCTION_FAILED\n", encoding="utf-8")
                raise RuntimeError("H11 reproduction gate failed; counterfactual variants were not run")

    (output_dir / "RUN.status").write_text("COMPLETE\n", encoding="utf-8")
    print(f"H11 complete: {prediction_path}", flush=True)


if __name__ == "__main__":
    main()

