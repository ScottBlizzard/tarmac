from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for item in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_dinov3_old_third_feature_cv_20260523 import (  # noqa: E402
    build_eval_transform,
    forward_features,
    move_to_device,
)
from thymic_baseline.cropping import detect_specimen_bbox, expand_bbox  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply locked old+third frozen-feature Task7 models to a registry.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--model-path", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def metric_dict(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "high_recall": float(tp / max(tp + fn, 1)),
        "low_specificity": float(tn / max(tn + fp, 1)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, prob))
    except Exception:
        out["auc"] = float("nan")
    return out


def prepare_registry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str, "original_case_id": str}).reset_index(drop=True)
    if "image_name" not in df.columns:
        df["image_name"] = df["image_path"].map(lambda p: Path(str(p)).name)
    required = {"case_id", "image_path", "image_name", "label_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Registry missing required columns: {sorted(missing)}")
    missing_paths = [str(p) for p in df["image_path"].tolist() if not Path(str(p)).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing image files, first 10: {missing_paths[:10]}")
    return df


def fast_specimen_crop_from_path(path: str, margin_ratio: float = 0.12, detect_max_dim: int = 384) -> Image.Image:
    with Image.open(path) as low:
        orig_w, orig_h = low.size
        low.draft("RGB", (detect_max_dim, detect_max_dim))
        low.thumbnail((detect_max_dim, detect_max_dim), Image.Resampling.BILINEAR)
        small = low.convert("RGB")
        small_arr = np.asarray(small)
        small_w, small_h = small.size
    bbox = detect_specimen_bbox(small_arr)
    bbox = expand_bbox(bbox, small_arr.shape[:2], margin_ratio=margin_ratio)
    sx = orig_w / max(small_w, 1)
    sy = orig_h / max(small_h, 1)
    x1, y1, x2, y2 = bbox
    full_box = (
        max(0, int(round(x1 * sx))),
        max(0, int(round(y1 * sy))),
        min(orig_w, int(round(x2 * sx))),
        min(orig_h, int(round(y2 * sy))),
    )
    with Image.open(path) as full:
        image = full.convert("RGB")
        return image.crop(full_box)


class FastRegistryDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, input_variant: str, transform: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.input_variant = input_variant
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Any, int, str, str]:
        row = self.frame.iloc[index]
        path = str(row["image_path"])
        with Image.open(path) as im:
            whole = im.convert("RGB")
        if self.input_variant == "whole":
            inputs = self.transform(whole)
        elif self.input_variant == "crop":
            inputs = self.transform(fast_specimen_crop_from_path(path))
        elif self.input_variant == "whole_plus_crop":
            inputs = (self.transform(whole), self.transform(fast_specimen_crop_from_path(path)))
        else:
            raise ValueError(f"Unsupported input variant: {self.input_variant}")
        return inputs, int(row["label_idx"]), str(row["case_id"]), str(row["image_name"])


def extract_features_fast(
    frame: pd.DataFrame,
    model: torch.nn.Module,
    transform: Any,
    input_variant: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    dataset = FastRegistryDataset(frame, input_variant=input_variant, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    labels: list[int] = []
    case_ids: list[str] = []
    image_names: list[str] = []
    model.eval()
    with torch.inference_mode():
        for inputs, batch_labels, batch_case_ids, batch_image_names in tqdm(loader, desc="feature_extract", leave=False):
            inputs = move_to_device(inputs, device)
            feats = forward_features(model, inputs, amp=amp)
            chunks.append(feats.detach().cpu().numpy().astype(np.float32))
            labels.extend(int(x) for x in batch_labels)
            case_ids.extend(str(x) for x in batch_case_ids)
            image_names.extend(str(x) for x in batch_image_names)
    feature_table = pd.DataFrame(
        {
            "case_id": case_ids,
            "label_idx": labels,
            "image_name": image_names,
            "feature_idx": np.arange(len(case_ids), dtype=int),
        }
    )
    return feature_table, np.concatenate(chunks, axis=0).astype(np.float32)


def model_tag(model_path: Path, obj: dict[str, Any]) -> str:
    return f"{model_path.parent.name}__{obj['feature_model'].replace('/', '_')}"


def apply_one(
    frame: pd.DataFrame,
    model_path: Path,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    device_arg: str,
    reuse_features: bool,
    no_amp: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obj = joblib.load(model_path)
    tag = model_tag(model_path, obj)
    tag_dir = out_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    feature_table_path = tag_dir / "feature_table.csv"
    feature_npy_path = tag_dir / "features.npy"

    if reuse_features and feature_table_path.exists() and feature_npy_path.exists():
        feature_table = pd.read_csv(feature_table_path, dtype={"case_id": str})
        features = np.load(feature_npy_path).astype(np.float32)
    else:
        device = torch.device(device_arg if torch.cuda.is_available() or device_arg == "cpu" else "cpu")
        print(f"[model] {tag} loading on {device}", flush=True)
        backbone = timm.create_model(
            str(obj["feature_model"]),
            pretrained=True,
            num_classes=0,
            global_pool=str(obj["global_pool"]),
        )
        backbone.to(device)
        transform = build_eval_transform(backbone, image_size=int(obj["image_size"]))
        amp = device.type == "cuda" and not no_amp
        feature_table, features = extract_features_fast(
            frame,
            backbone,
            transform,
            input_variant=str(obj["input_variant"]),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            amp=amp,
        )
        feature_table.to_csv(feature_table_path, index=False, encoding="utf-8-sig")
        np.save(feature_npy_path, features.astype(np.float32))
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prob = obj["model"].predict_proba(features.astype(np.float32))[:, 1].astype(float)
    threshold = float(obj["threshold"])
    pred = (prob >= threshold).astype(int)
    out = frame.copy()
    out["locked_feature_model_tag"] = tag
    out["prob_high"] = prob
    out["threshold"] = threshold
    out["pred_idx"] = pred
    out["correct"] = (pred == out["label_idx"].astype(int).to_numpy()).astype(int)
    out.to_csv(tag_dir / "locked_feature_predictions.csv", index=False, encoding="utf-8-sig")

    y = out["label_idx"].astype(int).to_numpy()
    rows = [
        {
            "model_tag": tag,
            "model_path": str(model_path),
            "group": "all",
            "feature_model": str(obj["feature_model"]),
            "input_variant": str(obj["input_variant"]),
            "image_size": int(obj["image_size"]),
            "selected_row": json.dumps(obj.get("selected_row", {}), ensure_ascii=False),
            **metric_dict(y, prob, threshold),
        }
    ]
    if "task_l6_label" in out.columns:
        for subtype, g in out.groupby("task_l6_label", sort=True):
            rows.append(
                {
                    "model_tag": tag,
                    "model_path": str(model_path),
                    "group": f"subtype:{subtype}",
                    "feature_model": str(obj["feature_model"]),
                    "input_variant": str(obj["input_variant"]),
                    "image_size": int(obj["image_size"]),
                    "selected_row": json.dumps(obj.get("selected_row", {}), ensure_ascii=False),
                    **metric_dict(
                        g["label_idx"].astype(int).to_numpy(),
                        g["prob_high"].astype(float).to_numpy(),
                        threshold,
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(tag_dir / "locked_feature_summary.csv", index=False, encoding="utf-8-sig")
    return out, summary


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare_registry(Path(args.registry_csv))
    all_summaries = []
    for model in args.model_path:
        _, summary = apply_one(
            frame=frame,
            model_path=Path(model),
            out_dir=out_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_arg=args.device,
            reuse_features=args.reuse_features,
            no_amp=args.no_amp,
        )
        all_summaries.append(summary)
    merged = pd.concat(all_summaries, ignore_index=True)
    merged.to_csv(out_dir / "locked_feature_models_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "registry_csv": args.registry_csv,
        "model_paths": args.model_path,
        "boundary": "Models and thresholds are locked from old+third development; this script only applies them to the supplied registry.",
    }
    (out_dir / "locked_feature_models_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(merged[merged["group"].eq("all")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
