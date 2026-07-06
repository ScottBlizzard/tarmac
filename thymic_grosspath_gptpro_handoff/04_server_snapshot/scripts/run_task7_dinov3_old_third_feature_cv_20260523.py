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
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thymic_baseline.cropping import extract_specimen_crop


BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
OLD_LABELS = BASE / "task7_curriculum_runs" / "09_case_mlp_schemeB_m060_salvagehard_full5fold" / "curriculum_case_table.csv"
OLD_REGISTRY = BASE / "frozen_inputs" / "combined_task567_registry_with_gross_findings_20260520.csv"
THIRD_REGISTRY = (
    BASE
    / "task7_external_runs"
    / "04_third_batch_whole_plus_crop_64style_20260521"
    / "third_batch_task7_registry.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 old+third DINOv3 frozen-feature CV.")
    parser.add_argument("--model-name", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--global-pool", default="avg", choices=("avg", "token"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-variant", default="whole_plus_crop", choices=("whole", "crop", "whole_plus_crop"))
    parser.add_argument("--out-name", default="54_old_third_dinov3_vitb16_wpc_20260523")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})


def make_frames(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_labels = read_csv(OLD_LABELS)
    old_registry = read_csv(OLD_REGISTRY)
    old_cols = [
        "case_id",
        "original_case_id",
        "source_dataset",
        "source_case_folder",
        "selected_original_image_name",
        "training_image_path",
        "task_l6_label",
        "task_l7_label",
    ]
    old = old_labels[["case_id", "fold_id", "label_idx", "difficulty", "difficulty_fine"]].merge(
        old_registry[old_cols],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    if old["training_image_path"].isna().any():
        missing = old.loc[old["training_image_path"].isna(), "case_id"].head(10).tolist()
        raise KeyError(f"Missing old image paths: {missing}")
    old["image_path"] = old["training_image_path"].astype(str)
    old["image_name"] = old["selected_original_image_name"].astype(str)
    old["domain"] = "old"
    old["source_folder"] = old["source_case_folder"].fillna("old").astype(str)

    third = read_csv(THIRD_REGISTRY)
    third = third[
        [
            "case_id",
            "original_case_id",
            "label_idx",
            "task_l6_label",
            "task_l7_label",
            "source_folder",
            "image_name",
            "image_path",
        ]
    ].copy()
    third["domain"] = "third"
    third["difficulty"] = ""
    third["difficulty_fine"] = ""
    third["fold_id"] = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, idx) in enumerate(skf.split(third, third["label_idx"].astype(int)), start=1):
        third.loc[idx, "fold_id"] = fold

    return old.reset_index(drop=True), third.reset_index(drop=True)


class CaseImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, input_variant: str, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.input_variant = input_variant
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int) -> tuple[Any, int, str, str]:
        row = self.frame.iloc[index]
        image = self._load_image(str(row["image_path"]))
        if self.input_variant == "whole":
            inputs = self.transform(image)
        elif self.input_variant == "crop":
            inputs = self.transform(extract_specimen_crop(image))
        elif self.input_variant == "whole_plus_crop":
            inputs = (self.transform(image), self.transform(extract_specimen_crop(image)))
        else:
            raise ValueError(f"Unsupported input variant: {self.input_variant}")
        return inputs, int(row["label_idx"]), str(row["case_id"]), str(row["image_name"])


def build_eval_transform(model: torch.nn.Module, image_size: int) -> transforms.Compose:
    data_config = timm.data.resolve_model_data_config(model)
    interpolation = str(data_config.get("interpolation", "bicubic")).lower()
    interp = InterpolationMode.BICUBIC if interpolation == "bicubic" else InterpolationMode.BILINEAR
    mean = data_config.get("mean", (0.485, 0.456, 0.406))
    std = data_config.get("std", (0.229, 0.224, 0.225))
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=interp),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def move_to_device(inputs: Any, device: torch.device) -> Any:
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=True)
    if isinstance(inputs, (tuple, list)):
        return [move_to_device(item, device) for item in inputs]
    raise TypeError(f"Unsupported input type: {type(inputs)!r}")


def forward_features(model: torch.nn.Module, inputs: Any, amp: bool) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            output = model(inputs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.float()
    if isinstance(inputs, (tuple, list)):
        parts = [forward_features(model, item, amp=amp) for item in inputs]
        return torch.cat(parts, dim=1)
    raise TypeError(f"Unsupported input type: {type(inputs)!r}")


def extract_features(
    frame: pd.DataFrame,
    model: torch.nn.Module,
    transform: transforms.Compose,
    input_variant: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    dataset = CaseImageDataset(frame, input_variant=input_variant, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feature_chunks: list[np.ndarray] = []
    labels: list[int] = []
    case_ids: list[str] = []
    image_names: list[str] = []

    model.eval()
    with torch.inference_mode():
        for inputs, batch_labels, batch_case_ids, batch_image_names in loader:
            inputs = move_to_device(inputs, device)
            feats = forward_features(model, inputs, amp=amp)
            feature_chunks.append(feats.detach().cpu().numpy().astype(np.float32))
            labels.extend(int(x) for x in batch_labels)
            case_ids.extend(str(x) for x in batch_case_ids)
            image_names.extend(str(x) for x in batch_image_names)

    features = np.concatenate(feature_chunks, axis=0).astype(np.float32)
    table = pd.DataFrame(
        {
            "case_id": case_ids,
            "label_idx": labels,
            "image_name": image_names,
            "feature_idx": np.arange(len(case_ids), dtype=int),
        }
    )
    return table, features


def save_feature_cache(directory: Path, table: pd.DataFrame, features: np.ndarray, third: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    table_name = "third_batch_dino_concat_feature_table.csv" if third else "case_dino_concat_feature_table.csv"
    npy_name = "third_batch_dino_concat_features.npy" if third else "case_dino_concat_features.npy"
    table.to_csv(directory / table_name, index=False, encoding="utf-8-sig")
    np.save(directory / npy_name, features.astype(np.float32))


def load_feature_cache(directory: Path, third: bool) -> tuple[pd.DataFrame, np.ndarray]:
    table_name = "third_batch_dino_concat_feature_table.csv" if third else "case_dino_concat_feature_table.csv"
    npy_name = "third_batch_dino_concat_features.npy" if third else "case_dino_concat_features.npy"
    table = read_csv(directory / table_name)
    features = np.load(directory / npy_name).astype(np.float32)
    return table, features


def align_features(order: pd.Series, table: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    idx = order.astype(str).to_frame("case_id").merge(
        table[["case_id", "feature_idx"]],
        on="case_id",
        how="left",
    )["feature_idx"]
    if idx.isna().any():
        missing = order[idx.isna()].head(10).tolist()
        raise KeyError(f"Missing feature rows: {missing}")
    return features[idx.astype(int).to_numpy()].astype(np.float32)


def metric_row(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    row: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }
    row["auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")
    return row


def choose_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, dict[str, float | int]]:
    best_key: tuple[float, ...] | None = None
    best_threshold = 0.5
    best_row: dict[str, float | int] = {}
    for threshold in np.linspace(0.05, 0.95, 181):
        row = metric_row(y, prob, float(threshold))
        if objective == "accuracy":
            key = (float(row["accuracy"]), float(row["balanced_accuracy"]), float(row["f1"]))
        elif objective == "balanced_accuracy":
            key = (float(row["balanced_accuracy"]), float(row["accuracy"]), float(row["f1"]))
        elif objective == "high_sensitivity":
            penalty = min(0.0, float(row["sensitivity_high"]) - 0.80)
            key = (float(row["balanced_accuracy"]) + 0.4 * penalty, float(row["accuracy"]), float(row["f1"]))
        else:
            raise ValueError(objective)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_row = row
    return best_threshold, best_row


def build_model_grid() -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}
    for c in [0.003, 0.01, 0.03, 0.1, 0.3]:
        models[f"logreg_bal_c{c:g}"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=128, svd_solver="randomized", random_state=20260523)),
                ("clf", LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=5000)),
            ]
        )
        models[f"logreg_plain_c{c:g}"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=128, svd_solver="randomized", random_state=20260523)),
                ("clf", LogisticRegression(C=c, solver="liblinear", max_iter=5000)),
            ]
        )
    for depth in [2, 3, 4]:
        models[f"extra_d{depth}"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=128, svd_solver="randomized", random_state=20260523)),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_depth=depth,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=20260523,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def sample_weights(frame: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    weights = np.ones(len(frame), dtype=float)
    y = frame["label_idx"].astype(int).to_numpy()
    if "label" in mode:
        for label in sorted(np.unique(y)):
            mask = y == label
            weights[mask] *= len(y) / (2.0 * mask.sum())
    if "domain" in mode:
        domains = frame["domain"].astype(str).to_numpy()
        for domain in sorted(np.unique(domains)):
            mask = domains == domain
            weights[mask] *= len(domains) / (2.0 * mask.sum())
    if "third_up" in mode:
        weights[frame["domain"].eq("third").to_numpy()] *= 1.5
    return weights


def fit_estimator(estimator: Pipeline, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> Pipeline:
    fitted = clone(estimator)
    n_components = min(128, x.shape[0] - 1, x.shape[1])
    fitted.set_params(pca__n_components=n_components)
    if weights is None:
        fitted.fit(x, y)
    else:
        fitted.fit(x, y, clf__sample_weight=weights)
    return fitted


def predict_prob(estimator: Pipeline, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    score = estimator.decision_function(x)
    return 1.0 / (1.0 + np.exp(-score))


def run_cv(
    out: Path,
    old: pd.DataFrame,
    third: pd.DataFrame,
    old_features: np.ndarray,
    third_features: np.ndarray,
    model_name: str,
    global_pool: str,
    input_variant: str,
    image_size: int,
) -> None:
    frame = pd.concat([old, third], ignore_index=True, sort=False)
    x = np.vstack([old_features, third_features]).astype(np.float32)
    y = frame["label_idx"].astype(int).to_numpy()
    folds = frame["fold_id"].astype(int).to_numpy()
    old_mask = frame["domain"].eq("old").to_numpy()
    third_mask = frame["domain"].eq("third").to_numpy()

    rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    for estimator_name, estimator in build_model_grid().items():
        for weight_mode in ["none", "label", "domain_label", "domain_label_third_up"]:
            oof = np.zeros(len(frame), dtype=np.float32)
            for fold in sorted(np.unique(folds)):
                train_mask = folds != fold
                val_mask = folds == fold
                weights = sample_weights(frame.loc[train_mask].reset_index(drop=True), weight_mode)
                fitted = fit_estimator(estimator, x[train_mask], y[train_mask], weights)
                oof[val_mask] = predict_prob(fitted, x[val_mask])

            for objective in ["balanced_accuracy", "accuracy", "high_sensitivity"]:
                threshold, all_row = choose_threshold(y, oof, objective)
                old_row = metric_row(y[old_mask], oof[old_mask], threshold)
                third_row = metric_row(y[third_mask], oof[third_mask], threshold)
                row = {
                    "feature_model": model_name,
                    "global_pool": global_pool,
                    "input_variant": input_variant,
                    "image_size": image_size,
                    "estimator": estimator_name,
                    "weight_mode": weight_mode,
                    "objective": objective,
                    **{f"all_{k}": v for k, v in all_row.items()},
                    **{f"old_{k}": v for k, v in old_row.items()},
                    **{f"third_{k}": v for k, v in third_row.items()},
                }
                row["selection_score"] = (
                    0.30 * float(row["old_balanced_accuracy"])
                    + 0.30 * float(row["third_balanced_accuracy"])
                    + 0.20 * float(row["old_accuracy"])
                    + 0.20 * float(row["third_accuracy"])
                    - max(0.0, 0.90 - float(row["old_accuracy"]))
                    - max(0.0, 0.70 - float(row["third_sensitivity_high"]))
                )
                row["old_guard_090"] = bool(row["old_accuracy"] >= 0.90 and row["old_balanced_accuracy"] >= 0.90)
                row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                rows.append(row)

            pred = frame[
                [
                    "case_id",
                    "original_case_id",
                    "domain",
                    "source_folder",
                    "fold_id",
                    "task_l6_label",
                    "task_l7_label",
                    "label_idx",
                    "difficulty",
                    "difficulty_fine",
                ]
            ].copy()
            pred["feature_model"] = model_name
            pred["global_pool"] = global_pool
            pred["input_variant"] = input_variant
            pred["image_size"] = image_size
            pred["estimator"] = estimator_name
            pred["weight_mode"] = weight_mode
            pred["oof_prob_high"] = oof
            pred_frames.append(pred)

    summary = pd.DataFrame(rows).sort_values(
        ["selection_score", "third_accuracy", "third_balanced_accuracy"],
        ascending=False,
    )
    predictions = pd.concat(pred_frames, ignore_index=True)
    summary.to_csv(out / "dinov3_feature_cv_summary.csv", index=False, encoding="utf-8-sig")
    summary.head(100).to_csv(out / "top100_dinov3_feature_cv.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "dinov3_feature_cv_all_oof_predictions.csv", index=False, encoding="utf-8-sig")

    guarded92 = summary[summary["old_guard_092"]].copy()
    guarded90 = summary[summary["old_guard_090"]].copy()
    selected = (guarded92 if not guarded92.empty else guarded90 if not guarded90.empty else summary).iloc[0]
    selected_pred = predictions[
        (predictions["estimator"] == selected["estimator"])
        & (predictions["weight_mode"] == selected["weight_mode"])
    ].copy()
    selected_pred["threshold"] = float(selected["all_threshold"])
    selected_pred["oof_pred_idx"] = (selected_pred["oof_prob_high"].astype(float) >= float(selected["all_threshold"])).astype(int)
    selected_pred["oof_correct"] = (selected_pred["oof_pred_idx"] == selected_pred["label_idx"].astype(int)).astype(int)
    selected_pred.to_csv(out / "selected_dinov3_feature_oof_predictions.csv", index=False, encoding="utf-8-sig")

    final_estimator = build_model_grid()[str(selected["estimator"])]
    final_weights = sample_weights(frame.reset_index(drop=True), str(selected["weight_mode"]))
    final_model = fit_estimator(final_estimator, x, y, final_weights)
    joblib.dump(
        {
            "model": final_model,
            "threshold": float(selected["all_threshold"]),
            "feature_model": model_name,
            "global_pool": global_pool,
            "input_variant": input_variant,
            "image_size": int(image_size),
            "selected_row": selected.to_dict(),
            "data_boundary": "trained and selected on old batch1+batch2 plus third-batch development cohort only; strict external folder is not used",
        },
        out / "selected_final_dinov3_feature_model_old_plus_third.joblib",
    )

    report = {
        "experiment": "Task7 DINOv3 frozen whole+crop feature development CV",
        "data_boundary": "Only old batch1+batch2 and third-batch development data are used. Strict external folder is not read.",
        "n_old": int(len(old)),
        "n_third": int(len(third)),
        "feature_shape_old": list(map(int, old_features.shape)),
        "feature_shape_third": list(map(int, third_features.shape)),
        "selected": selected.to_dict(),
        "top10": summary.head(10).to_dict(orient="records"),
    }
    (out / "dinov3_feature_cv_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    show_cols = [
        "estimator",
        "weight_mode",
        "objective",
        "selection_score",
        "all_threshold",
        "old_accuracy",
        "old_balanced_accuracy",
        "old_tn",
        "old_fp",
        "old_fn",
        "old_tp",
        "third_accuracy",
        "third_balanced_accuracy",
        "third_f1",
        "third_sensitivity_high",
        "third_specificity_low",
        "third_tn",
        "third_fp",
        "third_fn",
        "third_tp",
        "third_auc",
    ]
    print(summary[show_cols].head(20).to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    out = BASE / "task7_adaptation_runs" / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    old_feature_dir = out / "old_features"
    third_feature_dir = out / "third_features"
    old, third = make_frames(seed=args.seed)
    old.to_csv(out / "old_frame_with_folds.csv", index=False, encoding="utf-8-sig")
    third.to_csv(out / "third_frame_with_folds.csv", index=False, encoding="utf-8-sig")

    if args.reuse_features and (old_feature_dir / "case_dino_concat_features.npy").exists():
        old_table, old_features = load_feature_cache(old_feature_dir, third=False)
        third_table, third_features = load_feature_cache(third_feature_dir, third=True)
        old_features = align_features(old["case_id"], old_table, old_features)
        third_features = align_features(third["case_id"], third_table, third_features)
    else:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        print(f"[model] loading {args.model_name} on {device}", flush=True)
        model = timm.create_model(args.model_name, pretrained=True, num_classes=0, global_pool=args.global_pool)
        model.to(device)
        transform = build_eval_transform(model, image_size=args.image_size)
        amp = device.type == "cuda" and not args.no_amp
        print(f"[extract] old n={len(old)} variant={args.input_variant} size={args.image_size}", flush=True)
        old_table, old_features = extract_features(
            old,
            model,
            transform,
            input_variant=args.input_variant,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            amp=amp,
        )
        save_feature_cache(old_feature_dir, old_table, old_features, third=False)
        print(f"[extract] third n={len(third)} variant={args.input_variant} size={args.image_size}", flush=True)
        third_table, third_features = extract_features(
            third,
            model,
            transform,
            input_variant=args.input_variant,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            amp=amp,
        )
        save_feature_cache(third_feature_dir, third_table, third_features, third=True)

    print(
        f"[features] old={old_features.shape} third={third_features.shape} out={out}",
        flush=True,
    )
    run_cv(
        out=out,
        old=old,
        third=third,
        old_features=old_features,
        third_features=third_features,
        model_name=args.model_name,
        global_pool=args.global_pool,
        input_variant=args.input_variant,
        image_size=args.image_size,
    )
    print(f"[done] out={out}", flush=True)


if __name__ == "__main__":
    main()
