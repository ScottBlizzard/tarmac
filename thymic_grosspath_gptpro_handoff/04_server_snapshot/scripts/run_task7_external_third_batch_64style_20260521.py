from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for item in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_dinov2_frozen_probe import (  # noqa: E402
    aggregate_features_to_cases,
    extract_dataset_features,
    load_dinov2_model,
    set_seed,
)
from run_task7_concat_curriculum_probe import DEFAULT_DINO_IMAGE_SIZE  # noqa: E402
from thymic_baseline.train import resolve_device  # noqa: E402


LOW_LABEL = "low_risk_group"
HIGH_LABEL = "high_risk_group"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen external Task7 test on third batch with a 64-style image-only reviewer."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--third-images-root", default="datasets/third_batch_306_20260521")
    parser.add_argument(
        "--old-registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--old-curriculum-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument(
        "--old-dino-feature-table",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_feature_table.csv",
    )
    parser.add_argument(
        "--old-dino-feature-npy",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_features.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/01_third_batch_64style_image_only_20260521",
    )
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", default="dinov2_vits14,dinov2_vitb14")
    parser.add_argument("--feature-mode", default="cls_patchmean", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--input-variant", default="whole", choices=("whole", "crop", "whole_plus_crop"))
    parser.add_argument("--image-size", type=int, default=DEFAULT_DINO_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, object] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if prob is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, prob))
    return out


def entropy_binary(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob.astype(float), 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob.astype(float), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def best_threshold(y: np.ndarray, prob: np.ndarray, objective: str = "balanced_accuracy") -> tuple[float, float]:
    best_t = 0.5
    best_s = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        if objective == "accuracy":
            score = accuracy_score(y, pred)
        elif objective == "f1":
            score = f1_score(y, pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t = float(t)
            best_s = float(score)
    return best_t, best_s


def safe_stem_case_id(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"(\d+)", stem)
    return match.group(1) if match else re.sub(r"\W+", "_", stem)


def build_third_registry(images_root: Path) -> pd.DataFrame:
    label_map = {
        "AB 212": ("AB", LOW_LABEL, 0),
        "B1型 12": ("B1", LOW_LABEL, 0),
        "B2 29": ("B2", HIGH_LABEL, 1),
        "TC 53": ("TC", HIGH_LABEL, 1),
    }
    rows: list[dict[str, object]] = []
    for folder_name, (l6_label, l7_label, label_idx) in label_map.items():
        folder = images_root / folder_name
        if not folder.exists():
            raise FileNotFoundError(folder)
        for image_path in sorted([p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"}]):
            original_case_id = safe_stem_case_id(image_path.name)
            case_id = f"third_{l6_label}_{original_case_id}"
            rows.append(
                {
                    "case_id": case_id,
                    "original_case_id": original_case_id,
                    "source_dataset": "third_batch_306_20260521",
                    "source_folder": folder_name,
                    "image_count": 1,
                    "image_filenames": image_path.name,
                    "selected_original_image_name": image_path.name,
                    "selected_original_image_relpath": str(image_path.relative_to(images_root)),
                    "selected_original_image_path": str(image_path),
                    "training_image_path": str(image_path),
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "selection_rule": "third_single_image",
                    "task_l6_label": l6_label,
                    "task_l7_label": l7_label,
                    "label_idx": int(label_idx),
                }
            )
    df = pd.DataFrame(rows).sort_values(["task_l6_label", "original_case_id"]).reset_index(drop=True)
    if df["case_id"].duplicated().any():
        dup = df.loc[df["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"Duplicated third-batch case_id: {dup[:10]}")
    return df


def load_old_frame(project_root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    curriculum = pd.read_csv(project_root / args.old_curriculum_csv, dtype={"case_id": str})
    keep = [
        "case_id",
        "fold_id",
        "label_idx",
        "difficulty",
        "difficulty_fine",
        "correct_count",
        "mean_true_prob",
        "mean_margin",
    ]
    frame = curriculum[[c for c in keep if c in curriculum.columns]].copy()
    frame["fold_id"] = frame["fold_id"].astype(int)
    frame["label_idx"] = frame["label_idx"].astype(int)
    frame["hard_core"] = frame["difficulty_fine"].eq("hard_core").astype(int)

    registry = pd.read_csv(project_root / args.old_registry_csv, dtype={"case_id": str, "original_case_id": str})
    reg_keep = [
        "case_id",
        "original_case_id",
        "image_count",
        "selection_rule",
        "task_l6_label",
        "task_l7_label",
    ]
    frame = frame.merge(registry[[c for c in reg_keep if c in registry.columns]], on="case_id", how="left")

    feature_table = pd.read_csv(project_root / args.old_dino_feature_table, dtype={"case_id": str})
    old_features = np.load(project_root / args.old_dino_feature_npy).astype(np.float32)
    order = frame[["case_id"]].merge(feature_table[["case_id", "feature_idx"]], on="case_id", how="left")
    if order["feature_idx"].isna().any():
        missing = order.loc[order["feature_idx"].isna(), "case_id"].head().tolist()
        raise KeyError(f"Missing old DINO features: {missing}")
    aligned = old_features[order["feature_idx"].astype(int).to_numpy()]
    return frame.reset_index(drop=True), aligned.astype(np.float32)


def load_or_extract_third_features(
    project_root: Path,
    args: argparse.Namespace,
    third_registry: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    cache_table = output_dir / "third_batch_dino_concat_feature_table.csv"
    cache_npy = output_dir / "third_batch_dino_concat_features.npy"
    if cache_table.exists() and cache_npy.exists():
        table = pd.read_csv(cache_table, dtype={"case_id": str})
        features = np.load(cache_npy).astype(np.float32)
        order = third_registry[["case_id"]].merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
        if order["feature_idx"].isna().any():
            raise KeyError("Cached third-batch DINO feature table is incomplete.")
        return table, features[order["feature_idx"].astype(int).to_numpy()]

    set_seed(args.seed)
    device = resolve_device(args.device)
    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    image_df = third_registry[["case_id", "label_idx", "image_path", "image_name"]].copy()
    all_features: list[np.ndarray] = []
    labels_ref: np.ndarray | None = None
    case_ids_ref: list[str] | None = None
    image_names_ref: list[str] | None = None
    for model_name in model_names:
        print(f"[extract] loading {model_name}", flush=True)
        model = load_dinov2_model(project_root / args.repo_dir, model_name, device)
        features, labels, case_ids, image_names = extract_dataset_features(
            image_df=image_df,
            input_variant=args.input_variant,
            feature_mode=args.feature_mode,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model=model,
            device=device,
        )
        if labels_ref is None:
            labels_ref = labels
            case_ids_ref = case_ids
            image_names_ref = image_names
        else:
            if not np.array_equal(labels_ref, labels) or case_ids_ref != case_ids or image_names_ref != image_names:
                raise ValueError(f"Sample order mismatch while extracting {model_name}.")
        all_features.append(features.astype(np.float32))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    concat = np.concatenate(all_features, axis=1).astype(np.float32)
    case_features, case_labels, case_ids, case_image_names = aggregate_features_to_cases(
        concat,
        labels_ref,
        case_ids_ref,
        image_names_ref,
        "mean",
    )
    table = pd.DataFrame(
        {
            "case_id": case_ids,
            "label_idx": case_labels.astype(int),
            "image_name": case_image_names,
            "feature_idx": np.arange(len(case_ids), dtype=int),
        }
    )
    table.to_csv(cache_table, index=False, encoding="utf-8-sig")
    np.save(cache_npy, case_features.astype(np.float32))
    order = third_registry[["case_id"]].merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if order["feature_idx"].isna().any():
        raise KeyError("Extracted third-batch DINO feature table is incomplete.")
    return table, case_features[order["feature_idx"].astype(int).to_numpy()].astype(np.float32)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    params: tuple[tuple[str, object], ...] = ()


def make_model(spec: ModelSpec, seed: int):
    params = dict(spec.params)
    if spec.kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params.get("C", 0.1)),
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=seed,
            ),
        )
    if spec.kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=int(params.get("min_samples_leaf", 3)),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if spec.kind == "rf":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=int(params.get("min_samples_leaf", 3)),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if spec.kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 120)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            max_depth=int(params.get("max_depth", 2)),
            random_state=seed,
        )
    if spec.kind == "svc":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=float(params.get("C", 0.5)),
                gamma=params.get("gamma", "scale"),
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if spec.kind == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=int(params.get("n_neighbors", 9))))
    if spec.kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=params.get("hidden_layer_sizes", (96,)),
                alpha=float(params.get("alpha", 0.03)),
                learning_rate_init=float(params.get("learning_rate_init", 0.001)),
                max_iter=700,
                early_stopping=True,
                n_iter_no_change=30,
                random_state=seed,
            ),
        )
    raise ValueError(spec.kind)


def candidate_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for c in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        specs.append(ModelSpec(f"logreg_c{str(c).replace('.', '')}", "logreg", (("C", c),)))
    for depth in [3, 5, 7, None]:
        for leaf in [2, 4, 8]:
            depth_name = "none" if depth is None else str(depth)
            specs.append(
                ModelSpec(
                    f"extra_d{depth_name}_l{leaf}",
                    "extra",
                    (("max_depth", depth), ("min_samples_leaf", leaf)),
                )
            )
    for depth in [3, 5, 7]:
        specs.append(ModelSpec(f"rf_d{depth}", "rf", (("max_depth", depth), ("min_samples_leaf", 4))))
    specs.extend(
        [
            ModelSpec("gb_d1_lr03", "gb", (("max_depth", 1), ("learning_rate", 0.03), ("n_estimators", 160))),
            ModelSpec("gb_d2_lr03", "gb", (("max_depth", 2), ("learning_rate", 0.03), ("n_estimators", 140))),
            ModelSpec("svc_c03", "svc", (("C", 0.3),)),
            ModelSpec("svc_c1", "svc", (("C", 1.0),)),
            ModelSpec("knn7", "knn", (("n_neighbors", 7),)),
            ModelSpec("knn11", "knn", (("n_neighbors", 11),)),
            ModelSpec("mlp96_a003", "mlp", (("hidden_layer_sizes", (96,)), ("alpha", 0.03))),
            ModelSpec("mlp64_32_a003", "mlp", (("hidden_layer_sizes", (64, 32)), ("alpha", 0.03))),
        ]
    )
    return specs


def predict_proba_high(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        cls = int(model.classes_[0]) if hasattr(model, "classes_") else 0
        return np.ones(len(x), dtype=float) if cls == 1 else np.zeros(len(x), dtype=float)
    return proba[:, 1].astype(float)


def oof_and_external_probs(
    spec: ModelSpec,
    x_old: np.ndarray,
    y_old: np.ndarray,
    folds: np.ndarray,
    x_external: np.ndarray,
    seed: int,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if train_mask is None:
        train_mask = np.ones(len(y_old), dtype=bool)
    oof = np.full(len(y_old), np.nan, dtype=float)
    for fold in sorted(set(folds)):
        train = (folds != fold) & train_mask
        valid = folds == fold
        if train.sum() < 12 or len(np.unique(y_old[train])) < 2:
            train = folds != fold
        model = make_model(spec, seed + int(fold))
        model.fit(x_old[train], y_old[train])
        oof[valid] = predict_proba_high(model, x_old[valid])
    if np.isnan(oof).any():
        raise RuntimeError(f"OOF failed for {spec.name}")
    full_train = train_mask
    if full_train.sum() < 12 or len(np.unique(y_old[full_train])) < 2:
        full_train = np.ones(len(y_old), dtype=bool)
    model = make_model(spec, seed + 1000)
    model.fit(x_old[full_train], y_old[full_train])
    external = predict_proba_high(model, x_external)
    return oof, external


def build_prob_features(
    probs: np.ndarray,
    names: list[str],
    image_count: np.ndarray | None = None,
    selection_rule: pd.Series | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    feat = pd.DataFrame(index=np.arange(probs.shape[0]))
    for idx, name in enumerate(names):
        p = probs[:, idx].astype(float)
        feat[f"{prefix}p_{name}"] = p
        feat[f"{prefix}pred_{name}"] = (p >= 0.5).astype(float)
        feat[f"{prefix}margin_{name}"] = np.abs(p - 0.5)
        feat[f"{prefix}entropy_{name}"] = entropy_binary(p)
        feat[f"{prefix}logit_{name}"] = logit(p)
    if probs.shape[1] > 0:
        feat[f"{prefix}prob_mean"] = probs.mean(axis=1)
        feat[f"{prefix}prob_std"] = probs.std(axis=1)
        feat[f"{prefix}prob_min"] = probs.min(axis=1)
        feat[f"{prefix}prob_max"] = probs.max(axis=1)
        feat[f"{prefix}prob_range"] = probs.max(axis=1) - probs.min(axis=1)
        votes = (probs >= 0.5).astype(int)
        feat[f"{prefix}vote_sum"] = votes.sum(axis=1)
        feat[f"{prefix}vote_frac"] = votes.mean(axis=1)
        feat[f"{prefix}vote_disagree"] = ((votes.sum(axis=1) > 0) & (votes.sum(axis=1) < votes.shape[1])).astype(float)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            feat[f"{prefix}prob_q{int(q * 100)}"] = np.quantile(probs, q, axis=1)
    for i in range(probs.shape[1]):
        for j in range(i + 1, probs.shape[1]):
            diff = probs[:, i] - probs[:, j]
            feat[f"{prefix}diff_{names[i]}__{names[j]}"] = diff
            feat[f"{prefix}absdiff_{names[i]}__{names[j]}"] = np.abs(diff)
    if image_count is not None:
        feat[f"{prefix}image_count"] = image_count.astype(float)
    if selection_rule is not None:
        dummies = pd.get_dummies(selection_rule.fillna("").astype(str), prefix=f"{prefix}sel", dtype=float)
        feat = pd.concat([feat, dummies.reset_index(drop=True)], axis=1)
    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def align_columns(old: pd.DataFrame, external: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = sorted(set(old.columns).union(external.columns))
    return old.reindex(columns=cols, fill_value=0.0), external.reindex(columns=cols, fill_value=0.0)


def select_base_model(
    y_old: np.ndarray,
    folds: np.ndarray,
    candidate_old: np.ndarray,
    candidate_external: np.ndarray,
    candidate_names: list[str],
    seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    base_options: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for idx, name in enumerate(candidate_names):
        prob = candidate_old[:, idx]
        t, _ = best_threshold(y_old, prob, "balanced_accuracy")
        pred = (prob >= t).astype(int)
        row = metric_dict(y_old, pred, prob)
        row.update({"base_name": name, "threshold": t, "kind": "single"})
        rows.append(row)
        base_options.append((name, prob, candidate_external[:, idx], t))

    meta_old = build_prob_features(candidate_old, candidate_names)
    meta_external = build_prob_features(candidate_external, candidate_names)
    meta_old, meta_external = align_columns(meta_old, meta_external)
    meta_specs = [
        ModelSpec("meta_logreg_c003", "logreg", (("C", 0.03),)),
        ModelSpec("meta_logreg_c01", "logreg", (("C", 0.1),)),
        ModelSpec("meta_extra_d3", "extra", (("max_depth", 3), ("min_samples_leaf", 4))),
        ModelSpec("meta_extra_d5", "extra", (("max_depth", 5), ("min_samples_leaf", 4))),
        ModelSpec("meta_rf_d5", "rf", (("max_depth", 5), ("min_samples_leaf", 4))),
    ]
    x_meta = meta_old.to_numpy(dtype=np.float32)
    x_meta_external = meta_external.to_numpy(dtype=np.float32)
    for spec in meta_specs:
        prob, external_prob = oof_and_external_probs(spec, x_meta, y_old, folds, x_meta_external, seed + 2000)
        t, _ = best_threshold(y_old, prob, "balanced_accuracy")
        pred = (prob >= t).astype(int)
        row = metric_dict(y_old, pred, prob)
        row.update({"base_name": spec.name, "threshold": t, "kind": "stack"})
        rows.append(row)
        base_options.append((spec.name, prob, external_prob, t))

    summary = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "accuracy", "f1", "auc"], ascending=False
    ).reset_index(drop=True)
    best_name = str(summary.loc[0, "base_name"])
    best_threshold_value = float(summary.loc[0, "threshold"])
    for name, old_prob, external_prob, threshold in base_options:
        if name == best_name and math.isclose(threshold, best_threshold_value, rel_tol=1e-9, abs_tol=1e-9):
            old_pred = (old_prob >= threshold).astype(int)
            external_pred = (external_prob >= threshold).astype(int)
            return summary.loc[0].to_dict(), old_prob, old_pred, external_prob, external_pred
    raise RuntimeError("Selected base model not found.")


def feature_matrix(kind: str, model_old: pd.DataFrame, model_external: pd.DataFrame, dino_old: np.ndarray, dino_external: np.ndarray):
    if kind == "model":
        return model_old.to_numpy(dtype=np.float32), model_external.to_numpy(dtype=np.float32)
    if kind == "dino":
        return dino_old.astype(np.float32), dino_external.astype(np.float32)
    if kind == "dino_model":
        return (
            np.concatenate([dino_old.astype(np.float32), model_old.to_numpy(dtype=np.float32)], axis=1),
            np.concatenate([dino_external.astype(np.float32), model_external.to_numpy(dtype=np.float32)], axis=1),
        )
    raise ValueError(kind)


def scope_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "all":
        return np.ones(len(frame), dtype=bool)
    if scope == "non_easy":
        return ~frame["difficulty_fine"].eq("easy").to_numpy()
    if scope == "hard_all":
        return frame["difficulty_fine"].isin(["hard_core", "hard_salvage_teacher"]).to_numpy()
    if scope == "hard_core":
        return frame["difficulty_fine"].eq("hard_core").to_numpy()
    raise ValueError(scope)


def route_specs() -> list[ModelSpec]:
    return [
        ModelSpec("route_logreg_c003", "logreg", (("C", 0.03),)),
        ModelSpec("route_logreg_c01", "logreg", (("C", 0.1),)),
        ModelSpec("route_extra_d3", "extra", (("max_depth", 3), ("min_samples_leaf", 4))),
        ModelSpec("route_extra_d5", "extra", (("max_depth", 5), ("min_samples_leaf", 4))),
    ]


def corrector_specs() -> list[ModelSpec]:
    return [
        ModelSpec("corr_logreg_c001", "logreg", (("C", 0.01),)),
        ModelSpec("corr_logreg_c003", "logreg", (("C", 0.03),)),
        ModelSpec("corr_logreg_c01", "logreg", (("C", 0.1),)),
        ModelSpec("corr_extra_d3", "extra", (("max_depth", 3), ("min_samples_leaf", 4))),
        ModelSpec("corr_extra_d5", "extra", (("max_depth", 5), ("min_samples_leaf", 4))),
        ModelSpec("corr_rf_d5", "rf", (("max_depth", 5), ("min_samples_leaf", 4))),
    ]


def choose_route_threshold(score_old: np.ndarray, budget_pct: int) -> float:
    if budget_pct <= 0:
        return float("inf")
    k = max(1, int(round(len(score_old) * budget_pct / 100.0)))
    return float(np.sort(score_old)[-k])


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    old_frame, old_dino = load_old_frame(project_root, args)
    third_registry = build_third_registry(project_root / args.third_images_root)
    third_registry.to_csv(output_dir / "third_batch_task7_registry.csv", index=False, encoding="utf-8-sig")
    _, third_dino = load_or_extract_third_features(project_root, args, third_registry, output_dir)

    y_old = old_frame["label_idx"].astype(int).to_numpy()
    folds = old_frame["fold_id"].astype(int).to_numpy()
    y_external = third_registry["label_idx"].astype(int).to_numpy()

    print(f"[data] old={len(old_frame)} third={len(third_registry)} dino_dim={old_dino.shape[1]}", flush=True)

    cand_rows: list[dict[str, object]] = []
    cand_old_probs: list[np.ndarray] = []
    cand_external_probs: list[np.ndarray] = []
    cand_names: list[str] = []
    for idx, spec in enumerate(candidate_specs(), start=1):
        print(f"[candidate {idx:02d}] {spec.name}", flush=True)
        old_prob, external_prob = oof_and_external_probs(spec, old_dino, y_old, folds, third_dino, args.seed + idx * 17)
        t, _ = best_threshold(y_old, old_prob, "balanced_accuracy")
        old_pred = (old_prob >= t).astype(int)
        row = metric_dict(y_old, old_pred, old_prob)
        row.update({"candidate": spec.name, "threshold": t})
        cand_rows.append(row)
        cand_old_probs.append(old_prob)
        cand_external_probs.append(external_prob)
        cand_names.append(spec.name)

    cand_old = np.stack(cand_old_probs, axis=1)
    cand_external = np.stack(cand_external_probs, axis=1)
    pd.DataFrame(cand_rows).sort_values(["balanced_accuracy", "accuracy", "f1"], ascending=False).to_csv(
        output_dir / "old_oof_candidate_summary.csv", index=False, encoding="utf-8-sig"
    )

    base_summary, base_old_prob, base_old_pred, base_external_prob, base_external_pred = select_base_model(
        y_old, folds, cand_old, cand_external, cand_names, args.seed
    )
    base_external_metrics = metric_dict(y_external, base_external_pred, base_external_prob)

    old_model_feat = build_prob_features(
        cand_old,
        cand_names,
        image_count=pd.to_numeric(old_frame["image_count"], errors="coerce").fillna(1).to_numpy(),
        selection_rule=old_frame["selection_rule"],
    )
    third_model_feat = build_prob_features(
        cand_external,
        cand_names,
        image_count=np.ones(len(third_registry), dtype=float),
        selection_rule=third_registry["selection_rule"],
    )
    old_model_feat["p_selected_base"] = base_old_prob
    old_model_feat["pred_selected_base"] = base_old_pred.astype(float)
    old_model_feat["selected_base_margin"] = np.abs(base_old_prob - 0.5)
    old_model_feat["selected_base_entropy"] = entropy_binary(base_old_prob)
    old_model_feat["selected_base_logit"] = logit(base_old_prob)
    third_model_feat["p_selected_base"] = base_external_prob
    third_model_feat["pred_selected_base"] = base_external_pred.astype(float)
    third_model_feat["selected_base_margin"] = np.abs(base_external_prob - 0.5)
    third_model_feat["selected_base_entropy"] = entropy_binary(base_external_prob)
    third_model_feat["selected_base_logit"] = logit(base_external_prob)
    old_model_feat, third_model_feat = align_columns(old_model_feat, third_model_feat)

    route_scores_old: dict[str, np.ndarray] = {}
    route_scores_external: dict[str, np.ndarray] = {}
    route_targets = {
        "base_wrong": (base_old_pred != y_old).astype(int),
        "base_fn": ((y_old == 1) & (base_old_pred == 0)).astype(int),
        "hard_core": old_frame["hard_core"].astype(int).to_numpy(),
    }
    for feature_kind in ["model", "dino_model"]:
        x_old, x_external = feature_matrix(feature_kind, old_model_feat, third_model_feat, old_dino, third_dino)
        for target_name, target in route_targets.items():
            if len(np.unique(target)) < 2:
                continue
            for spec in route_specs():
                name = f"{target_name}_{feature_kind}_{spec.name}"
                print(f"[route] {name}", flush=True)
                old_score, external_score = oof_and_external_probs(
                    spec, x_old, target, folds, x_external, args.seed + 3000 + len(route_scores_old)
                )
                route_scores_old[name] = old_score
                route_scores_external[name] = external_score

    correctors: dict[str, dict[str, object]] = {}
    for feature_kind in ["model", "dino", "dino_model"]:
        x_old, x_external = feature_matrix(feature_kind, old_model_feat, third_model_feat, old_dino, third_dino)
        for scope in ["non_easy", "hard_all", "hard_core", "all"]:
            train_mask = scope_mask(old_frame, scope)
            for spec in corrector_specs():
                name = f"{feature_kind}_{scope}_{spec.name}"
                print(f"[corrector] {name}", flush=True)
                old_prob, external_prob = oof_and_external_probs(
                    spec,
                    x_old,
                    y_old,
                    folds,
                    x_external,
                    args.seed + 5000 + len(correctors),
                    train_mask=train_mask,
                )
                correctors[name] = {
                    "old_prob": old_prob,
                    "external_prob": external_prob,
                    "feature_kind": feature_kind,
                    "scope": scope,
                    "model": spec.name,
                }

    old_rows: list[dict[str, object]] = []
    budgets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
    corrector_thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    for route_name, route_score in route_scores_old.items():
        for corr_name, corr in correctors.items():
            corr_prob = corr["old_prob"]
            for corr_t in corrector_thresholds:
                corr_pred = (corr_prob >= corr_t).astype(int)
                for budget in budgets:
                    route_t = choose_route_threshold(route_score, budget)
                    routed = route_score >= route_t
                    final_prob = base_old_prob.copy()
                    final_pred = base_old_pred.copy()
                    final_prob[routed] = corr_prob[routed]
                    final_pred[routed] = corr_pred[routed]
                    row = metric_dict(y_old, final_pred, final_prob)
                    row.update(
                        {
                            "route_name": route_name,
                            "route_threshold": route_t,
                            "budget_pct": int(budget),
                            "corrector": corr_name,
                            "corrector_threshold": float(corr_t),
                            "routed_n": int(routed.sum()),
                            "routed_pct": float(routed.mean()),
                            "pass_n": int((~routed).sum()),
                            "pass_acc": float((final_pred[~routed] == y_old[~routed]).mean()) if (~routed).any() else np.nan,
                            "routed_acc": float((final_pred[routed] == y_old[routed]).mean()) if routed.any() else np.nan,
                            "hard_core_routed": int(old_frame["hard_core"].astype(int).to_numpy()[routed].sum()),
                            "rescue_n": int(((base_old_pred != y_old) & (final_pred == y_old) & routed).sum()),
                            "hurt_n": int(((base_old_pred == y_old) & (final_pred != y_old) & routed).sum()),
                        }
                    )
                    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
                    old_rows.append(row)

    old_summary = pd.DataFrame(old_rows).sort_values(
        ["accuracy", "balanced_accuracy", "f1", "net_rescue", "hurt_n", "routed_n"],
        ascending=[False, False, False, False, True, True],
    )
    old_summary.to_csv(output_dir / "old_oof_64style_selection_summary.csv", index=False, encoding="utf-8-sig")
    selected = old_summary.iloc[0].to_dict()

    selected_route = str(selected["route_name"])
    selected_corr = str(selected["corrector"])
    selected_route_t = float(selected["route_threshold"])
    selected_corr_t = float(selected["corrector_threshold"])
    external_route_score = route_scores_external[selected_route]
    external_corr_prob = correctors[selected_corr]["external_prob"]
    external_corr_pred = (external_corr_prob >= selected_corr_t).astype(int)
    external_routed = external_route_score >= selected_route_t
    external_final_prob = base_external_prob.copy()
    external_final_pred = base_external_pred.copy()
    external_final_prob[external_routed] = external_corr_prob[external_routed]
    external_final_pred[external_routed] = external_corr_pred[external_routed]
    external_metrics = metric_dict(y_external, external_final_pred, external_final_prob)
    external_metrics.update(
        {
            "routed_n": int(external_routed.sum()),
            "routed_pct": float(external_routed.mean()),
            "pass_n": int((~external_routed).sum()),
            "pass_acc": float((external_final_pred[~external_routed] == y_external[~external_routed]).mean())
            if (~external_routed).any()
            else np.nan,
            "routed_acc": float((external_final_pred[external_routed] == y_external[external_routed]).mean())
            if external_routed.any()
            else np.nan,
            "base_accuracy": base_external_metrics["accuracy"],
            "base_balanced_accuracy": base_external_metrics["balanced_accuracy"],
            "base_f1": base_external_metrics["f1"],
            "selected_base": base_summary,
            "selected_old_oof_policy": selected,
        }
    )

    case_out = third_registry[
        ["case_id", "original_case_id", "source_folder", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]
    ].copy()
    case_out["base_prob_high"] = base_external_prob
    case_out["base_pred_idx"] = base_external_pred
    case_out["route_score"] = external_route_score
    case_out["routed_to_reviewer"] = external_routed.astype(int)
    case_out["reviewer_prob_high"] = external_corr_prob
    case_out["reviewer_pred_idx"] = external_corr_pred
    case_out["final_prob_high"] = external_final_prob
    case_out["final_pred_idx"] = external_final_pred
    case_out["final_correct"] = (external_final_pred == y_external).astype(int)
    case_out["base_correct"] = (base_external_pred == y_external).astype(int)
    case_out.to_csv(output_dir / "third_batch_external_case_predictions.csv", index=False, encoding="utf-8-sig")

    subtype_rows = []
    for subtype, group in case_out.groupby("task_l6_label"):
        idx = group.index.to_numpy()
        row = metric_dict(y_external[idx], external_final_pred[idx], external_final_prob[idx])
        row.update({"task_l6_label": subtype, "n": int(len(group))})
        subtype_rows.append(row)
    pd.DataFrame(subtype_rows).sort_values("task_l6_label").to_csv(
        output_dir / "third_batch_external_metrics_by_subtype.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "boundary": {
            "third_batch_is_external_only": True,
            "no_doctor_gross_text": True,
            "no_case_id_lookup": True,
            "note": "The exact original No.64 candidate checkpoints were unavailable, so this is a frozen 64-style retraining on old 285 with third batch used only once for external scoring.",
        },
        "old_data_n": int(len(old_frame)),
        "third_data_n": int(len(third_registry)),
        "third_distribution": third_registry["task_l6_label"].value_counts().sort_index().to_dict(),
        "old_oof_selected_base": base_summary,
        "old_oof_selected_policy": selected,
        "third_external_base_metrics": base_external_metrics,
        "third_external_final_metrics": external_metrics,
    }
    (output_dir / "external_64style_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# Task7 第三批 306 例外部测试：64 号同规则图像版

## 实验边界

- 第三批 306 例只用于最后一次外部计分，不参与训练、模型选择或阈值选择。
- 本轮不使用医生肉眼所见文字，不按病例号查表，不使用病理文字。
- 原 64 号依赖的历史候选模型 checkpoint 未完整保存，因此这里采用“64 号同规则重训版”：在旧 285 例上重训图像候选委员会、路由器和复核器，再固定策略外推第三批。

## 旧 285 OOF 选择

- 选中的主模型：`{base_summary['base_name']}`，旧 OOF BACC = {base_summary['balanced_accuracy']:.4f}，ACC = {base_summary['accuracy']:.4f}
- 选中的路由器：`{selected_route}`
- 选中的复核器：`{selected_corr}`
- 旧 OOF 最终 ACC = {selected['accuracy']:.4f}，BACC = {selected['balanced_accuracy']:.4f}，F1 = {selected['f1']:.4f}

## 第三批外部结果

- 主模型单独：ACC = {base_external_metrics['accuracy']:.4f}，BACC = {base_external_metrics['balanced_accuracy']:.4f}，F1 = {base_external_metrics['f1']:.4f}
- 两段式最终：ACC = {external_metrics['accuracy']:.4f}，BACC = {external_metrics['balanced_accuracy']:.4f}，F1 = {external_metrics['f1']:.4f}
- 混淆矩阵 TN/FP/FN/TP = {external_metrics['tn']}/{external_metrics['fp']}/{external_metrics['fn']}/{external_metrics['tp']}
- 进入复核器：{external_metrics['routed_n']}/{len(third_registry)}，占 {external_metrics['routed_pct']:.1%}
- 直接放行准确率：{external_metrics['pass_acc']:.4f}
- 复核池准确率：{external_metrics['routed_acc']:.4f}

## 输出文件

- `third_batch_external_case_predictions.csv`：第三批逐例预测。
- `third_batch_external_metrics_by_subtype.csv`：AB/B1/B2/TC 分亚型结果。
- `old_oof_64style_selection_summary.csv`：旧 285 上策略选择依据。
- `external_64style_report.json`：结构化汇总。
"""
    (output_dir / "README_结果摘要.md").write_text(readme, encoding="utf-8")

    print(json.dumps(report["third_external_final_metrics"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
