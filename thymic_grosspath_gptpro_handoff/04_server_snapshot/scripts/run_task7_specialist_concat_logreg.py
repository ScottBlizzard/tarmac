from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_dinov2_frozen_probe import (
    build_prediction_df,
    extract_dataset_features,
    format_metric_value,
    load_available_folds,
    load_dinov2_model,
    save_split_outputs,
    set_seed,
    trim_image_df,
    write_json,
)
from run_task56_dinov2_probe import get_task, load_task56_image_df
from thymic_baseline.config import DEFAULT_RANDOM_SEED
from thymic_baseline.metrics import aggregate_case_predictions, summarize_prediction_frame
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import resolve_device


DEFAULT_DINO_IMAGE_SIZE = 518
TASK7_KEY = "task7_lowhigh_tc"
VIEW_TYPES = ("cut_surface", "outer_surface", "mixed", "unclear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task7 true scheme-B specialist logreg on frozen DINO concat features.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", default=TASK7_KEY, choices=(TASK7_KEY,))
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", required=True, help="Comma-separated two DINO model names.")
    parser.add_argument("--feature-mode", default="cls", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--viewtype-seed-csv", required=True)
    parser.add_argument("--full-template-csv", required=True)
    parser.add_argument("--image-size", type=int, default=DEFAULT_DINO_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--aggregate-methods", default="mean,max_prob,majority_vote")
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=None)
    parser.add_argument("--mixed-margin", type=float, default=0.08)
    parser.add_argument("--balanced-train-weight", type=float, default=0.45)
    parser.add_argument("--cut-c", type=float, default=1.0)
    parser.add_argument("--outer-c", type=float, default=1.0)
    parser.add_argument("--global-c", type=float, default=1.0)
    parser.add_argument("--selection-metric", default="accuracy", choices=("accuracy", "balanced_accuracy", "auc", "f1"))
    return parser.parse_args()


def extract_concat_features(
    image_df: pd.DataFrame,
    model_names: list[str],
    models: list[Any],
    args: argparse.Namespace,
    device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    features_list: list[np.ndarray] = []
    labels_ref = None
    case_ids_ref = None
    image_names_ref = None
    for model_name, model in zip(model_names, models):
        features, labels, case_ids, image_names = extract_dataset_features(
            image_df,
            input_variant="whole",
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
            if not np.array_equal(labels_ref, labels):
                raise ValueError(f"Label mismatch while concatenating features for {model_name}.")
            if case_ids_ref != case_ids or image_names_ref != image_names:
                raise ValueError(f"Sample order mismatch while concatenating features for {model_name}.")
        features_list.append(features)
    assert labels_ref is not None and case_ids_ref is not None and image_names_ref is not None
    return np.concatenate(features_list, axis=1), labels_ref, case_ids_ref, image_names_ref


def build_feature_frame(
    image_df: pd.DataFrame,
    full_features: np.ndarray,
    full_labels: np.ndarray,
    full_case_ids: list[str],
    full_image_names: list[str],
) -> pd.DataFrame:
    feat_df = pd.DataFrame(
        {
            "training_case_id": image_df["training_case_id"].astype(str).tolist(),
            "case_id": full_case_ids,
            "image_name": full_image_names,
            "label_idx": full_labels.tolist(),
        }
    )
    feat_df["feature_row"] = np.arange(len(feat_df))
    return feat_df


def fit_viewtype_classifier(features: np.ndarray, view_labels: np.ndarray) -> Pipeline:
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    solver="lbfgs",
                    C=1.0,
                    random_state=0,
                ),
            ),
        ]
    )
    clf.fit(features, view_labels)
    return clf


def build_view_probs_for_split(
    train_feat_df: pd.DataFrame,
    train_features: np.ndarray,
    all_feat_df: pd.DataFrame,
    all_features: np.ndarray,
    view_seed: pd.DataFrame,
) -> pd.DataFrame:
    view_to_idx = {name: idx for idx, name in enumerate(VIEW_TYPES)}
    labeled = train_feat_df.merge(view_seed, on="training_case_id", how="inner")
    labeled = labeled[labeled["view_type_seed"].isin(view_to_idx)].copy()
    labeled["view_idx"] = labeled["view_type_seed"].map(view_to_idx).astype(int)
    if len(labeled) < 20:
        raise ValueError("Not enough train-fold view-type seed labels.")

    view_clf = fit_viewtype_classifier(
        train_features[labeled["feature_row"].to_numpy()],
        labeled["view_idx"].to_numpy(),
    )
    raw_probs = view_clf.predict_proba(all_features)
    classes = view_clf.named_steps["logreg"].classes_
    probs = np.zeros((len(all_features), len(VIEW_TYPES)), dtype=np.float64)
    for src_col, class_idx in enumerate(classes):
        probs[:, int(class_idx)] = raw_probs[:, src_col]
    idx_to_view = {idx: name for name, idx in view_to_idx.items()}
    out = all_feat_df.copy()
    for idx, name in idx_to_view.items():
        out[f"view_prob_{name}"] = probs[:, idx]
    pred_idx = probs.argmax(axis=1)
    out["pred_view_type"] = [idx_to_view[int(x)] for x in pred_idx]
    out["pred_view_confidence"] = probs.max(axis=1)
    return out


def add_view_modes(feat_df: pd.DataFrame, mixed_margin: float) -> pd.DataFrame:
    out = feat_df.copy()
    modes: list[str] = []
    for row in out.itertuples(index=False):
        pred_view_type = getattr(row, "pred_view_type")
        cut_prob = float(getattr(row, "view_prob_cut_surface"))
        outer_prob = float(getattr(row, "view_prob_outer_surface"))
        if pred_view_type == "mixed":
            diff = cut_prob - outer_prob
            if diff >= mixed_margin:
                modes.append("cut_heavy")
            elif diff <= -mixed_margin:
                modes.append("outer_heavy")
            else:
                modes.append("balanced_mixed")
        else:
            modes.append(str(pred_view_type))
    out["pred_view_mode"] = modes
    return out


def fit_logreg_probe(features: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray, c_value: float, seed: int) -> Pipeline:
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    solver="lbfgs",
                    C=c_value,
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(features, labels, logreg__sample_weight=sample_weights)
    return clf


def specialist_train_weights(
    view_df: pd.DataFrame,
    balanced_train_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    mode = view_df["pred_view_mode"].astype(str).to_numpy()
    cut_prob = view_df["view_prob_cut_surface"].astype(float).to_numpy()
    outer_prob = view_df["view_prob_outer_surface"].astype(float).to_numpy()
    cut_outer_sum = np.maximum(cut_prob + outer_prob, 1e-6)
    balanced_cut_share = cut_prob / cut_outer_sum
    balanced_outer_share = outer_prob / cut_outer_sum

    cut_weight = np.zeros(len(view_df), dtype=np.float32)
    outer_weight = np.zeros(len(view_df), dtype=np.float32)

    cut_weight[np.isin(mode, ["cut_surface", "cut_heavy"])] = 1.0
    outer_weight[np.isin(mode, ["outer_surface", "outer_heavy"])] = 1.0

    balanced_mask = mode == "balanced_mixed"
    cut_weight[balanced_mask] = balanced_train_weight * balanced_cut_share[balanced_mask]
    outer_weight[balanced_mask] = balanced_train_weight * balanced_outer_share[balanced_mask]
    return cut_weight, outer_weight


def blend_probs(
    global_probs: np.ndarray,
    cut_probs: np.ndarray,
    outer_probs: np.ndarray,
    view_df: pd.DataFrame,
    pure_global_weight: float,
    heavy_global_weight: float,
    balanced_global_weight: float,
) -> np.ndarray:
    mode = view_df["pred_view_mode"].astype(str).to_numpy()
    cut_prob = view_df["view_prob_cut_surface"].astype(float).to_numpy()
    outer_prob = view_df["view_prob_outer_surface"].astype(float).to_numpy()
    cut_outer_sum = np.maximum(cut_prob + outer_prob, 1e-6)
    cut_share = cut_prob / cut_outer_sum
    outer_share = outer_prob / cut_outer_sum

    final_probs = np.zeros_like(global_probs, dtype=np.float64)
    for idx, mode_name in enumerate(mode):
        if mode_name == "cut_surface":
            final_probs[idx] = pure_global_weight * global_probs[idx] + (1.0 - pure_global_weight) * cut_probs[idx]
        elif mode_name == "outer_surface":
            final_probs[idx] = pure_global_weight * global_probs[idx] + (1.0 - pure_global_weight) * outer_probs[idx]
        elif mode_name == "cut_heavy":
            final_probs[idx] = heavy_global_weight * global_probs[idx] + (1.0 - heavy_global_weight) * cut_probs[idx]
        elif mode_name == "outer_heavy":
            final_probs[idx] = heavy_global_weight * global_probs[idx] + (1.0 - heavy_global_weight) * outer_probs[idx]
        elif mode_name == "balanced_mixed":
            specialist = cut_share[idx] * cut_probs[idx] + outer_share[idx] * outer_probs[idx]
            final_probs[idx] = balanced_global_weight * global_probs[idx] + (1.0 - balanced_global_weight) * specialist
        else:
            final_probs[idx] = global_probs[idx]
    return final_probs


def summarize_case_metric(
    probs: np.ndarray,
    labels: np.ndarray,
    case_ids: list[str],
    image_names: list[str],
    task,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred_df = build_prediction_df(
        probs=probs,
        labels=labels,
        case_ids=case_ids,
        image_names=image_names,
        class_names=task.class_names,
    )
    case_df = aggregate_case_predictions(pred_df, task.class_names, method="mean")
    metrics = summarize_prediction_frame(case_df, task.class_names)
    return case_df, metrics


def metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, float("nan"))
    try:
        return float(value)
    except Exception:
        return float("nan")


def choose_fusion_params(
    global_probs: np.ndarray,
    cut_probs: np.ndarray,
    outer_probs: np.ndarray,
    view_df: pd.DataFrame,
    labels: np.ndarray,
    case_ids: list[str],
    image_names: list[str],
    task,
    selection_metric: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    best_params: dict[str, float] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score: float | None = None
    for pure_global_weight in (0.0, 0.1, 0.2):
        for heavy_global_weight in (0.1, 0.2, 0.3):
            for balanced_global_weight in (0.3, 0.4, 0.5, 0.6):
                probs = blend_probs(
                    global_probs,
                    cut_probs,
                    outer_probs,
                    view_df,
                    pure_global_weight=pure_global_weight,
                    heavy_global_weight=heavy_global_weight,
                    balanced_global_weight=balanced_global_weight,
                )
                _, metrics = summarize_case_metric(probs, labels, case_ids, image_names, task)
                score = metric_value(metrics, selection_metric)
                if best_score is None or (not math.isnan(score) and score > best_score):
                    best_score = score
                    best_params = {
                        "pure_global_weight": float(pure_global_weight),
                        "heavy_global_weight": float(heavy_global_weight),
                        "balanced_global_weight": float(balanced_global_weight),
                    }
                    best_metrics = metrics
    if best_params is None or best_metrics is None:
        raise RuntimeError("Failed to choose fusion params.")
    return best_params, best_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]

    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    if len(model_names) != 2:
        raise ValueError("--model-names must contain exactly two model names.")

    full_template = pd.read_csv(args.full_template_csv, encoding="utf-8-sig")
    full_template["training_image_name"] = full_template["training_image_name"].astype(str)
    full_template["training_case_id"] = full_template["training_case_id"].astype(str)
    view_seed = pd.read_csv(args.viewtype_seed_csv, encoding="utf-8-sig")
    view_seed["training_case_id"] = view_seed["training_case_id"].astype(str)

    image_df = load_task56_image_df(args.registry_csv, args.split_csv, args.images_root, task).copy()
    image_df["training_image_name"] = image_df["image_name"].astype(str)
    template_map = full_template[["training_image_name", "training_case_id"]].drop_duplicates()
    image_df = image_df.merge(template_map, on="training_image_name", how="left")
    if image_df["training_case_id"].isna().any():
        missing = image_df.loc[image_df["training_case_id"].isna(), "training_image_name"].head(10).tolist()
        raise ValueError(f"Missing training_case_id mapping for images: {missing}")
    image_df["training_case_id"] = image_df["training_case_id"].astype(str)

    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]
    repo_dir = Path(args.repo_dir)
    models = [load_dinov2_model(repo_dir=repo_dir, model_name=name, device=device) for name in model_names]

    all_df = trim_image_df(image_df, None)
    full_features, full_labels, full_case_ids, full_image_names = extract_concat_features(all_df, model_names, models, args, device)
    feat_df = build_feature_frame(all_df, full_features, full_labels, full_case_ids, full_image_names)

    oof_case_frames: list[pd.DataFrame] = []
    fold_summary_rows: list[dict[str, Any]] = []
    fold_view_frames: list[pd.DataFrame] = []

    for fold_id in fold_ids:
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df, val_df, test_df = subset_by_fold(image_df, test_fold=fold_id, num_folds=len(available_folds))
        train_df = trim_image_df(train_df, args.max_train_images)
        val_df = trim_image_df(val_df, args.max_val_images)
        test_df = trim_image_df(test_df, args.max_test_images)

        train_case_ids = set(train_df["training_case_id"].astype(str))
        val_case_ids = set(val_df["training_case_id"].astype(str))
        test_case_ids = set(test_df["training_case_id"].astype(str))

        train_rows = feat_df["training_case_id"].isin(train_case_ids)
        val_rows = feat_df["training_case_id"].isin(val_case_ids)
        test_rows = feat_df["training_case_id"].isin(test_case_ids)

        fold_view_df = build_view_probs_for_split(
            train_feat_df=feat_df.loc[train_rows].copy(),
            train_features=full_features,
            all_feat_df=feat_df.copy(),
            all_features=full_features,
            view_seed=view_seed,
        )
        fold_view_df = add_view_modes(fold_view_df, mixed_margin=args.mixed_margin)
        fold_view_df["fold_id"] = fold_id
        fold_view_frames.append(fold_view_df.loc[val_rows | test_rows].copy())

        x_train = full_features[train_rows.to_numpy()]
        y_train = feat_df.loc[train_rows, "label_idx"].to_numpy(dtype=int)
        x_val = full_features[val_rows.to_numpy()]
        y_val = feat_df.loc[val_rows, "label_idx"].to_numpy(dtype=int)
        x_test = full_features[test_rows.to_numpy()]
        y_test = feat_df.loc[test_rows, "label_idx"].to_numpy(dtype=int)

        train_view_df = fold_view_df.loc[train_rows].copy()
        val_view_df = fold_view_df.loc[val_rows].copy()
        test_view_df = fold_view_df.loc[test_rows].copy()

        cut_weight_train, outer_weight_train = specialist_train_weights(train_view_df, balanced_train_weight=args.balanced_train_weight)
        cut_mask_train = cut_weight_train > 1e-6
        outer_mask_train = outer_weight_train > 1e-6

        global_model = fit_logreg_probe(
            features=x_train,
            labels=y_train,
            sample_weights=np.ones(len(y_train), dtype=np.float32),
            c_value=args.global_c,
            seed=args.seed,
        )
        cut_model = fit_logreg_probe(
            features=x_train[cut_mask_train],
            labels=y_train[cut_mask_train],
            sample_weights=cut_weight_train[cut_mask_train],
            c_value=args.cut_c,
            seed=args.seed + 11,
        )
        outer_model = fit_logreg_probe(
            features=x_train[outer_mask_train],
            labels=y_train[outer_mask_train],
            sample_weights=outer_weight_train[outer_mask_train],
            c_value=args.outer_c,
            seed=args.seed + 29,
        )

        global_val_probs = global_model.predict_proba(x_val)
        cut_val_probs = cut_model.predict_proba(x_val)
        outer_val_probs = outer_model.predict_proba(x_val)
        best_params, best_val_metrics = choose_fusion_params(
            global_val_probs,
            cut_val_probs,
            outer_val_probs,
            val_view_df,
            y_val,
            feat_df.loc[val_rows, "case_id"].astype(str).tolist(),
            feat_df.loc[val_rows, "image_name"].astype(str).tolist(),
            task,
            selection_metric=args.selection_metric,
        )

        global_test_probs = global_model.predict_proba(x_test)
        cut_test_probs = cut_model.predict_proba(x_test)
        outer_test_probs = outer_model.predict_proba(x_test)
        final_test_probs = blend_probs(
            global_test_probs,
            cut_test_probs,
            outer_test_probs,
            test_view_df,
            pure_global_weight=best_params["pure_global_weight"],
            heavy_global_weight=best_params["heavy_global_weight"],
            balanced_global_weight=best_params["balanced_global_weight"],
        )

        val_final_probs = blend_probs(
            global_val_probs,
            cut_val_probs,
            outer_val_probs,
            val_view_df,
            pure_global_weight=best_params["pure_global_weight"],
            heavy_global_weight=best_params["heavy_global_weight"],
            balanced_global_weight=best_params["balanced_global_weight"],
        )

        val_predictions = build_prediction_df(
            probs=val_final_probs,
            labels=y_val,
            case_ids=feat_df.loc[val_rows, "case_id"].astype(str).tolist(),
            image_names=feat_df.loc[val_rows, "image_name"].astype(str).tolist(),
            class_names=task.class_names,
        )
        test_predictions = build_prediction_df(
            probs=final_test_probs,
            labels=y_test,
            case_ids=feat_df.loc[test_rows, "case_id"].astype(str).tolist(),
            image_names=feat_df.loc[test_rows, "image_name"].astype(str).tolist(),
            class_names=task.class_names,
        )

        val_metrics = save_split_outputs(fold_dir, "val", val_predictions, task, aggregate_methods)
        test_metrics = save_split_outputs(fold_dir, "test", test_predictions, task, aggregate_methods)
        write_json(
            fold_dir / "fold_summary.json",
            {
                "task": task.key,
                "fold_id": fold_id,
                "selection_metric": args.selection_metric,
                "best_fusion_params": best_params,
                "best_val_selected_metrics": best_val_metrics,
                "train_images": int(train_rows.sum()),
                "val_images": int(val_rows.sum()),
                "test_images": int(test_rows.sum()),
                "pred_view_mode_counts_train": train_view_df["pred_view_mode"].value_counts().to_dict(),
                "pred_view_mode_counts_test": test_view_df["pred_view_mode"].value_counts().to_dict(),
                "val_case_mean": val_metrics["mean"],
                "test_case_mean": test_metrics["mean"],
            },
        )

        fold_case = pd.read_csv(fold_dir / "test_case_predictions_mean.csv")
        fold_case["fold_id"] = fold_id
        oof_case_frames.append(fold_case)
        fold_summary_rows.append({"fold_id": fold_id, **test_metrics["mean"]})

        print(
            f"[Fold {fold_id}] "
            f"val_{args.selection_metric}={format_metric_value(best_val_metrics.get(args.selection_metric))} | "
            f"test_auc={format_metric_value(test_metrics['mean'].get('auc'))} | "
            f"test_acc={format_metric_value(test_metrics['mean'].get('accuracy'))} | "
            f"test_bacc={format_metric_value(test_metrics['mean'].get('balanced_accuracy'))} | "
            f"test_f1={format_metric_value(test_metrics['mean'].get('f1'))}",
            flush=True,
        )

    pd.concat(fold_view_frames, ignore_index=True).to_csv(output_dir / "predicted_view_types_byfold.csv", index=False, encoding="utf-8-sig")

    oof_case = pd.concat(oof_case_frames, ignore_index=True).sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    y_true = oof_case["label_idx"].astype(int).to_numpy()
    y_pred = oof_case["pred_idx"].astype(int).to_numpy()
    prob_high = oof_case["prob_high_risk_group"].astype(float).to_numpy()
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(
            0.5
            * (
                (tp / (tp + fn) if (tp + fn) else float("nan"))
                + (tn / (tn + fp) if (tn + fp) else float("nan"))
            )
        ),
        "auc": float(roc_auc_score(y_true, prob_high)),
        "f1": float(f1_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn) if (tp + fn) else float("nan")),
        "specificity": float(tn / (tn + fp) if (tn + fp) else float("nan")),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    pd.DataFrame(fold_summary_rows).to_csv(output_dir / "cv_fold_summary_manual.csv", index=False)
    write_json(output_dir / "oof_metrics_manual.json", metrics)
    oof_case.to_csv(output_dir / "oof_case_predictions_mean.csv", index=False)
    print(f"Overall metrics: {json.dumps(metrics, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()

