from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_dinov2_frozen_probe import (
    aggregate_run_outputs,
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
from thymic_baseline.config import DEFAULT_RANDOM_SEED, INPUT_VARIANTS
from thymic_baseline.registry import subset_by_fold
from thymic_baseline.train import resolve_device


DEFAULT_DINO_IMAGE_SIZE = 518
TASK7_KEY = "task7_lowhigh_tc"
VIEW_TYPES = ("cut_surface", "outer_surface", "mixed", "unclear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task7 view-aware lightweight routed probe using frozen DINO features.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--task", default=TASK7_KEY, choices=(TASK7_KEY,))
    parser.add_argument("--fold", default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", required=True, help="Comma-separated two model names.")
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
    parser.add_argument("--cut-global-weight", type=float, default=0.8)
    parser.add_argument("--outer-global-weight", type=float, default=0.8)
    parser.add_argument("--mixed-cut-weight", type=float, default=0.4)
    parser.add_argument("--mixed-outer-weight", type=float, default=0.4)
    parser.add_argument("--mixed-global-weight", type=float, default=0.2)
    return parser.parse_args()


def extract_concat_features(
    image_df: pd.DataFrame,
    model_names: list[str],
    models: list[torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
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


def fit_binary_probe(features: np.ndarray, labels: np.ndarray) -> Pipeline:
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
    clf.fit(features, labels)
    return clf


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


def safe_predict_proba(model: Pipeline | None, features: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray | None:
    if model is None:
        return fallback
    return model.predict_proba(features)


def mix_probs(*items: tuple[np.ndarray | None, float]) -> np.ndarray:
    valid = [(arr, weight) for arr, weight in items if arr is not None and weight > 0]
    if not valid:
        raise ValueError("No probability arrays available to mix.")
    total = sum(weight for _, weight in valid)
    out = None
    for arr, weight in valid:
        part = arr * (weight / total)
        out = part if out is None else out + part
    return out


def compute_metrics_from_df(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["label_idx"].to_numpy(dtype=int)
    y_pred = df["pred_idx"].to_numpy(dtype=int)
    prob_high = df["prob_high_risk_group"].to_numpy(dtype=float)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, prob_high)),
        "f1": float(f1_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
    if len(model_names) != 2:
        raise ValueError("--model-names must contain exactly two model names.")

    full_template = pd.read_csv(args.full_template_csv, encoding="utf-8-sig")
    full_template["training_case_id"] = full_template["training_case_id"].astype(str)
    full_template["selected_image_name"] = full_template["selected_image_name"].astype(str)
    full_template["training_image_name"] = full_template["training_image_name"].astype(str)
    seed_df = pd.read_csv(args.viewtype_seed_csv, encoding="utf-8-sig")
    seed_df["training_case_id"] = seed_df["training_case_id"].astype(str)

    image_df = load_task56_image_df(args.registry_csv, args.split_csv, args.images_root, task)
    image_df = image_df.copy()
    image_df["training_image_name"] = image_df["image_name"].astype(str)
    mapping = full_template[["training_image_name", "training_case_id", "selected_image_name"]].drop_duplicates()
    image_df = image_df.merge(mapping, on="training_image_name", how="left")
    if image_df["training_case_id"].isna().any():
        missing = image_df.loc[image_df["training_case_id"].isna(), "training_image_name"].head(10).tolist()
        raise ValueError(f"Failed to map training_case_id for images: {missing}")

    available_folds = load_available_folds(args.split_csv)
    fold_ids = available_folds if args.fold == "all" else [int(args.fold)]
    repo_dir = Path(args.repo_dir)
    models = [load_dinov2_model(repo_dir=repo_dir, model_name=name, device=device) for name in model_names]

    # Extract once for full data to fit view classifier.
    all_df = trim_image_df(image_df, None)
    full_features, full_labels, full_case_ids, full_image_names = extract_concat_features(all_df, model_names, models, args, device)
    feat_df = pd.DataFrame(
        {
            "training_case_id": all_df["training_case_id"].astype(str).tolist(),
            "case_id": full_case_ids,
            "image_name": full_image_names,
            "label_idx": full_labels.tolist(),
        }
    )
    feat_df["feature_row"] = np.arange(len(feat_df))

    labeled = feat_df.merge(seed_df, on="training_case_id", how="inner")
    view_to_idx = {name: idx for idx, name in enumerate(VIEW_TYPES)}
    labeled = labeled[labeled["view_type_seed"].isin(view_to_idx)].copy()
    labeled["view_idx"] = labeled["view_type_seed"].map(view_to_idx).astype(int)
    if len(labeled) < 20:
        raise ValueError("Not enough labeled view-type samples to fit the routing classifier.")

    view_clf = fit_viewtype_classifier(full_features[labeled["feature_row"].to_numpy()], labeled["view_idx"].to_numpy())
    view_pred_idx = view_clf.predict(full_features)
    view_pred_prob = view_clf.predict_proba(full_features)
    idx_to_view = {idx: name for name, idx in view_to_idx.items()}
    feat_df["pred_view_type"] = [idx_to_view[int(x)] for x in view_pred_idx]
    feat_df["pred_view_confidence"] = view_pred_prob.max(axis=1)

    feat_df.to_csv(output_dir / "predicted_view_types.csv", index=False, encoding="utf-8-sig")
    (output_dir / "predicted_view_counts.json").write_text(
        json.dumps(feat_df["pred_view_type"].value_counts().to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    aggregate_methods = [item.strip() for item in args.aggregate_methods.split(",") if item.strip()]
    oof_image_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

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

        train_rows = feat_df["training_case_id"].astype(str).isin(train_case_ids)
        val_rows = feat_df["training_case_id"].astype(str).isin(val_case_ids)
        test_rows = feat_df["training_case_id"].astype(str).isin(test_case_ids)

        x_train = full_features[train_rows.to_numpy()]
        y_train = feat_df.loc[train_rows, "label_idx"].to_numpy(dtype=int)
        x_val = full_features[val_rows.to_numpy()]
        y_val = feat_df.loc[val_rows, "label_idx"].to_numpy(dtype=int)
        x_test = full_features[test_rows.to_numpy()]
        y_test = feat_df.loc[test_rows, "label_idx"].to_numpy(dtype=int)

        train_view = feat_df.loc[train_rows, "pred_view_type"].to_numpy(dtype=str)
        val_view = feat_df.loc[val_rows, "pred_view_type"].to_numpy(dtype=str)
        test_view = feat_df.loc[test_rows, "pred_view_type"].to_numpy(dtype=str)

        global_clf = fit_binary_probe(x_train, y_train)

        def maybe_fit(mask_name: str) -> Pipeline | None:
            mask = train_view == mask_name
            labels = y_train[mask]
            if mask.sum() < 12 or len(np.unique(labels)) < 2:
                return None
            return fit_binary_probe(x_train[mask], labels)

        cut_clf = maybe_fit("cut_surface")
        outer_clf = maybe_fit("outer_surface")

        global_val = global_clf.predict_proba(x_val)
        global_test = global_clf.predict_proba(x_test)
        cut_val = safe_predict_proba(cut_clf, x_val)
        cut_test = safe_predict_proba(cut_clf, x_test)
        outer_val = safe_predict_proba(outer_clf, x_val)
        outer_test = safe_predict_proba(outer_clf, x_test)

        def route_probs(view_names: np.ndarray, g: np.ndarray, c: np.ndarray | None, o: np.ndarray | None) -> np.ndarray:
            out = []
            for idx, view_name in enumerate(view_names):
                g_i = g[idx : idx + 1]
                c_i = c[idx : idx + 1] if c is not None else None
                o_i = o[idx : idx + 1] if o is not None else None
                if view_name == "cut_surface":
                    probs = mix_probs(
                        (c_i, args.cut_global_weight),
                        (g_i, 1.0 - args.cut_global_weight),
                    )
                elif view_name == "outer_surface":
                    probs = mix_probs(
                        (o_i, args.outer_global_weight),
                        (g_i, 1.0 - args.outer_global_weight),
                    )
                elif view_name == "mixed":
                    probs = mix_probs(
                        (c_i, args.mixed_cut_weight),
                        (o_i, args.mixed_outer_weight),
                        (g_i, args.mixed_global_weight),
                    )
                else:
                    probs = g_i
                out.append(probs)
            return np.concatenate(out, axis=0)

        val_probs = route_probs(val_view, global_val, cut_val, outer_val)
        test_probs = route_probs(test_view, global_test, cut_test, outer_test)

        val_predictions = build_prediction_df(
            probs=val_probs,
            labels=y_val,
            case_ids=feat_df.loc[val_rows, "case_id"].astype(str).tolist(),
            image_names=feat_df.loc[val_rows, "image_name"].astype(str).tolist(),
            class_names=task.class_names,
        )
        test_predictions = build_prediction_df(
            probs=test_probs,
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
                "train_images": int(train_rows.sum()),
                "val_images": int(val_rows.sum()),
                "test_images": int(test_rows.sum()),
                "pred_view_counts_train": pd.Series(train_view).value_counts().to_dict(),
                "pred_view_counts_test": pd.Series(test_view).value_counts().to_dict(),
                "cut_classifier_fitted": cut_clf is not None,
                "outer_classifier_fitted": outer_clf is not None,
                "val_case_mean": val_metrics["mean"],
                "test_case_mean": test_metrics["mean"],
            },
        )
        fold_test_case = pd.read_csv(fold_dir / "test_case_predictions_mean.csv")
        fold_test_case["fold_id"] = fold_id
        oof_image_frames.append(fold_test_case)
        summary_rows.append({"fold_id": fold_id, **test_metrics["mean"]})
        print(
            f"[Fold {fold_id}] "
            f"test_auc={format_metric_value(test_metrics['mean'].get('auc'))} | "
            f"test_acc={format_metric_value(test_metrics['mean'].get('accuracy'))} | "
            f"test_bacc={format_metric_value(test_metrics['mean'].get('balanced_accuracy'))} | "
            f"test_f1={format_metric_value(test_metrics['mean'].get('f1'))}",
            flush=True,
        )

    oof_case = pd.concat(oof_image_frames, ignore_index=True).sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    metrics = compute_metrics_from_df(oof_case)
    pd.DataFrame(summary_rows).to_csv(output_dir / "cv_fold_summary_manual.csv", index=False)
    oof_case.to_csv(output_dir / "oof_case_predictions_mean_manual.csv", index=False)
    (output_dir / "oof_metrics_manual.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if len(fold_ids) > 1:
            aggregate_run_outputs(output_dir, task, fold_ids, aggregate_methods)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[Warn] aggregate_run_outputs failed: {exc}", flush=True)
    print("Overall metrics:", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
