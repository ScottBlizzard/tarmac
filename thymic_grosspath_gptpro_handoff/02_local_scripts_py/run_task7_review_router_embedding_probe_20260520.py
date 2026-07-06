from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_dinov2_frozen_probe import load_dinov2_model, set_seed
from run_task56_dinov2_probe import get_task, load_task56_image_df
from run_task7_concat_curriculum_probe import (
    DEFAULT_DINO_IMAGE_SIZE,
    build_case_feature_df,
    extract_concat_features,
)
from run_task7_gross_feature_probe_20260520 import extract_gross_features
from run_task7_hardcore_gross_calibrator_20260520 import add_manual_gross_scores
from thymic_baseline.config import DEFAULT_RANDOM_SEED
from thymic_baseline.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 review-risk router with DINO embedding-neighborhood features.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--split-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_5fold_assignments.csv",
    )
    parser.add_argument("--images-root", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/selected_images")
    parser.add_argument(
        "--curriculum-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520",
    )
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", default="dinov2_vits14,dinov2_vitb14")
    parser.add_argument("--feature-mode", default="cls_patchmean", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--input-variant", default="whole")
    parser.add_argument("--image-size", type=int, default=DEFAULT_DINO_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def load_prob_source(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str})
    required = {"case_id", "prob_high_risk_group", "pred_idx"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    return df[["case_id", "prob_high_risk_group", "pred_idx"]].rename(
        columns={"prob_high_risk_group": f"p_{name}", "pred_idx": f"pred_{name}"}
    )


def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def build_frame(project_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    root = project_root / "outputs" / "batch1_batch2_task567_20260514"
    curriculum = pd.read_csv(project_root / args.curriculum_csv, dtype={"case_id": str})
    frame = curriculum[["case_id", "fold_id", "label_idx", "difficulty", "difficulty_fine"]].copy()
    frame["hard_core"] = (frame["difficulty_fine"] == "hard_core").astype(int)

    sources = {
        "stage2": root / "task7_curriculum_runs/07_case_mlp_schemeB_m060_stage2only_full5fold/oof_case_predictions_mean.csv",
        "stage3": root / "task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/oof_case_predictions_mean.csv",
        "main": root / "task7_curriculum_runs/12_stage2_salvage_foldwise_blend_noncore/oof_case_predictions_mean.csv",
        "upper": root / "task7_curriculum_runs/36_stage3_balcore_foldwise_blend_noncore/oof_case_predictions_mean.csv",
        "distill": root / "task7_curriculum_runs/52_gross_distill_trusted55_w005_full5fold/oof_case_predictions_mean.csv",
    }
    for name, path in sources.items():
        if path.exists():
            frame = frame.merge(load_prob_source(path, name), on="case_id", how="left")

    registry = pd.read_csv(project_root / args.registry_csv, dtype={"case_id": str, "original_case_id": str})
    frame = frame.merge(
        registry[["case_id", "original_case_id", "image_count", "selection_rule"]],
        on="case_id",
        how="left",
    )
    frame["upper_correct"] = (frame["pred_upper"].astype(int) == frame["label_idx"].astype(int)).astype(int)
    frame["upper_wrong"] = 1 - frame["upper_correct"]
    frame["upper_fn"] = ((frame["label_idx"].astype(int) == 1) & (frame["pred_upper"].astype(int) == 0)).astype(int)
    frame["upper_fp"] = ((frame["label_idx"].astype(int) == 0) & (frame["pred_upper"].astype(int) == 1)).astype(int)
    frame["upper_conf"] = np.maximum(frame["p_upper"].astype(float), 1.0 - frame["p_upper"].astype(float))
    return frame


def build_visible_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    prob_names = [col[2:] for col in frame.columns if col.startswith("p_")]
    feat = pd.DataFrame(index=frame.index)
    for name in prob_names:
        p = frame[f"p_{name}"].astype(float).fillna(0.5).to_numpy()
        feat[f"p_{name}"] = p
        feat[f"margin_{name}"] = np.abs(p - 0.5)
        feat[f"entropy_{name}"] = entropy_binary(p)
    for idx, a in enumerate(prob_names):
        for b in prob_names[idx + 1 :]:
            diff = frame[f"p_{a}"].astype(float).fillna(0.5).to_numpy() - frame[f"p_{b}"].astype(float).fillna(0.5).to_numpy()
            feat[f"diff_{a}_{b}"] = diff
            feat[f"absdiff_{a}_{b}"] = np.abs(diff)
    if prob_names:
        probs = frame[[f"p_{name}" for name in prob_names]].astype(float).fillna(0.5).to_numpy()
        feat["prob_mean"] = probs.mean(axis=1)
        feat["prob_std"] = probs.std(axis=1)
        feat["prob_range"] = probs.max(axis=1) - probs.min(axis=1)
        votes = (probs >= 0.5).astype(int)
        feat["vote_sum"] = votes.sum(axis=1)
        feat["vote_disagree"] = ((votes.sum(axis=1) > 0) & (votes.sum(axis=1) < votes.shape[1])).astype(float)
    feat["image_count"] = pd.to_numeric(frame["image_count"], errors="coerce").fillna(1.0)
    feat = pd.concat([feat, pd.get_dummies(frame[["selection_rule"]].fillna(""), dtype=float)], axis=1)
    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_or_extract_case_embeddings(project_root: Path, args: argparse.Namespace, frame: pd.DataFrame, output_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    cache_np = output_dir / "case_dino_concat_features.npy"
    cache_csv = output_dir / "case_dino_concat_feature_table.csv"
    if cache_np.exists() and cache_csv.exists():
        table = pd.read_csv(cache_csv, dtype={"case_id": str})
        features = np.load(cache_np)
    else:
        set_seed(args.seed)
        device = resolve_device(args.device)
        model_names = [item.strip() for item in args.model_names.split(",") if item.strip()]
        models = [load_dinov2_model(repo_dir=project_root / args.repo_dir, model_name=name, device=device) for name in model_names]
        task = get_task("task7_lowhigh_tc")
        image_df = load_task56_image_df(project_root / args.registry_csv, project_root / args.split_csv, project_root / args.images_root, task)
        features, labels, case_ids, image_names = extract_concat_features(image_df, model_names, models, args, device)
        table, case_features = build_case_feature_df(features, labels, case_ids, image_names, "mean")
        table.to_csv(cache_csv, index=False)
        np.save(cache_np, case_features.astype(np.float32))
        features = case_features.astype(np.float32)
    order = frame[["case_id"]].merge(table.reset_index().rename(columns={"index": "feature_row"})[["case_id", "feature_row"]], on="case_id", how="left")
    if order["feature_row"].isna().any():
        missing = order.loc[order["feature_row"].isna(), "case_id"].tolist()
        raise KeyError(f"Missing DINO case features for {missing[:10]}")
    aligned = features[order["feature_row"].astype(int).to_numpy()]
    return aligned.astype(np.float32), table


def pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = (a * a).sum(axis=1, keepdims=True)
    b2 = (b * b).sum(axis=1, keepdims=True).T
    dist2 = np.maximum(a2 + b2 - 2 * a @ b.T, 0.0)
    return np.sqrt(dist2)


def make_embedding_neighborhood_features(
    case_features: np.ndarray,
    frame: pd.DataFrame,
    detector_fold: int,
    include_history_error: bool,
) -> pd.DataFrame:
    folds = frame["fold_id"].astype(int).to_numpy()
    ref_mask = folds != int(detector_fold)
    scaler = StandardScaler()
    ref_x = scaler.fit_transform(case_features[ref_mask]).astype(np.float32)
    all_x = scaler.transform(case_features).astype(np.float32)
    ref_norm = ref_x / np.maximum(np.linalg.norm(ref_x, axis=1, keepdims=True), 1e-8)
    all_norm = all_x / np.maximum(np.linalg.norm(all_x, axis=1, keepdims=True), 1e-8)
    dist = pairwise_l2(all_norm, ref_norm)
    ref_indices = np.where(ref_mask)[0]
    ref_label = frame.loc[ref_mask, "label_idx"].astype(int).to_numpy()
    ref_hard = frame.loc[ref_mask, "hard_core"].astype(int).to_numpy()
    ref_wrong = frame.loc[ref_mask, "upper_wrong"].astype(int).to_numpy()
    ref_fn = frame.loc[ref_mask, "upper_fn"].astype(int).to_numpy()

    # Exclude self for training cases in the reference set.
    ref_pos = {idx: pos for pos, idx in enumerate(ref_indices)}
    for row_idx in ref_indices:
        dist[row_idx, ref_pos[row_idx]] = np.inf

    out = pd.DataFrame(index=frame.index)
    for k in [3, 5, 10, 20]:
        kk = min(k, max(1, ref_mask.sum() - 1))
        nn = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
        nn_dist = np.take_along_axis(dist, nn, axis=1)
        out[f"knn{k}_dist_min"] = np.nanmin(nn_dist, axis=1)
        out[f"knn{k}_dist_mean"] = np.nanmean(nn_dist, axis=1)
        out[f"knn{k}_high_frac"] = ref_label[nn].mean(axis=1)
        out[f"knn{k}_hardcore_frac"] = ref_hard[nn].mean(axis=1)
        if include_history_error:
            out[f"knn{k}_upperwrong_frac"] = ref_wrong[nn].mean(axis=1)
            out[f"knn{k}_upperfn_frac"] = ref_fn[nn].mean(axis=1)

    for name, mask in {
        "low": ref_label == 0,
        "high": ref_label == 1,
        "easy": frame.loc[ref_mask, "difficulty_fine"].to_numpy() == "easy",
        "hardcore": ref_hard == 1,
        "wrong": ref_wrong == 1,
    }.items():
        if not mask.any():
            continue
        centroid = ref_norm[mask].mean(axis=0, keepdims=True)
        centroid = centroid / np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-8)
        out[f"proto_{name}_dist"] = pairwise_l2(all_norm, centroid).ravel()
    if {"proto_low_dist", "proto_high_dist"}.issubset(out.columns):
        out["proto_high_minus_low_dist"] = out["proto_high_dist"] - out["proto_low_dist"]
    if {"proto_hardcore_dist", "proto_easy_dist"}.issubset(out.columns):
        out["proto_hard_minus_easy_dist"] = out["proto_hardcore_dist"] - out["proto_easy_dist"]
    return out.replace([np.inf, -np.inf], np.nan).fillna(out.replace([np.inf, -np.inf], np.nan).max().max())


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))


def fit_predict_oof(
    frame: pd.DataFrame,
    base_features: pd.DataFrame,
    case_features: np.ndarray,
    target_col: str,
    model_name: str,
    include_embedding: bool,
    include_history_error: bool,
) -> np.ndarray:
    y = frame[target_col].astype(int).to_numpy()
    folds = frame["fold_id"].astype(int).to_numpy()
    prob = np.full(len(frame), np.nan, dtype=float)
    for fold in sorted(set(folds)):
        train_mask = folds != fold
        test_mask = folds == fold
        pieces = [base_features]
        if include_embedding:
            pieces.append(make_embedding_neighborhood_features(case_features, frame, int(fold), include_history_error))
        x_all = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if model_name == "logreg":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=20260520),
            )
        elif model_name == "rf":
            model = RandomForestClassifier(
                n_estimators=400, max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=20260520
            )
        elif model_name == "extra":
            model = ExtraTreesClassifier(
                n_estimators=600, max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=20260520
            )
        else:
            raise ValueError(model_name)
        model.fit(x_all.loc[train_mask], y[train_mask])
        prob[test_mask] = model.predict_proba(x_all.loc[test_mask])[:, 1]
    return prob


def best_f1_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (prob >= threshold).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if (f1, -abs(threshold - 0.5)) > (best_f1, -abs(best_t - 0.5)):
            best_t = float(threshold)
            best_f1 = float(f1)
    return best_t, best_f1


def detector_metrics(name: str, frame: pd.DataFrame, target_col: str, prob: np.ndarray) -> dict[str, object]:
    y = frame[target_col].astype(int).to_numpy()
    threshold, _ = best_f1_threshold(y, prob)
    pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    order = np.argsort(-prob)
    topk = max(1, int(y.sum()))
    top = order[:topk]
    return {
        "score": name,
        "target": target_col,
        "positives": int(y.sum()),
        "auc": safe_auc(y, prob),
        "ap": float(average_precision_score(y, prob)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "topk_precision": float(y[top].mean()),
        "topk_recall": float(y[top].sum() / max(1, y.sum())),
    }


def selective_curves(frame: pd.DataFrame, scores: dict[str, np.ndarray], output_dir: Path) -> pd.DataFrame:
    y_true = frame["label_idx"].astype(int).to_numpy()
    upper_pred = frame["pred_upper"].astype(int).to_numpy()
    review_fracs = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    rows = []
    n = len(frame)
    for score_name, values in scores.items():
        order = np.argsort(-np.asarray(values, dtype=float))
        for frac in review_fracs:
            review_n = int(round(n * frac))
            review_mask = np.zeros(n, dtype=bool)
            review_mask[order[:review_n]] = True
            accept_mask = ~review_mask
            if accept_mask.sum():
                accept_acc = accuracy_score(y_true[accept_mask], upper_pred[accept_mask])
                accept_bacc = (
                    balanced_accuracy_score(y_true[accept_mask], upper_pred[accept_mask])
                    if len(np.unique(y_true[accept_mask])) > 1
                    else float("nan")
                )
                tn, fp, fn, tp = confusion_matrix(y_true[accept_mask], upper_pred[accept_mask], labels=[0, 1]).ravel()
            else:
                accept_acc = accept_bacc = float("nan")
                tn = fp = fn = tp = 0
            rows.append(
                {
                    "score": score_name,
                    "review_frac": float(frac),
                    "review_n": int(review_n),
                    "accept_n": int(accept_mask.sum()),
                    "accept_accuracy": float(accept_acc),
                    "accept_balanced_accuracy": float(accept_bacc),
                    "accept_tn": int(tn),
                    "accept_fp": int(fp),
                    "accept_fn": int(fn),
                    "accept_tp": int(tp),
                    "hardcore_recall": float((review_mask & (frame["hard_core"].to_numpy() == 1)).sum() / max(1, frame["hard_core"].sum())),
                    "error_recall": float((review_mask & (frame["upper_wrong"].to_numpy() == 1)).sum() / max(1, frame["upper_wrong"].sum())),
                    "fn_recall": float((review_mask & (frame["upper_fn"].to_numpy() == 1)).sum() / max(1, frame["upper_fn"].sum())),
                    "review_hardcore_precision": float(frame.loc[review_mask, "hard_core"].mean()) if review_mask.sum() else float("nan"),
                    "review_error_precision": float(frame.loc[review_mask, "upper_wrong"].mean()) if review_mask.sum() else float("nan"),
                }
            )
    curves = pd.DataFrame(rows)
    curves.to_csv(output_dir / "selective_review_curves.csv", index=False)
    summary = []
    for score_name, sub in curves.groupby("score", sort=False):
        row30 = sub.iloc[(sub["review_frac"] - 0.30).abs().argmin()]
        reach = sub[sub["accept_accuracy"] >= 0.90]
        first = reach.sort_values("review_frac").iloc[0] if not reach.empty else None
        summary.append(
            {
                "score": score_name,
                "accept_acc_at_30": row30["accept_accuracy"],
                "accept_bacc_at_30": row30["accept_balanced_accuracy"],
                "hardcore_recall_at_30": row30["hardcore_recall"],
                "error_recall_at_30": row30["error_recall"],
                "fn_recall_at_30": row30["fn_recall"],
                "review_frac_for_acc90": float(first["review_frac"]) if first is not None else float("nan"),
                "accept_n_for_acc90": int(first["accept_n"]) if first is not None else float("nan"),
                "hardcore_recall_at_acc90": float(first["hardcore_recall"]) if first is not None else float("nan"),
                "error_recall_at_acc90": float(first["error_recall"]) if first is not None else float("nan"),
            }
        )
    summary_df = pd.DataFrame(summary).sort_values(["accept_acc_at_30", "hardcore_recall_at_30"], ascending=False)
    summary_df.to_csv(output_dir / "selective_review_summary.csv", index=False)
    return summary_df


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_frame(project_root, args)
    base_features = build_visible_model_features(frame)
    registry = pd.read_csv(project_root / args.registry_csv, dtype={"case_id": str})
    gross_frame = frame[["case_id"]].merge(registry, on="case_id", how="left")
    gross_features = add_manual_gross_scores(extract_gross_features(gross_frame))
    case_features, case_table = load_or_extract_case_embeddings(project_root, args, frame, output_dir)

    feature_sets = {
        "model_only": (base_features, False, False),
        "model_embedding_basic": (base_features, True, False),
        "model_embedding_history": (base_features, True, True),
        "model_embedding_history_gross": (pd.concat([base_features, gross_features], axis=1), True, True),
    }
    detector_rows = []
    risk_scores: dict[str, np.ndarray] = {
        "upper_low_conf": 1.0 - frame["upper_conf"].astype(float).to_numpy(),
        "ensemble_prob_std": base_features["prob_std"].to_numpy(),
        "ensemble_prob_range": base_features["prob_range"].to_numpy(),
        "vote_disagree": base_features["vote_disagree"].to_numpy(),
    }
    for target in ["hard_core", "upper_wrong", "upper_fn"]:
        for feature_name, (features, use_embedding, use_history) in feature_sets.items():
            for model_name in ["logreg", "extra", "rf"]:
                score_name = f"learn_{target}_{feature_name}_{model_name}"
                prob = fit_predict_oof(
                    frame,
                    features,
                    case_features,
                    target_col=target,
                    model_name=model_name,
                    include_embedding=use_embedding,
                    include_history_error=use_history,
                )
                risk_scores[score_name] = prob
                detector_rows.append(detector_metrics(score_name, frame, target, prob))
                out = frame[
                    [
                        "case_id",
                        "original_case_id",
                        "fold_id",
                        "label_idx",
                        "difficulty_fine",
                        "hard_core",
                        "upper_wrong",
                        "upper_fn",
                    ]
                ].copy()
                out["risk_prob"] = prob
                out.to_csv(output_dir / f"{score_name}_oof_predictions.csv", index=False)

    detector_df = pd.DataFrame(detector_rows).sort_values(["target", "auc", "ap"], ascending=[True, False, False])
    detector_df.to_csv(output_dir / "detector_ranking.csv", index=False)
    review_summary = selective_curves(frame, risk_scores, output_dir)
    best_score = str(review_summary.iloc[0]["score"])
    case_priority = frame[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "difficulty_fine",
            "hard_core",
            "p_upper",
            "pred_upper",
            "upper_wrong",
            "upper_fn",
            "upper_conf",
        ]
    ].copy()
    for name in ["upper_low_conf", "ensemble_prob_std", "ensemble_prob_range", "vote_disagree", best_score]:
        if name in risk_scores:
            case_priority[name] = risk_scores[name]
    case_priority.sort_values(best_score, ascending=False).to_csv(output_dir / "case_review_priority_best.csv", index=False)
    print("Detector ranking top:")
    print(detector_df.head(20).to_string(index=False))
    print("\nSelective review top:")
    print(review_summary.head(20).to_string(index=False))
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "n_cases": int(len(frame)),
                "hard_core": int(frame["hard_core"].sum()),
                "upper_errors": int(frame["upper_wrong"].sum()),
                "upper_fn": int(frame["upper_fn"].sum()),
                "best_selective_score": best_score,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
