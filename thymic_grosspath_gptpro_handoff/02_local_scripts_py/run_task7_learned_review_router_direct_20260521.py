from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_task7_gross_text_stacking_20260521 import load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 direct learned router: decide when to use gross/text specialist.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", default="fast", choices=("fast", "full"))
    parser.add_argument("--registry-csv", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv")
    parser.add_argument("--split-csv", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_5fold_assignments.csv")
    parser.add_argument("--curriculum-csv", default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv")
    parser.add_argument("--review-score-csv", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv")
    parser.add_argument("--best41-csv", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/41_best_candidate_stacking_balanced_20260520/best_case_outputs_full.csv")
    parser.add_argument("--gross-specialist-oof", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/57_learned_gross_text_switch_20260521/oof_case_predictions_mean.csv")
    parser.add_argument("--embedding-table", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_feature_table.csv")
    parser.add_argument("--embedding-npy", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_features.npy")
    parser.add_argument("--output-dir", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/60_learned_review_router_direct_20260521")
    return parser.parse_args()


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_models(mode: str) -> dict[str, object]:
    models: dict[str, object] = {
        "logreg_c01": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=20260521)),
        "logreg_c03": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=20260521)),
        "extra_d3_l8": ExtraTreesClassifier(n_estimators=500, max_depth=3, min_samples_leaf=8, class_weight="balanced", random_state=20260521, n_jobs=-1),
        "rf_d3_l8": RandomForestClassifier(n_estimators=400, max_depth=3, min_samples_leaf=8, class_weight="balanced_subsample", random_state=20260521, n_jobs=-1),
        "gb_d1": GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=1, random_state=20260521),
    }
    if mode == "full":
        models.update(
            {
                "logreg_c003": make_pipeline(StandardScaler(), LogisticRegression(C=0.03, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=20260521)),
                "logreg_c1": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=20260521)),
                "extra_d2_l5": ExtraTreesClassifier(n_estimators=500, max_depth=2, min_samples_leaf=5, class_weight="balanced", random_state=20260521, n_jobs=-1),
                "rf_d2_l5": RandomForestClassifier(n_estimators=400, max_depth=2, min_samples_leaf=5, class_weight="balanced_subsample", random_state=20260521, n_jobs=-1),
                "gb_d2": GradientBoostingClassifier(n_estimators=80, learning_rate=0.035, max_depth=2, random_state=20260521),
            }
        )
    return models


def load_embedding(root: Path, args: argparse.Namespace, case_ids: pd.Series) -> np.ndarray:
    table = pd.read_csv(root / args.embedding_table, dtype={"case_id": str})
    arr = np.load(root / args.embedding_npy)
    aligned = case_ids.to_frame("case_id").merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if aligned["feature_idx"].isna().any():
        missing = aligned.loc[aligned["feature_idx"].isna(), "case_id"].head().tolist()
        raise ValueError(f"Missing embedding for cases: {missing}")
    return arr[aligned["feature_idx"].astype(int).to_numpy()]


def build_numeric(df: pd.DataFrame, numeric_all: pd.DataFrame, specialist: pd.DataFrame) -> pd.DataFrame:
    x = numeric_all.copy()
    x["base_prob"] = df["p_best41"].astype(float)
    x["base_pred"] = df["pred_best41"].astype(float)
    x["base_margin"] = np.abs(x["base_prob"] - 0.5)
    x["calib_prob"] = specialist["calib_prob_high_risk_group"].astype(float).to_numpy()
    x["calib_pred"] = specialist["calib_pred_idx"].astype(float).to_numpy()
    x["calib_margin"] = np.abs(x["calib_prob"] - 0.5)
    x["calib_minus_base"] = x["calib_prob"] - x["base_prob"]
    x["abs_calib_minus_base"] = np.abs(x["calib_minus_base"])
    x["base_calib_disagree"] = (x["base_pred"] != x["calib_pred"]).astype(float)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def get_targets(df: pd.DataFrame, base_pred: np.ndarray, calib_pred: np.ndarray) -> dict[str, np.ndarray]:
    y = df["label_idx"].to_numpy(dtype=int)
    base_correct = base_pred == y
    calib_correct = calib_pred == y
    return {
        "hard_core": df["difficulty_fine"].eq("hard_core").astype(int).to_numpy(),
        "all_hard": df["difficulty_fine"].isin(["hard_core", "hard_salvage_teacher"]).astype(int).to_numpy(),
        "base_wrong": (~base_correct).astype(int),
        "switch_gain": ((~base_correct) & calib_correct).astype(int),
    }


def prepare_feature_matrix(
    numeric: pd.DataFrame,
    emb: np.ndarray | None,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    feature_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    pieces_train = []
    pieces_test = []
    if "numeric" in feature_name:
        scaler = StandardScaler()
        pieces_train.append(scaler.fit_transform(numeric.loc[train_mask]))
        pieces_test.append(scaler.transform(numeric.loc[test_mask]))
    if "emb" in feature_name:
        n_comp = int(feature_name.split("emb")[-1])
        assert emb is not None
        scaler = StandardScaler()
        emb_train = scaler.fit_transform(emb[train_mask])
        emb_test = scaler.transform(emb[test_mask])
        n_comp = min(n_comp, emb_train.shape[0] - 1, emb_train.shape[1])
        pca = PCA(n_components=n_comp, random_state=20260521)
        pieces_train.append(pca.fit_transform(emb_train))
        pieces_test.append(pca.transform(emb_test))
    return np.hstack(pieces_train), np.hstack(pieces_test)


def score_router(
    numeric: pd.DataFrame,
    emb: np.ndarray | None,
    folds: np.ndarray,
    route_target: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    feature_name: str,
    model: object,
) -> np.ndarray:
    if train_mask.sum() < 10 or len(np.unique(route_target[train_mask])) < 2:
        return np.zeros(test_mask.sum(), dtype=float)
    x_train, x_test = prepare_feature_matrix(numeric, emb, train_mask, test_mask, feature_name)
    clf = clone(model)
    clf.fit(x_train, route_target[train_mask])
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x_test)[:, 1]
    return clf.decision_function(x_test)


def inner_router_oof(
    numeric: pd.DataFrame,
    emb: np.ndarray | None,
    folds: np.ndarray,
    route_target: np.ndarray,
    outer_train: np.ndarray,
    feature_name: str,
    model: object,
) -> np.ndarray:
    scores = np.full(len(folds), np.nan, dtype=float)
    for inner_fold in sorted(set(folds[outer_train])):
        tr = outer_train & (folds != inner_fold)
        va = outer_train & (folds == inner_fold)
        scores[va] = score_router(numeric, emb, folds, route_target, tr, va, feature_name, model)
    return scores


def apply_policy(base_pred: np.ndarray, calib_pred: np.ndarray, scores: np.ndarray, policy: dict[str, object]) -> np.ndarray:
    switch = scores >= float(policy["threshold"])
    if policy["direction"] == "disagree":
        switch &= base_pred != calib_pred
    elif policy["direction"] == "low_to_high":
        switch &= (base_pred == 0) & (calib_pred == 1)
    elif policy["direction"] == "high_to_low":
        switch &= (base_pred == 1) & (calib_pred == 0)
    pred = base_pred.copy()
    pred[switch] = calib_pred[switch]
    return pred


def select_policy(y: np.ndarray, base_pred: np.ndarray, calib_pred: np.ndarray, scores: np.ndarray, train_mask: np.ndarray, objective: str) -> dict[str, object]:
    valid = train_mask & ~np.isnan(scores)
    best: dict[str, object] | None = None
    if valid.sum() == 0:
        return {"direction": "none", "threshold": 1e9, "train_accuracy": float(accuracy_score(y[train_mask], base_pred[train_mask])), "train_balanced_accuracy": float(balanced_accuracy_score(y[train_mask], base_pred[train_mask])), "train_switch_n": 0}
    values = np.unique(np.quantile(scores[valid], np.linspace(0.05, 0.95, 91)))
    for direction in ["any", "disagree", "low_to_high", "high_to_low"]:
        for threshold in values:
            pred = base_pred.copy()
            pred[valid] = apply_policy(base_pred[valid], calib_pred[valid], scores[valid], {"direction": direction, "threshold": float(threshold)})
            acc = float(accuracy_score(y[train_mask], pred[train_mask]))
            bacc = float(balanced_accuracy_score(y[train_mask], pred[train_mask]))
            switched = int((pred[train_mask] != base_pred[train_mask]).sum())
            primary = acc if objective == "accuracy" else bacc
            key = (primary, bacc, acc, -switched)
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "direction": direction,
                    "threshold": float(threshold),
                    "train_accuracy": acc,
                    "train_balanced_accuracy": bacc,
                    "train_switch_n": switched,
                }
    assert best is not None
    best.pop("key")
    return best


def run_one(
    df: pd.DataFrame,
    numeric: pd.DataFrame,
    emb: np.ndarray | None,
    feature_name: str,
    target_name: str,
    route_target: np.ndarray,
    model_name: str,
    model: object,
    objective: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    y = df["label_idx"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy(dtype=int)
    base_pred = df["pred_best41"].to_numpy(dtype=int)
    base_prob = df["p_best41"].to_numpy(dtype=float)
    calib_pred = df["calib_pred_idx"].to_numpy(dtype=int)
    calib_prob = df["calib_prob_high_risk_group"].to_numpy(dtype=float)
    final_pred = base_pred.copy()
    final_prob = base_prob.copy()
    route_score = np.full(len(df), np.nan, dtype=float)
    switched = np.zeros(len(df), dtype=bool)
    fold_rows = []
    for fold in sorted(set(folds)):
        outer_train = folds != fold
        outer_test = folds == fold
        train_scores = inner_router_oof(numeric, emb, folds, route_target, outer_train, feature_name, model)
        policy = select_policy(y, base_pred, calib_pred, train_scores, outer_train, objective)
        test_scores = score_router(numeric, emb, folds, route_target, outer_train, outer_test, feature_name, model)
        route_score[outer_test] = test_scores
        test_pred = apply_policy(base_pred[outer_test], calib_pred[outer_test], test_scores, policy)
        final_pred[outer_test] = test_pred
        use_calib = test_pred != base_pred[outer_test]
        final_prob[outer_test] = np.where(use_calib, calib_prob[outer_test], base_prob[outer_test])
        switched[outer_test] = use_calib
        fold_rows.append(
            {
                "fold_id": int(fold),
                "test_n": int(outer_test.sum()),
                "test_switch_n": int(use_calib.sum()),
                **policy,
            }
        )
    metrics = metric_dict(y, final_pred, final_prob)
    rescue = ((base_pred != y) & (final_pred == y)).sum()
    hurt = ((base_pred == y) & (final_pred != y)).sum()
    routed_acc = float(accuracy_score(y[switched], final_pred[switched])) if switched.sum() else float("nan")
    pass_acc = float(accuracy_score(y[~switched], final_pred[~switched])) if (~switched).sum() else float("nan")
    metrics.update(
        {
            "feature_set": feature_name,
            "target": target_name,
            "model": model_name,
            "objective": objective,
            "routed_n": int(switched.sum()),
            "routed_frac": float(switched.mean()),
            "routed_acc": routed_acc,
            "pass_n": int((~switched).sum()),
            "pass_frac": float((~switched).mean()),
            "pass_acc": pass_acc,
            "rescue_n": int(rescue),
            "hurt_n": int(hurt),
            "net_rescue": int(rescue - hurt),
        }
    )
    oof = df[["case_id", "original_case_id", "fold_id", "label_idx", "difficulty_fine", "task_l6_label", "task_l7_label"]].copy()
    oof["base_pred_idx"] = base_pred
    oof["base_prob_high_risk_group"] = base_prob
    oof["calib_pred_idx"] = calib_pred
    oof["calib_prob_high_risk_group"] = calib_prob
    oof["route_score"] = route_score
    oof["switched"] = switched
    oof["pred_idx"] = final_pred
    oof["prob_high_risk_group"] = final_prob
    oof["correct"] = final_pred == y
    return metrics, oof, pd.DataFrame(fold_rows)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df, _text, numeric_all = load_data(root, args)
    specialist = pd.read_csv(root / args.gross_specialist_oof, dtype={"case_id": str})
    specialist = df[["case_id"]].merge(
        specialist[["case_id", "calib_prob_high_risk_group", "calib_pred_idx"]],
        on="case_id",
        how="left",
    )
    df = pd.concat([df.reset_index(drop=True), specialist[["calib_prob_high_risk_group", "calib_pred_idx"]].reset_index(drop=True)], axis=1)
    numeric = build_numeric(df, numeric_all.reset_index(drop=True), specialist)
    emb = load_embedding(root, args, df["case_id"])
    targets = get_targets(df, df["pred_best41"].to_numpy(dtype=int), df["calib_pred_idx"].to_numpy(dtype=int))
    models = make_models(args.mode)
    if args.mode == "fast":
        feature_sets = ["numeric", "numeric_emb8", "numeric_emb16", "emb16"]
        target_names = ["hard_core", "base_wrong", "switch_gain"]
        objectives = ["accuracy"]
    else:
        feature_sets = ["numeric", "numeric_emb8", "numeric_emb16", "numeric_emb32", "emb8", "emb16", "emb32"]
        target_names = ["hard_core", "all_hard", "base_wrong", "switch_gain"]
        objectives = ["accuracy", "balanced_accuracy"]
    rows = []
    best_oof: pd.DataFrame | None = None
    best_key: tuple[float, float, int] | None = None
    for feature_name in feature_sets:
        for target_name in target_names:
            for model_name, model in models.items():
                for objective in objectives:
                    metrics, oof, fold_df = run_one(df, numeric, emb, feature_name, target_name, targets[target_name], model_name, model, objective)
                    rows.append(metrics)
                    run_name = f"{feature_name}__{target_name}__{model_name}__{objective}"
                    run_dir = output_dir / run_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    oof.to_csv(run_dir / "oof_case_predictions_mean.csv", index=False, encoding="utf-8-sig")
                    fold_df.to_csv(run_dir / "fold_policy.csv", index=False, encoding="utf-8-sig")
                    (run_dir / "overall_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                    key = (float(metrics["accuracy"]), float(metrics["balanced_accuracy"]), int(metrics["net_rescue"]))
                    if best_key is None or key > best_key:
                        best_key = key
                        best_oof = oof.copy()
                    pd.DataFrame(rows).sort_values(["accuracy", "balanced_accuracy", "net_rescue"], ascending=False).to_csv(
                        output_dir / "learned_review_router_direct_summary.partial.csv", index=False, encoding="utf-8-sig"
                    )
    summary = pd.DataFrame(rows).sort_values(["accuracy", "balanced_accuracy", "net_rescue"], ascending=False)
    summary.to_csv(output_dir / "learned_review_router_direct_summary.csv", index=False, encoding="utf-8-sig")
    if best_oof is not None:
        best_oof.to_csv(output_dir / "best_oof_case_predictions_mean.csv", index=False, encoding="utf-8-sig")
    (output_dir / "best_summary.json").write_text(json.dumps(summary.iloc[0].to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
