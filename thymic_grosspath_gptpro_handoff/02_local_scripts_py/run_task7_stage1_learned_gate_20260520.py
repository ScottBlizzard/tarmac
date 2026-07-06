from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested OOF learned gate for Task7 stage-1 direct-pass decisions.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/14_stage1_learned_gate_20260520",
    )
    parser.add_argument("--targets", default="0.90,0.95")
    parser.add_argument("--selection-modes", default="raw,wilson90")
    parser.add_argument("--feature-sets", default="raw_outputs,raw_plus_all_review")
    parser.add_argument("--models", default="logreg_l2,extra_shallow")
    parser.add_argument("--min-train-accept", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def wilson_lower_bound(successes: np.ndarray, totals: np.ndarray, z: float) -> np.ndarray:
    phat = successes / totals
    z2 = z * z
    denom = 1.0 + z2 / totals
    center = phat + z2 / (2.0 * totals)
    spread = z * np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * totals)) / totals)
    return (center - spread) / denom


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    if len(y) == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "specificity_low": np.nan,
            "sensitivity_high": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(
    score: np.ndarray,
    correct: np.ndarray,
    target_acc: float,
    min_accept: int,
    selection_mode: str,
) -> dict[str, float | int]:
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_correct = correct[order].astype(float)
    cumsum = np.cumsum(sorted_correct)
    ks = np.arange(1, len(order) + 1)
    acc = cumsum / ks
    if selection_mode == "raw":
        criterion = acc
    elif selection_mode == "wilson90":
        criterion = wilson_lower_bound(cumsum, ks, z=1.6448536269514722)
    elif selection_mode == "wilson95":
        criterion = wilson_lower_bound(cumsum, ks, z=1.959963984540054)
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")
    ok = (criterion >= target_acc) & (ks >= min_accept)
    if not ok.any():
        return {"threshold": np.inf, "train_accept_n": 0, "train_accept_acc": np.nan, "train_accept_criterion": np.nan}
    best_idx = np.where(ok)[0][-1]
    return {
        "threshold": float(sorted_score[best_idx]),
        "train_accept_n": int(best_idx + 1),
        "train_accept_acc": float(acc[best_idx]),
        "train_accept_criterion": float(criterion[best_idx]),
    }


def build_feature_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    pred_cols = [c for c in df.columns if c.startswith("pred_") and c != "pred_upper"]
    review_cols = [c for c in df.columns if c.startswith("review_score_")]

    features = pd.DataFrame(index=df.index)
    for col in prob_cols:
        p = pd.to_numeric(df[col], errors="coerce").fillna(0.5).to_numpy(dtype=float)
        name = col[2:]
        features[f"p_{name}"] = p
        features[f"margin_{name}"] = np.abs(p - 0.5)
        features[f"entropy_{name}"] = entropy_binary(p)
    for col in pred_cols:
        features[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    probs = df[prob_cols].astype(float).fillna(0.5).to_numpy()
    votes = (probs >= 0.5).astype(float)
    p_upper = pd.to_numeric(df["p_upper"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    pred_upper = pd.to_numeric(df["pred_upper"], errors="coerce").fillna(0).to_numpy(dtype=float)
    upper_conf = np.where(pred_upper >= 0.5, p_upper, 1.0 - p_upper)
    features["prob_mean"] = probs.mean(axis=1)
    features["prob_median"] = np.median(probs, axis=1)
    features["prob_std"] = probs.std(axis=1)
    features["prob_range"] = probs.max(axis=1) - probs.min(axis=1)
    features["vote_frac"] = votes.mean(axis=1)
    features["vote_disagree"] = ((votes.sum(axis=1) > 0) & (votes.sum(axis=1) < votes.shape[1])).astype(float)
    features["upper_conf"] = upper_conf
    features["upper_low_conf"] = 1.0 - upper_conf
    features["upper_pred_is_high"] = pred_upper
    features["predlow_p_upper"] = np.where(pred_upper < 0.5, p_upper, 0.0)
    features["predhigh_1_minus_p_upper"] = np.where(pred_upper >= 0.5, 1.0 - p_upper, 0.0)
    features["image_count"] = pd.to_numeric(df.get("image_count", 1), errors="coerce").fillna(1.0)
    if "selection_rule" in df.columns:
        features = pd.concat([features, pd.get_dummies(df["selection_rule"].fillna(""), prefix="selection", dtype=float)], axis=1)

    raw = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    simple_review = df[[c for c in review_cols if not c.startswith("review_score_learn_")]].astype(float).fillna(0.0)
    all_review = df[review_cols].astype(float).fillna(0.0)
    return {
        "raw_outputs": raw,
        "raw_plus_simple_review": pd.concat([raw, simple_review], axis=1),
        "raw_plus_all_review": pd.concat([raw, all_review], axis=1),
    }


def make_models(seed: int) -> dict[str, object]:
    return {
        "logreg_l2": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "rf_shallow": RandomForestClassifier(
            n_estimators=260,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_shallow": ExtraTreesClassifier(
            n_estimators=320,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gb_small": GradientBoostingClassifier(max_depth=2, learning_rate=0.035, n_estimators=80, random_state=seed),
    }


def fit_predict_prob(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    clf = clone(model)
    clf.fit(x_train, y_train)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x_test)[:, 1]
    decision = clf.decision_function(x_test)
    return 1.0 / (1.0 + np.exp(-decision))


def inner_cv_scores(model: object, x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan, dtype=float)
    for inner_fold in sorted(np.unique(folds)):
        train = folds != inner_fold
        test = folds == inner_fold
        if len(np.unique(y[train])) < 2:
            out[test] = float(np.mean(y[train]))
        else:
            out[test] = fit_predict_prob(model, x[train], y[train], x[test])
    return np.nan_to_num(out, nan=float(np.nanmean(out)))


def run_one(
    df: pd.DataFrame,
    feature_name: str,
    feature_df: pd.DataFrame,
    model_name: str,
    model: object,
    target_acc: float,
    selection_mode: str,
    min_accept: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    y_label = df["label_idx"].astype(int).to_numpy()
    pred_upper = df["pred_upper"].astype(int).to_numpy()
    correct = (y_label == pred_upper).astype(int)
    folds = df["fold_id"].astype(int).to_numpy()
    x = feature_df.to_numpy(dtype=float)

    score = np.full(len(df), np.nan, dtype=float)
    accept = np.zeros(len(df), dtype=bool)
    fold_rows: list[dict[str, float | int | str]] = []

    for outer_fold in sorted(np.unique(folds)):
        train = folds != outer_fold
        test = folds == outer_fold
        inner_score = inner_cv_scores(model, x[train], correct[train], folds[train])
        selected = select_threshold(inner_score, correct[train], target_acc, min_accept, selection_mode)
        threshold = float(selected["threshold"])
        if len(np.unique(correct[train])) < 2:
            test_score = np.full(test.sum(), float(np.mean(correct[train])))
        else:
            test_score = fit_predict_prob(model, x[train], correct[train], x[test])
        score[test] = test_score
        accept[test] = test_score >= threshold
        fold_rows.append(
            {
                "feature_set": feature_name,
                "model": model_name,
                "target_accept_acc": float(target_acc),
                "selection_mode": selection_mode,
                "fold_id": int(outer_fold),
                **selected,
                "test_accept_n": int(accept[test].sum()),
                "test_n": int(test.sum()),
            }
        )

    reviewed = ~accept
    hard_core = df["hard_core"].astype(int).to_numpy().astype(bool)
    upper_fn = df["upper_fn"].astype(int).to_numpy().astype(bool)
    wrong = pred_upper != y_label
    accept_metrics = metric_row(y_label[accept], pred_upper[accept])
    review_metrics = metric_row(y_label[reviewed], pred_upper[reviewed])
    final_if_review_corrected = pred_upper.copy()
    final_if_review_corrected[reviewed] = y_label[reviewed]
    final_metrics = metric_row(y_label, final_if_review_corrected)
    try:
        auc = float(roc_auc_score(correct, score))
    except ValueError:
        auc = np.nan
    row = {
        "feature_set": feature_name,
        "model": model_name,
        "target_accept_acc": float(target_acc),
        "selection_mode": selection_mode,
        "correct_auc": auc,
        "accept_n": int(accept.sum()),
        "accept_frac": float(accept.mean()),
        "review_n": int(reviewed.sum()),
        "review_frac": float(reviewed.mean()),
        "review_error_precision": float((reviewed & wrong).sum() / reviewed.sum()) if reviewed.sum() else np.nan,
        "error_recall_in_review": float((reviewed & wrong).sum() / wrong.sum()) if wrong.sum() else np.nan,
        "fn_recall_in_review": float((reviewed & upper_fn).sum() / upper_fn.sum()) if upper_fn.sum() else np.nan,
        "hardcore_recall_in_review": float((reviewed & hard_core).sum() / hard_core.sum()) if hard_core.sum() else np.nan,
    }
    row.update({f"accept_{k}": v for k, v in accept_metrics.items()})
    row.update({f"review_base_{k}": v for k, v in review_metrics.items()})
    row.update({f"final_if_review_corrected_{k}": v for k, v in final_metrics.items()})

    case_df = df[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "pred_upper",
            "p_upper",
            "difficulty_fine",
            "hard_core",
            "upper_wrong",
            "upper_fn",
            "upper_fp",
        ]
    ].copy()
    case_df["feature_set"] = feature_name
    case_df["model"] = model_name
    case_df["target_accept_acc"] = float(target_acc)
    case_df["selection_mode"] = selection_mode
    case_df["correct_prob"] = score
    case_df["stage1_accept"] = accept.astype(int)
    return row, pd.DataFrame(fold_rows), case_df


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
    selection_modes = [x.strip() for x in args.selection_modes.split(",") if x.strip()]
    feature_sets = build_feature_sets(df)
    models = make_models(args.seed)
    feature_keep = {x.strip() for x in args.feature_sets.split(",") if x.strip()}
    model_keep = {x.strip() for x in args.models.split(",") if x.strip()}
    feature_sets = {k: v for k, v in feature_sets.items() if k in feature_keep}
    models = {k: v for k, v in models.items() if k in model_keep}
    if not feature_sets:
        raise ValueError("No feature sets selected.")
    if not models:
        raise ValueError("No models selected.")

    summary_rows = []
    fold_rows = []
    case_rows = []
    for feature_name, feature_df in feature_sets.items():
        for model_name, model in models.items():
            for target in targets:
                for selection_mode in selection_modes:
                    row, fold_df, case_df = run_one(
                        df,
                        feature_name,
                        feature_df,
                        model_name,
                        model,
                        target,
                        selection_mode,
                        args.min_train_accept,
                    )
                    summary_rows.append(row)
                    fold_rows.append(fold_df)
                    case_rows.append(case_df)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["target_accept_acc", "selection_mode", "accept_accuracy", "accept_frac"],
        ascending=[True, True, False, False],
    )
    fold_summary = pd.concat(fold_rows, ignore_index=True)
    case_decisions = pd.concat(case_rows, ignore_index=True)

    summary.to_csv(out_dir / "stage1_learned_gate_summary.csv", index=False)
    fold_summary.to_csv(out_dir / "stage1_learned_gate_fold_thresholds.csv", index=False)
    case_decisions.to_csv(out_dir / "stage1_learned_gate_case_decisions.csv", index=False)
    baseline = metric_row(df["label_idx"].astype(int).to_numpy(), df["pred_upper"].astype(int).to_numpy())
    pd.DataFrame([baseline]).to_csv(out_dir / "baseline_upper_metrics.csv", index=False)

    print("Baseline upper:", baseline)
    for target in targets:
        print(f"\nTarget direct-pass accuracy >= {target:.2f}")
        show = summary[summary["target_accept_acc"] == target].copy()
        show = show.sort_values(["accept_accuracy", "accept_frac"], ascending=[False, False])
        cols = [
            "feature_set",
            "model",
            "selection_mode",
            "correct_auc",
            "accept_n",
            "accept_frac",
            "review_frac",
            "accept_accuracy",
            "accept_balanced_accuracy",
            "accept_sensitivity_high",
            "accept_specificity_low",
            "review_error_precision",
            "error_recall_in_review",
            "fn_recall_in_review",
            "hardcore_recall_in_review",
        ]
        print(show[cols].head(20).to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
