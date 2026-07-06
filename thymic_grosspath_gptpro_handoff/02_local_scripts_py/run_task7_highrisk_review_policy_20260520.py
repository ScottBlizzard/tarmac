from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task7 human-in-the-loop review policy focused on overall errors and high-risk misses."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--curriculum-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--sources-json",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/11_review_router_large_ensemble_20260520/sources_used.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520",
    )
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def metric_from_pred(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_prob_source(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str})
    missing = {"case_id", "prob_high_risk_group", "pred_idx"}.difference(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    out = df[["case_id", "prob_high_risk_group", "pred_idx"]].copy()
    out = out.rename(columns={"prob_high_risk_group": f"p_{name}", "pred_idx": f"pred_{name}"})
    return out


def load_frame(project_root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    curriculum = pd.read_csv(project_root / args.curriculum_csv, dtype={"case_id": str})
    frame = curriculum[["case_id", "fold_id", "label_idx", "difficulty", "difficulty_fine"]].copy()
    frame["fold_id"] = frame["fold_id"].astype(int)
    frame["label_idx"] = frame["label_idx"].astype(int)
    frame["hard_core"] = (frame["difficulty_fine"] == "hard_core").astype(int)

    sources_path = project_root / args.sources_json
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    source_names: list[str] = []
    for name, rel_path in sources.items():
        path = project_root / rel_path
        if not path.exists():
            continue
        frame = frame.merge(load_prob_source(path, name), on="case_id", how="left")
        source_names.append(name)
    if "upper" not in source_names:
        raise KeyError("The source list must include the current best model named 'upper'.")

    registry_path = project_root / args.registry_csv
    if registry_path.exists():
        registry = pd.read_csv(registry_path, dtype={"case_id": str, "original_case_id": str})
        keep = [col for col in ["case_id", "original_case_id", "image_count", "selection_rule"] if col in registry.columns]
        frame = frame.merge(registry[keep], on="case_id", how="left")
    else:
        frame["original_case_id"] = frame["case_id"]
        frame["image_count"] = 1
        frame["selection_rule"] = ""

    frame["p_upper"] = frame["p_upper"].astype(float)
    frame["pred_upper"] = frame["pred_upper"].astype(int)
    frame["upper_correct"] = (frame["pred_upper"] == frame["label_idx"]).astype(int)
    frame["upper_wrong"] = 1 - frame["upper_correct"]
    frame["upper_fn"] = ((frame["label_idx"] == 1) & (frame["pred_upper"] == 0)).astype(int)
    frame["upper_fp"] = ((frame["label_idx"] == 0) & (frame["pred_upper"] == 1)).astype(int)
    frame["upper_conf"] = np.where(frame["pred_upper"] == 1, frame["p_upper"], 1.0 - frame["p_upper"])
    return frame, source_names


def build_features(frame: pd.DataFrame, source_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat = pd.DataFrame(index=frame.index)
    prob_cols = [f"p_{name}" for name in source_names]
    pred_cols = [f"pred_{name}" for name in source_names]
    probs = frame[prob_cols].astype(float).fillna(0.5).to_numpy()
    preds = frame[pred_cols].astype(float).fillna(0.0).to_numpy()

    for idx, name in enumerate(source_names):
        p = probs[:, idx]
        feat[f"p_{name}"] = p
        feat[f"margin_{name}"] = np.abs(p - 0.5)
        feat[f"entropy_{name}"] = entropy_binary(p)
        feat[f"pred_{name}"] = preds[:, idx]

    for idx, name_a in enumerate(source_names):
        for jdx in range(idx + 1, len(source_names)):
            name_b = source_names[jdx]
            diff = probs[:, idx] - probs[:, jdx]
            feat[f"absdiff_{name_a}_{name_b}"] = np.abs(diff)

    prob_mean = probs.mean(axis=1)
    prob_median = np.median(probs, axis=1)
    prob_std = probs.std(axis=1)
    prob_range = probs.max(axis=1) - probs.min(axis=1)
    vote_frac = preds.mean(axis=1)
    vote_disagree = ((preds.sum(axis=1) > 0) & (preds.sum(axis=1) < preds.shape[1])).astype(float)
    p_upper = frame["p_upper"].astype(float).to_numpy()
    pred_upper = frame["pred_upper"].astype(int).to_numpy()

    simple_scores = pd.DataFrame(index=frame.index)
    simple_scores["upper_low_conf"] = 1.0 - frame["upper_conf"].astype(float)
    simple_scores["all_std"] = prob_std
    simple_scores["all_range"] = prob_range
    simple_scores["all_vote_disagree"] = vote_disagree
    simple_scores["mean_absdiff_to_upper"] = np.abs(probs - p_upper[:, None]).mean(axis=1)
    simple_scores["max_absdiff_to_upper"] = np.abs(probs - p_upper[:, None]).max(axis=1)
    simple_scores["upper_predlow_pupper"] = np.where(pred_upper == 0, p_upper, -1.0)
    simple_scores["predlow_ensemble_mean"] = np.where(pred_upper == 0, prob_mean, -1.0)
    simple_scores["predlow_ensemble_median"] = np.where(pred_upper == 0, prob_median, -1.0)
    simple_scores["predlow_range"] = np.where(pred_upper == 0, prob_range, -1.0)
    simple_scores["predlow_disagree_mean"] = np.where(pred_upper == 0, prob_mean + prob_range, -1.0)
    simple_scores["predlow_vote_frac"] = np.where(pred_upper == 0, vote_frac, -1.0)

    feat["prob_mean"] = prob_mean
    feat["prob_median"] = prob_median
    feat["prob_std"] = prob_std
    feat["prob_range"] = prob_range
    feat["vote_frac"] = vote_frac
    feat["vote_disagree"] = vote_disagree
    feat["image_count"] = pd.to_numeric(frame.get("image_count", 1), errors="coerce").fillna(1.0)
    if "selection_rule" in frame.columns:
        feat = pd.concat([feat, pd.get_dummies(frame["selection_rule"].fillna(""), prefix="selection", dtype=float)], axis=1)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat, simple_scores


def make_learned_scores(frame: pd.DataFrame, feat: pd.DataFrame, seed: int) -> pd.DataFrame:
    folds = frame["fold_id"].astype(int).to_numpy()
    targets = {
        "hard_core": frame["hard_core"].astype(int).to_numpy(),
        "upper_wrong": frame["upper_wrong"].astype(int).to_numpy(),
        "upper_fn": frame["upper_fn"].astype(int).to_numpy(),
    }
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000, C=0.25, random_state=seed),
        ),
        "rf": RandomForestClassifier(
            n_estimators=600,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra": ExtraTreesClassifier(
            n_estimators=700,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(max_depth=2, learning_rate=0.03, n_estimators=120, random_state=seed),
    }
    x = feat.to_numpy(dtype=float)
    out = pd.DataFrame(index=frame.index)
    for target_name, y in targets.items():
        for model_name, model in models.items():
            score = np.full(len(frame), np.nan, dtype=float)
            for fold in sorted(np.unique(folds)):
                train = folds != fold
                test = folds == fold
                if len(np.unique(y[train])) < 2:
                    continue
                clf = clone(model)
                clf.fit(x[train], y[train])
                score[test] = clf.predict_proba(x[test])[:, 1]
            out[f"learn_{target_name}_{model_name}"] = np.nan_to_num(score, nan=float(np.nanmean(score)))
    return out


def percentile_rank(values: pd.Series) -> np.ndarray:
    values = values.astype(float).fillna(-1e9)
    return values.rank(method="first", pct=True).to_numpy(dtype=float)


def add_hybrid_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Combine broad error-disagreement and high-risk-miss signals.

    The individual signals have different roles: all_range is better for overall
    error capture, while upper_predlow_pupper is better for high-risk false
    negatives. Percentile ranks keep the blend scale-free.
    """
    out = scores.copy()
    all_range = percentile_rank(out["all_range"])
    upper_fn = percentile_rank(out["upper_predlow_pupper"])
    all_std = percentile_rank(out["all_std"])
    max_absdiff = percentile_rank(out["max_absdiff_to_upper"])
    out["hybrid_rank_allrange_075_fn_025"] = 0.75 * all_range + 0.25 * upper_fn
    out["hybrid_rank_allrange_050_fn_050"] = 0.50 * all_range + 0.50 * upper_fn
    out["hybrid_rank_allrange_025_fn_075"] = 0.25 * all_range + 0.75 * upper_fn
    out["hybrid_max_allrange_fn"] = np.maximum(all_range, upper_fn)
    out["hybrid_mean_all_std_abs_fn"] = (all_range + all_std + max_absdiff + upper_fn) / 4.0
    out["hybrid_overall_then_fn"] = all_range + 0.25 * upper_fn
    out["hybrid_fn_then_overall"] = upper_fn + 0.25 * all_range
    return out


def evaluate_review_score(frame: pd.DataFrame, score_name: str, score: np.ndarray, fractions: np.ndarray) -> list[dict[str, float | int | str]]:
    y = frame["label_idx"].astype(int).to_numpy()
    base_pred = frame["pred_upper"].astype(int).to_numpy()
    base_wrong = base_pred != y
    upper_fn = frame["upper_fn"].astype(int).to_numpy().astype(bool)
    upper_fp = frame["upper_fp"].astype(int).to_numpy().astype(bool)
    hard_core = frame["hard_core"].astype(int).to_numpy().astype(bool)
    n = len(frame)
    order = np.argsort(-np.nan_to_num(score, nan=-1e9), kind="mergesort")
    rows: list[dict[str, float | int | str]] = []

    for frac in fractions:
        k = int(round(float(frac) * n))
        k = max(0, min(n, k))
        reviewed = np.zeros(n, dtype=bool)
        if k:
            reviewed[order[:k]] = True
        final_pred = base_pred.copy()
        final_pred[reviewed] = y[reviewed]
        final_metrics = metric_from_pred(y, final_pred)

        accepted = ~reviewed
        if accepted.any():
            accepted_metrics = metric_from_pred(y[accepted], base_pred[accepted])
            accepted_acc = accepted_metrics["accuracy"]
            accepted_bacc = accepted_metrics["balanced_accuracy"]
        else:
            accepted_acc = float("nan")
            accepted_bacc = float("nan")

        reviewed_errors = reviewed & base_wrong
        rows.append(
            {
                "score": score_name,
                "review_frac": float(frac),
                "review_n": int(k),
                "review_precision_error": float(reviewed_errors.sum() / k) if k else float("nan"),
                "error_recall": float(reviewed_errors.sum() / base_wrong.sum()) if base_wrong.sum() else float("nan"),
                "fn_recall": float((reviewed & upper_fn).sum() / upper_fn.sum()) if upper_fn.sum() else float("nan"),
                "fp_recall": float((reviewed & upper_fp).sum() / upper_fp.sum()) if upper_fp.sum() else float("nan"),
                "hardcore_recall": float((reviewed & hard_core).sum() / hard_core.sum()) if hard_core.sum() else float("nan"),
                "accepted_acc": float(accepted_acc),
                "accepted_bacc": float(accepted_bacc),
                "final_acc_if_review_corrected": final_metrics["accuracy"],
                "final_bacc_if_review_corrected": final_metrics["balanced_accuracy"],
                "final_specificity_low": final_metrics["specificity_low"],
                "final_sensitivity_high": final_metrics["sensitivity_high"],
                "final_fp": final_metrics["fp"],
                "final_fn": final_metrics["fn"],
            }
        )
    return rows


def summarize_curves(curves: pd.DataFrame, score_diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score, g in curves.groupby("score", sort=False):
        row: dict[str, float | int | str] = {"score": score}
        for frac in [0.1, 0.2, 0.3, 0.4]:
            nearest = g.iloc[(g["review_frac"] - frac).abs().argsort()[:1]].iloc[0]
            suffix = f"at_{int(frac * 100)}pct"
            row[f"final_acc_{suffix}"] = nearest["final_acc_if_review_corrected"]
            row[f"final_bacc_{suffix}"] = nearest["final_bacc_if_review_corrected"]
            row[f"high_sens_{suffix}"] = nearest["final_sensitivity_high"]
            row[f"fn_recall_{suffix}"] = nearest["fn_recall"]
            row[f"error_recall_{suffix}"] = nearest["error_recall"]
        for metric, threshold in [
            ("final_acc_if_review_corrected", 0.90),
            ("final_bacc_if_review_corrected", 0.90),
            ("final_sensitivity_high", 0.90),
            ("final_sensitivity_high", 0.95),
            ("accepted_acc", 0.90),
        ]:
            hit = g[g[metric] >= threshold]
            key = metric.replace("final_", "").replace("_if_review_corrected", "")
            if hit.empty:
                row[f"min_review_for_{key}_ge_{int(threshold * 100)}"] = np.nan
                row[f"review_n_for_{key}_ge_{int(threshold * 100)}"] = np.nan
            else:
                best = hit.sort_values(["review_frac", "review_n"]).iloc[0]
                row[f"min_review_for_{key}_ge_{int(threshold * 100)}"] = best["review_frac"]
                row[f"review_n_for_{key}_ge_{int(threshold * 100)}"] = best["review_n"]
        for sens_threshold in [0.90, 0.95]:
            hit = g[
                (g["final_acc_if_review_corrected"] >= 0.90)
                & (g["final_sensitivity_high"] >= sens_threshold)
            ]
            suffix = int(sens_threshold * 100)
            if hit.empty:
                row[f"min_review_for_acc90_and_sens{suffix}"] = np.nan
                row[f"review_n_for_acc90_and_sens{suffix}"] = np.nan
            else:
                best = hit.sort_values(["review_frac", "review_n"]).iloc[0]
                row[f"min_review_for_acc90_and_sens{suffix}"] = best["review_frac"]
                row[f"review_n_for_acc90_and_sens{suffix}"] = best["review_n"]
        diag = score_diag[score_diag["score"] == score]
        if not diag.empty:
            row["auc_upper_wrong"] = diag.iloc[0]["auc_upper_wrong"]
            row["auc_upper_fn"] = diag.iloc[0]["auc_upper_fn"]
            row["auc_hard_core"] = diag.iloc[0]["auc_hard_core"]
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["final_acc_at_30pct", "high_sens_at_30pct", "fn_recall_at_30pct"], ascending=[False, False, False]
    )


def threshold_sweep(frame: pd.DataFrame, source_names: list[str]) -> pd.DataFrame:
    y = frame["label_idx"].astype(int).to_numpy()
    rows = []
    prob_inputs = {"upper": frame["p_upper"].astype(float).to_numpy()}
    prob_cols = [f"p_{name}" for name in source_names]
    probs = frame[prob_cols].astype(float).fillna(0.5).to_numpy()
    prob_inputs["all_mean"] = probs.mean(axis=1)
    prob_inputs["all_median"] = np.median(probs, axis=1)
    for name, p in prob_inputs.items():
        for threshold in np.linspace(0.05, 0.95, 91):
            pred = (p >= threshold).astype(int)
            row = metric_from_pred(y, pred)
            row["score"] = name
            row["threshold"] = float(threshold)
            rows.append(row)
    return pd.DataFrame(rows)


def write_priority_lists(frame: pd.DataFrame, scores: pd.DataFrame, curves: pd.DataFrame, out_dir: Path) -> None:
    for objective, metric in [
        ("overall_final_acc", "final_acc_if_review_corrected"),
        ("highrisk_sensitivity", "final_sensitivity_high"),
        ("accepted_acc", "accepted_acc"),
    ]:
        at30 = curves[np.isclose(curves["review_frac"], 0.30)].sort_values(
            [metric, "fn_recall", "error_recall"], ascending=[False, False, False]
        )
        if at30.empty:
            continue
        score_name = str(at30.iloc[0]["score"])
        score = scores[score_name].to_numpy(dtype=float)
        order = np.argsort(-np.nan_to_num(score, nan=-1e9), kind="mergesort")
        priority = frame[
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
                "upper_fp",
            ]
        ].copy()
        priority["review_score"] = score
        rank = np.empty(len(priority), dtype=int)
        rank[order] = np.arange(1, len(priority) + 1)
        priority["review_rank"] = rank
        priority = priority.sort_values("review_rank")
        priority.to_csv(out_dir / f"case_review_priority_{objective}_{score_name}.csv", index=False)

        top30 = set(priority.head(int(round(0.30 * len(priority))))["case_id"])
        missed = priority[(priority["upper_fn"] == 1) & (~priority["case_id"].isin(top30))].copy()
        missed.to_csv(out_dir / f"highrisk_misses_not_reviewed_at30_{objective}_{score_name}.csv", index=False)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, source_names = load_frame(project_root, args)
    feat, simple_scores = build_features(frame, source_names)
    learned_scores = make_learned_scores(frame, feat, args.seed)
    scores = add_hybrid_scores(pd.concat([simple_scores, learned_scores], axis=1))

    y_wrong = frame["upper_wrong"].astype(int).to_numpy()
    y_fn = frame["upper_fn"].astype(int).to_numpy()
    y_hard = frame["hard_core"].astype(int).to_numpy()
    diag_rows = []
    for col in scores.columns:
        s = scores[col].to_numpy(dtype=float)
        diag_rows.append(
            {
                "score": col,
                "auc_upper_wrong": safe_auc(y_wrong, s),
                "ap_upper_wrong": safe_ap(y_wrong, s),
                "auc_upper_fn": safe_auc(y_fn, s),
                "ap_upper_fn": safe_ap(y_fn, s),
                "auc_hard_core": safe_auc(y_hard, s),
                "ap_hard_core": safe_ap(y_hard, s),
            }
        )
    score_diag = pd.DataFrame(diag_rows).sort_values(["auc_upper_fn", "auc_upper_wrong"], ascending=False)

    fractions = np.round(np.arange(0.0, 0.851, 0.01), 2)
    curve_rows: list[dict[str, float | int | str]] = []
    for col in scores.columns:
        curve_rows.extend(evaluate_review_score(frame, col, scores[col].to_numpy(dtype=float), fractions))
    curves = pd.DataFrame(curve_rows)
    summary = summarize_curves(curves, score_diag)
    threshold = threshold_sweep(frame, source_names)

    baseline = metric_from_pred(frame["label_idx"].to_numpy(dtype=int), frame["pred_upper"].to_numpy(dtype=int))
    baseline_df = pd.DataFrame([baseline])

    frame_out = frame.copy()
    for col in scores.columns:
        frame_out[f"review_score_{col}"] = scores[col].to_numpy(dtype=float)

    baseline_df.to_csv(out_dir / "baseline_upper_metrics.csv", index=False)
    score_diag.to_csv(out_dir / "review_score_diagnostics.csv", index=False)
    curves.to_csv(out_dir / "human_review_policy_curves.csv", index=False)
    summary.to_csv(out_dir / "human_review_policy_summary.csv", index=False)
    threshold.to_csv(out_dir / "probability_threshold_sweep.csv", index=False)
    frame_out.to_csv(out_dir / "case_review_scores_all.csv", index=False)
    write_priority_lists(frame, scores, curves, out_dir)

    print("Baseline upper:", baseline)
    print("Top review-policy summary:")
    print(summary.head(12).to_string(index=False))
    print("Top score diagnostics:")
    print(score_diag.head(12).to_string(index=False))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
