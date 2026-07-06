from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fold-wise validation-selected blend for two Task7 curriculum outputs. "
            "The blend weight and threshold are selected on each fold's validation split, "
            "then applied once to that fold's held-out test split."
        )
    )
    parser.add_argument("--run-a", required=True, help="First run directory, usually stage2 easy+medium.")
    parser.add_argument("--run-b", required=True, help="Second run directory, usually stage3 salvage-hard.")
    parser.add_argument("--run-a-name", default="stage2_easy_medium")
    parser.add_argument("--run-b-name", default="stage3_salvage_hard")
    parser.add_argument("--curriculum-table", default="", help="Optional curriculum_case_table.csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument(
        "--objective",
        default="noncore_acc",
        choices=(
            "all_acc",
            "all_bacc",
            "all_f1",
            "noncore_acc",
            "noncore_bacc",
            "noncore_f1",
            "salvage_acc",
            "salvage_bacc",
            "easy_medium_acc",
            "easy_medium_bacc",
        ),
        help="Validation objective used to choose alpha and threshold inside each fold.",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.01,
        help="Grid step for alpha. Final probability = (1-alpha)*run_a + alpha*run_b.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Grid step for decision threshold.",
    )
    return parser.parse_args()


def load_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = ["case_id", "label_idx", "prob_high_risk_group"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    keep_cols = ["case_id", "label_idx", "prob_high_risk_group"]
    if "n_images" in df.columns:
        keep_cols.append("n_images")
    return df[keep_cols].copy()


def merge_pair(run_a: Path, run_b: Path, fold_id: int, split: str, name_a: str, name_b: str) -> pd.DataFrame:
    a = load_prediction(run_a / f"fold_{fold_id}" / f"{split}_case_predictions_mean.csv")
    b = load_prediction(run_b / f"fold_{fold_id}" / f"{split}_case_predictions_mean.csv")
    a = a.rename(columns={"prob_high_risk_group": f"prob_{name_a}"})
    b = b.rename(columns={"prob_high_risk_group": f"prob_{name_b}"})
    merged = a.merge(b[["case_id", "label_idx", f"prob_{name_b}"]], on="case_id", how="inner", suffixes=("", "_b"))
    bad_label = merged["label_idx"] != merged["label_idx_b"]
    if bad_label.any():
        bad_cases = merged.loc[bad_label, "case_id"].head(5).tolist()
        raise ValueError(f"Label mismatch between runs in fold {fold_id} {split}: {bad_cases}")
    merged = merged.drop(columns=["label_idx_b"])
    merged["fold_id"] = fold_id
    return merged


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def safe_bacc(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float(accuracy_score(y_true, pred))
    return float(balanced_accuracy_score(y_true, pred))


def score_frame(df: pd.DataFrame, prob: np.ndarray, threshold: float, objective: str) -> float:
    if objective.startswith("all_"):
        sub = df
        metric = objective.removeprefix("all_")
    elif objective.startswith("noncore_"):
        sub = df[df["difficulty_fine"] != "hard_core"].copy()
        metric = objective.removeprefix("noncore_")
    elif objective.startswith("salvage_"):
        sub = df[df["difficulty_fine"] == "hard_salvage_teacher"].copy()
        metric = objective.removeprefix("salvage_")
    elif objective.startswith("easy_medium_"):
        sub = df[df["difficulty"].isin(["easy", "medium"])].copy()
        metric = objective.removeprefix("easy_medium_")
    else:
        raise ValueError(objective)

    if sub.empty:
        return -1.0
    sub_prob = prob[sub.index.to_numpy()]
    y_true = sub["label_idx"].to_numpy(dtype=int)
    pred = (sub_prob >= threshold).astype(int)
    if metric == "acc":
        return float(accuracy_score(y_true, pred))
    if metric == "bacc":
        return safe_bacc(y_true, pred)
    if metric == "f1":
        if len(np.unique(y_true)) < 2 and y_true[0] == 0:
            return 0.0
        return float(f1_score(y_true, pred, zero_division=0))
    raise ValueError(metric)


def metric_row(name: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    y_true = df["label_idx"].to_numpy(dtype=int)
    prob = df["prob_high_risk_group"].to_numpy(dtype=float)
    pred = df["pred_idx"].to_numpy(dtype=int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    return {
        "group": name,
        "n": int(len(df)),
        "n_low": int((y_true == 0).sum()),
        "n_high": int((y_true == 1).sum()),
        "auc": safe_auc(y_true, prob),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": safe_bacc(y_true, pred),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_params_on_validation(
    val_df: pd.DataFrame,
    name_a: str,
    name_b: str,
    objective: str,
    alpha_step: float,
    threshold_step: float,
) -> tuple[float, float, float]:
    alphas = np.arange(0.0, 1.0 + alpha_step / 2.0, alpha_step)
    thresholds = np.arange(0.0, 1.0 + threshold_step / 2.0, threshold_step)
    prob_a = val_df[f"prob_{name_a}"].to_numpy(dtype=float)
    prob_b = val_df[f"prob_{name_b}"].to_numpy(dtype=float)

    best_key: tuple[float, float, float, float] | None = None
    best_alpha = 0.0
    best_threshold = 0.5
    for alpha in alphas:
        prob = (1.0 - alpha) * prob_a + alpha * prob_b
        for threshold in thresholds:
            score = score_frame(val_df, prob, float(threshold), objective)
            # Prefer higher validation score; then keep threshold close to 0.5 and avoid extreme alpha.
            key = (score, -abs(float(threshold) - 0.5), -abs(float(alpha) - 0.5), safe_auc(val_df["label_idx"].to_numpy(dtype=int), prob))
            if best_key is None or key > best_key:
                best_key = key
                best_alpha = float(alpha)
                best_threshold = float(threshold)
    assert best_key is not None
    return best_alpha, best_threshold, float(best_key[0])


def main() -> None:
    args = parse_args()
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]

    curriculum = None
    if args.curriculum_table:
        curriculum = pd.read_csv(args.curriculum_table)
        required = ["case_id", "difficulty", "difficulty_fine"]
        missing = [col for col in required if col not in curriculum.columns]
        if missing:
            raise KeyError(f"{args.curriculum_table} missing columns: {missing}")
        curriculum = curriculum[required + [col for col in ["correct_count", "mean_true_prob", "min_true_prob"] if col in curriculum.columns]]

    choices = []
    test_outputs = []
    for fold_id in folds:
        val_df = merge_pair(run_a, run_b, fold_id, "val", args.run_a_name, args.run_b_name)
        test_df = merge_pair(run_a, run_b, fold_id, "test", args.run_a_name, args.run_b_name)
        if curriculum is not None:
            val_df = val_df.merge(curriculum, on="case_id", how="left")
            test_df = test_df.merge(curriculum, on="case_id", how="left")
        for df in (val_df, test_df):
            df["difficulty"] = df.get("difficulty", pd.Series(["unknown"] * len(df))).fillna("unknown")
            df["difficulty_fine"] = df.get("difficulty_fine", pd.Series(["unknown"] * len(df))).fillna("unknown")

        alpha, threshold, val_score = best_params_on_validation(
            val_df=val_df,
            name_a=args.run_a_name,
            name_b=args.run_b_name,
            objective=args.objective,
            alpha_step=args.alpha_step,
            threshold_step=args.threshold_step,
        )
        prob_a = test_df[f"prob_{args.run_a_name}"].to_numpy(dtype=float)
        prob_b = test_df[f"prob_{args.run_b_name}"].to_numpy(dtype=float)
        prob = (1.0 - alpha) * prob_a + alpha * prob_b
        pred = (prob >= threshold).astype(int)

        out = test_df.copy()
        out["alpha_stage3"] = alpha
        out["threshold"] = threshold
        out["prob_high_risk_group"] = prob
        out["prob_low_risk_group"] = 1.0 - prob
        out["pred_idx"] = pred
        test_outputs.append(out)

        fold_eval = out[["case_id", "label_idx", "prob_high_risk_group", "pred_idx"]].copy()
        fold_metrics = metric_row(f"fold_{fold_id}", fold_eval)
        choices.append(
            {
                "fold_id": fold_id,
                "alpha_stage3": alpha,
                "weight_run_a": 1.0 - alpha,
                "weight_run_b": alpha,
                "threshold": threshold,
                "val_objective_score": val_score,
                **{f"test_{k}": v for k, v in fold_metrics.items() if k != "group"},
            }
        )

    oof = pd.concat(test_outputs, ignore_index=True).drop_duplicates("case_id")
    oof = oof.sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    save_cols = [
        "case_id",
        "fold_id",
        "label_idx",
        "pred_idx",
        "n_images",
        "prob_low_risk_group",
        "prob_high_risk_group",
        f"prob_{args.run_a_name}",
        f"prob_{args.run_b_name}",
        "alpha_stage3",
        "threshold",
        "difficulty",
        "difficulty_fine",
    ]
    save_cols = [col for col in save_cols if col in oof.columns]
    oof[save_cols].to_csv(output_dir / "oof_case_predictions_mean.csv", index=False)

    metrics = [metric_row("overall", oof)]
    if "difficulty" in oof.columns:
        for group, sub in oof.groupby("difficulty", sort=True):
            metrics.append(metric_row(f"difficulty={group}", sub))
    if "difficulty_fine" in oof.columns:
        for group, sub in oof.groupby("difficulty_fine", sort=True):
            metrics.append(metric_row(f"difficulty_fine={group}", sub))
    if "difficulty" in oof.columns:
        non_core = oof[oof["difficulty_fine"] != "hard_core"].copy()
        if not non_core.empty:
            metrics.append(metric_row("non_hard_core", non_core))
        easy_medium = oof[oof["difficulty"].isin(["easy", "medium"])].copy()
        if not easy_medium.empty:
            metrics.append(metric_row("easy_plus_medium", easy_medium))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "oof_metrics_by_group.csv", index=False)
    pd.DataFrame(choices).to_csv(output_dir / "fold_blend_choices.csv", index=False)

    overall = metrics[0] | {
        "objective": args.objective,
        "run_a": str(run_a),
        "run_b": str(run_b),
        "run_a_name": args.run_a_name,
        "run_b_name": args.run_b_name,
        "alpha_step": args.alpha_step,
        "threshold_step": args.threshold_step,
    }
    (output_dir / "overall_metrics.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
