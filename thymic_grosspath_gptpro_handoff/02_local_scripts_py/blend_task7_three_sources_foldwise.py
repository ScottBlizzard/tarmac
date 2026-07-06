from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fold-wise validation-selected blend for three Task7 probability sources."
    )
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--run-c", required=True)
    parser.add_argument("--run-a-name", default="stage2")
    parser.add_argument("--run-b-name", default="stage3")
    parser.add_argument("--run-c-name", default="task6_folded")
    parser.add_argument("--curriculum-table", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument(
        "--objective",
        default="all_bacc",
        choices=(
            "all_acc",
            "all_bacc",
            "all_f1",
            "noncore_acc",
            "noncore_bacc",
            "noncore_f1",
            "easy_medium_acc",
            "easy_medium_bacc",
            "salvage_acc",
            "salvage_bacc",
        ),
    )
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def load_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"case_id": str})
    required = {"case_id", "label_idx", "prob_high_risk_group"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    keep = ["case_id", "label_idx", "prob_high_risk_group"]
    if "n_images" in df.columns:
        keep.append("n_images")
    return df[keep].copy()


def merge_three(
    run_a: Path,
    run_b: Path,
    run_c: Path,
    fold_id: int,
    split: str,
    name_a: str,
    name_b: str,
    name_c: str,
) -> pd.DataFrame:
    a = load_prediction(run_a / f"fold_{fold_id}" / f"{split}_case_predictions_mean.csv")
    b = load_prediction(run_b / f"fold_{fold_id}" / f"{split}_case_predictions_mean.csv")
    c = load_prediction(run_c / f"fold_{fold_id}" / f"{split}_case_predictions_mean.csv")
    a = a.rename(columns={"prob_high_risk_group": f"prob_{name_a}"})
    b = b.rename(columns={"prob_high_risk_group": f"prob_{name_b}", "label_idx": "label_idx_b"})
    c = c.rename(columns={"prob_high_risk_group": f"prob_{name_c}", "label_idx": "label_idx_c"})
    merged = a.merge(b[["case_id", "label_idx_b", f"prob_{name_b}"]], on="case_id", how="inner")
    merged = merged.merge(c[["case_id", "label_idx_c", f"prob_{name_c}"]], on="case_id", how="inner")
    if not (merged["label_idx"].astype(int) == merged["label_idx_b"].astype(int)).all():
        raise ValueError(f"Label mismatch between run-a and run-b in fold {fold_id} {split}.")
    if not (merged["label_idx"].astype(int) == merged["label_idx_c"].astype(int)).all():
        raise ValueError(f"Label mismatch between run-a and run-c in fold {fold_id} {split}.")
    merged = merged.drop(columns=["label_idx_b", "label_idx_c"])
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


def metric_row(name: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    y_true = df["label_idx"].to_numpy(dtype=int)
    prob = df["prob_high_risk_group"].to_numpy(dtype=float)
    pred = df["pred_idx"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
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


def score_frame(df: pd.DataFrame, prob: np.ndarray, threshold: float, objective: str) -> float:
    if objective.startswith("all_"):
        sub = df
        metric = objective.removeprefix("all_")
    elif objective.startswith("noncore_"):
        sub = df[df["difficulty_fine"] != "hard_core"].copy()
        metric = objective.removeprefix("noncore_")
    elif objective.startswith("easy_medium_"):
        sub = df[df["difficulty"].isin(["easy", "medium"])].copy()
        metric = objective.removeprefix("easy_medium_")
    elif objective.startswith("salvage_"):
        sub = df[df["difficulty_fine"] == "hard_salvage_teacher"].copy()
        metric = objective.removeprefix("salvage_")
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
        return float(f1_score(y_true, pred, zero_division=0))
    raise ValueError(metric)


def best_params(
    val_df: pd.DataFrame,
    name_a: str,
    name_b: str,
    name_c: str,
    objective: str,
    weight_step: float,
    threshold_step: float,
) -> tuple[float, float, float, float]:
    probs_a = val_df[f"prob_{name_a}"].to_numpy(dtype=float)
    probs_b = val_df[f"prob_{name_b}"].to_numpy(dtype=float)
    probs_c = val_df[f"prob_{name_c}"].to_numpy(dtype=float)
    thresholds = np.arange(0.0, 1.0 + threshold_step / 2.0, threshold_step)
    weights = np.arange(0.0, 1.0 + weight_step / 2.0, weight_step)
    best_key: tuple[float, float, float, float] | None = None
    best = (1.0, 0.0, 0.0, 0.50)
    y_true = val_df["label_idx"].to_numpy(dtype=int)
    for wa in weights:
        for wb in weights:
            wc = 1.0 - float(wa) - float(wb)
            if wc < -1e-9:
                continue
            wc = max(0.0, wc)
            prob = float(wa) * probs_a + float(wb) * probs_b + float(wc) * probs_c
            auc = safe_auc(y_true, prob)
            for threshold in thresholds:
                score = score_frame(val_df, prob, float(threshold), objective)
                key = (score, -abs(float(threshold) - 0.5), auc, -abs(float(wc)))
                if best_key is None or key > best_key:
                    best_key = key
                    best = (float(wa), float(wb), float(wc), float(threshold))
    assert best_key is not None
    return best[0], best[1], best[2], best[3]


def main() -> None:
    args = parse_args()
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    run_c = Path(args.run_c)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]

    curriculum = None
    if args.curriculum_table:
        curriculum = pd.read_csv(args.curriculum_table, dtype={"case_id": str})
        curriculum = curriculum[[col for col in ["case_id", "difficulty", "difficulty_fine"] if col in curriculum.columns]]

    choices: list[dict[str, float | int]] = []
    test_outputs: list[pd.DataFrame] = []
    for fold_id in folds:
        val_df = merge_three(run_a, run_b, run_c, fold_id, "val", args.run_a_name, args.run_b_name, args.run_c_name)
        test_df = merge_three(run_a, run_b, run_c, fold_id, "test", args.run_a_name, args.run_b_name, args.run_c_name)
        if curriculum is not None:
            val_df = val_df.merge(curriculum, on="case_id", how="left")
            test_df = test_df.merge(curriculum, on="case_id", how="left")
        for df in (val_df, test_df):
            df["difficulty"] = df.get("difficulty", pd.Series(["unknown"] * len(df))).fillna("unknown")
            df["difficulty_fine"] = df.get("difficulty_fine", pd.Series(["unknown"] * len(df))).fillna("unknown")

        wa, wb, wc, threshold = best_params(
            val_df=val_df,
            name_a=args.run_a_name,
            name_b=args.run_b_name,
            name_c=args.run_c_name,
            objective=args.objective,
            weight_step=args.weight_step,
            threshold_step=args.threshold_step,
        )
        prob = (
            wa * test_df[f"prob_{args.run_a_name}"].to_numpy(dtype=float)
            + wb * test_df[f"prob_{args.run_b_name}"].to_numpy(dtype=float)
            + wc * test_df[f"prob_{args.run_c_name}"].to_numpy(dtype=float)
        )
        out = test_df.copy()
        out["weight_a"] = wa
        out["weight_b"] = wb
        out["weight_c"] = wc
        out["threshold"] = threshold
        out["prob_high_risk_group"] = prob
        out["prob_low_risk_group"] = 1.0 - prob
        out["pred_idx"] = (prob >= threshold).astype(int)
        test_outputs.append(out)
        fold_metrics = metric_row(f"fold_{fold_id}", out)
        choices.append(
            {
                "fold_id": fold_id,
                "weight_a": wa,
                "weight_b": wb,
                "weight_c": wc,
                "threshold": threshold,
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
        f"prob_{args.run_c_name}",
        "weight_a",
        "weight_b",
        "weight_c",
        "threshold",
        "difficulty",
        "difficulty_fine",
    ]
    save_cols = [col for col in save_cols if col in oof.columns]
    oof[save_cols].to_csv(output_dir / "oof_case_predictions_mean.csv", index=False)

    metrics = [metric_row("overall", oof)]
    for group, sub in oof.groupby("difficulty", sort=True):
        metrics.append(metric_row(f"difficulty={group}", sub))
    for group, sub in oof.groupby("difficulty_fine", sort=True):
        metrics.append(metric_row(f"difficulty_fine={group}", sub))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "oof_metrics_by_group.csv", index=False)
    pd.DataFrame(choices).to_csv(output_dir / "fold_blend_choices.csv", index=False)

    overall = metrics[0] | {
        "objective": args.objective,
        "run_a": str(run_a),
        "run_b": str(run_b),
        "run_c": str(run_c),
        "run_a_name": args.run_a_name,
        "run_b_name": args.run_b_name,
        "run_c_name": args.run_c_name,
        "weight_step": args.weight_step,
        "threshold_step": args.threshold_step,
    }
    (output_dir / "overall_metrics.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
