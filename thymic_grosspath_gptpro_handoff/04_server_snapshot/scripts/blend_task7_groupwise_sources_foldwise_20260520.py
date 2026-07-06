from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-wise fold validation blend for Task7 sources.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-col", default="difficulty_fine", choices=("difficulty", "difficulty_fine"))
    parser.add_argument("--objective", default="balanced_accuracy", choices=("accuracy", "balanced_accuracy", "f1"))
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--folds", default="1,2,3,4,5")
    return parser.parse_args()


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def safe_bacc(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float(accuracy_score(y_true, pred))
    return float(balanced_accuracy_score(y_true, pred))


def metric_row(name: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    y = df["label_idx"].to_numpy(dtype=int)
    p = df["prob_high_risk_group"].to_numpy(dtype=float)
    pred = df["pred_idx"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "group": name,
        "n": int(len(df)),
        "n_low": int((y == 0).sum()),
        "n_high": int((y == 1).sum()),
        "auc": safe_auc(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": safe_bacc(y, pred),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def score(y: np.ndarray, pred: np.ndarray, objective: str) -> float:
    if objective == "accuracy":
        return float(accuracy_score(y, pred))
    if objective == "balanced_accuracy":
        return safe_bacc(y, pred)
    if objective == "f1":
        return float(f1_score(y, pred, zero_division=0))
    raise ValueError(objective)


def load_source(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str})
    required = {"case_id", "fold_id", "label_idx", "prob_high_risk_group"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    return df[["case_id", "fold_id", "label_idx", "prob_high_risk_group"]].rename(
        columns={"prob_high_risk_group": f"prob_{name}"}
    )


def build_frame(project_root: Path) -> tuple[pd.DataFrame, list[str]]:
    root = project_root / "outputs" / "batch1_batch2_task567_20260514"
    sources = {
        "stage2": root / "task7_curriculum_runs/07_case_mlp_schemeB_m060_stage2only_full5fold/oof_case_predictions_mean.csv",
        "stage3": root / "task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/oof_case_predictions_mean.csv",
        "main": root / "task7_curriculum_runs/12_stage2_salvage_foldwise_blend_noncore/oof_case_predictions_mean.csv",
        "upper": root / "task7_curriculum_runs/36_stage3_balcore_foldwise_blend_noncore/oof_case_predictions_mean.csv",
        "task6": root / "task6_curriculum_runs/04_task6_curriculum_vits_vitb_stage3_salvage/task7_folded_from_task6/oof_case_predictions_mean.csv",
    }
    frame = None
    for name, path in sources.items():
        src = load_source(path, name)
        if frame is None:
            frame = src
        else:
            frame = frame.merge(src.drop(columns=["label_idx", "fold_id"]), on="case_id", how="inner")
    assert frame is not None
    curriculum = pd.read_csv(
        root / "task6_curriculum_runs/04_task6_curriculum_vits_vitb_stage3_salvage/curriculum_case_table.csv",
        dtype={"case_id": str},
    )
    frame = frame.merge(curriculum[["case_id", "difficulty", "difficulty_fine"]], on="case_id", how="left")
    frame["difficulty"] = frame["difficulty"].fillna("unknown")
    frame["difficulty_fine"] = frame["difficulty_fine"].fillna("unknown")
    return frame, list(sources)


def weight_grid(n: int, step: float) -> list[np.ndarray]:
    vals = np.arange(0.0, 1.0 + step / 2, step)
    out = []
    if n == 1:
        return [np.array([1.0])]
    if n == 2:
        for w0 in vals:
            out.append(np.array([float(w0), float(1.0 - w0)]))
        return out
    if n == 3:
        for w0 in vals:
            for w1 in vals:
                w2 = 1.0 - float(w0) - float(w1)
                if w2 >= -1e-9:
                    out.append(np.array([float(w0), float(w1), max(0.0, w2)]))
        return out
    raise ValueError("This helper expects up to 3 sources.")


def best_params(val: pd.DataFrame, source_names: list[str], objective: str, weight_step: float, threshold_step: float) -> tuple[list[str], np.ndarray, float, float]:
    y = val["label_idx"].to_numpy(dtype=int)
    thresholds = np.arange(0.05, 0.95 + threshold_step / 2, threshold_step)
    candidate_source_sets = []
    for n in [1, 2, 3]:
        for combo in __import__("itertools").combinations(source_names, n):
            candidate_source_sets.append(list(combo))
    best = None
    best_sources = [source_names[0]]
    best_weights = np.array([1.0])
    best_threshold = 0.5
    for sources in candidate_source_sets:
        prob_mat = val[[f"prob_{name}" for name in sources]].to_numpy(dtype=float)
        for weights in weight_grid(len(sources), weight_step):
            prob = prob_mat @ weights
            auc = safe_auc(y, prob)
            for threshold in thresholds:
                pred = (prob >= threshold).astype(int)
                s = score(y, pred, objective)
                key = (s, -abs(float(threshold) - 0.5), auc, -len(sources))
                if best is None or key > best:
                    best = key
                    best_sources = sources
                    best_weights = weights
                    best_threshold = float(threshold)
    assert best is not None
    return best_sources, best_weights, best_threshold, float(best[0])


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    frame, source_names = build_frame(project_root)
    outputs = []
    choices = []
    for fold in folds:
        val_all = frame[frame["fold_id"] != fold].copy()
        test_all = frame[frame["fold_id"] == fold].copy()
        for group, test_sub in test_all.groupby(args.group_col, sort=True):
            val_sub = val_all[val_all[args.group_col] == group].copy()
            if len(val_sub) < 8 or len(test_sub) == 0:
                val_sub = val_all
                group_key = "__fallback_all__"
            else:
                group_key = str(group)
            sources, weights, threshold, val_score = best_params(
                val_sub,
                source_names,
                args.objective,
                args.weight_step,
                args.threshold_step,
            )
            prob_mat = test_sub[[f"prob_{name}" for name in sources]].to_numpy(dtype=float)
            prob = prob_mat @ weights
            out = test_sub[["case_id", "fold_id", "label_idx", "difficulty", "difficulty_fine"]].copy()
            out["prob_high_risk_group"] = prob
            out["prob_low_risk_group"] = 1.0 - prob
            out["pred_idx"] = (prob >= threshold).astype(int)
            outputs.append(out)
            choices.append(
                {
                    "fold_id": fold,
                    "group_col": args.group_col,
                    "group_value": group,
                    "selected_on": group_key,
                    "sources": "+".join(sources),
                    "weights": "+".join(f"{w:.3f}" for w in weights),
                    "threshold": threshold,
                    "val_score": val_score,
                    "test_n": int(len(test_sub)),
                }
            )
    oof = pd.concat(outputs, ignore_index=True).sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    oof.to_csv(output_dir / "oof_case_predictions_mean.csv", index=False)
    pd.DataFrame(choices).to_csv(output_dir / "fold_group_choices.csv", index=False)
    metrics = [metric_row("overall", oof)]
    for group, sub in oof.groupby("difficulty_fine", sort=True):
        metrics.append(metric_row(f"difficulty_fine={group}", sub))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "oof_metrics_by_group.csv", index=False)
    overall = metrics[0] | {
        "group_col": args.group_col,
        "objective": args.objective,
        "weight_step": args.weight_step,
        "threshold_step": args.threshold_step,
    }
    (output_dir / "overall_metrics.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
