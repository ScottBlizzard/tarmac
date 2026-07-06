from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested fold OOF stacking probe for Task7 probability sources.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objective", default="balanced_accuracy", choices=("accuracy", "balanced_accuracy", "f1"))
    parser.add_argument("--folds", default="1,2,3,4,5")
    return parser.parse_args()


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def score(y_true: np.ndarray, pred: np.ndarray, objective: str) -> float:
    if objective == "accuracy":
        return float(accuracy_score(y_true, pred))
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, pred))
    if objective == "f1":
        return float(f1_score(y_true, pred, zero_division=0))
    raise ValueError(objective)


def best_threshold(y_true: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_s = -1.0
    for t in thresholds:
        pred = (prob >= t).astype(int)
        s = score(y_true, pred, objective)
        key = (s, -abs(float(t) - 0.5))
        if key > (best_s, -abs(best_t - 0.5)):
            best_s = s
            best_t = float(t)
    return best_t, best_s


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
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_source(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str})
    required = {"case_id", "fold_id", "label_idx", "prob_high_risk_group"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    return df[["case_id", "fold_id", "label_idx", "prob_high_risk_group"]].rename(
        columns={"prob_high_risk_group": f"prob_{name}"}
    )


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def build_frame(project_root: Path) -> pd.DataFrame:
    root = project_root / "outputs" / "batch1_batch2_task567_20260514"
    sources = {
        "stage2": root / "task7_curriculum_runs/07_case_mlp_schemeB_m060_stage2only_full5fold/oof_case_predictions_mean.csv",
        "stage3": root / "task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/oof_case_predictions_mean.csv",
        "main_blend": root / "task7_curriculum_runs/12_stage2_salvage_foldwise_blend_noncore/oof_case_predictions_mean.csv",
        "upper_blend": root / "task7_curriculum_runs/36_stage3_balcore_foldwise_blend_noncore/oof_case_predictions_mean.csv",
        "task6_folded": root / "task6_curriculum_runs/04_task6_curriculum_vits_vitb_stage3_salvage/task7_folded_from_task6/oof_case_predictions_mean.csv",
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
    keep_cols = ["case_id", "difficulty", "difficulty_fine", "correct_count", "mean_true_prob", "min_true_prob", "mean_margin"]
    frame = frame.merge(curriculum[[col for col in keep_cols if col in curriculum.columns]], on="case_id", how="left")
    frame["difficulty"] = frame["difficulty"].fillna("unknown")
    frame["difficulty_fine"] = frame["difficulty_fine"].fillna("unknown")
    return frame


def feature_columns(frame: pd.DataFrame, source_names: list[str], use_curriculum: bool) -> tuple[pd.DataFrame, list[str]]:
    feat = pd.DataFrame(index=frame.index)
    for name in source_names:
        p = frame[f"prob_{name}"].to_numpy(dtype=float)
        feat[f"p_{name}"] = p
        feat[f"logit_{name}"] = logit(p)
        feat[f"margin_{name}"] = np.abs(p - 0.5)
    for a, b in combinations(source_names, 2):
        feat[f"diff_{a}_{b}"] = frame[f"prob_{a}"].to_numpy(dtype=float) - frame[f"prob_{b}"].to_numpy(dtype=float)
    if use_curriculum:
        for col in ["correct_count", "mean_true_prob", "min_true_prob", "mean_margin"]:
            if col in frame.columns:
                feat[col] = frame[col].fillna(0).to_numpy(dtype=float)
        onehot = pd.get_dummies(frame[["difficulty", "difficulty_fine"]], columns=["difficulty", "difficulty_fine"], dtype=float)
        feat = pd.concat([feat, onehot], axis=1)
    return feat, feat.columns.tolist()


def inner_select(
    frame: pd.DataFrame,
    x_all: pd.DataFrame,
    train_mask: np.ndarray,
    folds: list[int],
    objective: str,
) -> tuple[float, float]:
    y = frame["label_idx"].to_numpy(dtype=int)
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_c = 1.0
    best_t = 0.5
    best_s = -1.0
    train_folds = [f for f in folds if np.any(train_mask & (frame["fold_id"].to_numpy(dtype=int) == f))]
    for c in c_grid:
        inner_probs = np.full(len(frame), np.nan, dtype=float)
        for val_fold in train_folds:
            inner_train = train_mask & (frame["fold_id"].to_numpy(dtype=int) != val_fold)
            inner_val = train_mask & (frame["fold_id"].to_numpy(dtype=int) == val_fold)
            if inner_val.sum() == 0 or inner_train.sum() == 0:
                continue
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_all.loc[inner_train])
            x_va = scaler.transform(x_all.loc[inner_val])
            clf = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=20260520)
            clf.fit(x_tr, y[inner_train])
            inner_probs[inner_val] = clf.predict_proba(x_va)[:, 1]
        valid = train_mask & ~np.isnan(inner_probs)
        if valid.sum() == 0:
            continue
        t, s = best_threshold(y[valid], inner_probs[valid], objective)
        key = (s, -abs(t - 0.5), -abs(np.log10(c)))
        if key > (best_s, -abs(best_t - 0.5), -abs(np.log10(best_c))):
            best_s = s
            best_c = c
            best_t = t
    return best_c, best_t


def run_config(frame: pd.DataFrame, source_names: list[str], use_curriculum: bool, folds: list[int], objective: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_all, columns = feature_columns(frame, source_names, use_curriculum)
    y = frame["label_idx"].to_numpy(dtype=int)
    fold_arr = frame["fold_id"].to_numpy(dtype=int)
    outputs = []
    choices = []
    for fold in folds:
        train_mask = fold_arr != fold
        test_mask = fold_arr == fold
        c, threshold = inner_select(frame, x_all, train_mask, folds, objective)
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_all.loc[train_mask])
        x_te = scaler.transform(x_all.loc[test_mask])
        clf = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=20260520 + fold)
        clf.fit(x_tr, y[train_mask])
        prob = clf.predict_proba(x_te)[:, 1]
        pred = (prob >= threshold).astype(int)
        out = frame.loc[test_mask, ["case_id", "fold_id", "label_idx", "difficulty", "difficulty_fine"]].copy()
        out["prob_high_risk_group"] = prob
        out["prob_low_risk_group"] = 1.0 - prob
        out["pred_idx"] = pred
        outputs.append(out)
        choices.append({"fold_id": fold, "c": c, "threshold": threshold, "n_features": len(columns)})
    oof = pd.concat(outputs, ignore_index=True).sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    return oof, pd.DataFrame(choices)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    frame = build_frame(project_root)

    configs = {
        "task7_sources": (["stage2", "stage3", "main_blend", "upper_blend"], False),
        "task7_sources_curriculum_meta": (["stage2", "stage3", "main_blend", "upper_blend"], True),
        "task7_plus_task6": (["stage2", "stage3", "main_blend", "upper_blend", "task6_folded"], False),
        "task7_plus_task6_curriculum_meta": (["stage2", "stage3", "main_blend", "upper_blend", "task6_folded"], True),
        "main_plus_task6_curriculum_meta": (["main_blend", "task6_folded"], True),
    }
    summary = []
    for name, (sources, use_curriculum) in configs.items():
        run_dir = output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        oof, choices = run_config(frame, sources, use_curriculum, folds, args.objective)
        oof.to_csv(run_dir / "oof_case_predictions_mean.csv", index=False)
        choices.to_csv(run_dir / "fold_stack_choices.csv", index=False)
        metrics = [metric_row("overall", oof)]
        for group, sub in oof.groupby("difficulty_fine", sort=True):
            metrics.append(metric_row(f"difficulty_fine={group}", sub))
        pd.DataFrame(metrics).to_csv(run_dir / "oof_metrics_by_group.csv", index=False)
        overall = metrics[0] | {"sources": sources, "use_curriculum_meta": use_curriculum, "objective": args.objective}
        (run_dir / "overall_metrics.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(overall | {"config": name})
    summary_df = pd.DataFrame(summary).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False)
    summary_df.to_csv(output_dir / "stacking_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
