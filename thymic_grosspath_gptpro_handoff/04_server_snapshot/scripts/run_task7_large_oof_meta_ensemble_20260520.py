from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXCLUDE_PATH_PARTS = (
    "03_hardcore_gross_calibrator",
    "04_gross_auto_router",
    "08_review_router",
    "09_review_router",
    "10_review_router",
    "11_review_router",
    "12_highrisk_review_policy",
    "13_stage1",
    "14_stage1",
    "15_stage2_auto",
    "16_stage2_selective",
    "17_hardcore_split",
    "18_selectivenet",
    "large_oof_meta_ensemble",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict outer-fold Task7 meta ensemble over legal OOF sources and gross cues.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--top-k-list", default="5,10,20,40,all")
    parser.add_argument("--models", default="logreg,extratrees,hgb")
    parser.add_argument("--objective", default="balanced_accuracy", choices=("accuracy", "balanced_accuracy", "f1"))
    parser.add_argument("--include-gross-cues", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-review-scores", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def safe_name(path: Path, root: Path, index: int) -> str:
    rel = str(path.relative_to(root)).replace("\\", "/")
    name = re.sub(r"[^0-9A-Za-z]+", "_", rel).strip("_").lower()
    return f"s{index:03d}_{name[:120]}"


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def score_predictions(y_true: np.ndarray, pred: np.ndarray, objective: str) -> float:
    if objective == "accuracy":
        return float(accuracy_score(y_true, pred))
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, pred))
    if objective == "f1":
        return float(f1_score(y_true, pred, zero_division=0))
    raise ValueError(objective)


def choose_threshold(y_true: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (prob >= threshold).astype(int)
        value = score_predictions(y_true, pred, objective)
        key = (value, -abs(float(threshold) - 0.5))
        if key > (best_score, -abs(best_threshold - 0.5)):
            best_score = value
            best_threshold = float(threshold)
    return best_threshold, best_score


def metric_row(name: str, df: pd.DataFrame) -> dict[str, Any]:
    y_true = df["label_idx"].to_numpy(dtype=int)
    prob = df["prob_high_risk_group"].to_numpy(dtype=float)
    pred = df["pred_idx"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "group": name,
        "n": int(len(df)),
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


def is_excluded(path: Path) -> bool:
    text = str(path).replace("\\", "/")
    return any(part in text for part in EXCLUDE_PATH_PARTS)


def load_oof_sources(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_frame: pd.DataFrame | None = None
    feature_frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    source_index = 0
    for path in sorted(root.rglob("oof_case_predictions_mean.csv")):
        if is_excluded(path):
            continue
        try:
            df = pd.read_csv(path, dtype={"case_id": str})
        except Exception:
            continue
        required = {"case_id", "fold_id", "label_idx", "prob_high_risk_group"}
        if not required.issubset(df.columns):
            continue
        if len(df) != 285 or df["case_id"].nunique() != 285:
            continue
        if df["prob_high_risk_group"].isna().any():
            continue
        df = df[["case_id", "fold_id", "label_idx", "prob_high_risk_group"]].copy()
        df["fold_id"] = df["fold_id"].astype(int)
        df["label_idx"] = df["label_idx"].astype(int)
        if base_frame is None:
            base_frame = df[["case_id", "fold_id", "label_idx"]].copy()
        else:
            check = base_frame.merge(df[["case_id", "fold_id", "label_idx"]], on="case_id", suffixes=("", "_new"))
            if not (check["fold_id"].eq(check["fold_id_new"]).all() and check["label_idx"].eq(check["label_idx_new"]).all()):
                continue
        source_index += 1
        name = safe_name(path, root, source_index)
        prob = df["prob_high_risk_group"].clip(1e-5, 1 - 1e-5).to_numpy(dtype=float)
        feature_frames.append(
            pd.DataFrame(
                {
                    "case_id": df["case_id"],
                    f"{name}__prob": prob,
                    f"{name}__logit": np.log(prob / (1 - prob)),
                    f"{name}__margin": np.abs(prob - 0.5),
                }
            )
        )
        pred = (prob >= 0.5).astype(int)
        inventory.append(
            {
                "source_name": name,
                "path": str(path),
                "accuracy": float(accuracy_score(df["label_idx"], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(df["label_idx"], pred)),
                "auc": safe_auc(df["label_idx"].to_numpy(dtype=int), prob),
            }
        )
    if base_frame is None:
        raise RuntimeError("No legal OOF sources found.")
    frame = base_frame.copy()
    for feat in feature_frames:
        frame = frame.merge(feat, on="case_id", how="inner")
    return frame, pd.DataFrame(inventory)


def add_gross_cues(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    gross_path = root / "frozen_inputs/gross_boundary_aux_20260520/task7_gross_boundary_aux_labels_allcases_20260520.csv"
    if not gross_path.exists():
        return frame
    gross = pd.read_csv(gross_path, dtype={"case_id": str})
    keep = [
        "case_id",
        "gross_score",
        "exp_manual_pale_uniform",
        "exp_manual_round_smooth",
        "exp_manual_microcystic",
        "exp_manual_multinodular",
        "exp_manual_hemonec",
        "exp_manual_irregularity",
        "exp_manual_view_limit",
    ]
    gross = gross[[col for col in keep if col in gross.columns]].drop_duplicates("case_id")
    if "gross_score" in gross.columns:
        gross["gross_score"] = pd.to_numeric(gross["gross_score"], errors="coerce")
    categorical_cols = [col for col in gross.columns if col not in {"case_id", "gross_score"}]
    gross = pd.get_dummies(gross, columns=categorical_cols, dummy_na=True, dtype=float)
    return frame.merge(gross, on="case_id", how="left")


def add_review_scores(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    review_path = root / "task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv"
    if not review_path.exists():
        return frame
    review = pd.read_csv(review_path, dtype={"case_id": str})
    numeric_cols = [
        col
        for col in review.columns
        if col.startswith("p_")
        or col.startswith("pred_")
        or col.startswith("review_score_")
        or col in {"upper_conf", "image_count"}
    ]
    blocked = {
        "pred_upper",
        "p_upper",
    }
    numeric_cols = [col for col in numeric_cols if col not in blocked]
    keep = ["case_id", *numeric_cols]
    review = review[keep].drop_duplicates("case_id")
    for col in numeric_cols:
        review[col] = pd.to_numeric(review[col], errors="coerce")
    rename = {col: f"review__{col}" for col in numeric_cols}
    review = review.rename(columns=rename)
    return frame.merge(review, on="case_id", how="left")


def add_difficulty_for_reporting(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    path = root / "task7_curriculum_runs/53_gross_boundary_aux_allcases_w0005_full5fold/curriculum_case_table.csv"
    if not path.exists():
        return frame
    diff = pd.read_csv(path, dtype={"case_id": str})
    cols = [col for col in ["case_id", "difficulty", "difficulty_fine"] if col in diff.columns]
    return frame.merge(diff[cols].drop_duplicates("case_id"), on="case_id", how="left")


def source_prob_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.endswith("__prob")]


def select_sources_by_train_bacc(frame: pd.DataFrame, train_mask: np.ndarray, top_k: str) -> list[str]:
    prob_cols = source_prob_columns(frame)
    scores: list[tuple[float, str]] = []
    y = frame.loc[train_mask, "label_idx"].to_numpy(dtype=int)
    for prob_col in prob_cols:
        pred = (frame.loc[train_mask, prob_col].to_numpy(dtype=float) >= 0.5).astype(int)
        scores.append((float(balanced_accuracy_score(y, pred)), prob_col.replace("__prob", "")))
    scores.sort(reverse=True)
    if top_k == "all":
        selected = [name for _, name in scores]
    else:
        selected = [name for _, name in scores[: int(top_k)]]
    return selected


def make_features(frame: pd.DataFrame, selected_sources: list[str], include_gross: bool) -> pd.DataFrame:
    cols: list[str] = []
    for name in selected_sources:
        cols.extend([f"{name}__prob", f"{name}__logit", f"{name}__margin"])
    if include_gross:
        cols.extend(
            [
                col
                for col in frame.columns
                if col.startswith("gross_score")
                or col.startswith("exp_manual_")
            ]
        )
    cols.extend([col for col in frame.columns if col.startswith("review__")])
    return frame[cols].copy()


def build_model(model_name: str, seed: int):
    if model_name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        )
    if model_name == "extratrees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=4,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if model_name == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, l2_regularization=0.1, random_state=seed),
        )
    raise ValueError(model_name)


def inner_threshold(
    frame: pd.DataFrame,
    x_all: pd.DataFrame,
    train_mask: np.ndarray,
    folds: list[int],
    model_name: str,
    objective: str,
) -> tuple[float, float]:
    y = frame["label_idx"].to_numpy(dtype=int)
    fold_arr = frame["fold_id"].to_numpy(dtype=int)
    inner_prob = np.full(len(frame), np.nan, dtype=float)
    for inner_fold in folds:
        inner_train = train_mask & (fold_arr != inner_fold)
        inner_val = train_mask & (fold_arr == inner_fold)
        if inner_train.sum() == 0 or inner_val.sum() == 0:
            continue
        model = build_model(model_name, seed=20260520 + inner_fold)
        model.fit(x_all.loc[inner_train], y[inner_train])
        inner_prob[inner_val] = model.predict_proba(x_all.loc[inner_val])[:, 1]
    valid = train_mask & np.isfinite(inner_prob)
    if valid.sum() == 0:
        return 0.5, float("nan")
    return choose_threshold(y[valid], inner_prob[valid], objective)


def run_config(
    frame: pd.DataFrame,
    folds: list[int],
    top_k: str,
    model_name: str,
    include_gross: bool,
    objective: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame["label_idx"].to_numpy(dtype=int)
    fold_arr = frame["fold_id"].to_numpy(dtype=int)
    outputs: list[pd.DataFrame] = []
    choices: list[dict[str, Any]] = []
    for fold in folds:
        train_mask = fold_arr != fold
        test_mask = fold_arr == fold
        selected = select_sources_by_train_bacc(frame, train_mask, top_k=top_k)
        x_all = make_features(frame, selected, include_gross=include_gross)
        threshold, inner_score = inner_threshold(frame, x_all, train_mask, folds, model_name, objective)
        model = build_model(model_name, seed=20260520 + fold)
        model.fit(x_all.loc[train_mask], y[train_mask])
        prob = model.predict_proba(x_all.loc[test_mask])[:, 1]
        pred = (prob >= threshold).astype(int)
        out_cols = ["case_id", "fold_id", "label_idx"]
        for col in ["difficulty", "difficulty_fine"]:
            if col in frame.columns:
                out_cols.append(col)
        out = frame.loc[test_mask, out_cols].copy()
        out["prob_high_risk_group"] = prob
        out["prob_low_risk_group"] = 1.0 - prob
        out["pred_idx"] = pred
        outputs.append(out)
        choices.append(
            {
                "fold_id": fold,
                "top_k": top_k,
                "model": model_name,
                "include_gross": include_gross,
                "threshold": threshold,
                "inner_score": inner_score,
                "n_sources": len(selected),
                "n_features": int(x_all.shape[1]),
                "selected_sources": ";".join(selected),
            }
        )
    oof = pd.concat(outputs, ignore_index=True).sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    return oof, pd.DataFrame(choices)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    root = project_root / "outputs/batch1_batch2_task567_20260514"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    top_k_list = [item.strip() for item in args.top_k_list.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]

    frame, inventory = load_oof_sources(root)
    if args.include_gross_cues:
        frame = add_gross_cues(frame, root)
    if args.include_review_scores:
        frame = add_review_scores(frame, root)
    frame = add_difficulty_for_reporting(frame, root)
    frame.to_csv(output_dir / "meta_feature_frame_preview.csv", index=False)
    inventory.sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False).to_csv(output_dir / "source_inventory.csv", index=False)

    summary: list[dict[str, Any]] = []
    for top_k in top_k_list:
        for model_name in models:
            config_name = f"top{top_k}_{model_name}_{'gross' if args.include_gross_cues else 'nogross'}"
            run_dir = output_dir / config_name
            run_dir.mkdir(parents=True, exist_ok=True)
            oof, choices = run_config(frame, folds, top_k, model_name, args.include_gross_cues, args.objective)
            oof.to_csv(run_dir / "oof_case_predictions_mean.csv", index=False)
            choices.to_csv(run_dir / "fold_meta_choices.csv", index=False)
            metrics = [metric_row("overall", oof)]
            if "difficulty_fine" in oof.columns:
                for group, sub in oof.groupby("difficulty_fine", sort=True):
                    metrics.append(metric_row(f"difficulty_fine={group}", sub))
            pd.DataFrame(metrics).to_csv(run_dir / "oof_metrics_by_group.csv", index=False)
            overall = metrics[0] | {"config": config_name, "top_k": top_k, "model": model_name, "include_gross": args.include_gross_cues}
            (run_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
            summary.append(overall)
    summary_df = pd.DataFrame(summary).sort_values(["balanced_accuracy", "accuracy", "auc"], ascending=False)
    summary_df.to_csv(output_dir / "large_oof_meta_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
