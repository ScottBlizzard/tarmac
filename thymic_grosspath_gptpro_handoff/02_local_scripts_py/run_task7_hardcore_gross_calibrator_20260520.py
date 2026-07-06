from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from run_task7_gross_feature_probe_20260520 import (
    TASK7_CLASS_NAMES,
    build_feature_frame,
    extract_gross_features,
    load_image_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 hard-core gross-finding calibrator.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--split-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_5fold_assignments.csv",
    )
    parser.add_argument(
        "--curriculum-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/03_hardcore_gross_calibrator",
    )
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
    best_t = 0.5
    best_s = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (prob >= threshold).astype(int)
        current = score(y_true, pred, objective)
        key = (current, -abs(float(threshold) - 0.5))
        if key > (best_s, -abs(best_t - 0.5)):
            best_s = current
            best_t = float(threshold)
    return best_t, best_s


def metric_row(group: str, df: pd.DataFrame) -> dict[str, object]:
    y = df["label_idx"].to_numpy(dtype=int)
    prob = df["prob_high_risk_group"].to_numpy(dtype=float)
    pred = df["pred_idx"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "group": group,
        "n": int(len(df)),
        "n_low": int((y == 0).sum()),
        "n_high": int((y == 1).sum()),
        "auc": safe_auc(y, prob),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def write_metrics(oof: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = [metric_row("overall", oof)]
    for col in ["difficulty", "difficulty_fine", "task_l6_label"]:
        if col not in oof.columns:
            continue
        for value, sub in oof.groupby(col, sort=True):
            rows.append(metric_row(f"{col}={value}", sub))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "oof_metrics_by_group.csv", index=False)
    return metrics


def json_ready(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def add_manual_gross_scores(gross_feat: pd.DataFrame) -> pd.DataFrame:
    feat = gross_feat.copy()
    z_size = np.log1p(feat.get("tumor_max_dim_mm", 0.0).astype(float))
    z_area = np.log1p(feat.get("tumor_max_area_mm2", 0.0).astype(float))
    if float(z_size.std()) > 1e-8:
        z_size = (z_size - z_size.mean()) / z_size.std()
    if float(z_area.std()) > 1e-8:
        z_area = (z_area - z_area.mean()) / z_area.std()
    risk = (
        1.4 * feat.get("kw_boundary_unclear", 0.0)
        + 1.1 * feat.get("kw_capsule_absent", 0.0)
        + 1.2 * feat.get("kw_capsule_involved", 0.0)
        + 1.0 * feat.get("kw_lung_attached", 0.0)
        + 0.8 * feat.get("kw_pericardium_attached", 0.0)
        + 0.5 * feat.get("kw_pleura_attached", 0.0)
        + 0.5 * feat.get("kw_necrosis", 0.0)
        + 0.3 * feat.get("kw_hemorrhage", 0.0)
        + 0.3 * z_size.fillna(0.0)
        + 0.2 * z_area.fillna(0.0)
        - 1.2 * feat.get("kw_capsule_complete", 0.0)
        - 1.0 * feat.get("kw_boundary_clear", 0.0)
        - 0.3 * feat.get("kw_texture_tender", 0.0)
    )
    feat["manual_gross_highrisk_score"] = risk.astype(float)
    feat["manual_capsule_boundary_balance"] = (
        feat.get("kw_boundary_unclear", 0.0)
        + feat.get("kw_capsule_absent", 0.0)
        + feat.get("kw_capsule_involved", 0.0)
        - feat.get("kw_boundary_clear", 0.0)
        - feat.get("kw_capsule_complete", 0.0)
    ).astype(float)
    return feat


def select_gross_features(gross_feat: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return gross_feat.iloc[:, 0:0].copy()
    if mode == "full":
        return gross_feat.copy()
    core_cols = [
        "age",
        "sex_male",
        "sex_female",
        "gross_text_len",
        "tumor_n_size_mentions",
        "tumor_max_dim_mm",
        "tumor_max_area_mm2",
        "tumor_max_volume_mm3",
        "log1p_tumor_max_dim_mm",
        "log1p_tumor_max_area_mm2",
        "log1p_tumor_max_volume_mm3",
        "kw_boundary_clear",
        "kw_boundary_unclear",
        "kw_capsule_any",
        "kw_capsule_complete",
        "kw_capsule_absent",
        "kw_capsule_involved",
        "kw_lung_attached",
        "kw_pericardium_attached",
        "kw_pleura_attached",
        "kw_fat_attached",
        "kw_hemorrhage",
        "kw_necrosis",
        "kw_cystic",
        "kw_lobulated",
        "kw_septum",
        "kw_texture_tender",
        "kw_texture_medium",
        "kw_texture_tough",
        "kw_texture_fragile",
        "manual_gross_highrisk_score",
        "manual_capsule_boundary_balance",
    ]
    compact_cols = [
        "tumor_max_dim_mm",
        "log1p_tumor_max_dim_mm",
        "kw_boundary_clear",
        "kw_boundary_unclear",
        "kw_capsule_complete",
        "kw_capsule_absent",
        "kw_capsule_involved",
        "kw_lung_attached",
        "kw_pericardium_attached",
        "kw_necrosis",
        "kw_texture_tender",
        "kw_texture_tough",
        "manual_gross_highrisk_score",
        "manual_capsule_boundary_balance",
    ]
    selected = compact_cols if mode == "compact" else core_cols
    return gross_feat[[col for col in selected if col in gross_feat.columns]].copy()


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def build_features(base_df: pd.DataFrame, gross_feat: pd.DataFrame, image_sources: list[str], gross_mode: str) -> pd.DataFrame:
    pieces = []
    if image_sources:
        pieces.append(build_feature_frame(base_df, gross_feat.iloc[:, 0:0], image_sources, False))
    selected_gross = select_gross_features(gross_feat, gross_mode)
    if not selected_gross.empty:
        pieces.append(selected_gross)
    if not pieces:
        raise ValueError("No features selected.")
    return pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def inner_select(
    x_all: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    train_mask: np.ndarray,
    objective: str,
) -> tuple[float, float]:
    c_grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    best_c = 0.1
    best_t = 0.5
    best_s = -1.0
    fold_ids = sorted(set(folds[train_mask]))
    for c in c_grid:
        inner_prob = np.full(len(y), np.nan, dtype=float)
        for val_fold in fold_ids:
            tr = train_mask & (folds != val_fold)
            va = train_mask & (folds == val_fold)
            if tr.sum() < 8 or va.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_all.loc[tr])
            x_va = scaler.transform(x_all.loc[va])
            clf = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=20260520)
            clf.fit(x_tr, y[tr])
            inner_prob[va] = clf.predict_proba(x_va)[:, 1]
        valid = train_mask & ~np.isnan(inner_prob)
        if valid.sum() < 8 or len(np.unique(y[valid])) < 2:
            continue
        threshold, current = best_threshold(y[valid], inner_prob[valid], objective)
        key = (current, -abs(threshold - 0.5), -abs(math.log10(c)))
        if key > (best_s, -abs(best_t - 0.5), -abs(math.log10(best_c))):
            best_s = current
            best_c = c
            best_t = threshold
    return best_c, best_t


def load_baseline(path: Path, base_df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"case_id": str})
    keep = ["case_id", "fold_id", "label_idx", "pred_idx", "prob_low_risk_group", "prob_high_risk_group"]
    out = df[keep].copy()
    meta_cols = ["case_id", "original_case_id", "task_l6_label", "task_l7_label", "difficulty", "difficulty_fine"]
    out = out.merge(base_df[meta_cols], on="case_id", how="left")
    out["source"] = name
    return out


def run_hard_route(
    name: str,
    baseline: pd.DataFrame,
    base_df: pd.DataFrame,
    x_all: pd.DataFrame,
    train_groups: set[str],
    apply_groups: set[str],
    objective: str,
    output_root: Path,
) -> dict[str, object]:
    y = base_df["label_idx"].to_numpy(dtype=int)
    folds = base_df["fold_id"].to_numpy(dtype=int)
    train_scope = base_df["difficulty_fine"].isin(train_groups).to_numpy()
    apply_scope = base_df["difficulty_fine"].isin(apply_groups).to_numpy()

    oof = baseline.copy().sort_values(["fold_id", "case_id"]).reset_index(drop=True)
    aligned = base_df[["case_id"]].reset_index(drop=True)
    if not aligned["case_id"].equals(oof.sort_values(["fold_id", "case_id"]).reset_index(drop=True)["case_id"]):
        oof = aligned.merge(oof.drop(columns=["fold_id", "label_idx"], errors="ignore"), on="case_id", how="left")
        oof["fold_id"] = base_df["fold_id"].values
        oof["label_idx"] = y

    oof["base_pred_idx"] = oof["pred_idx"].astype(int)
    oof["base_prob_high_risk_group"] = oof["prob_high_risk_group"].astype(float)
    oof["routed"] = False
    fold_choices = []

    for fold in sorted(set(folds)):
        test_mask = (folds == fold) & apply_scope
        train_mask = (folds != fold) & train_scope
        if test_mask.sum() == 0:
            continue
        if train_mask.sum() < 10 or len(np.unique(y[train_mask])) < 2:
            continue
        c, threshold = inner_select(x_all, y, folds, train_mask, objective)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_all.loc[train_mask])
        x_test = scaler.transform(x_all.loc[test_mask])
        clf = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=20260520 + int(fold))
        clf.fit(x_train, y[train_mask])
        prob = clf.predict_proba(x_test)[:, 1]
        case_ids = set(base_df.loc[test_mask, "case_id"])
        rows = oof["case_id"].isin(case_ids)
        oof.loc[rows, "prob_high_risk_group"] = prob
        oof.loc[rows, "prob_low_risk_group"] = 1.0 - prob
        oof.loc[rows, "pred_idx"] = (prob >= threshold).astype(int)
        oof.loc[rows, "routed"] = True
        fold_choices.append(
            {
                "fold_id": int(fold),
                "c": c,
                "threshold": threshold,
                "train_n": int(train_mask.sum()),
                "test_n": int(test_mask.sum()),
                "n_features": int(x_all.shape[1]),
            }
        )

    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(run_dir / "oof_case_predictions_mean.csv", index=False)
    pd.DataFrame(fold_choices).to_csv(run_dir / "fold_choices.csv", index=False)
    metrics = write_metrics(oof, run_dir)

    routed = oof[oof["routed"]].copy()
    routed["base_correct"] = routed["base_pred_idx"].astype(int) == routed["label_idx"].astype(int)
    routed["new_correct"] = routed["pred_idx"].astype(int) == routed["label_idx"].astype(int)
    rescue = routed[(~routed["base_correct"]) & (routed["new_correct"])]
    hurt = routed[(routed["base_correct"]) & (~routed["new_correct"])]
    summary = dict(metrics.iloc[0])
    summary.update(
        {
            "config": name,
            "objective": objective,
            "train_groups": "|".join(sorted(train_groups)),
            "apply_groups": "|".join(sorted(apply_groups)),
            "routed_n": int(len(routed)),
            "rescue_n": int(len(rescue)),
            "hurt_n": int(len(hurt)),
            "net_rescue": int(len(rescue) - len(hurt)),
            "rescue_high_to_low_fn": int(((rescue["label_idx"] == 1) & (rescue["base_pred_idx"] == 0)).sum()),
            "rescue_low_to_high_fp": int(((rescue["label_idx"] == 0) & (rescue["base_pred_idx"] == 1)).sum()),
            "hurt_to_fn": int(((hurt["label_idx"] == 1) & (hurt["pred_idx"] == 0)).sum()),
            "hurt_to_fp": int(((hurt["label_idx"] == 0) & (hurt["pred_idx"] == 1)).sum()),
        }
    )
    (run_dir / "overall_metrics.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    routed.to_csv(run_dir / "routed_case_delta.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(project_root / args.registry_csv, dtype={"case_id": str, "original_case_id": str})
    split = pd.read_csv(project_root / args.split_csv, dtype={"case_id": str})
    split = split[["case_id", "master_fold_id"]].rename(columns={"master_fold_id": "fold_id"})
    curriculum = pd.read_csv(project_root / args.curriculum_csv, dtype={"case_id": str})
    curriculum = curriculum[["case_id", "difficulty", "difficulty_fine", "correct_count", "mean_true_prob", "mean_margin"]]

    base_df = registry.merge(split, on="case_id", how="inner")
    base_df = base_df.merge(curriculum, on="case_id", how="left")
    base_df = base_df[base_df["task_l7_label"].isin(TASK7_CLASS_NAMES)].copy()
    base_df["label_idx"] = (base_df["task_l7_label"] == "high_risk_group").astype(int)
    base_df["fold_id"] = base_df["fold_id"].astype(int)
    base_df["肉眼所见"] = base_df["肉眼所见"].fillna("")

    root = project_root / "outputs" / "batch1_batch2_task567_20260514"
    image_sources = {
        "stage2": root / "task7_curriculum_runs/07_case_mlp_schemeB_m060_stage2only_full5fold/oof_case_predictions_mean.csv",
        "stage3": root / "task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/oof_case_predictions_mean.csv",
        "main": root / "task7_curriculum_runs/12_stage2_salvage_foldwise_blend_noncore/oof_case_predictions_mean.csv",
        "upper": root / "task7_curriculum_runs/36_stage3_balcore_foldwise_blend_noncore/oof_case_predictions_mean.csv",
    }
    for source_name, source_path in image_sources.items():
        base_df = base_df.merge(load_image_source(source_path, source_name), on="case_id", how="left")

    gross_feat = add_manual_gross_scores(extract_gross_features(base_df))
    gross_feat.to_csv(output_dir / "gross_structured_features_with_manual_scores.csv", index=False, encoding="utf-8-sig")

    baselines = {
        "main": load_baseline(image_sources["main"], base_df, "main"),
        "upper": load_baseline(image_sources["upper"], base_df, "upper"),
    }
    summary_rows = []
    for baseline_name, baseline in baselines.items():
        run_dir = output_dir / f"baseline_{baseline_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        baseline.to_csv(run_dir / "oof_case_predictions_mean.csv", index=False)
        metrics = write_metrics(baseline, run_dir)
        row = dict(metrics.iloc[0])
        row.update({"config": f"baseline_{baseline_name}", "routed_n": 0, "rescue_n": 0, "hurt_n": 0, "net_rescue": 0})
        summary_rows.append(row)

    feature_configs = {
        "gross_compact": ([], "compact"),
        "gross_core": ([], "core"),
        "img_upper_gross_compact": (["upper"], "compact"),
        "img_upper_gross_core": (["upper"], "core"),
        "img_stack_gross_compact": (["stage2", "stage3", "upper"], "compact"),
        "img_stack_gross_core": (["stage2", "stage3", "upper"], "core"),
        "img_stack_gross_full": (["stage2", "stage3", "upper"], "full"),
    }
    route_configs = [
        ("train_hardcore_apply_hardcore", {"hard_core"}, {"hard_core"}),
        ("train_allhard_apply_hardcore", {"hard_salvage_teacher", "hard_core"}, {"hard_core"}),
        ("train_allhard_apply_allhard", {"hard_salvage_teacher", "hard_core"}, {"hard_salvage_teacher", "hard_core"}),
    ]
    for baseline_name, baseline in baselines.items():
        for feature_name, (source_names, gross_mode) in feature_configs.items():
            x_all = build_features(base_df, gross_feat, source_names, gross_mode)
            for route_name, train_groups, apply_groups in route_configs:
                for objective in ["balanced_accuracy", "f1"]:
                    name = f"{baseline_name}__{feature_name}__{route_name}__{objective}"
                    summary_rows.append(
                        run_hard_route(
                            name=name,
                            baseline=baseline,
                            base_df=base_df,
                            x_all=x_all,
                            train_groups=train_groups,
                            apply_groups=apply_groups,
                            objective=objective,
                            output_root=output_dir,
                        )
                    )

    summary = pd.DataFrame(summary_rows)
    sort_cols = ["balanced_accuracy", "accuracy", "net_rescue", "auc"]
    summary = summary.sort_values(sort_cols, ascending=False)
    summary.to_csv(output_dir / "hardcore_gross_calibrator_summary.csv", index=False)
    print(summary.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
