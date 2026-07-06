from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_task7_gross_feature_probe_20260520 import TASK7_CLASS_NAMES, extract_gross_features, load_image_source
from run_task7_hardcore_gross_calibrator_20260520 import (
    add_manual_gross_scores,
    build_features,
    inner_select,
    json_ready,
    load_baseline,
    metric_row,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 deployable gross auto-router probe.")
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
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/04_gross_auto_router",
    )
    return parser.parse_args()


def load_base(project_root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
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
    return base_df, gross_feat, image_sources


def route_masks(base_df: pd.DataFrame, baseline: pd.DataFrame, gross_feat: pd.DataFrame) -> dict[str, np.ndarray]:
    probs = np.vstack(
        [
            base_df["prob_stage2"].to_numpy(dtype=float),
            base_df["prob_stage3"].to_numpy(dtype=float),
            base_df["prob_upper"].to_numpy(dtype=float),
        ]
    ).T
    votes = (probs >= 0.5).astype(int)
    vote_sum = votes.sum(axis=1)
    min_margin = np.min(np.abs(probs - 0.5), axis=1)
    mean_margin = np.mean(np.abs(probs - 0.5), axis=1)
    prob_range = np.max(probs, axis=1) - np.min(probs, axis=1)
    base_pred = baseline.sort_values(["fold_id", "case_id"]).reset_index(drop=True)["pred_idx"].to_numpy(dtype=int)
    risk = gross_feat["manual_gross_highrisk_score"].to_numpy(dtype=float)

    strong_gross_conflict = ((risk >= 0.5) & (base_pred == 0)) | ((risk <= -0.5) & (base_pred == 1))
    gross_conflict = ((risk >= 0.0) & (base_pred == 0)) | ((risk < 0.0) & (base_pred == 1))
    disagreement = (vote_sum > 0) & (vote_sum < probs.shape[1])

    return {
        "all_cases": np.ones(len(base_df), dtype=bool),
        "vote_disagreement": disagreement,
        "low_min_margin_020": min_margin <= 0.20,
        "low_min_margin_030": min_margin <= 0.30,
        "low_mean_margin_025": mean_margin <= 0.25,
        "prob_range_025": prob_range >= 0.25,
        "prob_range_035": prob_range >= 0.35,
        "gross_conflict": gross_conflict,
        "strong_gross_conflict": strong_gross_conflict,
        "disagree_or_gross_conflict": disagreement | gross_conflict,
        "lowmargin_or_gross_conflict": (min_margin <= 0.30) | gross_conflict,
        "strong_conflict_or_range": strong_gross_conflict | (prob_range >= 0.35),
    }


def run_auto_route(
    name: str,
    baseline: pd.DataFrame,
    base_df: pd.DataFrame,
    x_all: pd.DataFrame,
    candidate_mask: np.ndarray,
    train_groups: set[str],
    objective: str,
    output_root: Path,
) -> dict[str, object]:
    y = base_df["label_idx"].to_numpy(dtype=int)
    folds = base_df["fold_id"].to_numpy(dtype=int)
    train_scope = base_df["difficulty_fine"].isin(train_groups).to_numpy()
    oof = baseline.sort_values(["fold_id", "case_id"]).reset_index(drop=True).copy()
    oof["base_pred_idx"] = oof["pred_idx"].astype(int)
    oof["base_prob_high_risk_group"] = oof["prob_high_risk_group"].astype(float)
    oof["routed"] = False
    fold_choices = []

    for fold in sorted(set(folds)):
        test_mask = (folds == fold) & candidate_mask
        train_mask = (folds != fold) & train_scope
        if test_mask.sum() == 0 or train_mask.sum() < 10 or len(np.unique(y[train_mask])) < 2:
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
        fold_choices.append({"fold_id": int(fold), "c": c, "threshold": threshold, "train_n": int(train_mask.sum()), "test_n": int(test_mask.sum())})

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
    hard_core_mask = base_df["difficulty_fine"].eq("hard_core").to_numpy()
    route_hard_core_precision = float((candidate_mask & hard_core_mask).sum() / candidate_mask.sum()) if candidate_mask.sum() else float("nan")
    route_hard_core_recall = float((candidate_mask & hard_core_mask).sum() / hard_core_mask.sum()) if hard_core_mask.sum() else float("nan")
    summary = dict(metrics.iloc[0])
    summary.update(
        {
            "config": name,
            "objective": objective,
            "train_groups": "|".join(sorted(train_groups)),
            "routed_n": int(len(routed)),
            "rescue_n": int(len(rescue)),
            "hurt_n": int(len(hurt)),
            "net_rescue": int(len(rescue) - len(hurt)),
            "route_hard_core_precision": route_hard_core_precision,
            "route_hard_core_recall": route_hard_core_recall,
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

    base_df, gross_feat, image_sources = load_base(project_root, args)
    baseline = load_baseline(image_sources["upper"], base_df, "upper")
    x_all = build_features(base_df, gross_feat, ["stage2", "stage3", "upper"], "core")
    masks = route_masks(base_df, baseline, gross_feat)

    rows = []
    baseline_dir = output_dir / "baseline_upper"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(baseline_dir / "oof_case_predictions_mean.csv", index=False)
    baseline_metrics = write_metrics(baseline, baseline_dir)
    baseline_row = dict(baseline_metrics.iloc[0])
    baseline_row.update({"config": "baseline_upper", "routed_n": 0, "rescue_n": 0, "hurt_n": 0, "net_rescue": 0})
    rows.append(baseline_row)

    for mask_name, mask in masks.items():
        for train_name, train_groups in {
            "train_allhard": {"hard_salvage_teacher", "hard_core"},
            "train_hardcore": {"hard_core"},
        }.items():
            for objective in ["balanced_accuracy", "f1"]:
                rows.append(
                    run_auto_route(
                        name=f"upper__img_stack_gross_core__{train_name}__apply_{mask_name}__{objective}",
                        baseline=baseline,
                        base_df=base_df,
                        x_all=x_all,
                        candidate_mask=mask,
                        train_groups=train_groups,
                        objective=objective,
                        output_root=output_dir,
                    )
                )

    summary = pd.DataFrame(rows).sort_values(["balanced_accuracy", "accuracy", "net_rescue", "auc"], ascending=False)
    summary.to_csv(output_dir / "gross_auto_router_summary.csv", index=False)
    print(summary.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
