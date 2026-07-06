from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from run_task7_gross_text_stacking_20260521 import (
    build_sparse,
    inner_select,
    load_data,
    metric_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 oracle hard-scope gross/text calibrator on top of best41.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/55_oracle_hard_gross_text_calibrator_20260521",
    )
    parser.add_argument("--registry-csv", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv")
    parser.add_argument("--split-csv", default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_5fold_assignments.csv")
    parser.add_argument("--curriculum-csv", default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv")
    parser.add_argument("--review-score-csv", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv")
    parser.add_argument("--best41-csv", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/41_best_candidate_stacking_balanced_20260520/best_case_outputs_full.csv")
    return parser.parse_args()


def select_numeric(numeric: pd.DataFrame, mode: str) -> pd.DataFrame:
    hand_cols = [c for c in numeric.columns if not (c.startswith("p_") or c.startswith("pred_") or c.startswith("review_score_"))]
    image_core = [
        c
        for c in [
            "p_best41",
            "p_best41_upper",
            "p_mv_first",
            "p_mv_all",
            "p_stage2",
            "p_stage3",
            "p_main",
            "p_upper",
            "p_threeway",
            "p_anti22",
            "p_anti25",
            "p_anti30",
            "p_anti34",
            "p_convnext",
            "p_view_specialist",
        ]
        if c in numeric.columns
    ]
    review_core = [
        c
        for c in numeric.columns
        if c.startswith("review_score_all_")
        or c.startswith("review_score_predlow_")
        or c in ["review_score_upper_low_conf", "review_score_upper_predlow_pupper"]
    ]
    if mode == "gross_hand":
        cols = hand_cols
    elif mode == "image_core":
        cols = image_core
    elif mode == "image_core_gross_hand":
        cols = image_core + hand_cols
    elif mode == "image_review_gross_hand":
        cols = image_core + review_core + hand_cols
    else:
        raise ValueError(mode)
    return numeric.loc[:, cols].copy()


def run_config(
    df: pd.DataFrame,
    text: pd.Series,
    numeric: pd.DataFrame,
    train_groups: set[str],
    apply_groups: set[str],
    cfg: dict[str, object],
    objective: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    y = df["label_idx"].to_numpy(dtype=int)
    folds = df["fold_id"].to_numpy(dtype=int)
    train_scope = df["difficulty_fine"].isin(train_groups).to_numpy()
    apply_scope = df["difficulty_fine"].isin(apply_groups).to_numpy()
    prob = df["p_best41"].to_numpy(dtype=float).copy()
    pred = df["pred_best41"].to_numpy(dtype=int).copy()
    routed = np.zeros(len(df), dtype=bool)
    choices: list[dict[str, object]] = []

    for fold in sorted(set(folds)):
        train_mask = (folds != fold) & train_scope
        test_mask = (folds == fold) & apply_scope
        if test_mask.sum() == 0 or train_mask.sum() < 16 or len(np.unique(y[train_mask])) < 2:
            continue
        c, threshold, inner_score = inner_select(df, text, numeric, train_mask, cfg, objective)
        xtr, xte, detail = build_sparse(
            text[train_mask],
            text[test_mask],
            numeric.loc[train_mask] if cfg["use_numeric"] else None,
            numeric.loc[test_mask] if cfg["use_numeric"] else None,
            cfg,
        )
        clf = LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=20260521 + int(fold),
        )
        clf.fit(xtr, y[train_mask])
        fold_prob = clf.predict_proba(xte)[:, 1]
        prob[test_mask] = fold_prob
        pred[test_mask] = (fold_prob >= threshold).astype(int)
        routed[test_mask] = True
        choices.append(
            {
                "fold_id": int(fold),
                "c": c,
                "threshold": threshold,
                "inner_score": inner_score,
                "train_n": int(train_mask.sum()),
                "test_n": int(test_mask.sum()),
                **detail,
            }
        )

    oof = df[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "task_l6_label",
            "task_l7_label",
            "difficulty",
            "difficulty_fine",
        ]
    ].copy()
    oof["base_pred_idx"] = df["pred_best41"].astype(int)
    oof["base_prob_high_risk_group"] = df["p_best41"].astype(float)
    oof["pred_idx"] = pred
    oof["prob_high_risk_group"] = prob
    oof["prob_low_risk_group"] = 1.0 - prob
    oof["routed"] = routed
    oof["base_correct"] = oof["base_pred_idx"].astype(int) == oof["label_idx"].astype(int)
    oof["new_correct"] = oof["pred_idx"].astype(int) == oof["label_idx"].astype(int)
    return oof, choices


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df, text, numeric_all = load_data(project_root, args)
    numeric_all = numeric_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    configs = []
    for numeric_mode in ["gross_hand", "image_core", "image_core_gross_hand", "image_review_gross_hand"]:
        for use_text in [False, True]:
            if not use_text and numeric_mode == "gross_hand":
                name = numeric_mode
            else:
                name = f"{numeric_mode}_{'text' if use_text else 'notext'}"
            configs.append(
                {
                    "name": name,
                    "numeric_mode": numeric_mode,
                    "use_text": use_text,
                    "use_numeric": True,
                    "ngram_range": (2, 4),
                    "min_df": 1,
                    "max_df": 0.95,
                    "c_grid": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                }
            )

    route_configs = [
        ("train_hardcore_apply_hardcore", {"hard_core"}, {"hard_core"}),
        ("train_allhard_apply_hardcore", {"hard_salvage_teacher", "hard_core"}, {"hard_core"}),
        ("train_allhard_apply_allhard", {"hard_salvage_teacher", "hard_core"}, {"hard_salvage_teacher", "hard_core"}),
    ]
    rows = []
    for route_name, train_groups, apply_groups in route_configs:
        for cfg in configs:
            numeric = select_numeric(numeric_all, cfg["numeric_mode"])
            for objective in ["balanced_accuracy", "accuracy", "f1"]:
                run_name = f"{route_name}__{cfg['name']}__{objective}"
                run_dir = output_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                oof, choices = run_config(df, text, numeric, train_groups, apply_groups, cfg, objective)
                metrics = metric_dict(
                    oof["label_idx"].to_numpy(dtype=int),
                    oof["pred_idx"].to_numpy(dtype=int),
                    oof["prob_high_risk_group"].to_numpy(dtype=float),
                )
                routed = oof[oof["routed"]].copy()
                rescue = routed[(~routed["base_correct"]) & (routed["new_correct"])]
                hurt = routed[(routed["base_correct"]) & (~routed["new_correct"])]
                row = {
                    **metrics,
                    "config": run_name,
                    "route": route_name,
                    "feature": cfg["name"],
                    "objective": objective,
                    "routed_n": int(len(routed)),
                    "rescue_n": int(len(rescue)),
                    "hurt_n": int(len(hurt)),
                    "net_rescue": int(len(rescue) - len(hurt)),
                    "train_groups": "|".join(sorted(train_groups)),
                    "apply_groups": "|".join(sorted(apply_groups)),
                }
                rows.append(row)
                oof.to_csv(run_dir / "oof_case_predictions_mean.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(choices).to_csv(run_dir / "fold_choices.csv", index=False)
                (run_dir / "overall_metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                pd.DataFrame(rows).sort_values(["balanced_accuracy", "accuracy", "net_rescue"], ascending=False).to_csv(
                    output_dir / "oracle_hard_gross_text_summary.partial.csv", index=False, encoding="utf-8-sig"
                )

    summary = pd.DataFrame(rows).sort_values(["balanced_accuracy", "accuracy", "net_rescue"], ascending=False)
    summary.to_csv(output_dir / "oracle_hard_gross_text_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
