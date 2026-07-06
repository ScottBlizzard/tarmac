from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_task7_score_split_corrector_20260520 import evaluate_config
from run_task7_stage2_auto_corrector_20260520 import build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter grid for Task7 score-split corrector.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score", default="review_score_hybrid_max_allrange_fn")
    parser.add_argument("--recall-targets", default="0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--feature-sets", default="raw_plus_review_gross")
    parser.add_argument("--train-scopes", default="flag_train,all_train")
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def build_model_grid(seed: int) -> dict[str, object]:
    models: dict[str, object] = {}
    for depth in [1, 2, 3]:
        for lr in [0.02, 0.035, 0.05, 0.08]:
            for n_estimators in [60, 100, 140]:
                models[f"gb_d{depth}_lr{lr:g}_n{n_estimators}"] = GradientBoostingClassifier(
                    max_depth=depth,
                    learning_rate=lr,
                    n_estimators=n_estimators,
                    random_state=seed,
                )
    for max_leaf_nodes in [3, 5, 7, 9]:
        for l2 in [0.0, 0.05, 0.1, 0.3]:
            models[f"hgb_leaf{max_leaf_nodes}_l2{l2:g}"] = HistGradientBoostingClassifier(
                max_iter=80,
                max_leaf_nodes=max_leaf_nodes,
                l2_regularization=l2,
                random_state=seed,
            )
    for depth in [2, 3, 4, 5]:
        for leaf in [4, 6, 8, 12]:
            models[f"extra_d{depth}_leaf{leaf}"] = ExtraTreesClassifier(
                n_estimators=500,
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            models[f"rf_d{depth}_leaf{leaf}"] = RandomForestClassifier(
                n_estimators=400,
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
    for c in [0.05, 0.1, 0.3, 0.5, 1.0]:
        models[f"logreg_c{c:g}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        )
    return models


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    feature_sets = build_features(df, Path(args.registry_csv))
    recall_targets = [float(item.strip()) for item in args.recall_targets.split(",") if item.strip()]
    feature_names = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    train_scopes = [item.strip() for item in args.train_scopes.split(",") if item.strip()]
    models = build_model_grid(args.seed)

    summaries: list[dict[str, Any]] = []
    cases: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for feature_name in feature_names:
        x_df = feature_sets[feature_name]
        for recall_target in recall_targets:
            for model_name, model in models.items():
                for train_scope in train_scopes:
                    summary, case_df, fold_df = evaluate_config(
                        df=df,
                        x_df=x_df,
                        score_col=args.score,
                        recall_target=recall_target,
                        model_name=model_name,
                        model=model,
                        train_scope=train_scope,
                    )
                    summary["feature_set"] = feature_name
                    summaries.append(summary)
                    case_df["feature_set"] = feature_name
                    fold_df["feature_set"] = feature_name
                    fold_df["score"] = args.score
                    fold_df["recall_target"] = recall_target
                    fold_df["model"] = model_name
                    fold_df["train_scope"] = train_scope
                    cases.append(case_df)
                    folds.append(fold_df)

    summary_df = pd.DataFrame(summaries).sort_values(["final_bacc", "final_acc", "final_auc"], ascending=False)
    summary_df.to_csv(out_dir / "hypergrid_summary.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(out_dir / "hypergrid_case_outputs.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(out_dir / "hypergrid_fold_thresholds.csv", index=False)
    print(summary_df.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
