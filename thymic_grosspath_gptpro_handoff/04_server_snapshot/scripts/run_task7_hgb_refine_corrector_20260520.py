from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from run_task7_score_split_corrector_20260520 import evaluate_config
from run_task7_stage2_auto_corrector_20260520 import build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused HGB refinement for the Task7 score-split corrector.")
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
    parser.add_argument("--recall-targets", default="0.58,0.60,0.62")
    parser.add_argument("--feature-set", default="raw_plus_review_gross")
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def build_hgb_grid(seed: int) -> dict[str, HistGradientBoostingClassifier]:
    models: dict[str, HistGradientBoostingClassifier] = {}
    for leaf_nodes in [5, 6, 7, 8, 9]:
        for l2 in [0.03, 0.05, 0.08]:
            for min_leaf in [16, 20, 24]:
                for max_iter in [60, 80, 100]:
                    for lr in [0.08, 0.1, 0.12]:
                        name = f"hgb_leaf{leaf_nodes}_l2{l2:g}_min{min_leaf}_iter{max_iter}_lr{lr:g}"
                        models[name] = HistGradientBoostingClassifier(
                            max_leaf_nodes=leaf_nodes,
                            l2_regularization=l2,
                            min_samples_leaf=min_leaf,
                            max_iter=max_iter,
                            learning_rate=lr,
                            random_state=seed,
                        )
    return models


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    feature_sets = build_features(df, Path(args.registry_csv))
    x_df = feature_sets[args.feature_set]
    recall_targets = [float(item.strip()) for item in args.recall_targets.split(",") if item.strip()]
    models = build_hgb_grid(args.seed)

    summaries: list[dict[str, Any]] = []
    cases: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for recall_target in recall_targets:
        for model_name, model in models.items():
            summary, case_df, fold_df = evaluate_config(
                df=df,
                x_df=x_df,
                score_col=args.score,
                recall_target=recall_target,
                model_name=model_name,
                model=model,
                train_scope="flag_train",
            )
            summary["feature_set"] = args.feature_set
            summaries.append(summary)
            case_df["feature_set"] = args.feature_set
            fold_df["feature_set"] = args.feature_set
            fold_df["score"] = args.score
            fold_df["recall_target"] = recall_target
            fold_df["model"] = model_name
            fold_df["train_scope"] = "flag_train"
            cases.append(case_df)
            folds.append(fold_df)

    summary_df = pd.DataFrame(summaries).sort_values(["final_bacc", "final_acc", "final_auc"], ascending=False)
    summary_df.to_csv(out_dir / "hgb_refine_summary.csv", index=False)
    pd.concat(cases, ignore_index=True).to_csv(out_dir / "hgb_refine_case_outputs.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(out_dir / "hgb_refine_fold_thresholds.csv", index=False)
    print(summary_df.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
