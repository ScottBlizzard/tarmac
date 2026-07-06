from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_task7_image_only_hardcore_reviewer_20260521 import load_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No.64-guarded conservative overlay with third-adapted frozen-feature candidates.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--run64-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521",
    )
    parser.add_argument(
        "--adapt-cache-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/11_unified_two_stage_adapt72_20260521",
    )
    parser.add_argument(
        "--third-external-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/21_no64_guarded_adapt_overlay_20260522",
    )
    return parser.parse_args()


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, object] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if prob is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, prob))
    return out


def reconstruct_no64_old(project_root: Path, run64_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    # Reuse the exact data-loading code from the No.64 experiment, then replay its top nested-route policy.
    no64_args = SimpleNamespace(
        curriculum_csv="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
        registry_csv="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
        best41_csv="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/41_best_candidate_stacking_balanced_20260520/best_case_outputs_full.csv",
        review_score_csv="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
        dino_feature_table="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_feature_table.csv",
        dino_feature_npy="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/10_review_router_embedding_probe_20260520/case_dino_concat_features.npy",
        route_case_table="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/63_gross_hardcore_signal_fixed_20260521/case_level_gross_signal_table.csv",
    )
    df, _, _, route_scores = load_data(project_root, no64_args)
    corr = pd.read_csv(run64_dir / "model_non_easy_extra_oof.csv", dtype={"case_id": str})
    fold_choices = pd.read_csv(
        run64_dir / "model_non_easy_extra__route_best41_wrong_model_visible_extra_fold_choices.csv"
    )
    threshold_by_fold = dict(zip(fold_choices["fold_id"].astype(int), fold_choices["threshold"].astype(float)))
    route = route_scores["best41_wrong_model_visible_extra"]

    out = df[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "task_l6_label",
            "task_l7_label",
            "difficulty",
            "difficulty_fine",
            "hard_core",
            "p_best41",
            "pred_best41",
        ]
    ].copy()
    out = out.merge(corr[["case_id", "corrector_prob", "corrector_pred"]], on="case_id", how="left")
    out["no64_base_prob_high"] = out["p_best41"].astype(float)
    out["no64_base_pred_idx"] = out["pred_best41"].astype(int)
    out["no64_route_score"] = route
    out["no64_routed"] = [
        bool(score >= threshold_by_fold[int(fold)]) for score, fold in zip(out["no64_route_score"], out["fold_id"])
    ]
    out["no64_final_prob_high"] = out["no64_base_prob_high"].to_numpy()
    out["no64_final_pred_idx"] = out["no64_base_pred_idx"].to_numpy()
    routed = out["no64_routed"].to_numpy(bool)
    out.loc[routed, "no64_final_prob_high"] = out.loc[routed, "corrector_prob"].astype(float)
    out.loc[routed, "no64_final_pred_idx"] = out.loc[routed, "corrector_pred"].astype(int)
    out["no64_final_correct"] = (out["no64_final_pred_idx"].astype(int) == out["label_idx"].astype(int)).astype(int)
    return out


def add_route_scores(frame: pd.DataFrame, candidate_cols: list[str], base_prob_col: str, adapt_prob_col: str, adapt_t: float) -> dict[str, np.ndarray]:
    base_prob = frame[base_prob_col].to_numpy(float)
    adapt_prob = frame[adapt_prob_col].to_numpy(float)
    base_pred = frame["base_pred_for_overlay"].to_numpy(int)
    adapt_pred = (adapt_prob >= adapt_t).astype(int)
    disagreement = (base_pred != adapt_pred).astype(float)
    base_low_margin = 1.0 - np.minimum(np.abs(base_prob - 0.5) / 0.5, 1.0)
    adapt_conf = np.abs(adapt_prob - adapt_t)
    candidate_std = frame[candidate_cols].astype(float).std(axis=1).to_numpy()
    candidate_range = frame[candidate_cols].astype(float).max(axis=1).to_numpy() - frame[candidate_cols].astype(float).min(axis=1).to_numpy()
    return {
        "base_low_margin": base_low_margin,
        "candidate_std": candidate_std,
        "candidate_range": candidate_range,
        "disagree_base_low_margin": disagreement * (1.0 + base_low_margin),
        "disagree_adapt_conf": disagreement * (1.0 + adapt_conf),
        "base_low_adapt_high": ((base_pred == 0) & (adapt_pred == 1)).astype(float) * (1.0 + adapt_conf),
        "base_high_adapt_low": ((base_pred == 1) & (adapt_pred == 0)).astype(float) * (1.0 + adapt_conf),
        "mixed_uncertainty": 0.45 * base_low_margin + 0.35 * candidate_std + 0.20 * candidate_range,
    }


def route_threshold_from_old(score: np.ndarray, budget_pct: int) -> float:
    if budget_pct <= 0:
        return float("inf")
    k = max(1, int(round(len(score) * budget_pct / 100.0)))
    return float(np.sort(score)[-k])


def apply_overlay(
    y: np.ndarray,
    base_prob: np.ndarray,
    base_pred: np.ndarray,
    adapt_prob: np.ndarray,
    adapt_t: float,
    route_score: np.ndarray,
    route_threshold: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    routed = route_score >= route_threshold if np.isfinite(route_threshold) else np.zeros(len(y), dtype=bool)
    adapt_pred = (adapt_prob >= adapt_t).astype(int)
    final_prob = base_prob.copy()
    final_pred = base_pred.copy()
    final_prob[routed] = adapt_prob[routed]
    final_pred[routed] = adapt_pred[routed]
    row = metric_dict(y, final_pred, final_prob)
    row.update(
        {
            "routed_n": int(routed.sum()),
            "routed_pct": float(routed.mean()),
            "pass_n": int((~routed).sum()),
            "pass_acc": float((final_pred[~routed] == y[~routed]).mean()) if (~routed).any() else np.nan,
            "routed_acc": float((final_pred[routed] == y[routed]).mean()) if routed.any() else np.nan,
            "rescue_n": int(((base_pred != y) & (final_pred == y) & routed).sum()),
            "hurt_n": int(((base_pred == y) & (final_pred != y) & routed).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    return row, final_pred, final_prob, routed


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run64_dir = project_root / args.run64_dir
    adapt_cache = project_root / args.adapt_cache_dir
    third_external = project_root / args.third_external_dir
    out = project_root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    old_no64 = reconstruct_no64_old(project_root, run64_dir, args)
    old_adapt = pd.read_csv(adapt_cache / "candidate_train_oof_probs.csv", dtype={"case_id": str})
    old = old_no64.merge(old_adapt, on="case_id", how="inner")
    old["base_pred_for_overlay"] = old["no64_final_pred_idx"].astype(int)

    third_base = pd.read_csv(third_external / "third_batch_external_case_predictions.csv", dtype={"case_id": str, "original_case_id": str})
    third_adapt = pd.read_csv(adapt_cache / "candidate_holdout_probs.csv", dtype={"case_id": str})
    third = third_base.merge(third_adapt, on="case_id", how="inner")
    third["base_pred_for_overlay"] = third["final_pred_idx"].astype(int)

    candidate_cols = [c for c in old_adapt.columns if c != "case_id"]
    adapt_candidates = ["adapt_r4_c0.0003", "adapt_r2_c0.0003", "adapt_r2_c0.01", "oldonly_c0.001"]
    adapt_candidates = [c for c in adapt_candidates if c in candidate_cols]
    adapt_thresholds = [0.50, 0.54, 0.57, 0.58, 0.60, 0.62]
    budgets = [0, 1, 2, 3, 5, 8, 10, 15, 20]

    y_old = old["label_idx"].to_numpy(int)
    old_base_prob = old["no64_final_prob_high"].to_numpy(float)
    old_base_pred = old["no64_final_pred_idx"].to_numpy(int)
    y_third = third["label_idx"].to_numpy(int)
    third_base_prob = third["final_prob_high"].to_numpy(float)
    third_base_pred = third["final_pred_idx"].to_numpy(int)

    old_base_metrics = metric_dict(y_old, old_base_pred, old_base_prob)
    third_base_metrics = metric_dict(y_third, third_base_pred, third_base_prob)

    rows: list[dict[str, object]] = []
    case_outputs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for adapt_name in adapt_candidates:
        for adapt_t in adapt_thresholds:
            old_scores = add_route_scores(old, candidate_cols, "no64_final_prob_high", adapt_name, adapt_t)
            third_scores = add_route_scores(third, candidate_cols, "final_prob_high", adapt_name, adapt_t)
            for route_name, old_route_score in old_scores.items():
                third_route_score = third_scores[route_name]
                for budget in budgets:
                    threshold = route_threshold_from_old(old_route_score, budget)
                    old_row, old_final_pred, old_final_prob, old_routed = apply_overlay(
                        y_old,
                        old_base_prob,
                        old_base_pred,
                        old[adapt_name].to_numpy(float),
                        adapt_t,
                        old_route_score,
                        threshold,
                    )
                    third_row, third_final_pred, third_final_prob, third_routed = apply_overlay(
                        y_third,
                        third_base_prob,
                        third_base_pred,
                        third[adapt_name].to_numpy(float),
                        adapt_t,
                        third_route_score,
                        threshold,
                    )
                    row = {
                        "adapt_candidate": adapt_name,
                        "adapt_threshold": adapt_t,
                        "route_name": route_name,
                        "old_budget_pct": budget,
                        "route_threshold_from_old": threshold,
                    }
                    row.update({f"old_{k}": v for k, v in old_row.items()})
                    row.update({f"third_holdout_{k}": v for k, v in third_row.items()})
                    row["old_guard_0p92_pass"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                    row["old_guard_0p90_pass"] = bool(row["old_accuracy"] >= 0.90 and row["old_balanced_accuracy"] >= 0.90)
                    row["old_delta_acc_vs_no64"] = float(row["old_accuracy"] - old_base_metrics["accuracy"])
                    row["third_delta_acc_vs_base"] = float(row["third_holdout_accuracy"] - third_base_metrics["accuracy"])
                    row["third_delta_bacc_vs_base"] = float(row["third_holdout_balanced_accuracy"] - third_base_metrics["balanced_accuracy"])
                    rows.append(row)
                    case_outputs[len(rows) - 1] = (
                        old_final_pred,
                        old_final_prob,
                        old_routed,
                        third_final_pred,
                        third_final_prob,
                        third_routed,
                    )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "no64_guarded_adapt_overlay_all_policies.csv", index=False, encoding="utf-8-sig")

    guarded92 = summary[summary["old_guard_0p92_pass"]].copy()
    guarded90 = summary[summary["old_guard_0p90_pass"]].copy()
    if guarded92.empty:
        selected = guarded90.sort_values(
            ["old_accuracy", "old_balanced_accuracy", "third_holdout_accuracy", "third_holdout_balanced_accuracy"],
            ascending=False,
        ).iloc[0]
        selected_guard = "0.90"
    else:
        # This choice uses old data only except for a tie-breaker after the old guard; the reference best-third row is reported separately.
        selected = guarded92.sort_values(
            ["old_accuracy", "old_balanced_accuracy", "old_routed_n", "third_holdout_accuracy"],
            ascending=False,
        ).iloc[0]
        selected_guard = "0.92"

    best_third_guarded92 = (
        guarded92.sort_values(["third_holdout_accuracy", "third_holdout_balanced_accuracy", "old_accuracy"], ascending=False).iloc[0]
        if not guarded92.empty
        else None
    )
    best_third_guarded90 = (
        guarded90.sort_values(["third_holdout_accuracy", "third_holdout_balanced_accuracy", "old_accuracy"], ascending=False).iloc[0]
        if not guarded90.empty
        else None
    )

    def save_case_table(prefix: str, row: pd.Series) -> None:
        idx = int(row.name)
        old_pred, old_prob, old_routed, third_pred, third_prob, third_routed = case_outputs[idx]
        old_case = old[
            [
                "case_id",
                "original_case_id",
                "label_idx",
                "task_l6_label",
                "task_l7_label",
                "difficulty",
                "difficulty_fine",
                "no64_final_prob_high",
                "no64_final_pred_idx",
            ]
        ].copy()
        old_case["overlay_routed"] = old_routed.astype(int)
        old_case["overlay_final_prob_high"] = old_prob
        old_case["overlay_final_pred_idx"] = old_pred
        old_case["overlay_correct"] = (old_pred == y_old).astype(int)
        old_case.to_csv(out / f"{prefix}_old_case_predictions.csv", index=False, encoding="utf-8-sig")

        third_case = third[
            [
                "case_id",
                "original_case_id",
                "source_folder",
                "task_l6_label",
                "task_l7_label",
                "label_idx",
                "image_name",
                "final_prob_high",
                "final_pred_idx",
            ]
        ].copy()
        third_case["overlay_routed"] = third_routed.astype(int)
        third_case["overlay_final_prob_high"] = third_prob
        third_case["overlay_final_pred_idx"] = third_pred
        third_case["overlay_correct"] = (third_pred == y_third).astype(int)
        third_case.to_csv(out / f"{prefix}_third_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")

    save_case_table("selected_old_guard", selected)
    if best_third_guarded92 is not None:
        save_case_table("best_third_under_old_guard92_reference", best_third_guarded92)
    if best_third_guarded90 is not None:
        save_case_table("best_third_under_old_guard90_reference", best_third_guarded90)

    top_cols = [
        "adapt_candidate",
        "adapt_threshold",
        "route_name",
        "old_budget_pct",
        "old_accuracy",
        "old_balanced_accuracy",
        "old_routed_pct",
        "old_rescue_n",
        "old_hurt_n",
        "third_holdout_accuracy",
        "third_holdout_balanced_accuracy",
        "third_holdout_routed_pct",
        "third_holdout_rescue_n",
        "third_holdout_hurt_n",
        "third_delta_acc_vs_base",
        "third_delta_bacc_vs_base",
    ]
    guarded92.sort_values(["third_holdout_accuracy", "third_holdout_balanced_accuracy"], ascending=False).head(50).to_csv(
        out / "top_third_policies_under_old_guard92.csv", index=False, encoding="utf-8-sig"
    )
    guarded90.sort_values(["third_holdout_accuracy", "third_holdout_balanced_accuracy"], ascending=False).head(50).to_csv(
        out / "top_third_policies_under_old_guard90.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "protocol": {
            "old_main": "Exact No.64 old OOF nested route is reconstructed and used as the protected main prediction.",
            "overlay": "Adapt72 frozen-feature candidate can overwrite only routed cases. Route thresholds are derived from old-data route-score quantiles.",
            "third_eval_scope": "Only the 234-case third holdout from the adapt72 split has adapt candidate probabilities, so third metrics here are holdout-only.",
            "selection_warning": "best_third_under_old_guard rows use third labels for reference only; selected_old_guard is the deployable old-guarded choice.",
        },
        "old_no64_reconstructed": old_base_metrics,
        "third_old_only_proxy_base": third_base_metrics,
        "selected_guard": selected_guard,
        "selected_old_guard": selected.to_dict(),
        "best_third_under_old_guard92_reference": None if best_third_guarded92 is None else best_third_guarded92.to_dict(),
        "best_third_under_old_guard90_reference": None if best_third_guarded90 is None else best_third_guarded90.to_dict(),
        "n_policies": int(len(summary)),
        "n_guard92": int(len(guarded92)),
        "n_guard90": int(len(guarded90)),
        "output_dir": str(out),
    }
    (out / "no64_guarded_adapt_overlay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nTop third policies under old guard 0.92")
    if not guarded92.empty:
        print(
            guarded92.sort_values(["third_holdout_accuracy", "third_holdout_balanced_accuracy"], ascending=False)[top_cols]
            .head(20)
            .to_string(index=False)
        )
    print("\nTop third policies under old guard 0.90")
    if not guarded90.empty:
        print(
            guarded90.sort_values(["third_holdout_accuracy", "third_holdout_balanced_accuracy"], ascending=False)[top_cols]
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
