from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_task7_no64_guarded_adapt_overlay_20260522 import (  # noqa: E402
    add_route_scores,
    apply_overlay,
    metric_dict,
    reconstruct_no64_old,
)


def threshold_from_scores(score: np.ndarray, budget_pct: int) -> float:
    eligible = score[np.isfinite(score)]
    eligible = eligible[eligible > 0]
    if budget_pct <= 0 or len(eligible) == 0:
        return float("inf")
    k = max(1, int(round(len(score) * budget_pct / 100.0)))
    k = min(k, len(eligible))
    return float(np.sort(eligible)[-k])


def split_candidate_train(train_probs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old = train_probs[~train_probs["case_id"].str.startswith("third_")].copy()
    adapt = train_probs[train_probs["case_id"].str.startswith("third_")].copy()
    return old, adapt


def main() -> None:
    root = Path(".").resolve()
    run64 = root / "outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521"
    adapt_cache = root / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/11_unified_two_stage_adapt72_20260521"
    third_external = root / "outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521"
    out = root / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/25_no64_adapt_tuned_overlay_20260522"
    out.mkdir(parents=True, exist_ok=True)

    old_no64 = reconstruct_no64_old(root, run64, None)
    train_probs = pd.read_csv(adapt_cache / "candidate_train_oof_probs.csv", dtype={"case_id": str})
    hold_probs = pd.read_csv(adapt_cache / "candidate_holdout_probs.csv", dtype={"case_id": str})
    old_prob_rows, adapt_prob_rows = split_candidate_train(train_probs)

    third_base_all = pd.read_csv(third_external / "third_batch_external_case_predictions.csv", dtype={"case_id": str, "original_case_id": str})

    old = old_no64.merge(old_prob_rows, on="case_id", how="inner")
    old["base_pred_for_overlay"] = old["no64_final_pred_idx"].astype(int)
    adapt = third_base_all.merge(adapt_prob_rows, on="case_id", how="inner")
    adapt["base_pred_for_overlay"] = adapt["final_pred_idx"].astype(int)
    hold = third_base_all.merge(hold_probs, on="case_id", how="inner")
    hold["base_pred_for_overlay"] = hold["final_pred_idx"].astype(int)

    candidate_cols = [c for c in train_probs.columns if c != "case_id"]
    adapt_candidates = ["adapt_r4_c0.0003", "adapt_r2_c0.0003", "adapt_r2_c0.01", "oldonly_c0.001"]
    adapt_candidates = [c for c in adapt_candidates if c in candidate_cols]
    adapt_thresholds = [0.50, 0.54, 0.57, 0.58, 0.60, 0.62]
    budgets = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]
    threshold_sources = ["old", "adapt", "old_adapt"]

    y_old = old["label_idx"].to_numpy(int)
    old_base_prob = old["no64_final_prob_high"].to_numpy(float)
    old_base_pred = old["no64_final_pred_idx"].to_numpy(int)
    y_adapt = adapt["label_idx"].to_numpy(int)
    adapt_base_prob = adapt["final_prob_high"].to_numpy(float)
    adapt_base_pred = adapt["final_pred_idx"].to_numpy(int)
    y_hold = hold["label_idx"].to_numpy(int)
    hold_base_prob = hold["final_prob_high"].to_numpy(float)
    hold_base_pred = hold["final_pred_idx"].to_numpy(int)

    old_base = metric_dict(y_old, old_base_pred, old_base_prob)
    adapt_base = metric_dict(y_adapt, adapt_base_pred, adapt_base_prob)
    hold_base = metric_dict(y_hold, hold_base_pred, hold_base_prob)

    rows: list[dict[str, object]] = []
    case_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for adapt_name in adapt_candidates:
        for adapt_t in adapt_thresholds:
            old_scores = add_route_scores(old, candidate_cols, "no64_final_prob_high", adapt_name, adapt_t)
            adapt_scores = add_route_scores(adapt, candidate_cols, "final_prob_high", adapt_name, adapt_t)
            hold_scores = add_route_scores(hold, candidate_cols, "final_prob_high", adapt_name, adapt_t)
            for route_name in old_scores:
                for source in threshold_sources:
                    score_source = {
                        "old": old_scores[route_name],
                        "adapt": adapt_scores[route_name],
                        "old_adapt": np.concatenate([old_scores[route_name], adapt_scores[route_name]]),
                    }[source]
                    for budget in budgets:
                        rt = threshold_from_scores(score_source, budget)
                        old_row, old_pred, old_prob, old_routed = apply_overlay(
                            y_old,
                            old_base_prob,
                            old_base_pred,
                            old[adapt_name].to_numpy(float),
                            adapt_t,
                            old_scores[route_name],
                            rt,
                        )
                        adapt_row, adapt_pred, adapt_prob, adapt_routed = apply_overlay(
                            y_adapt,
                            adapt_base_prob,
                            adapt_base_pred,
                            adapt[adapt_name].to_numpy(float),
                            adapt_t,
                            adapt_scores[route_name],
                            rt,
                        )
                        hold_row, hold_pred, hold_prob, hold_routed = apply_overlay(
                            y_hold,
                            hold_base_prob,
                            hold_base_pred,
                            hold[adapt_name].to_numpy(float),
                            adapt_t,
                            hold_scores[route_name],
                            rt,
                        )
                        row = {
                            "adapt_candidate": adapt_name,
                            "adapt_threshold": adapt_t,
                            "route_name": route_name,
                            "threshold_source": source,
                            "budget_pct": budget,
                            "route_threshold": rt,
                        }
                        row.update({f"old_{k}": v for k, v in old_row.items()})
                        row.update({f"adapt_{k}": v for k, v in adapt_row.items()})
                        row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                        row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                        row["old_guard_090"] = bool(row["old_accuracy"] >= 0.90 and row["old_balanced_accuracy"] >= 0.90)
                        row["adapt_tp_preserved"] = bool(row["adapt_tp"] >= adapt_base["tp"])
                        row["adapt_acc_gain"] = float(row["adapt_accuracy"] - adapt_base["accuracy"])
                        row["adapt_bacc_gain"] = float(row["adapt_balanced_accuracy"] - adapt_base["balanced_accuracy"])
                        row["holdout_tp_preserved"] = bool(row["holdout_tp"] >= hold_base["tp"])
                        row["holdout_acc_gain"] = float(row["holdout_accuracy"] - hold_base["accuracy"])
                        row["holdout_bacc_gain"] = float(row["holdout_balanced_accuracy"] - hold_base["balanced_accuracy"])
                        # Selection does not use holdout metrics.
                        row["selection_score_safe"] = (
                            float(row["adapt_accuracy"])
                            + 0.6 * float(row["adapt_balanced_accuracy"])
                            + 0.02 * float(row["adapt_net_rescue"])
                            - 0.04 * float(row["old_routed_pct"])
                        )
                        rows.append(row)
                        case_cache[len(rows) - 1] = (
                            old_pred,
                            old_prob,
                            old_routed,
                            adapt_pred,
                            adapt_prob,
                            adapt_routed,
                            hold_pred,
                            hold_prob,
                            hold_routed,
                        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "adapt_tuned_overlay_all_policies.csv", index=False, encoding="utf-8-sig")
    guard92 = summary[summary["old_guard_092"]].copy()
    # Deployable selection: old guard + adapt-set TP preserved + adapt-set non-degrading accuracy/BACC.
    safe_adapt = guard92[
        guard92["adapt_tp_preserved"] & (guard92["adapt_acc_gain"] >= 0) & (guard92["adapt_bacc_gain"] >= 0)
    ].copy()
    selected_df = safe_adapt.sort_values(
        ["selection_score_safe", "adapt_accuracy", "adapt_balanced_accuracy", "old_accuracy"], ascending=False
    )
    if selected_df.empty:
        selected_df = guard92.sort_values(["adapt_accuracy", "adapt_balanced_accuracy", "old_accuracy"], ascending=False)

    holdout_ref = guard92.sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
    holdout_safe_ref = guard92[guard92["holdout_tp_preserved"]].sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
    selected_df.head(100).to_csv(out / "top_selected_by_old_plus_adapt_only.csv", index=False, encoding="utf-8-sig")
    holdout_ref.head(100).to_csv(out / "top_holdout_reference_under_old_guard92.csv", index=False, encoding="utf-8-sig")
    holdout_safe_ref.head(100).to_csv(out / "top_holdout_tp_preserved_reference_under_old_guard92.csv", index=False, encoding="utf-8-sig")

    def row_or_none(df: pd.DataFrame) -> dict[str, object] | None:
        return None if df.empty else df.iloc[0].to_dict()

    selected = row_or_none(selected_df)
    best_holdout_ref = row_or_none(holdout_ref)
    best_holdout_safe_ref = row_or_none(holdout_safe_ref)

    def save_cases(prefix: str, row: dict[str, object] | None) -> None:
        if row is None:
            return
        matches = summary[
            (summary["adapt_candidate"] == row["adapt_candidate"])
            & (summary["adapt_threshold"] == row["adapt_threshold"])
            & (summary["route_name"] == row["route_name"])
            & (summary["threshold_source"] == row["threshold_source"])
            & (summary["budget_pct"] == row["budget_pct"])
        ]
        idx = int(matches.index[0])
        old_pred, old_prob, old_routed, adapt_pred, adapt_prob, adapt_routed, hold_pred, hold_prob, hold_routed = case_cache[idx]
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

        hold_case = hold[
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
        hold_case["overlay_routed"] = hold_routed.astype(int)
        hold_case["overlay_final_prob_high"] = hold_prob
        hold_case["overlay_final_pred_idx"] = hold_pred
        hold_case["overlay_correct"] = (hold_pred == y_hold).astype(int)
        hold_case.to_csv(out / f"{prefix}_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")

    save_cases("selected_by_old_plus_adapt", selected)
    save_cases("best_holdout_reference", best_holdout_ref)
    save_cases("best_holdout_tp_preserved_reference", best_holdout_safe_ref)

    comp_rows = [
        {
            "name": "base",
            "old_accuracy": old_base["accuracy"],
            "old_balanced_accuracy": old_base["balanced_accuracy"],
            "adapt_accuracy": adapt_base["accuracy"],
            "adapt_balanced_accuracy": adapt_base["balanced_accuracy"],
            "adapt_tn": adapt_base["tn"],
            "adapt_fp": adapt_base["fp"],
            "adapt_fn": adapt_base["fn"],
            "adapt_tp": adapt_base["tp"],
            "holdout_accuracy": hold_base["accuracy"],
            "holdout_balanced_accuracy": hold_base["balanced_accuracy"],
            "holdout_tn": hold_base["tn"],
            "holdout_fp": hold_base["fp"],
            "holdout_fn": hold_base["fn"],
            "holdout_tp": hold_base["tp"],
            "policy": "No.64 protected old + old-only proxy base on third",
        }
    ]
    for name, row in [
        ("selected_by_old_plus_adapt", selected),
        ("best_holdout_reference", best_holdout_ref),
        ("best_holdout_tp_preserved_reference", best_holdout_safe_ref),
    ]:
        if row is None:
            continue
        comp_rows.append(
            {
                "name": name,
                "old_accuracy": row["old_accuracy"],
                "old_balanced_accuracy": row["old_balanced_accuracy"],
                "adapt_accuracy": row["adapt_accuracy"],
                "adapt_balanced_accuracy": row["adapt_balanced_accuracy"],
                "adapt_tn": row["adapt_tn"],
                "adapt_fp": row["adapt_fp"],
                "adapt_fn": row["adapt_fn"],
                "adapt_tp": row["adapt_tp"],
                "holdout_accuracy": row["holdout_accuracy"],
                "holdout_balanced_accuracy": row["holdout_balanced_accuracy"],
                "holdout_tn": row["holdout_tn"],
                "holdout_fp": row["holdout_fp"],
                "holdout_fn": row["holdout_fn"],
                "holdout_tp": row["holdout_tp"],
                "policy": f"{row['adapt_candidate']} t={row['adapt_threshold']} route={row['route_name']} source={row['threshold_source']} budget={row['budget_pct']}",
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "adapt_tuned_overlay_key_comparison.csv", index=False, encoding="utf-8-sig")

    report = {
        "protocol": {
            "selection_uses_holdout": False,
            "selection_data": "old OOF + third adapt72 only",
            "holdout_data": "third adapt72 holdout 234 cases",
        },
        "old_base": old_base,
        "adapt_base": adapt_base,
        "holdout_base": hold_base,
        "selected_by_old_plus_adapt": selected,
        "best_holdout_reference_under_old_guard92": best_holdout_ref,
        "best_holdout_tp_preserved_reference_under_old_guard92": best_holdout_safe_ref,
        "n_policies": int(len(summary)),
        "n_guard92": int(len(guard92)),
        "n_safe_adapt": int(len(safe_adapt)),
        "output_dir": str(out),
    }
    (out / "adapt_tuned_overlay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nKey comparison")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
