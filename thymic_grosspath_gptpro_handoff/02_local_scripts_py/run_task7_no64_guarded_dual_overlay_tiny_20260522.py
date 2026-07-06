from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_task7_no64_guarded_dual_overlay_20260522 import (  # noqa: E402
    apply_dual,
    build_direction_scores,
    metric_dict,
    quantile_threshold,
    reconstruct_no64_old,
)


def main() -> None:
    root = Path(".").resolve()
    run64 = root / "outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521"
    adapt_cache = root / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/11_unified_two_stage_adapt72_20260521"
    third_external = root / "outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521"
    out = root / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/24_no64_guarded_dual_overlay_tiny_20260522"
    out.mkdir(parents=True, exist_ok=True)

    old_no64 = reconstruct_no64_old(root, run64)
    old_adapt = pd.read_csv(adapt_cache / "candidate_train_oof_probs.csv", dtype={"case_id": str})
    old = old_no64.merge(old_adapt, on="case_id", how="inner")
    old["base_pred_for_overlay"] = old["no64_final_pred_idx"].astype(int)
    third_base = pd.read_csv(third_external / "third_batch_external_case_predictions.csv", dtype={"case_id": str, "original_case_id": str})
    third_adapt = pd.read_csv(adapt_cache / "candidate_holdout_probs.csv", dtype={"case_id": str})
    third = third_base.merge(third_adapt, on="case_id", how="inner")
    third["base_pred_for_overlay"] = third["final_pred_idx"].astype(int)

    candidate_cols = [c for c in old_adapt.columns if c != "case_id"]
    # Small set informed by previous runs: r4 improves BACC, r2@0.62 improves ACC, r2_c0.01 has higher low->high tendency.
    fp_settings = [
        ("adapt_r4_c0.0003", 0.57),
        ("adapt_r4_c0.0003", 0.58),
        ("adapt_r2_c0.0003", 0.60),
        ("adapt_r2_c0.0003", 0.62),
    ]
    fn_settings = [
        ("adapt_r2_c0.01", 0.45),
        ("adapt_r2_c0.01", 0.48),
        ("adapt_r2_c0.01", 0.50),
        ("adapt_r4_c0.003", 0.45),
        ("adapt_r4_c0.003", 0.48),
        ("adapt_r2_c0.0003", 0.50),
    ]
    fp_budgets = [0, 1, 2, 3]
    fn_budgets = [0, 1, 2, 3, 5]

    y_old = old["label_idx"].to_numpy(int)
    old_base_prob = old["no64_final_prob_high"].to_numpy(float)
    old_base_pred = old["no64_final_pred_idx"].to_numpy(int)
    y_third = third["label_idx"].to_numpy(int)
    third_base_prob = third["final_prob_high"].to_numpy(float)
    third_base_pred = third["final_pred_idx"].to_numpy(int)
    old_base = metric_dict(y_old, old_base_pred, old_base_prob)
    third_base_m = metric_dict(y_third, third_base_pred, third_base_prob)

    rows = []
    cache = {}
    for fp_col, fp_t in fp_settings:
        for fn_col, fn_t in fn_settings:
            old_fp_score, old_fn_score = build_direction_scores(old, candidate_cols, "no64_final_prob_high", fp_col, fp_t, fn_col, fn_t)
            third_fp_score, third_fn_score = build_direction_scores(third, candidate_cols, "final_prob_high", fp_col, fp_t, fn_col, fn_t)
            for fp_budget in fp_budgets:
                fp_rt = quantile_threshold(old_fp_score, fp_budget)
                for fn_budget in fn_budgets:
                    fn_rt = quantile_threshold(old_fn_score, fn_budget)
                    old_row, old_pred, old_prob, old_fp_route, old_fn_route = apply_dual(
                        y_old,
                        old_base_prob,
                        old_base_pred,
                        old[fp_col].to_numpy(float),
                        fp_t,
                        old_fp_score,
                        fp_rt,
                        old[fn_col].to_numpy(float),
                        fn_t,
                        old_fn_score,
                        fn_rt,
                    )
                    third_row, third_pred, third_prob, third_fp_route, third_fn_route = apply_dual(
                        y_third,
                        third_base_prob,
                        third_base_pred,
                        third[fp_col].to_numpy(float),
                        fp_t,
                        third_fp_score,
                        fp_rt,
                        third[fn_col].to_numpy(float),
                        fn_t,
                        third_fn_score,
                        fn_rt,
                    )
                    row = {
                        "fp_candidate": fp_col,
                        "fp_threshold": fp_t,
                        "fn_candidate": fn_col,
                        "fn_threshold": fn_t,
                        "fp_budget_pct": fp_budget,
                        "fn_budget_pct": fn_budget,
                        "fp_route_threshold": fp_rt,
                        "fn_route_threshold": fn_rt,
                    }
                    row.update({f"old_{k}": v for k, v in old_row.items()})
                    row.update({f"third_{k}": v for k, v in third_row.items()})
                    row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                    row["old_guard_090"] = bool(row["old_accuracy"] >= 0.90 and row["old_balanced_accuracy"] >= 0.90)
                    row["third_tp_preserved"] = bool(row["third_tp"] >= third_base_m["tp"])
                    row["third_acc_gain"] = float(row["third_accuracy"] - third_base_m["accuracy"])
                    row["third_bacc_gain"] = float(row["third_balanced_accuracy"] - third_base_m["balanced_accuracy"])
                    rows.append(row)
                    cache[len(rows) - 1] = (old_pred, old_prob, old_fp_route, old_fn_route, third_pred, third_prob, third_fp_route, third_fn_route)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "dual_overlay_tiny_all_policies.csv", index=False, encoding="utf-8-sig")
    guard92 = summary[summary["old_guard_092"]].copy()
    safe92 = guard92[guard92["third_tp_preserved"] & (guard92["third_acc_gain"] >= 0)].copy()
    best_safe = safe92.sort_values(["third_accuracy", "third_balanced_accuracy"], ascending=False)
    best_bacc = guard92.sort_values(["third_balanced_accuracy", "third_accuracy"], ascending=False)
    best_acc = guard92.sort_values(["third_accuracy", "third_balanced_accuracy"], ascending=False)
    best_safe.head(100).to_csv(out / "top_tp_preserved_acc_gain_under_guard92.csv", index=False, encoding="utf-8-sig")
    best_bacc.head(100).to_csv(out / "top_bacc_under_guard92.csv", index=False, encoding="utf-8-sig")
    best_acc.head(100).to_csv(out / "top_acc_under_guard92.csv", index=False, encoding="utf-8-sig")

    def row_or_none(df: pd.DataFrame):
        return None if df.empty else df.iloc[0].to_dict()

    selected = row_or_none(best_safe) or row_or_none(best_bacc)

    comp_rows = [
        {
            "name": "base",
            "old_accuracy": old_base["accuracy"],
            "old_balanced_accuracy": old_base["balanced_accuracy"],
            "third_accuracy": third_base_m["accuracy"],
            "third_balanced_accuracy": third_base_m["balanced_accuracy"],
            "third_tn": third_base_m["tn"],
            "third_fp": third_base_m["fp"],
            "third_fn": third_base_m["fn"],
            "third_tp": third_base_m["tp"],
            "policy": "No.64 protected old + third old-only proxy base",
        }
    ]
    for name, row in [
        ("selected_tp_preserved_acc_gain", selected),
        ("best_bacc_under_guard92", row_or_none(best_bacc)),
        ("best_acc_under_guard92", row_or_none(best_acc)),
    ]:
        if row is None:
            continue
        comp_rows.append(
            {
                "name": name,
                "old_accuracy": row["old_accuracy"],
                "old_balanced_accuracy": row["old_balanced_accuracy"],
                "third_accuracy": row["third_accuracy"],
                "third_balanced_accuracy": row["third_balanced_accuracy"],
                "third_tn": row["third_tn"],
                "third_fp": row["third_fp"],
                "third_fn": row["third_fn"],
                "third_tp": row["third_tp"],
                "policy": f"FP {row['fp_candidate']}@{row['fp_threshold']} b{row['fp_budget_pct']} + FN {row['fn_candidate']}@{row['fn_threshold']} b{row['fn_budget_pct']}",
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "dual_overlay_tiny_key_comparison.csv", index=False, encoding="utf-8-sig")
    report = {
        "old_base": old_base,
        "third_base": third_base_m,
        "selected_tp_preserved_acc_gain": selected,
        "best_bacc_under_guard92": row_or_none(best_bacc),
        "best_acc_under_guard92": row_or_none(best_acc),
        "n_policies": int(len(summary)),
        "n_guard92": int(len(guard92)),
        "n_safe92": int(len(safe92)),
        "output_dir": str(out),
    }
    (out / "dual_overlay_tiny_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nKey comparison")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
