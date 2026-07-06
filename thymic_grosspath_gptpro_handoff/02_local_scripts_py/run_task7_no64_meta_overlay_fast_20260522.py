from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_task7_no64_guarded_adapt_overlay_20260522 import reconstruct_no64_old  # noqa: E402
from run_task7_no64_meta_overlay_20260522 import apply_dual_overlay, metric_dict, threshold_from_budget  # noqa: E402


def prepare_frames(root: Path, out_prev: Path):
    run64 = root / "outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521"
    adapt_cache = root / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/11_unified_two_stage_adapt72_20260521"
    third_external = root / "outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521"
    old_no64 = reconstruct_no64_old(root, run64, None)
    train_probs_all = pd.read_csv(adapt_cache / "candidate_train_oof_probs.csv", dtype={"case_id": str})
    third_all = pd.read_csv(third_external / "third_batch_external_case_predictions.csv", dtype={"case_id": str, "original_case_id": str})
    old_prob_rows = train_probs_all[~train_probs_all["case_id"].str.startswith("third_")].copy()
    adapt_prob_rows = train_probs_all[train_probs_all["case_id"].str.startswith("third_")].copy()
    old = old_no64.merge(old_prob_rows[["case_id"]], on="case_id", how="inner")
    old["base_prob"] = old["no64_final_prob_high"].astype(float)
    old["base_pred"] = old["no64_final_pred_idx"].astype(int)
    adapt = third_all.merge(adapt_prob_rows[["case_id"]], on="case_id", how="inner")
    adapt["base_prob"] = adapt["final_prob_high"].astype(float)
    adapt["base_pred"] = adapt["final_pred_idx"].astype(int)
    hold = third_all.merge(pd.read_csv(out_prev / "meta_holdout_probs.csv", dtype={"case_id": str})[["case_id"]], on="case_id", how="inner")
    hold["base_prob"] = hold["final_prob_high"].astype(float)
    hold["base_pred"] = hold["final_pred_idx"].astype(int)

    old_meta = pd.read_csv(out_prev / "meta_train_oof_probs.csv", dtype={"case_id": str})
    hold_meta = pd.read_csv(out_prev / "meta_holdout_probs.csv", dtype={"case_id": str})
    old_meta_part = old_meta[~old_meta["case_id"].str.startswith("third_")].reset_index(drop=True)
    adapt_meta_part = old_meta[old_meta["case_id"].str.startswith("third_")].reset_index(drop=True)
    return old.reset_index(drop=True), adapt.reset_index(drop=True), hold.reset_index(drop=True), old_meta_part, adapt_meta_part, hold_meta


def source_arrays(source: str, old_arr: np.ndarray, adapt_arr: np.ndarray) -> np.ndarray:
    if source == "old":
        return old_arr
    if source == "adapt":
        return adapt_arr
    return np.concatenate([old_arr, adapt_arr])


def main() -> None:
    root = Path(".").resolve()
    prev = root / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/26_no64_meta_overlay_20260522"
    out = root / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/26_no64_meta_overlay_fast_20260522"
    out.mkdir(parents=True, exist_ok=True)
    old, adapt, hold, old_meta, adapt_meta, hold_meta = prepare_frames(root, prev)
    cand_summary = pd.read_csv(prev / "meta_candidate_oof_summary.csv")

    cand_summary["selection_rank_score"] = (
        cand_summary["adapt_accuracy"].astype(float)
        + 0.35 * cand_summary["balanced_accuracy"].astype(float)
        + 0.10 * cand_summary["old_accuracy"].astype(float)
    )
    top_names = []
    for sort_cols in [
        ["selection_rank_score", "adapt_accuracy", "balanced_accuracy"],
        ["adapt_accuracy", "balanced_accuracy", "old_accuracy"],
        ["balanced_accuracy", "old_accuracy", "adapt_accuracy"],
    ]:
        top_names.extend(
            cand_summary.sort_values(sort_cols, ascending=False)["candidate"].head(5).astype(str).tolist()
        )
    meta_names = list(dict.fromkeys(top_names))[:4]

    y_old = old["label_idx"].to_numpy(int)
    y_adapt = adapt["label_idx"].to_numpy(int)
    y_hold = hold["label_idx"].to_numpy(int)
    old_base_prob = old["base_prob"].to_numpy(float)
    adapt_base_prob = adapt["base_prob"].to_numpy(float)
    hold_base_prob = hold["base_prob"].to_numpy(float)
    old_base_pred = old["base_pred"].to_numpy(int)
    adapt_base_pred = adapt["base_pred"].to_numpy(int)
    hold_base_pred = hold["base_pred"].to_numpy(int)
    old_base = metric_dict(y_old, old_base_pred, old_base_prob)
    adapt_base = metric_dict(y_adapt, adapt_base_pred, adapt_base_prob)
    hold_base = metric_dict(y_hold, hold_base_pred, hold_base_prob)

    old_base_margin = 1.0 - np.minimum(np.abs(old_base_prob - 0.5) / 0.5, 1.0)
    adapt_base_margin = 1.0 - np.minimum(np.abs(adapt_base_prob - 0.5) / 0.5, 1.0)
    hold_base_margin = 1.0 - np.minimum(np.abs(hold_base_prob - 0.5) / 0.5, 1.0)
    budgets = [0, 2, 5, 10, 15]
    threshold_sources = ["adapt", "old_adapt"]
    route_modes = ["confidence"]
    low_to_high_thresholds = [0.50, 0.55]
    high_to_low_thresholds = [0.45, 0.50]

    rows: list[dict[str, object]] = []
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for meta_name in meta_names:
        old_p = old_meta[meta_name].to_numpy(float)
        adapt_p = adapt_meta[meta_name].to_numpy(float)
        hold_p = hold_meta[meta_name].to_numpy(float)
        for th_hi in low_to_high_thresholds:
            for th_lo in high_to_low_thresholds:
                if th_lo > th_hi:
                    continue
                for mode in route_modes:
                    if mode == "confidence":
                        old_lh_score = np.maximum(0.0, old_p - th_hi)
                        adapt_lh_score = np.maximum(0.0, adapt_p - th_hi)
                        hold_lh_score = np.maximum(0.0, hold_p - th_hi)
                        old_hl_score = np.maximum(0.0, th_lo - old_p)
                        adapt_hl_score = np.maximum(0.0, th_lo - adapt_p)
                        hold_hl_score = np.maximum(0.0, th_lo - hold_p)
                    else:
                        old_lh_score = np.maximum(0.0, old_p - th_hi) * (1.0 + old_base_margin)
                        adapt_lh_score = np.maximum(0.0, adapt_p - th_hi) * (1.0 + adapt_base_margin)
                        hold_lh_score = np.maximum(0.0, hold_p - th_hi) * (1.0 + hold_base_margin)
                        old_hl_score = np.maximum(0.0, th_lo - old_p) * (1.0 + old_base_margin)
                        adapt_hl_score = np.maximum(0.0, th_lo - adapt_p) * (1.0 + adapt_base_margin)
                        hold_hl_score = np.maximum(0.0, th_lo - hold_p) * (1.0 + hold_base_margin)
                    for source in threshold_sources:
                        lh_source = source_arrays(source, old_lh_score, adapt_lh_score)
                        hl_source = source_arrays(source, old_hl_score, adapt_hl_score)
                        lh_mask = source_arrays(source, old_base_pred == 0, adapt_base_pred == 0).astype(bool)
                        hl_mask = source_arrays(source, old_base_pred == 1, adapt_base_pred == 1).astype(bool)
                        for lh_budget in budgets:
                            lh_t = threshold_from_budget(lh_source, lh_mask, lh_budget)
                            for hl_budget in budgets:
                                hl_t = threshold_from_budget(hl_source, hl_mask, hl_budget)
                                old_row, old_pred, old_prob, old_lh, old_hl = apply_dual_overlay(
                                    y_old,
                                    old_base_prob,
                                    old_base_pred,
                                    old_p,
                                    th_hi,
                                    th_lo,
                                    old_lh_score,
                                    old_hl_score,
                                    lh_t,
                                    hl_t,
                                )
                                adapt_row, _, _, _, _ = apply_dual_overlay(
                                    y_adapt,
                                    adapt_base_prob,
                                    adapt_base_pred,
                                    adapt_p,
                                    th_hi,
                                    th_lo,
                                    adapt_lh_score,
                                    adapt_hl_score,
                                    lh_t,
                                    hl_t,
                                )
                                hold_row, hold_pred, hold_prob, hold_lh, hold_hl = apply_dual_overlay(
                                    y_hold,
                                    hold_base_prob,
                                    hold_base_pred,
                                    hold_p,
                                    th_hi,
                                    th_lo,
                                    hold_lh_score,
                                    hold_hl_score,
                                    lh_t,
                                    hl_t,
                                )
                                row = {
                                    "meta_candidate": meta_name,
                                    "low_to_high_threshold": th_hi,
                                    "high_to_low_threshold": th_lo,
                                    "route_mode": mode,
                                    "threshold_source": source,
                                    "low_to_high_budget_pct": lh_budget,
                                    "high_to_low_budget_pct": hl_budget,
                                    "low_to_high_route_threshold": lh_t,
                                    "high_to_low_route_threshold": hl_t,
                                }
                                row.update({f"old_{k}": v for k, v in old_row.items()})
                                row.update({f"adapt_{k}": v for k, v in adapt_row.items()})
                                row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                                row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                                row["adapt_tp_preserved"] = bool(row["adapt_tp"] >= adapt_base["tp"])
                                row["holdout_tp_preserved"] = bool(row["holdout_tp"] >= hold_base["tp"])
                                row["adapt_acc_gain"] = float(row["adapt_accuracy"] - adapt_base["accuracy"])
                                row["adapt_bacc_gain"] = float(row["adapt_balanced_accuracy"] - adapt_base["balanced_accuracy"])
                                row["holdout_acc_gain"] = float(row["holdout_accuracy"] - hold_base["accuracy"])
                                row["holdout_bacc_gain"] = float(row["holdout_balanced_accuracy"] - hold_base["balanced_accuracy"])
                                row["selection_score_safe"] = (
                                    float(row["adapt_accuracy"])
                                    + 0.8 * float(row["adapt_balanced_accuracy"])
                                    + 0.04 * float(row["adapt_net_rescue"])
                                    + 0.012 * float(row["adapt_tp"] - adapt_base["tp"])
                                    - 0.035 * float(row["old_hurt_n"])
                                )
                                rows.append(row)
                                cache[len(rows) - 1] = (hold_pred, hold_prob, hold_lh, hold_hl, hold_p)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "meta_overlay_fast_all_policies.csv", index=False, encoding="utf-8-sig")
    guard = summary[summary["old_guard_092"]].copy()
    safe = guard[
        guard["adapt_tp_preserved"]
        & (guard["adapt_acc_gain"] >= 0)
        & (guard["adapt_bacc_gain"] >= 0)
    ].copy()
    selected_df = safe.sort_values(
        ["selection_score_safe", "adapt_accuracy", "adapt_balanced_accuracy", "old_accuracy"], ascending=False
    )
    if selected_df.empty:
        selected_df = guard.sort_values(["adapt_accuracy", "adapt_balanced_accuracy", "old_accuracy"], ascending=False)
    hold_ref = guard.sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
    hold_tp = guard[guard["holdout_tp_preserved"]].sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
    selected_df.head(100).to_csv(out / "top_selected_by_old_plus_adapt_only.csv", index=False, encoding="utf-8-sig")
    hold_ref.head(100).to_csv(out / "top_holdout_reference_under_old_guard92.csv", index=False, encoding="utf-8-sig")
    hold_tp.head(100).to_csv(out / "top_holdout_tp_preserved_reference_under_old_guard92.csv", index=False, encoding="utf-8-sig")

    def row_or_none(df: pd.DataFrame) -> dict[str, object] | None:
        return None if df.empty else df.iloc[0].to_dict()

    selected = row_or_none(selected_df)
    best_hold = row_or_none(hold_ref)
    best_hold_tp = row_or_none(hold_tp)

    def save_holdout_cases(prefix: str, row: dict[str, object] | None) -> None:
        if row is None:
            return
        m = summary[
            (summary["meta_candidate"] == row["meta_candidate"])
            & (summary["low_to_high_threshold"] == row["low_to_high_threshold"])
            & (summary["high_to_low_threshold"] == row["high_to_low_threshold"])
            & (summary["route_mode"] == row["route_mode"])
            & (summary["threshold_source"] == row["threshold_source"])
            & (summary["low_to_high_budget_pct"] == row["low_to_high_budget_pct"])
            & (summary["high_to_low_budget_pct"] == row["high_to_low_budget_pct"])
        ]
        idx = int(m.index[0])
        hold_pred, hold_prob, hold_lh, hold_hl, hold_p = cache[idx]
        case = hold[[
            "case_id",
            "original_case_id",
            "source_folder",
            "task_l6_label",
            "task_l7_label",
            "label_idx",
            "image_name",
            "base_prob",
            "base_pred",
        ]].copy()
        case["meta_prob_high"] = hold_p
        case["overlay_low_to_high"] = hold_lh.astype(int)
        case["overlay_high_to_low"] = hold_hl.astype(int)
        case["overlay_final_prob_high"] = hold_prob
        case["overlay_final_pred_idx"] = hold_pred
        case["overlay_correct"] = (hold_pred == y_hold).astype(int)
        case.to_csv(out / f"{prefix}_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")

    save_holdout_cases("selected_by_old_plus_adapt", selected)
    save_holdout_cases("best_holdout_reference", best_hold)
    save_holdout_cases("best_holdout_tp_preserved_reference", best_hold_tp)

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
        ("best_holdout_reference", best_hold),
        ("best_holdout_tp_preserved_reference", best_hold_tp),
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
                "policy": (
                    f"{row['meta_candidate']} hi={row['low_to_high_threshold']} lo={row['high_to_low_threshold']} "
                    f"{row['route_mode']} {row['threshold_source']} "
                    f"lh={row['low_to_high_budget_pct']} hl={row['high_to_low_budget_pct']}"
                ),
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "meta_overlay_fast_key_comparison.csv", index=False, encoding="utf-8-sig")
    report = {
        "protocol": {
            "selection_uses_holdout": False,
            "selection_data": "old OOF + third adapt72 only",
            "holdout_data": "third adapt72 holdout 234 cases",
            "method": "fast meta overlay search over top OOF meta candidates",
        },
        "meta_candidates_used": meta_names,
        "old_base": old_base,
        "adapt_base": adapt_base,
        "holdout_base": hold_base,
        "selected_by_old_plus_adapt": selected,
        "best_holdout_reference_under_old_guard92": best_hold,
        "best_holdout_tp_preserved_reference_under_old_guard92": best_hold_tp,
        "n_policies": int(len(summary)),
        "n_guard92": int(len(guard)),
        "n_safe_adapt": int(len(safe)),
        "output_dir": str(out),
    }
    (out / "meta_overlay_fast_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nKey comparison")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
