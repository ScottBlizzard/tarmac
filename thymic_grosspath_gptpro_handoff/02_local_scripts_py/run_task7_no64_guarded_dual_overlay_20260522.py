from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_task7_image_only_hardcore_reviewer_20260521 import load_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual false-positive repair and false-negative rescue overlay under No.64 guard.")
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
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/23_no64_guarded_dual_overlay_20260522",
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


def reconstruct_no64_old(project_root: Path, run64_dir: Path) -> pd.DataFrame:
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
    return out


def quantile_threshold(score: np.ndarray, pct: int) -> float:
    eligible = score[score > 0]
    if pct <= 0 or len(eligible) == 0:
        return float("inf")
    k = max(1, int(round(len(score) * pct / 100.0)))
    k = min(k, len(eligible))
    return float(np.sort(eligible)[-k])


def build_direction_scores(df: pd.DataFrame, candidate_cols: list[str], base_prob_col: str, fp_col: str, fp_t: float, fn_col: str, fn_t: float) -> tuple[np.ndarray, np.ndarray]:
    cand = df[candidate_cols].astype(float)
    base_prob = df[base_prob_col].to_numpy(float)
    base_pred = df["base_pred_for_overlay"].to_numpy(int)
    fp_prob = df[fp_col].to_numpy(float)
    fn_prob = df[fn_col].to_numpy(float)
    high_vote = (cand >= fn_t).mean(axis=1).to_numpy()
    low_vote = (cand < fp_t).mean(axis=1).to_numpy()
    base_unc = 1.0 - np.minimum(np.abs(base_prob - 0.5) / 0.5, 1.0)
    # FP repair: base says high, adapted evidence says low.
    fp_score = ((base_pred == 1) & (fp_prob < fp_t)).astype(float) * (
        1.0 + (fp_t - fp_prob).clip(min=0) + 0.35 * low_vote + 0.20 * base_unc
    )
    # FN rescue: base says low, adapted evidence says high.
    fn_score = ((base_pred == 0) & (fn_prob >= fn_t)).astype(float) * (
        1.0 + (fn_prob - fn_t).clip(min=0) + 0.35 * high_vote + 0.20 * base_unc
    )
    return fp_score, fn_score


def apply_dual(
    y: np.ndarray,
    base_prob: np.ndarray,
    base_pred: np.ndarray,
    fp_prob: np.ndarray,
    fp_t: float,
    fp_score: np.ndarray,
    fp_rt: float,
    fn_prob: np.ndarray,
    fn_t: float,
    fn_score: np.ndarray,
    fn_rt: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fp_route = fp_score >= fp_rt if np.isfinite(fp_rt) else np.zeros(len(y), dtype=bool)
    fn_route = fn_score >= fn_rt if np.isfinite(fn_rt) else np.zeros(len(y), dtype=bool)
    # If both fire, prefer FN rescue to avoid suppressing high-risk calls.
    fp_route = fp_route & ~fn_route
    pred = base_pred.copy()
    prob = base_prob.copy()
    fp_pred = (fp_prob >= fp_t).astype(int)
    fn_pred = (fn_prob >= fn_t).astype(int)
    pred[fp_route] = fp_pred[fp_route]
    prob[fp_route] = fp_prob[fp_route]
    pred[fn_route] = fn_pred[fn_route]
    prob[fn_route] = fn_prob[fn_route]
    routed = fp_route | fn_route
    row = metric_dict(y, pred, prob)
    row.update(
        {
            "routed_n": int(routed.sum()),
            "routed_pct": float(routed.mean()),
            "fp_route_n": int(fp_route.sum()),
            "fn_route_n": int(fn_route.sum()),
            "rescue_n": int(((base_pred != y) & (pred == y) & routed).sum()),
            "hurt_n": int(((base_pred == y) & (pred != y) & routed).sum()),
            "fp_rescue_n": int(((base_pred == 1) & (pred == 0) & (y == 0) & fp_route).sum()),
            "fn_rescue_n": int(((base_pred == 0) & (pred == 1) & (y == 1) & fn_route).sum()),
            "fp_hurt_n": int(((base_pred == 1) & (pred == 0) & (y == 1) & fp_route).sum()),
            "fn_hurt_n": int(((base_pred == 0) & (pred == 1) & (y == 0) & fn_route).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    return row, pred, prob, fp_route, fn_route


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    run64 = root / args.run64_dir
    adapt_cache = root / args.adapt_cache_dir
    third_external = root / args.third_external_dir
    out = root / args.output_dir
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
    fp_candidates = [c for c in ["adapt_r4_c0.0003", "adapt_r2_c0.0003", "oldonly_c0.001"] if c in candidate_cols]
    fn_candidates = [c for c in ["adapt_r2_c0.01", "adapt_r4_c0.003", "adapt_r2_c0.0003", "adapt_r4_c0.0003"] if c in candidate_cols]
    fp_thresholds = [0.50, 0.54, 0.57, 0.58, 0.60, 0.62]
    fn_thresholds = [0.45, 0.48, 0.50, 0.52, 0.54, 0.57]
    fp_budgets = [0, 1, 2, 3, 5, 8]
    fn_budgets = [0, 1, 2, 3, 5, 8]

    y_old = old["label_idx"].to_numpy(int)
    old_base_prob = old["no64_final_prob_high"].to_numpy(float)
    old_base_pred = old["no64_final_pred_idx"].to_numpy(int)
    y_third = third["label_idx"].to_numpy(int)
    third_base_prob = third["final_prob_high"].to_numpy(float)
    third_base_pred = third["final_pred_idx"].to_numpy(int)
    old_base = metric_dict(y_old, old_base_pred, old_base_prob)
    third_base = metric_dict(y_third, third_base_pred, third_base_prob)

    rows: list[dict[str, object]] = []
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fp_col in fp_candidates:
        for fp_t in fp_thresholds:
            for fn_col in fn_candidates:
                for fn_t in fn_thresholds:
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
                            row["third_tp_preserved"] = bool(row["third_tp"] >= third_base["tp"])
                            row["third_acc_gain"] = float(row["third_accuracy"] - third_base["accuracy"])
                            row["third_bacc_gain"] = float(row["third_balanced_accuracy"] - third_base["balanced_accuracy"])
                            rows.append(row)
                            cache[len(rows) - 1] = (
                                old_pred,
                                old_prob,
                                old_fp_route,
                                old_fn_route,
                                third_pred,
                                third_prob,
                                third_fp_route,
                                third_fn_route,
                            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "dual_overlay_all_policies.csv", index=False, encoding="utf-8-sig")
    guard92 = summary[summary["old_guard_092"]].copy()
    safe92 = guard92[guard92["third_tp_preserved"] & (guard92["third_acc_gain"] >= 0)].copy()
    best_safe = safe92.sort_values(["third_accuracy", "third_balanced_accuracy"], ascending=False)
    best_bacc = guard92.sort_values(["third_balanced_accuracy", "third_accuracy"], ascending=False)
    best_acc = guard92.sort_values(["third_accuracy", "third_balanced_accuracy"], ascending=False)
    best_safe.head(100).to_csv(out / "top_tp_preserved_acc_gain_under_guard92.csv", index=False, encoding="utf-8-sig")
    best_bacc.head(100).to_csv(out / "top_bacc_under_guard92.csv", index=False, encoding="utf-8-sig")
    best_acc.head(100).to_csv(out / "top_acc_under_guard92.csv", index=False, encoding="utf-8-sig")

    def row_or_none(df: pd.DataFrame) -> dict[str, object] | None:
        return None if df.empty else df.iloc[0].to_dict()

    selected = row_or_none(best_safe)
    if selected is None:
        selected = row_or_none(best_bacc)

    def save_case(prefix: str, row_dict: dict[str, object] | None) -> None:
        if row_dict is None:
            return
        matches = summary[
            (summary["fp_candidate"] == row_dict["fp_candidate"])
            & (summary["fp_threshold"] == row_dict["fp_threshold"])
            & (summary["fn_candidate"] == row_dict["fn_candidate"])
            & (summary["fn_threshold"] == row_dict["fn_threshold"])
            & (summary["fp_budget_pct"] == row_dict["fp_budget_pct"])
            & (summary["fn_budget_pct"] == row_dict["fn_budget_pct"])
        ]
        idx = int(matches.index[0])
        old_pred, old_prob, old_fp_route, old_fn_route, third_pred, third_prob, third_fp_route, third_fn_route = cache[idx]
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
        old_case["fp_route"] = old_fp_route.astype(int)
        old_case["fn_route"] = old_fn_route.astype(int)
        old_case["final_prob_high"] = old_prob
        old_case["final_pred_idx"] = old_pred
        old_case["correct"] = (old_pred == y_old).astype(int)
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
        third_case["fp_route"] = third_fp_route.astype(int)
        third_case["fn_route"] = third_fn_route.astype(int)
        third_case["dual_final_prob_high"] = third_prob
        third_case["dual_final_pred_idx"] = third_pred
        third_case["correct"] = (third_pred == y_third).astype(int)
        third_case.to_csv(out / f"{prefix}_third_case_predictions.csv", index=False, encoding="utf-8-sig")

    save_case("selected", selected)
    save_case("best_bacc_reference", row_or_none(best_bacc))
    save_case("best_acc_reference", row_or_none(best_acc))

    comp_rows = [
        {
            "name": "base",
            "old_accuracy": old_base["accuracy"],
            "old_balanced_accuracy": old_base["balanced_accuracy"],
            "third_accuracy": third_base["accuracy"],
            "third_balanced_accuracy": third_base["balanced_accuracy"],
            "third_tn": third_base["tn"],
            "third_fp": third_base["fp"],
            "third_fn": third_base["fn"],
            "third_tp": third_base["tp"],
            "policy": "No.64 protected old + third old-only proxy base",
        }
    ]
    for name, row_dict in [
        ("selected_tp_preserved_acc_gain", selected),
        ("best_bacc_under_guard92", row_or_none(best_bacc)),
        ("best_acc_under_guard92", row_or_none(best_acc)),
    ]:
        if row_dict is None:
            continue
        comp_rows.append(
            {
                "name": name,
                "old_accuracy": row_dict["old_accuracy"],
                "old_balanced_accuracy": row_dict["old_balanced_accuracy"],
                "third_accuracy": row_dict["third_accuracy"],
                "third_balanced_accuracy": row_dict["third_balanced_accuracy"],
                "third_tn": row_dict["third_tn"],
                "third_fp": row_dict["third_fp"],
                "third_fn": row_dict["third_fn"],
                "third_tp": row_dict["third_tp"],
                "policy": f"FP {row_dict['fp_candidate']}@{row_dict['fp_threshold']} b{row_dict['fp_budget_pct']} + FN {row_dict['fn_candidate']}@{row_dict['fn_threshold']} b{row_dict['fn_budget_pct']}",
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "dual_overlay_key_comparison.csv", index=False, encoding="utf-8-sig")
    report = {
        "old_base": old_base,
        "third_base": third_base,
        "selected_tp_preserved_acc_gain": selected,
        "best_bacc_under_guard92": row_or_none(best_bacc),
        "best_acc_under_guard92": row_or_none(best_acc),
        "n_policies": int(len(summary)),
        "n_guard92": int(len(guard92)),
        "n_safe92": int(len(safe92)),
        "output_dir": str(out),
    }
    (out / "dual_overlay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nKey comparison")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
