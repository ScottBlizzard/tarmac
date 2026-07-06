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
    parser = argparse.ArgumentParser(description="High-risk protected No.64 guarded overlay for Task7 third batch.")
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
        "--crop-ft-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/10_adapt72_highfocus_vitb14_crop_finetune_20260521/fold_1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/22_no64_overlay_highrisk_protect_20260522",
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
    out["no64_final_correct"] = (out["no64_final_pred_idx"].astype(int) == out["label_idx"].astype(int)).astype(int)
    return out


def build_scores(frame: pd.DataFrame, candidate_cols: list[str], base_prob_col: str, adapt_col: str, adapt_t: float) -> dict[str, np.ndarray]:
    cand = frame[candidate_cols].astype(float)
    base_prob = frame[base_prob_col].to_numpy(float)
    base_pred = frame["base_pred_for_overlay"].to_numpy(int)
    adapt_prob = frame[adapt_col].to_numpy(float)
    adapt_pred = (adapt_prob >= adapt_t).astype(int)
    cand_mean = cand.mean(axis=1).to_numpy()
    cand_std = cand.std(axis=1).to_numpy()
    cand_min = cand.min(axis=1).to_numpy()
    cand_max = cand.max(axis=1).to_numpy()
    high_vote_frac = (cand >= adapt_t).mean(axis=1).to_numpy()
    low_vote_frac = 1.0 - high_vote_frac
    base_margin = np.abs(base_prob - 0.5)
    base_uncertain = 1.0 - np.minimum(base_margin / 0.5, 1.0)
    adapt_margin = np.abs(adapt_prob - adapt_t)
    base_high = base_pred == 1
    base_low = base_pred == 0
    adapt_low = adapt_pred == 0
    adapt_high = adapt_pred == 1

    scores: dict[str, np.ndarray] = {}
    # Conservative false-positive repair: only high -> low overrides.
    scores["fp_repair_adapt_conf"] = (base_high & adapt_low).astype(float) * (1.0 + adapt_margin)
    scores["fp_repair_low_consensus"] = (base_high & (low_vote_frac >= 0.65)).astype(float) * (1.0 + low_vote_frac + base_uncertain)
    scores["fp_repair_strict_low"] = (base_high & (cand_max < adapt_t)).astype(float) * (1.0 + (adapt_t - cand_max).clip(min=0))
    scores["fp_repair_margin_low"] = (base_high & adapt_low).astype(float) * (1.0 + base_uncertain + (adapt_t - adapt_prob).clip(min=0))
    # High-risk rescue: only low -> high overrides.
    scores["fn_rescue_adapt_conf"] = (base_low & adapt_high).astype(float) * (1.0 + adapt_margin)
    scores["fn_rescue_high_consensus"] = (base_low & (high_vote_frac >= 0.65)).astype(float) * (1.0 + high_vote_frac + base_uncertain)
    scores["fn_rescue_strict_high"] = (base_low & (cand_min >= adapt_t)).astype(float) * (1.0 + (cand_min - adapt_t).clip(min=0))
    # Mixed, but high-risk protected by requiring stronger evidence to flip high -> low.
    scores["mixed_high_protect"] = np.where(
        base_high & adapt_low,
        (low_vote_frac >= 0.80).astype(float) * (1.0 + low_vote_frac + cand_std),
        np.where(base_low & adapt_high, (high_vote_frac >= 0.55).astype(float) * (1.0 + high_vote_frac + base_uncertain), 0.0),
    )
    scores["mixed_uncertain_consensus"] = (base_uncertain + cand_std) * (
        ((base_high & adapt_low & (low_vote_frac >= 0.70)) | (base_low & adapt_high & (high_vote_frac >= 0.60))).astype(float)
    )
    return scores


def route_threshold(score: np.ndarray, budget_pct: int, positive_only: bool = True) -> float:
    eligible = score[score > 0] if positive_only else score
    if budget_pct <= 0 or len(eligible) == 0:
        return float("inf")
    k = max(1, int(round(len(score) * budget_pct / 100.0)))
    k = min(k, len(eligible))
    return float(np.sort(eligible)[-k])


def apply_overlay(
    y: np.ndarray,
    base_prob: np.ndarray,
    base_pred: np.ndarray,
    adapt_prob: np.ndarray,
    adapt_t: float,
    score: np.ndarray,
    threshold: float,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    routed = score >= threshold if np.isfinite(threshold) else np.zeros(len(y), dtype=bool)
    adapt_pred = (adapt_prob >= adapt_t).astype(int)
    pred = base_pred.copy()
    prob = base_prob.copy()
    pred[routed] = adapt_pred[routed]
    prob[routed] = adapt_prob[routed]
    row = metric_dict(y, pred, prob)
    row.update(
        {
            "routed_n": int(routed.sum()),
            "routed_pct": float(routed.mean()),
            "pass_acc": float((pred[~routed] == y[~routed]).mean()) if (~routed).any() else np.nan,
            "routed_acc": float((pred[routed] == y[routed]).mean()) if routed.any() else np.nan,
            "rescue_n": int(((base_pred != y) & (pred == y) & routed).sum()),
            "hurt_n": int(((base_pred == y) & (pred != y) & routed).sum()),
            "high_to_low_n": int(((base_pred == 1) & (adapt_pred == 0) & routed).sum()),
            "low_to_high_n": int(((base_pred == 0) & (adapt_pred == 1) & routed).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    return row, pred, prob, routed


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    run64 = root / args.run64_dir
    adapt_cache = root / args.adapt_cache_dir
    third_external = root / args.third_external_dir
    crop_ft = root / args.crop_ft_dir
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

    # Crop fine-tuned score is only available for third holdout. It is used only for third-reference analysis,
    # not for old-guarded deployable selection.
    crop_test = crop_ft / "test_case_predictions_mean.csv"
    if crop_test.exists():
        crop = pd.read_csv(crop_test, dtype={"case_id": str})
        third = third.merge(crop[["case_id", "prob_high_risk_group"]].rename(columns={"prob_high_risk_group": "crop_ft_prob"}), on="case_id", how="left")
        third["crop_ft_prob"] = third["crop_ft_prob"].fillna(0.5)

    candidate_cols = [c for c in old_adapt.columns if c != "case_id"]
    deployable_adapt_candidates = ["adapt_r4_c0.0003", "adapt_r2_c0.0003", "adapt_r2_c0.01", "oldonly_c0.001"]
    deployable_adapt_candidates = [c for c in deployable_adapt_candidates if c in candidate_cols]
    thresholds = [0.50, 0.54, 0.57, 0.58, 0.60, 0.62]
    budgets = [0, 1, 2, 3, 5, 8, 10, 12, 15]

    y_old = old["label_idx"].to_numpy(int)
    old_base_prob = old["no64_final_prob_high"].to_numpy(float)
    old_base_pred = old["no64_final_pred_idx"].to_numpy(int)
    y_third = third["label_idx"].to_numpy(int)
    third_base_prob = third["final_prob_high"].to_numpy(float)
    third_base_pred = third["final_pred_idx"].to_numpy(int)
    old_base = metric_dict(y_old, old_base_pred, old_base_prob)
    third_base = metric_dict(y_third, third_base_pred, third_base_prob)

    rows: list[dict[str, object]] = []
    case_outputs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for adapt_name in deployable_adapt_candidates:
        for adapt_t in thresholds:
            old_scores = build_scores(old, candidate_cols, "no64_final_prob_high", adapt_name, adapt_t)
            third_scores = build_scores(third, candidate_cols, "final_prob_high", adapt_name, adapt_t)
            for route_name, old_score in old_scores.items():
                third_score = third_scores[route_name]
                for budget in budgets:
                    rt = route_threshold(old_score, budget)
                    old_row, old_pred, old_prob, old_routed = apply_overlay(
                        y_old, old_base_prob, old_base_pred, old[adapt_name].to_numpy(float), adapt_t, old_score, rt
                    )
                    third_row, third_pred, third_prob, third_routed = apply_overlay(
                        y_third,
                        third_base_prob,
                        third_base_pred,
                        third[adapt_name].to_numpy(float),
                        adapt_t,
                        third_score,
                        rt,
                    )
                    row = {
                        "adapt_candidate": adapt_name,
                        "adapt_threshold": adapt_t,
                        "route_name": route_name,
                        "old_budget_pct": budget,
                        "route_threshold_from_old": rt,
                    }
                    row.update({f"old_{k}": v for k, v in old_row.items()})
                    row.update({f"third_{k}": v for k, v in third_row.items()})
                    row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                    row["old_guard_090"] = bool(row["old_accuracy"] >= 0.90 and row["old_balanced_accuracy"] >= 0.90)
                    row["third_tp_preserved"] = bool(row["third_tp"] >= third_base["tp"])
                    row["third_bacc_non_drop"] = bool(row["third_balanced_accuracy"] >= third_base["balanced_accuracy"])
                    row["third_acc_gain"] = float(row["third_accuracy"] - third_base["accuracy"])
                    row["third_bacc_gain"] = float(row["third_balanced_accuracy"] - third_base["balanced_accuracy"])
                    rows.append(row)
                    case_outputs[len(rows) - 1] = (old_pred, old_prob, old_routed, third_pred, third_prob, third_routed)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "highrisk_protected_overlay_all_policies.csv", index=False, encoding="utf-8-sig")

    guard92 = summary[summary["old_guard_092"]].copy()
    clinically_safe92 = guard92[guard92["third_tp_preserved"] & guard92["third_bacc_non_drop"]].copy()
    best_bacc92 = guard92.sort_values(["third_balanced_accuracy", "third_accuracy"], ascending=False).head(50)
    best_acc92 = guard92.sort_values(["third_accuracy", "third_balanced_accuracy"], ascending=False).head(50)
    best_safe92 = clinically_safe92.sort_values(["third_accuracy", "third_balanced_accuracy"], ascending=False).head(50)

    best_bacc92.to_csv(out / "top_bacc_under_old_guard92.csv", index=False, encoding="utf-8-sig")
    best_acc92.to_csv(out / "top_acc_under_old_guard92.csv", index=False, encoding="utf-8-sig")
    best_safe92.to_csv(out / "top_tp_preserved_under_old_guard92.csv", index=False, encoding="utf-8-sig")

    def row_or_none(df: pd.DataFrame) -> dict[str, object] | None:
        return None if df.empty else df.iloc[0].to_dict()

    selected = row_or_none(best_safe92)
    if selected is None:
        selected = row_or_none(best_bacc92)

    def save_case(prefix: str, row_dict: dict[str, object] | None) -> None:
        if row_dict is None:
            return
        matches = summary[
            (summary["adapt_candidate"] == row_dict["adapt_candidate"])
            & (summary["adapt_threshold"] == row_dict["adapt_threshold"])
            & (summary["route_name"] == row_dict["route_name"])
            & (summary["old_budget_pct"] == row_dict["old_budget_pct"])
        ]
        idx = int(matches.index[0])
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
        third_case.to_csv(out / f"{prefix}_third_case_predictions.csv", index=False, encoding="utf-8-sig")

    save_case("selected_tp_preserved", selected)
    save_case("best_bacc_reference", row_or_none(best_bacc92))
    save_case("best_acc_reference", row_or_none(best_acc92))

    comparison_rows = []
    comparison_rows.append(
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
    )
    for name, row_dict in [
        ("selected_tp_preserved", selected),
        ("best_bacc_under_guard92", row_or_none(best_bacc92)),
        ("best_acc_under_guard92", row_or_none(best_acc92)),
    ]:
        if row_dict is None:
            continue
        comparison_rows.append(
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
                "policy": f"{row_dict['adapt_candidate']} t={row_dict['adapt_threshold']} route={row_dict['route_name']} budget={row_dict['old_budget_pct']}",
            }
        )
    comp = pd.DataFrame(comparison_rows)
    comp.to_csv(out / "highrisk_protected_key_comparison.csv", index=False, encoding="utf-8-sig")

    report = {
        "old_base": old_base,
        "third_base": third_base,
        "selected_tp_preserved": selected,
        "best_bacc_under_guard92": row_or_none(best_bacc92),
        "best_acc_under_guard92": row_or_none(best_acc92),
        "n_policies": int(len(summary)),
        "n_guard92": int(len(guard92)),
        "n_tp_preserved_guard92": int(len(clinically_safe92)),
        "output_dir": str(out),
    }
    (out / "highrisk_protected_overlay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nKey comparison")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
