from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_task7_adapted_two_stage_router_20260521 import (  # noqa: E402
    evaluate_policy,
    metric_dict,
    read_old,
    read_third,
    route_by_budget,
    split_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast confidence/disagreement two-stage policy from cached candidate probs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--candidate-dir", default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/11_unified_two_stage_adapt72_20260521")
    parser.add_argument("--old-feature-dir", default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521")
    parser.add_argument("--third-feature-dir", default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521")
    parser.add_argument("--output-dir", default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/12_unified_two_stage_fast_policy_20260521")
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def load_probs(path: Path) -> tuple[pd.Series, np.ndarray, list[str]]:
    df = pd.read_csv(path, dtype={"case_id": str})
    names = [c for c in df.columns if c != "case_id"]
    return df["case_id"], df[names].to_numpy(dtype=float), names


def route_scores(base_prob: np.ndarray, base_pred: np.ndarray, probs: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    votes = (probs >= 0.5).astype(int)
    out = {
        "low_confidence": 1.0 - np.abs(base_prob - 0.5) * 2.0,
        "candidate_std": probs.std(axis=1),
        "candidate_range": probs.max(axis=1) - probs.min(axis=1),
        "vote_disagreement": ((votes.sum(axis=1) > 0) & (votes.sum(axis=1) < votes.shape[1])).astype(float),
    }
    # Route predicted-low cases when another candidate strongly argues high-risk, and
    # predicted-high cases when another candidate strongly argues low-risk.
    max_high = probs.max(axis=1)
    min_high = probs.min(axis=1)
    out["opposing_candidate"] = np.where(base_pred == 0, max_high, 1.0 - min_high)
    oldonly_cols = [i for i, n in enumerate(names) if n.startswith("oldonly")]
    adapt_cols = [i for i, n in enumerate(names) if n.startswith("adapt")]
    if oldonly_cols and adapt_cols:
        old_mean = probs[:, oldonly_cols].mean(axis=1)
        adapt_mean = probs[:, adapt_cols].mean(axis=1)
        out["old_adapt_disagreement"] = np.abs(old_mean - adapt_mean)
    return out


def subset_metric_row(y: np.ndarray, pred: np.ndarray, prob: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, object]:
    m = metric_dict(y[mask], pred[mask], prob[mask])
    return {f"{prefix}_{k}": v for k, v in m.items()}


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    cand_dir = root / args.candidate_dir
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    old, _ = read_old(root, args.old_feature_dir)
    third, _ = read_third(root, args.third_feature_dir)
    adapt_idx, hold_idx = split_profile(third, args.seed)
    adapt = third.iloc[adapt_idx].reset_index(drop=True)
    hold = third.iloc[hold_idx].reset_index(drop=True)
    y_train = np.concatenate([old["label_idx"].to_numpy(int), adapt["label_idx"].to_numpy(int)])
    y_hold = hold["label_idx"].to_numpy(int)
    is_old = np.concatenate([np.ones(len(old), dtype=bool), np.zeros(len(adapt), dtype=bool)])
    is_adapt = ~is_old

    train_ids, train_probs, names = load_probs(cand_dir / "candidate_train_oof_probs.csv")
    hold_ids, hold_probs, hold_names = load_probs(cand_dir / "candidate_holdout_probs.csv")
    if names != hold_names:
        raise ValueError("Train/hold candidate names mismatch.")
    base_summary = pd.read_csv(cand_dir / "base_candidate_summary.csv")
    base_summary = base_summary.sort_values(["train_balanced_accuracy", "old_oof_balanced_accuracy", "adapt_oof_balanced_accuracy"], ascending=False)
    base_rows = base_summary[
        (base_summary["old_oof_balanced_accuracy"] >= 0.64) & (base_summary["adapt_oof_balanced_accuracy"] >= 0.58)
    ].head(16)
    if base_rows.empty:
        base_rows = base_summary.head(16)
    corrector_idxs = sorted(set(base_summary.head(14)["base_idx"].astype(int).tolist() + base_rows["base_idx"].astype(int).tolist()))
    budgets = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60]
    corr_thresholds = np.round(np.linspace(0.30, 0.70, 17), 3)
    rows = []
    for _, b in base_rows.iterrows():
        base_idx = int(b["base_idx"])
        base_name = str(b["base_name"])
        base_t = float(b["threshold"])
        base_prob = train_probs[:, base_idx]
        base_pred = (base_prob >= base_t).astype(int)
        base_hold_prob = hold_probs[:, base_idx]
        base_hold_pred = (base_hold_prob >= base_t).astype(int)
        train_route_scores = route_scores(base_prob, base_pred, train_probs, names)
        hold_route_scores = route_scores(base_hold_prob, base_hold_pred, hold_probs, names)
        for route_name, r_train in train_route_scores.items():
            r_hold = hold_route_scores[route_name]
            for corr_idx in corrector_idxs:
                corr_name = names[corr_idx]
                corr_prob = train_probs[:, corr_idx]
                corr_hold_prob = hold_probs[:, corr_idx]
                for corr_t in corr_thresholds:
                    for budget in budgets:
                        routed_train, route_t = route_by_budget(r_train, budget)
                        train_row, _, _ = evaluate_policy(
                            y_train,
                            base_pred,
                            base_prob,
                            corr_prob,
                            float(corr_t),
                            routed_train,
                            is_old=is_old,
                            is_adapt=is_adapt,
                        )
                        old_bacc = float(train_row["old_oof_balanced_accuracy"])
                        adapt_bacc = float(train_row["adapt_oof_balanced_accuracy"])
                        train_row["selection_score"] = min(old_bacc, adapt_bacc) + 0.08 * float(train_row["accuracy"]) + 0.04 * float(train_row["pass_acc"])
                        routed_hold = r_hold >= route_t if np.isfinite(route_t) else np.zeros(len(r_hold), dtype=bool)
                        hold_row, _, _ = evaluate_policy(
                            y_hold,
                            base_hold_pred,
                            base_hold_prob,
                            corr_hold_prob,
                            float(corr_t),
                            routed_hold,
                        )
                        train_row.update(
                            {
                                "base_idx": base_idx,
                                "base_name": base_name,
                                "base_threshold": base_t,
                                "route_name": route_name,
                                "route_threshold": route_t,
                                "budget_pct": budget,
                                "corrector_idx": corr_idx,
                                "corrector_name": corr_name,
                                "corrector_threshold": float(corr_t),
                            }
                        )
                        train_row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                        rows.append(train_row)
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["selection_score", "old_oof_balanced_accuracy", "adapt_oof_balanced_accuracy", "accuracy"], ascending=False)
    summary.to_csv(out / "fast_two_stage_policy_summary.csv", index=False, encoding="utf-8-sig")

    selected = summary.iloc[0].to_dict()
    oracle = summary.sort_values(["holdout_balanced_accuracy", "holdout_accuracy"], ascending=False).iloc[0].to_dict()

    def export_policy(policy: dict[str, object], prefix: str) -> dict[str, object]:
        base_idx = int(policy["base_idx"])
        corr_idx = int(policy["corrector_idx"])
        base_t = float(policy["base_threshold"])
        corr_t = float(policy["corrector_threshold"])
        base_prob = train_probs[:, base_idx]
        base_pred = (base_prob >= base_t).astype(int)
        base_hold_prob = hold_probs[:, base_idx]
        base_hold_pred = (base_hold_prob >= base_t).astype(int)
        train_route = route_scores(base_prob, base_pred, train_probs, names)[str(policy["route_name"])]
        hold_route = route_scores(base_hold_prob, base_hold_pred, hold_probs, names)[str(policy["route_name"])]
        routed_train, route_t = route_by_budget(train_route, int(policy["budget_pct"]))
        routed_hold = hold_route >= route_t if np.isfinite(route_t) else np.zeros(len(hold_route), dtype=bool)
        train_eval, train_final_pred, train_final_prob = evaluate_policy(
            y_train,
            base_pred,
            base_prob,
            train_probs[:, corr_idx],
            corr_t,
            routed_train,
            is_old=is_old,
            is_adapt=is_adapt,
        )
        hold_eval, hold_final_pred, hold_final_prob = evaluate_policy(
            y_hold,
            base_hold_pred,
            base_hold_prob,
            hold_probs[:, corr_idx],
            corr_t,
            routed_hold,
        )
        hold_case = hold[["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]].copy()
        hold_case["base_prob_high"] = base_hold_prob
        hold_case["base_pred_idx"] = base_hold_pred
        hold_case["route_score"] = hold_route
        hold_case["routed_to_reviewer"] = routed_hold.astype(int)
        hold_case["reviewer_prob_high"] = hold_probs[:, corr_idx]
        hold_case["final_prob_high"] = hold_final_prob
        hold_case["final_pred_idx"] = hold_final_pred
        hold_case["final_correct"] = (hold_final_pred == y_hold).astype(int)
        hold_case.to_csv(out / f"{prefix}_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")
        subtype = hold_case.groupby("task_l6_label").agg(n=("case_id", "size"), correct=("final_correct", "sum"), accuracy=("final_correct", "mean"), routed=("routed_to_reviewer", "sum")).reset_index()
        subtype.to_csv(out / f"{prefix}_holdout_metrics_by_subtype.csv", index=False, encoding="utf-8-sig")
        return {"policy": policy, "train_oof": train_eval, "holdout": hold_eval, "subtype": subtype.to_dict("records")}

    selected_report = export_policy(selected, "selected")
    oracle_report = export_policy(oracle, "holdout_oracle_for_analysis")
    report = {
        "boundary": {
            "selection_uses_holdout": False,
            "holdout_oracle_is_for_analysis_only": True,
            "old_n": int(len(old)),
            "third_adapt_n": int(len(adapt)),
            "third_holdout_n": int(len(hold)),
        },
        "selected_by_train_oof": selected_report,
        "best_holdout_oracle_for_analysis": oracle_report,
    }
    (out / "fast_two_stage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nTop selected-score policies")
    print(summary.head(20).to_string(index=False))
    print("\nTop holdout policies (analysis only)")
    print(summary.sort_values(["holdout_balanced_accuracy", "holdout_accuracy"], ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
