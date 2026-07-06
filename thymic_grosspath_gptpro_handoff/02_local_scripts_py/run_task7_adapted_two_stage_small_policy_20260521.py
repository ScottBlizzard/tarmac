from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_task7_adapted_two_stage_router_20260521 import (  # noqa: E402
    metric_dict,
    read_old,
    read_third,
    split_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small-grid two-stage Task7 policy using cached old+adapt frozen-feature candidates."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--cache-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/11_unified_two_stage_adapt72_20260521",
    )
    parser.add_argument(
        "--old-feature-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521",
    )
    parser.add_argument(
        "--third-feature-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/12_unified_two_stage_small_policy_20260521",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def best_thresholds(y: np.ndarray, prob: np.ndarray) -> list[float]:
    rows: list[tuple[float, float]] = []
    for t in np.linspace(0.40, 0.70, 31):
        pred = (prob >= t).astype(int)
        m = metric_dict(y, pred, prob)
        rows.append((float(m["balanced_accuracy"]), float(t)))
    best = sorted(rows, reverse=True)[:3]
    fixed = [0.50, 0.54, 0.57, 0.58, 0.60, 0.62]
    return sorted({round(t, 3) for _, t in best} | set(fixed))


def eval_policy(
    y: np.ndarray,
    base_prob: np.ndarray,
    base_threshold: float,
    corr_prob: np.ndarray,
    corr_threshold: float,
    routed: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    base_pred = (base_prob >= base_threshold).astype(int)
    corr_pred = (corr_prob >= corr_threshold).astype(int)
    final_pred = base_pred.copy()
    final_prob = base_prob.copy()
    final_pred[routed] = corr_pred[routed]
    final_prob[routed] = corr_prob[routed]
    row = metric_dict(y, final_pred, final_prob)
    pass_mask = ~routed
    row.update(
        {
            "routed_n": int(routed.sum()),
            "routed_pct": float(routed.mean()),
            "pass_n": int(pass_mask.sum()),
            "pass_acc": float((final_pred[pass_mask] == y[pass_mask]).mean()) if pass_mask.any() else np.nan,
            "routed_acc": float((final_pred[routed] == y[routed]).mean()) if routed.any() else np.nan,
            "rescue_n": int(((base_pred != y) & (final_pred == y) & routed).sum()),
            "hurt_n": int(((base_pred == y) & (final_pred != y) & routed).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    return row, final_pred, final_prob


def route_by_budget(score: np.ndarray, budget_pct: int) -> tuple[np.ndarray, float]:
    if budget_pct <= 0:
        return np.zeros(len(score), dtype=bool), float("inf")
    threshold = float(np.quantile(score, 1.0 - budget_pct / 100.0))
    return score >= threshold, threshold


def prefixed_metrics(prefix: str, y: np.ndarray, pred: np.ndarray, prob: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    if not mask.any():
        return {f"{prefix}_n": 0}
    m = metric_dict(y[mask], pred[mask], prob[mask])
    return {f"{prefix}_{k}": v for k, v in m.items()} | {f"{prefix}_n": int(mask.sum())}


def make_route_scores(
    probs: pd.DataFrame,
    base_name: str,
    base_prob: np.ndarray,
    base_threshold: float,
) -> dict[str, np.ndarray]:
    numeric = probs.drop(columns=["case_id"], errors="ignore").astype(float)
    arr = numeric.to_numpy()
    votes = (arr >= 0.5).astype(int)
    base_pred = (base_prob >= base_threshold).astype(int)
    score: dict[str, np.ndarray] = {}
    score["low_margin"] = 1.0 - np.minimum(np.abs(base_prob - base_threshold) / 0.25, 1.0)
    score["candidate_std"] = numeric.std(axis=1).to_numpy()
    score["vote_disagreement"] = ((votes.sum(axis=1) > 0) & (votes.sum(axis=1) < votes.shape[1])).astype(float)
    if "oldonly_c0.001" in numeric.columns and "adapt_r4_c0.0003" in numeric.columns:
        score["old_adapt_gap"] = np.abs(numeric["oldonly_c0.001"].to_numpy() - numeric["adapt_r4_c0.0003"].to_numpy())
    score["base_low_high_consensus"] = np.where(base_pred == 0, numeric.max(axis=1).to_numpy(), 0.0)
    score["base_high_low_consensus"] = np.where(base_pred == 1, 1.0 - numeric.min(axis=1).to_numpy(), 0.0)
    score["mixed_disagreement_margin"] = 0.55 * score["candidate_std"] + 0.45 * score["low_margin"]
    return score


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    cache = root / args.cache_dir
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    old, _ = read_old(root, args.old_feature_dir)
    third, _ = read_third(root, args.third_feature_dir)
    adapt_idx, hold_idx = split_profile(third, args.seed)
    adapt = third.iloc[adapt_idx].reset_index(drop=True)
    hold = third.iloc[hold_idx].reset_index(drop=True)

    y_train = np.concatenate([old["label_idx"].to_numpy(int), adapt["label_idx"].to_numpy(int)])
    is_old = np.concatenate([np.ones(len(old), dtype=bool), np.zeros(len(adapt), dtype=bool)])
    is_adapt = ~is_old
    y_hold = hold["label_idx"].to_numpy(int)

    train_probs = pd.read_csv(cache / "candidate_train_oof_probs.csv", dtype={"case_id": str})
    hold_probs = pd.read_csv(cache / "candidate_holdout_probs.csv", dtype={"case_id": str})
    candidate_names = [c for c in train_probs.columns if c != "case_id"]

    important = [
        "adapt_r4_c0.0003",
        "adapt_r2_c0.0003",
        "oldonly_c0.001",
        "oldonly_c0.0003",
        "adapt_r4_c0.003",
        "adapt_r2_c0.01",
    ]
    base_names = [c for c in important if c in candidate_names]
    corr_names = [c for c in important if c in candidate_names]
    budgets = [0, 5, 10, 15, 20, 25, 30, 40]
    corr_thresholds = [0.45, 0.50, 0.54, 0.57, 0.58, 0.60, 0.62]

    rows: list[dict[str, object]] = []
    train_outputs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    hold_outputs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for base_name in base_names:
        base_prob = train_probs[base_name].to_numpy(float)
        hold_base_prob = hold_probs[base_name].to_numpy(float)
        for base_threshold in best_thresholds(y_train, base_prob):
            route_train_scores = make_route_scores(train_probs, base_name, base_prob, base_threshold)
            route_hold_scores = make_route_scores(hold_probs, base_name, hold_base_prob, base_threshold)
            for route_name, route_score in route_train_scores.items():
                hold_route_score = route_hold_scores[route_name]
                for budget in budgets:
                    routed_train, route_threshold = route_by_budget(route_score, budget)
                    routed_hold = hold_route_score >= route_threshold if np.isfinite(route_threshold) else np.zeros(
                        len(y_hold), dtype=bool
                    )
                    for corr_name in corr_names:
                        corr_prob = train_probs[corr_name].to_numpy(float)
                        hold_corr_prob = hold_probs[corr_name].to_numpy(float)
                        for corr_threshold in corr_thresholds:
                            train_row, train_pred, train_final_prob = eval_policy(
                                y_train, base_prob, base_threshold, corr_prob, corr_threshold, routed_train
                            )
                            old_m = prefixed_metrics("old_oof", y_train, train_pred, train_final_prob, is_old)
                            adapt_m = prefixed_metrics("adapt_oof", y_train, train_pred, train_final_prob, is_adapt)
                            # Selection deliberately uses only OOF/adaptation data, not third holdout.
                            old_bacc = float(old_m.get("old_oof_balanced_accuracy", 0.0))
                            adapt_bacc = float(adapt_m.get("adapt_oof_balanced_accuracy", 0.0))
                            old_acc = float(old_m.get("old_oof_accuracy", 0.0))
                            adapt_acc = float(adapt_m.get("adapt_oof_accuracy", 0.0))
                            selection_score = (
                                min(old_bacc, adapt_bacc)
                                + 0.15 * float(train_row["balanced_accuracy"])
                                + 0.08 * min(old_acc, adapt_acc)
                                - 0.03 * float(train_row["routed_pct"])
                                + 0.01 * float(train_row["net_rescue"])
                            )
                            hold_row, hold_pred, hold_final_prob = eval_policy(
                                y_hold, hold_base_prob, base_threshold, hold_corr_prob, corr_threshold, routed_hold
                            )
                            row = {
                                "base_name": base_name,
                                "base_threshold": base_threshold,
                                "route_name": route_name,
                                "route_threshold": route_threshold,
                                "budget_pct": budget,
                                "corrector_name": corr_name,
                                "corrector_threshold": corr_threshold,
                                "selection_score": selection_score,
                            }
                            row.update({f"train_{k}": v for k, v in train_row.items()})
                            row.update(old_m)
                            row.update(adapt_m)
                            row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                            rows.append(row)
                            idx = len(rows) - 1
                            train_outputs[idx] = (train_pred, train_final_prob, routed_train)
                            hold_outputs[idx] = (hold_pred, hold_final_prob, routed_hold)

    summary = pd.DataFrame(rows).sort_values(
        [
            "selection_score",
            "old_oof_balanced_accuracy",
            "adapt_oof_balanced_accuracy",
            "train_balanced_accuracy",
            "train_accuracy",
        ],
        ascending=False,
    )
    summary.to_csv(out / "small_two_stage_policy_summary.csv", index=False, encoding="utf-8-sig")

    selected_pos = int(summary.index[0])
    selected = summary.iloc[0].to_dict()
    train_pred, train_final_prob, routed_train = train_outputs[selected_pos]
    hold_pred, hold_final_prob, routed_hold = hold_outputs[selected_pos]

    train_case = pd.concat(
        [
            old[["case_id", "label_idx"]].assign(source_split="old_oof"),
            adapt[["case_id", "label_idx"]].assign(source_split="third_adapt_oof"),
        ],
        ignore_index=True,
    )
    train_case["routed_to_reviewer"] = routed_train.astype(int)
    train_case["final_prob_high"] = train_final_prob
    train_case["final_pred_idx"] = train_pred
    train_case["final_correct"] = (train_pred == y_train).astype(int)
    train_case.to_csv(out / "selected_train_oof_case_predictions.csv", index=False, encoding="utf-8-sig")

    hold_case = hold[["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]].copy()
    hold_case["routed_to_reviewer"] = routed_hold.astype(int)
    hold_case["final_prob_high"] = hold_final_prob
    hold_case["final_pred_idx"] = hold_pred
    hold_case["final_correct"] = (hold_pred == y_hold).astype(int)
    hold_case.to_csv(out / "selected_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")

    subtype = (
        hold_case.groupby("task_l6_label")
        .agg(
            n=("case_id", "size"),
            correct=("final_correct", "sum"),
            accuracy=("final_correct", "mean"),
            routed=("routed_to_reviewer", "sum"),
        )
        .reset_index()
    )
    subtype.to_csv(out / "selected_holdout_metrics_by_subtype.csv", index=False, encoding="utf-8-sig")

    report = {
        "selection_uses_third_holdout": False,
        "old_n": int(len(old)),
        "third_adapt_n": int(len(adapt)),
        "third_holdout_n": int(len(hold)),
        "selected_policy": selected,
        "holdout_subtype": subtype.to_dict("records"),
        "output_dir": str(out),
    }
    (out / "small_two_stage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nTop 20 policies")
    cols = [
        "base_name",
        "base_threshold",
        "route_name",
        "budget_pct",
        "corrector_name",
        "corrector_threshold",
        "old_oof_accuracy",
        "old_oof_balanced_accuracy",
        "adapt_oof_accuracy",
        "adapt_oof_balanced_accuracy",
        "holdout_accuracy",
        "holdout_balanced_accuracy",
        "holdout_f1",
        "holdout_tn",
        "holdout_fp",
        "holdout_fn",
        "holdout_tp",
        "holdout_routed_pct",
    ]
    print(summary[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
