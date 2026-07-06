from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROFILE = {"AB": 24, "B1": 4, "B2": 22, "TC": 22}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast tiny two-stage policy search for Task7 third-batch adaptation.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--cache-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/11_unified_two_stage_adapt72_20260521",
    )
    parser.add_argument(
        "--old-table",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521/case_dino_concat_feature_table.csv",
    )
    parser.add_argument(
        "--third-registry",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521/third_batch_task7_registry.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/13_unified_two_stage_tiny_policy_20260521",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def stable_key(case_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def split_profile(third: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    chosen: list[int] = []
    for subtype, n in PROFILE.items():
        g = third[third["task_l6_label"].eq(subtype)].copy()
        g["_key"] = g["case_id"].map(lambda x: stable_key(str(x), seed))
        chosen.extend(g.sort_values("_key").head(n).index.tolist())
    mask = np.zeros(len(third), dtype=bool)
    mask[np.array(chosen, dtype=int)] = True
    return np.where(mask)[0], np.where(~mask)[0]


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, object]:
    y = y.astype(int)
    pred = pred.astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    n = max(len(y), 1)
    acc = (tp + tn) / n
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float((spec + sens) / 2),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def prefixed(prefix: str, y: np.ndarray, pred: np.ndarray, prob: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    if not mask.any():
        return {f"{prefix}_n": 0}
    return {f"{prefix}_{k}": v for k, v in metrics(y[mask], pred[mask], prob[mask]).items()} | {
        f"{prefix}_n": int(mask.sum())
    }


def route_score(probs: pd.DataFrame, base_prob: np.ndarray, threshold: float, name: str) -> np.ndarray:
    numeric = probs.drop(columns=["case_id"], errors="ignore").astype(float)
    arr = numeric.to_numpy()
    base_pred = (base_prob >= threshold).astype(int)
    if name == "low_margin":
        return 1.0 - np.minimum(np.abs(base_prob - threshold) / 0.25, 1.0)
    if name == "candidate_std":
        return numeric.std(axis=1).to_numpy()
    if name == "old_adapt_gap":
        return np.abs(numeric["oldonly_c0.001"].to_numpy() - numeric["adapt_r4_c0.0003"].to_numpy())
    if name == "base_low_high_consensus":
        return np.where(base_pred == 0, numeric.max(axis=1).to_numpy(), 0.0)
    if name == "mixed_disagreement_margin":
        std = numeric.std(axis=1).to_numpy()
        margin = 1.0 - np.minimum(np.abs(base_prob - threshold) / 0.25, 1.0)
        return 0.55 * std + 0.45 * margin
    raise ValueError(name)


def route_mask(score: np.ndarray, budget: int) -> tuple[np.ndarray, float]:
    if budget <= 0:
        return np.zeros(len(score), dtype=bool), float("inf")
    threshold = float(np.quantile(score, 1.0 - budget / 100.0))
    return score >= threshold, threshold


def eval_two_stage(
    y: np.ndarray,
    base_prob: np.ndarray,
    base_t: float,
    corr_prob: np.ndarray,
    corr_t: float,
    routed: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    base_pred = (base_prob >= base_t).astype(int)
    corr_pred = (corr_prob >= corr_t).astype(int)
    final_pred = base_pred.copy()
    final_prob = base_prob.copy()
    final_pred[routed] = corr_pred[routed]
    final_prob[routed] = corr_prob[routed]
    row = metrics(y, final_pred, final_prob)
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


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    cache = root / args.cache_dir
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(root / args.old_table, dtype={"case_id": str})
    third = pd.read_csv(root / args.third_registry, dtype={"case_id": str, "original_case_id": str})
    adapt_idx, hold_idx = split_profile(third, args.seed)
    adapt = third.iloc[adapt_idx].reset_index(drop=True)
    hold = third.iloc[hold_idx].reset_index(drop=True)

    y_train = np.concatenate([old["label_idx"].to_numpy(int), adapt["label_idx"].to_numpy(int)])
    y_hold = hold["label_idx"].to_numpy(int)
    is_old = np.concatenate([np.ones(len(old), dtype=bool), np.zeros(len(adapt), dtype=bool)])
    is_adapt = ~is_old

    train_probs = pd.read_csv(cache / "candidate_train_oof_probs.csv", dtype={"case_id": str})
    hold_probs = pd.read_csv(cache / "candidate_holdout_probs.csv", dtype={"case_id": str})

    base_grid = {
        "adapt_r4_c0.0003": [0.54, 0.57, 0.58, 0.60],
        "adapt_r2_c0.0003": [0.52, 0.53, 0.57],
        "oldonly_c0.001": [0.56, 0.59, 0.62],
        "oldonly_c0.0003": [0.56, 0.60, 0.62],
    }
    corr_grid = {
        "adapt_r4_c0.0003": [0.50, 0.54, 0.57, 0.58, 0.60],
        "adapt_r2_c0.0003": [0.52, 0.53, 0.57],
        "oldonly_c0.001": [0.56, 0.59, 0.62],
    }
    route_names = ["low_margin", "candidate_std", "old_adapt_gap", "base_low_high_consensus", "mixed_disagreement_margin"]
    budgets = [0, 10, 20, 30]

    rows: list[dict[str, object]] = []
    selected_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for base_name, base_thresholds in base_grid.items():
        base_prob = train_probs[base_name].to_numpy(float)
        hold_base_prob = hold_probs[base_name].to_numpy(float)
        for base_t in base_thresholds:
            for route_name in route_names:
                train_score = route_score(train_probs, base_prob, base_t, route_name)
                hold_score = route_score(hold_probs, hold_base_prob, base_t, route_name)
                for budget in budgets:
                    routed_train, route_t = route_mask(train_score, budget)
                    routed_hold = hold_score >= route_t if np.isfinite(route_t) else np.zeros(len(y_hold), dtype=bool)
                    for corr_name, corr_thresholds in corr_grid.items():
                        corr_prob = train_probs[corr_name].to_numpy(float)
                        hold_corr_prob = hold_probs[corr_name].to_numpy(float)
                        for corr_t in corr_thresholds:
                            train_row, train_pred, train_final_prob = eval_two_stage(
                                y_train, base_prob, base_t, corr_prob, corr_t, routed_train
                            )
                            old_m = prefixed("old_oof", y_train, train_pred, train_final_prob, is_old)
                            adapt_m = prefixed("adapt_oof", y_train, train_pred, train_final_prob, is_adapt)
                            old_bacc = float(old_m["old_oof_balanced_accuracy"])
                            adapt_bacc = float(adapt_m["adapt_oof_balanced_accuracy"])
                            old_acc = float(old_m["old_oof_accuracy"])
                            adapt_acc = float(adapt_m["adapt_oof_accuracy"])
                            selection_score = (
                                min(old_bacc, adapt_bacc)
                                + 0.15 * float(train_row["balanced_accuracy"])
                                + 0.08 * min(old_acc, adapt_acc)
                                + 0.008 * float(train_row["net_rescue"])
                                - 0.025 * float(train_row["routed_pct"])
                            )
                            hold_row, hold_pred, hold_final_prob = eval_two_stage(
                                y_hold, hold_base_prob, base_t, hold_corr_prob, corr_t, routed_hold
                            )
                            row = {
                                "base_name": base_name,
                                "base_threshold": base_t,
                                "route_name": route_name,
                                "route_threshold": route_t,
                                "budget_pct": budget,
                                "corrector_name": corr_name,
                                "corrector_threshold": corr_t,
                                "selection_score": selection_score,
                            }
                            row.update({f"train_{k}": v for k, v in train_row.items()})
                            row.update(old_m)
                            row.update(adapt_m)
                            row.update({f"holdout_{k}": v for k, v in hold_row.items()})
                            rows.append(row)
                            selected_cache[len(rows) - 1] = (
                                train_pred,
                                train_final_prob,
                                routed_train,
                                hold_pred,
                                hold_final_prob,
                                routed_hold,
                            )

    summary = pd.DataFrame(rows).sort_values(
        [
            "selection_score",
            "old_oof_balanced_accuracy",
            "adapt_oof_balanced_accuracy",
            "train_balanced_accuracy",
        ],
        ascending=False,
    )
    summary.to_csv(out / "tiny_two_stage_policy_summary.csv", index=False, encoding="utf-8-sig")
    selected_idx = int(summary.index[0])
    train_pred, train_final_prob, routed_train, hold_pred, hold_final_prob, routed_hold = selected_cache[selected_idx]

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

    hold_case = hold[
        ["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]
    ].copy()
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
        "selected_policy": summary.iloc[0].to_dict(),
        "best_holdout_accuracy_policy_for_reference": summary.sort_values(
            ["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False
        ).iloc[0].to_dict(),
        "best_holdout_balanced_accuracy_policy_for_reference": summary.sort_values(
            ["holdout_balanced_accuracy", "holdout_accuracy"], ascending=False
        ).iloc[0].to_dict(),
        "holdout_subtype": subtype.to_dict("records"),
        "output_dir": str(out),
    }
    (out / "tiny_two_stage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nTop selected-by-OOF policies")
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
