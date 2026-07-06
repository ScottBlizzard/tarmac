from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v38_risk_gate_feature_ablation_20260527"
TARGETS = [0.90, 0.92, 0.95, 0.97]


FEATURE_SETS = {
    "all_common": "prob+image+binary+categorical",
    "prob_and_binary_no_image": "prob+binary+categorical",
    "prob_only": "probability features only",
    "core_probs_only": "six model probabilities only",
    "disagreement_only": "probability disagreement and margin only",
}


CORE_PROBS = [
    "prob_base162",
    "prob103_vitl",
    "prob107_qkvb",
    "prob_mean_core",
    "prob_stack_plain",
    "prob_stack_balanced",
]

DISAGREE = [
    "core_prob_std",
    "core_prob_range",
    "abs_162_103",
    "abs_162_107",
    "abs_103_107",
    "margin162",
    "margin_mean_core",
    "core_agree_count",
    "score_margin_agree",
    "score_v0_simple",
    "main_margin_abs",
    "robust_margin_abs",
    "main_robust_abs_diff",
    "main_robust_disagree",
    "low_main_high_robust",
    "high_main_low_robust",
    "safety_trigger",
]


def common(dev: pd.DataFrame, ext: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in dev.columns and c in ext.columns]


def feature_list(name: str, dev: pd.DataFrame, ext: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    if name == "all_common":
        numeric = common(dev, ext, v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES)
        categorical = common(dev, ext, v30.CAT_FEATURES)
    elif name == "prob_and_binary_no_image":
        numeric = common(dev, ext, v30.PROB_FEATURES + v30.BIN_FEATURES)
        categorical = common(dev, ext, v30.CAT_FEATURES)
    elif name == "prob_only":
        numeric = common(dev, ext, v30.PROB_FEATURES)
        categorical = []
    elif name == "core_probs_only":
        numeric = common(dev, ext, CORE_PROBS)
        categorical = []
    elif name == "disagreement_only":
        numeric = common(dev, ext, DISAGREE)
        categorical = []
    else:
        raise ValueError(name)
    return numeric, categorical, numeric + categorical


def metrics_with_review(df: pd.DataFrame, review: np.ndarray) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    wrong = p2 != y
    final = p2.copy()
    final[review] = y[review]
    m = v30.metrics_binary(y, final)
    m.update(
        {
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "captured_p2_wrong_n": int((review & wrong).sum()),
            "captured_p2_wrong_rate": float((review & wrong).sum() / wrong.sum()) if wrong.sum() else 0.0,
            "missed_p2_wrong_n": int((~review & wrong).sum()),
            "review_precision_vs_p2_error": float((review & wrong).sum() / review.sum()) if review.sum() else 0.0,
        }
    )
    return m


def choose_rank_budget(dev: pd.DataFrame, dev_score: np.ndarray, target: float) -> tuple[float, dict[str, float | int]]:
    best_budget = 1.0
    best_metrics = None
    for budget in np.round(np.arange(0.0, 0.805, 0.005), 3):
        review = v30.top_budget(dev_score, float(budget))
        m = metrics_with_review(dev, review)
        if m["balanced_accuracy"] >= target:
            best_budget = float(budget)
            best_metrics = m
            break
    if best_metrics is None:
        best_metrics = metrics_with_review(dev, v30.top_budget(dev_score, 1.0))
    return best_budget, best_metrics


def choose_abs_threshold(dev: pd.DataFrame, dev_score: np.ndarray, target: float) -> tuple[float, dict[str, float | int]]:
    best_thr = -np.inf
    best_metrics = None
    for thr in np.r_[np.inf, np.sort(np.unique(dev_score))[::-1], -np.inf]:
        review = dev_score >= float(thr)
        m = metrics_with_review(dev, review)
        if m["balanced_accuracy"] >= target:
            if best_metrics is None or m["review_rate"] < best_metrics["review_rate"]:
                best_thr = float(thr)
                best_metrics = m
    if best_metrics is None:
        best_metrics = metrics_with_review(dev, np.ones(len(dev), dtype=bool))
    return best_thr, best_metrics


def score_quantile_gap(dev_score: np.ndarray, ext_score: np.ndarray) -> dict[str, float]:
    return {
        "dev_score_median": float(np.median(dev_score)),
        "external_score_median": float(np.median(ext_score)),
        "dev_score_p95": float(np.quantile(dev_score, 0.95)),
        "external_score_p95": float(np.quantile(ext_score, 0.95)),
        "external_over_dev_median": float(np.median(ext_score) / np.median(dev_score)) if np.median(dev_score) else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = v30.load_development()
    ext = v30.load_external()
    rows = []
    score_rows = []

    for fs_name, fs_desc in FEATURE_SETS.items():
        numeric, categorical, features = feature_list(fs_name, dev, ext)
        model = v30.make_models(numeric, categorical)["hard_logistic"]
        dev_score, ext_score = v30.oof_and_external_scores(dev, ext, features, model)
        dev_auc = roc_auc_score(dev["p2_wrong"].astype(int), dev_score)
        ext_auc = roc_auc_score(ext["p2_wrong"].astype(int), ext_score)
        gap = score_quantile_gap(dev_score, ext_score)
        score_rows.append({"feature_set": fs_name, "description": fs_desc, "n_features": len(features), "dev_auc": dev_auc, "external_auc": ext_auc, **gap})

        for target in TARGETS:
            budget, dev_rank_m = choose_rank_budget(dev, dev_score, target)
            ext_rank_m = metrics_with_review(ext, v30.top_budget(ext_score, budget))
            rows.append(
                {
                    "feature_set": fs_name,
                    "description": fs_desc,
                    "selection": "rank_budget",
                    "target_dev_bacc": target,
                    "n_features": len(features),
                    "dev_auc": dev_auc,
                    "external_auc": ext_auc,
                    "selected_budget": budget,
                    "selected_threshold": np.nan,
                    **{f"dev_{k}": v for k, v in dev_rank_m.items()},
                    **{f"external_{k}": v for k, v in ext_rank_m.items()},
                }
            )

            thr, dev_thr_m = choose_abs_threshold(dev, dev_score, target)
            ext_thr_m = metrics_with_review(ext, ext_score >= thr)
            rows.append(
                {
                    "feature_set": fs_name,
                    "description": fs_desc,
                    "selection": "absolute_threshold",
                    "target_dev_bacc": target,
                    "n_features": len(features),
                    "dev_auc": dev_auc,
                    "external_auc": ext_auc,
                    "selected_budget": np.nan,
                    "selected_threshold": thr,
                    **{f"dev_{k}": v for k, v in dev_thr_m.items()},
                    **{f"external_{k}": v for k, v in ext_thr_m.items()},
                }
            )

    metrics = pd.DataFrame(rows)
    scores = pd.DataFrame(score_rows)
    metrics.to_csv(OUT_DIR / "v38_feature_ablation_target_transfer_metrics.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(OUT_DIR / "v38_feature_ablation_score_distribution.csv", index=False, encoding="utf-8-sig")

    show = metrics.loc[
        metrics["target_dev_bacc"].isin([0.95, 0.97]),
        [
            "feature_set",
            "selection",
            "target_dev_bacc",
            "dev_auc",
            "external_auc",
            "selected_budget",
            "selected_threshold",
            "external_review_rate",
            "external_balanced_accuracy",
            "external_accuracy",
            "external_fn",
            "external_fp",
        ],
    ].sort_values(["target_dev_bacc", "selection", "external_balanced_accuracy"], ascending=[True, True, False])
    print(show.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
