from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v40_risk_ranker_baselines_20260527"
FIG_DIR = OUT_DIR / "figures"
BUDGET_GRID = np.round(np.arange(0.0, 0.805, 0.025), 3)
TARGETS = [0.90, 0.92, 0.95, 0.97]
RANDOM_REPEATS = 300


def metrics_with_review(df: pd.DataFrame, score: np.ndarray, budget: float) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    wrong = p2 != y
    review = v30.top_budget(score, float(budget))
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
            "review_on_p2_correct_n": int((review & ~wrong).sum()),
            "review_precision_vs_p2_error": float((review & wrong).sum() / review.sum()) if review.sum() else 0.0,
        }
    )
    return m


def score_dict(dev: pd.DataFrame, ext: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical
    hard_model = v30.make_models(numeric, categorical)["hard_logistic"]
    hard_dev, hard_ext = v30.oof_and_external_scores(dev, ext, features, hard_model)

    dev_scores: dict[str, np.ndarray] = {"learned_hard_gate": hard_dev}
    ext_scores: dict[str, np.ndarray] = {"learned_hard_gate": hard_ext}

    candidates = {
        "low_main_margin": lambda d: -d["main_margin_abs"].to_numpy(dtype=float),
        "low_robust_margin": lambda d: -d["robust_margin_abs"].to_numpy(dtype=float),
        "model_prob_std": lambda d: d["core_prob_std"].to_numpy(dtype=float),
        "model_prob_range": lambda d: d["core_prob_range"].to_numpy(dtype=float),
        "main_robust_abs_diff": lambda d: d["main_robust_abs_diff"].to_numpy(dtype=float),
        "safety_trigger_first": lambda d: d["safety_trigger"].to_numpy(dtype=float),
        "high_main_prob": lambda d: d["main_prob"].to_numpy(dtype=float),
        "low_main_prob": lambda d: -d["main_prob"].to_numpy(dtype=float),
        "low_score_margin_agree": lambda d: -d["score_margin_agree"].to_numpy(dtype=float),
        "high_core_mean": lambda d: d["prob_mean_core"].to_numpy(dtype=float),
        "low_core_mean": lambda d: -d["prob_mean_core"].to_numpy(dtype=float),
    }
    for name, fn in candidates.items():
        if all(c in dev.columns and c in ext.columns for c in required_cols(name)):
            dev_scores[name] = fn(dev)
            ext_scores[name] = fn(ext)
    return dev_scores, ext_scores


def required_cols(name: str) -> list[str]:
    mapping = {
        "low_main_margin": ["main_margin_abs"],
        "low_robust_margin": ["robust_margin_abs"],
        "model_prob_std": ["core_prob_std"],
        "model_prob_range": ["core_prob_range"],
        "main_robust_abs_diff": ["main_robust_abs_diff"],
        "safety_trigger_first": ["safety_trigger"],
        "high_main_prob": ["main_prob"],
        "low_main_prob": ["main_prob"],
        "low_score_margin_agree": ["score_margin_agree"],
        "high_core_mean": ["prob_mean_core"],
        "low_core_mean": ["prob_mean_core"],
    }
    return mapping.get(name, [])


def choose_budget_for_target(dev: pd.DataFrame, score: np.ndarray, target: float) -> tuple[float, dict[str, float | int]]:
    best_m = None
    for budget in BUDGET_GRID:
        m = metrics_with_review(dev, score, float(budget))
        if m["balanced_accuracy"] >= target:
            return float(budget), m
        best_m = m
    return float(BUDGET_GRID[-1]), best_m if best_m is not None else metrics_with_review(dev, score, float(BUDGET_GRID[-1]))


def random_baseline(df: pd.DataFrame, budget: float, rng: np.random.Generator) -> dict[str, float]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    n = len(df)
    k = int(round(n * budget))
    vals = []
    for _ in range(RANDOM_REPEATS):
        score = rng.random(n)
        m = metrics_with_review(df, score, float(k / n))
        vals.append(m)
    out: dict[str, float] = {}
    for key in ["accuracy", "balanced_accuracy", "captured_p2_wrong_rate", "review_precision_vs_p2_error", "fn", "fp"]:
        arr = np.asarray([v[key] for v in vals], dtype=float)
        out[f"random_{key}_mean"] = float(arr.mean())
        out[f"random_{key}_sd"] = float(arr.std(ddof=1))
    return out


def make_plots(fixed: pd.DataFrame, target: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    key_rankers = [
        "learned_hard_gate",
        "low_main_margin",
        "model_prob_std",
        "model_prob_range",
        "main_robust_abs_diff",
        "safety_trigger_first",
    ]
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for ranker in key_rankers:
        sub = fixed.loc[(fixed["ranker"].eq(ranker)) & (fixed["split"].eq("external"))].sort_values("budget")
        if sub.empty:
            continue
        ax.plot(sub["review_rate"] * 100, sub["balanced_accuracy"] * 100, marker="o", linewidth=1.8, label=ranker)
    ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("External review rate (%)")
    ax.set_ylabel("External balanced accuracy (%)")
    ax.set_title("Risk-ranker baselines under equal review burden")
    ax.set_xlim(-2, 82)
    ax.set_ylim(68, 98)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v40_ranker_fixed_budget_external_bacc.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v40_ranker_fixed_budget_external_bacc.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    focus = target.loc[target["target_dev_bacc"].isin([0.95, 0.97])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), sharey=True)
    for ax, target_value in zip(axes, [0.95, 0.97]):
        sub = focus.loc[focus["target_dev_bacc"].eq(target_value)].sort_values("external_balanced_accuracy", ascending=False)
        ax.scatter(sub["external_review_rate"] * 100, sub["external_balanced_accuracy"] * 100, s=66, color="#117a65", edgecolor="white")
        for _, row in sub.iterrows():
            ax.text(row["external_review_rate"] * 100 + 0.8, row["external_balanced_accuracy"] * 100, row["ranker"], fontsize=7)
        ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"Dev target BAcc {int(target_value * 100)}%")
        ax.set_xlabel("External review rate (%)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(-2, 82)
    axes[0].set_ylabel("External balanced accuracy (%)")
    axes[0].set_ylim(68, 98)
    fig.suptitle("Dev-target transfer by risk ranker", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v40_ranker_dev_target_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v40_ranker_dev_target_transfer.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pareto_frontier(fixed: pd.DataFrame) -> pd.DataFrame:
    pts = fixed.loc[(fixed["split"].eq("external")) & (~fixed["ranker"].eq("random_review"))].copy()
    pts = pts[
        [
            "ranker",
            "budget",
            "review_rate",
            "balanced_accuracy",
            "accuracy",
            "fn",
            "fp",
            "captured_p2_wrong_n",
            "missed_p2_wrong_n",
            "review_precision_vs_p2_error",
        ]
    ].copy()
    keep = []
    for idx, row in pts.iterrows():
        dominated = (
            (pts["review_rate"] <= row["review_rate"] + 1e-12)
            & (pts["balanced_accuracy"] >= row["balanced_accuracy"] - 1e-12)
            & (
                (pts["review_rate"] < row["review_rate"] - 1e-12)
                | (pts["balanced_accuracy"] > row["balanced_accuracy"] + 1e-12)
            )
        ).any()
        if not dominated:
            keep.append(idx)
    return pts.loc[keep].sort_values(["review_rate", "balanced_accuracy"]).reset_index(drop=True)


def efficiency_summary(fixed: pd.DataFrame) -> pd.DataFrame:
    ext = fixed.loc[(fixed["split"].eq("external")) & (~fixed["ranker"].eq("random_review"))].copy()
    rows = []
    for ranker, group in ext.groupby("ranker"):
        group = group.sort_values("review_rate")
        denom = group["review_rate"].max() - group["review_rate"].min()
        auc = np.trapz(group["balanced_accuracy"], group["review_rate"]) / denom if denom else np.nan

        def min_review_for(target: float) -> float:
            ok = group.loc[group["balanced_accuracy"].ge(target)]
            return float(ok["review_rate"].min()) if not ok.empty else np.nan

        rows.append(
            {
                "ranker": ranker,
                "auc_bacc_over_review_0_80": float(auc),
                "max_external_bacc": float(group["balanced_accuracy"].max()),
                "min_review_for_bacc90": min_review_for(0.90),
                "min_review_for_bacc93": min_review_for(0.93),
                "min_review_for_bacc95": min_review_for(0.95),
            }
        )
    return pd.DataFrame(rows).sort_values(["min_review_for_bacc90", "auc_bacc_over_review_0_80"], ascending=[True, False])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260527)
    dev = v30.load_development()
    ext = v30.load_external()
    dev_scores, ext_scores = score_dict(dev, ext)

    fixed_rows = []
    for ranker in dev_scores:
        for budget in BUDGET_GRID:
            dev_m = metrics_with_review(dev, dev_scores[ranker], float(budget))
            ext_m = metrics_with_review(ext, ext_scores[ranker], float(budget))
            fixed_rows.append({"ranker": ranker, "split": "development", "budget": float(budget), **dev_m})
            fixed_rows.append({"ranker": ranker, "split": "external", "budget": float(budget), **ext_m})

    # Add a stochastic random baseline for interpretability at the same budgets.
    for budget in BUDGET_GRID:
        ext_random = random_baseline(ext, float(budget), rng)
        fixed_rows.append({"ranker": "random_review", "split": "external", "budget": float(budget), "review_rate": float(round(len(ext) * budget) / len(ext)), **ext_random})

    target_rows = []
    for ranker in dev_scores:
        for target in TARGETS:
            budget, dev_m = choose_budget_for_target(dev, dev_scores[ranker], target)
            ext_m = metrics_with_review(ext, ext_scores[ranker], budget)
            target_rows.append(
                {
                    "ranker": ranker,
                    "target_dev_bacc": target,
                    "selected_budget": budget,
                    **{f"dev_{k}": v for k, v in dev_m.items()},
                    **{f"external_{k}": v for k, v in ext_m.items()},
                }
            )

    fixed = pd.DataFrame(fixed_rows)
    target = pd.DataFrame(target_rows)
    pareto = pareto_frontier(fixed)
    efficiency = efficiency_summary(fixed)
    fixed.to_csv(OUT_DIR / "v40_ranker_fixed_budget_metrics.csv", index=False, encoding="utf-8-sig")
    target.to_csv(OUT_DIR / "v40_ranker_dev_target_transfer_metrics.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(OUT_DIR / "v40_external_pareto_rankers.csv", index=False, encoding="utf-8-sig")
    efficiency.to_csv(OUT_DIR / "v40_ranker_efficiency_summary.csv", index=False, encoding="utf-8-sig")
    make_plots(fixed, target)

    show = target.loc[target["target_dev_bacc"].isin([0.95, 0.97]), [
        "ranker",
        "target_dev_bacc",
        "selected_budget",
        "external_review_rate",
        "external_balanced_accuracy",
        "external_accuracy",
        "external_fn",
        "external_fp",
        "external_captured_p2_wrong_n",
        "external_missed_p2_wrong_n",
    ]].sort_values(["target_dev_bacc", "external_balanced_accuracy"], ascending=[True, False])
    print(show.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
