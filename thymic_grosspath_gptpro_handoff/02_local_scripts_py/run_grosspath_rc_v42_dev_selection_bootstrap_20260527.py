from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v42_dev_selection_bootstrap_20260527"
FIG_DIR = OUT_DIR / "figures"
BUDGET_GRID = np.round(np.arange(0.0, 0.805, 0.005), 3)
TARGETS = [0.90, 0.92, 0.95, 0.97]
N_BOOT = 2000
SEED = 20260527


def binary_metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float | int]:
    n = tn + fp + fn + tp
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sens / (precision + sens) if precision + sens else 0.0
    return {
        "accuracy": (tp + tn) / n if n else np.nan,
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def metrics_with_oracle_review(y: np.ndarray, p2: np.ndarray, score: np.ndarray, budget: float) -> dict[str, float | int]:
    wrong = p2 != y
    review = v30.top_budget(score, float(budget))
    final = p2.copy()
    final[review] = y[review]
    m = v30.metrics_binary(y, final)
    m.update(
        {
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "captured_wrong_n": int((review & wrong).sum()),
            "missed_wrong_n": int((~review & wrong).sum()),
            "review_precision_vs_error": float((review & wrong).sum() / review.sum()) if review.sum() else 0.0,
        }
    )
    return m


def budget_curve(y: np.ndarray, p2: np.ndarray, score: np.ndarray) -> dict[float, dict[str, float | int]]:
    y = np.asarray(y, dtype=int)
    p2 = np.asarray(p2, dtype=int)
    score = np.asarray(score, dtype=float)
    wrong = p2 != y
    order = np.argsort(-score, kind="mergesort")
    wrong_sorted = wrong[order]
    fn_wrong_sorted = wrong_sorted & (y[order] == 1) & (p2[order] == 0)
    fp_wrong_sorted = wrong_sorted & (y[order] == 0) & (p2[order] == 1)

    base_tn = int(((y == 0) & (p2 == 0)).sum())
    base_fp = int(((y == 0) & (p2 == 1)).sum())
    base_fn = int(((y == 1) & (p2 == 0)).sum())
    base_tp = int(((y == 1) & (p2 == 1)).sum())
    cum_fn = np.r_[0, np.cumsum(fn_wrong_sorted.astype(int))]
    cum_fp = np.r_[0, np.cumsum(fp_wrong_sorted.astype(int))]
    n = len(y)
    out: dict[float, dict[str, float | int]] = {}
    for budget in BUDGET_GRID:
        k = int(round(n * float(budget)))
        fn_caught = int(cum_fn[k])
        fp_caught = int(cum_fp[k])
        tn = base_tn + fp_caught
        fp = base_fp - fp_caught
        fn = base_fn - fn_caught
        tp = base_tp + fn_caught
        m = binary_metrics_from_counts(tn, fp, fn, tp)
        review = np.zeros(n, dtype=bool)
        if k:
            review[order[:k]] = True
        m.update(
            {
                "review_n": int(k),
                "review_rate": float(k / n) if n else 0.0,
                "captured_wrong_n": int(fn_caught + fp_caught),
                "missed_wrong_n": int(wrong.sum() - fn_caught - fp_caught),
                "review_precision_vs_error": float((fn_caught + fp_caught) / k) if k else 0.0,
            }
        )
        out[float(budget)] = m
    return out


def choose_budget_from_curve(curve: dict[float, dict[str, float | int]], target: float) -> tuple[float, dict[str, float | int]]:
    best_budget = float(BUDGET_GRID[-1])
    best_m = curve[best_budget]
    for budget in BUDGET_GRID:
        m = curve[float(budget)]
        if m["balanced_accuracy"] >= target:
            return float(budget), m
    return best_budget, best_m


def stratified_boot_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for target, g in rows.groupby("target_dev_bacc"):
        summary_rows.append(
            {
                "target_dev_bacc": target,
                "n_boot": len(g),
                "selected_budget_median": g["selected_budget"].median(),
                "selected_budget_ci025": g["selected_budget"].quantile(0.025),
                "selected_budget_ci975": g["selected_budget"].quantile(0.975),
                "external_bacc_median": g["external_balanced_accuracy"].median(),
                "external_bacc_ci025": g["external_balanced_accuracy"].quantile(0.025),
                "external_bacc_ci975": g["external_balanced_accuracy"].quantile(0.975),
                "external_acc_median": g["external_accuracy"].median(),
                "external_fn_median": g["external_fn"].median(),
                "external_fp_median": g["external_fp"].median(),
                "p_external_bacc_ge_90": float((g["external_balanced_accuracy"] >= 0.90).mean()),
                "p_external_bacc_ge_93": float((g["external_balanced_accuracy"] >= 0.93).mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def make_plots(rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for target, color in zip(TARGETS, ["#566573", "#2471a3", "#117a65", "#c0392b"]):
        g = rows.loc[rows["target_dev_bacc"].eq(target)]
        axes[0].hist(g["selected_budget"] * 100, bins=24, alpha=0.45, label=f"{int(target * 100)}%", color=color)
        axes[1].hist(g["external_balanced_accuracy"] * 100, bins=24, alpha=0.45, label=f"{int(target * 100)}%", color=color)
    axes[0].set_xlabel("Selected review budget from bootstrapped development (%)")
    axes[0].set_ylabel("Bootstrap count")
    axes[0].set_title("Development target -> selected review budget")
    axes[1].set_xlabel("External BAcc after applying selected budget (%)")
    axes[1].set_ylabel("Bootstrap count")
    axes[1].set_title("External performance induced by budget uncertainty")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(title="Dev target")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v42_bootstrap_budget_and_external_bacc.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v42_bootstrap_budget_and_external_bacc.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(summary))
    ax.errorbar(
        x,
        summary["external_bacc_median"] * 100,
        yerr=[
            (summary["external_bacc_median"] - summary["external_bacc_ci025"]) * 100,
            (summary["external_bacc_ci975"] - summary["external_bacc_median"]) * 100,
        ],
        fmt="o",
        capsize=4,
        color="#117a65",
    )
    ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(93, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t * 100)}%" for t in summary["target_dev_bacc"]])
    ax.set_xlabel("Development BAcc target")
    ax.set_ylabel("External BAcc median and 95% interval (%)")
    ax.set_title("Bootstrap stability of dev-selected review budget")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v42_bootstrap_external_bacc_ci_by_target.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v42_bootstrap_external_bacc_ci_by_target.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    dev = v30.load_development()
    ext = v30.load_external()
    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical
    model = v30.make_models(numeric, categorical)["hard_logistic"]
    dev_score, ext_score = v30.oof_and_external_scores(dev, ext, features, model)

    y_dev = dev["label_idx"].to_numpy(dtype=int)
    p2_dev = dev["p2_pred"].to_numpy(dtype=int)
    y_ext = ext["label_idx"].to_numpy(dtype=int)
    p2_ext = ext["p2_pred"].to_numpy(dtype=int)
    ext_curve = budget_curve(y_ext, p2_ext, ext_score)

    rows = []
    for boot_id in range(N_BOOT):
        idx = stratified_boot_indices(y_dev, rng)
        y_b = y_dev[idx]
        p2_b = p2_dev[idx]
        score_b = dev_score[idx]
        dev_curve = budget_curve(y_b, p2_b, score_b)
        for target in TARGETS:
            budget, dev_m = choose_budget_from_curve(dev_curve, target)
            ext_m = ext_curve[budget]
            rows.append(
                {
                    "boot_id": boot_id,
                    "target_dev_bacc": target,
                    "selected_budget": budget,
                    **{f"boot_dev_{k}": v for k, v in dev_m.items()},
                    **{f"external_{k}": v for k, v in ext_m.items()},
                }
            )

    out = pd.DataFrame(rows)
    summary = summarize(out)
    out.to_csv(OUT_DIR / "v42_bootstrap_selected_budget_external_eval.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v42_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    make_plots(out, summary)

    print(summary.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
