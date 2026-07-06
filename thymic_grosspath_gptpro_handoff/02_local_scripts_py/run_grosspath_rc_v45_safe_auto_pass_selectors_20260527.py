from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v45_safe_auto_pass_selectors_20260527"
FIG_DIR = OUT_DIR / "figures"
COVERAGE_GRID = np.round(np.arange(0.05, 1.005, 0.025), 3)
TARGETS = [0.85, 0.88, 0.90, 0.92, 0.95]


def safe_scores(dev: pd.DataFrame, ext: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical
    hard = v30.make_models(numeric, categorical)["hard_logistic"]
    hard_dev, hard_ext = v30.oof_and_external_scores(dev, ext, features, hard)

    def p2_conf(d: pd.DataFrame, prob_col: str) -> np.ndarray:
        prob = d[prob_col].to_numpy(dtype=float)
        p2 = d["p2_pred"].to_numpy(dtype=int)
        return np.where(p2 == 1, prob, 1.0 - prob)

    dev_scores = {
        "safe_by_learned_hard_gate": -hard_dev,
        "safe_by_main_margin": dev["main_margin_abs"].to_numpy(dtype=float),
        "safe_by_robust_margin": dev["robust_margin_abs"].to_numpy(dtype=float),
        "safe_by_p2_main_confidence": p2_conf(dev, "main_prob"),
        "safe_by_p2_robust_confidence": p2_conf(dev, "robust_prob"),
        "safe_by_p2_core_confidence": p2_conf(dev, "prob_mean_core"),
        "safe_by_score_margin_agree": dev["score_margin_agree"].to_numpy(dtype=float),
        "safe_by_core_agreement": dev["core_agree_count"].to_numpy(dtype=float),
        "safe_by_quality_score": dev["quality_score"].to_numpy(dtype=float) if "quality_score" in dev.columns else np.zeros(len(dev)),
    }
    ext_scores = {
        "safe_by_learned_hard_gate": -hard_ext,
        "safe_by_main_margin": ext["main_margin_abs"].to_numpy(dtype=float),
        "safe_by_robust_margin": ext["robust_margin_abs"].to_numpy(dtype=float),
        "safe_by_p2_main_confidence": p2_conf(ext, "main_prob"),
        "safe_by_p2_robust_confidence": p2_conf(ext, "robust_prob"),
        "safe_by_p2_core_confidence": p2_conf(ext, "prob_mean_core"),
        "safe_by_score_margin_agree": ext["score_margin_agree"].to_numpy(dtype=float),
        "safe_by_core_agreement": ext["core_agree_count"].to_numpy(dtype=float),
        "safe_by_quality_score": ext["quality_score"].to_numpy(dtype=float) if "quality_score" in ext.columns else np.zeros(len(ext)),
    }
    return dev_scores, ext_scores


def select_top_safe(score: np.ndarray, coverage: float) -> np.ndarray:
    n = len(score)
    k = int(round(n * float(coverage)))
    flag = np.zeros(n, dtype=bool)
    if k <= 0:
        return flag
    order = np.argsort(-score, kind="mergesort")
    flag[order[:k]] = True
    return flag


def evaluate_auto_subset(df: pd.DataFrame, auto_flag: np.ndarray) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    review = ~auto_flag
    final = p2.copy()
    final[review] = y[review]
    auto_m = v30.metrics_binary(y[auto_flag], p2[auto_flag]) if auto_flag.any() else {
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
    }
    workflow = v30.metrics_binary(y, final)
    wrong = p2 != y
    return {
        "auto_n": int(auto_flag.sum()),
        "auto_rate": float(auto_flag.mean()),
        "review_n": int(review.sum()),
        "review_rate": float(review.mean()),
        "auto_accuracy": float(auto_m["accuracy"]),
        "auto_balanced_accuracy": float(auto_m["balanced_accuracy"]),
        "auto_sensitivity": float(auto_m["sensitivity"]),
        "auto_specificity": float(auto_m["specificity"]),
        "auto_fn": int(auto_m["fn"]),
        "auto_fp": int(auto_m["fp"]),
        "auto_wrong_n": int((auto_flag & wrong).sum()),
        "workflow_accuracy": float(workflow["accuracy"]),
        "workflow_balanced_accuracy": float(workflow["balanced_accuracy"]),
        "workflow_fn": int(workflow["fn"]),
        "workflow_fp": int(workflow["fp"]),
    }


def build_curves(df: pd.DataFrame, scores: dict[str, np.ndarray], split: str) -> pd.DataFrame:
    rows = []
    for selector, score in scores.items():
        for cov in COVERAGE_GRID:
            auto = select_top_safe(score, float(cov))
            rows.append({"split": split, "selector": selector, "coverage": float(cov), **evaluate_auto_subset(df, auto)})
    return pd.DataFrame(rows)


def select_by_dev(dev_curve: pd.DataFrame, ext_curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for selector in sorted(dev_curve["selector"].unique()):
        dsel = dev_curve.loc[dev_curve["selector"].eq(selector)].copy()
        esel = ext_curve.loc[ext_curve["selector"].eq(selector)].copy()
        for target in TARGETS:
            ok = dsel.loc[dsel["auto_accuracy"].ge(target)].copy()
            if ok.empty:
                continue
            # Maximize auto coverage while satisfying dev auto accuracy.
            chosen = ok.sort_values("auto_rate", ascending=False).iloc[0]
            ext = esel.loc[esel["coverage"].eq(chosen["coverage"])].iloc[0]
            rows.append(
                {
                    "selector": selector,
                    "target_dev_auto_accuracy": target,
                    "selected_coverage": float(chosen["coverage"]),
                    **{f"dev_{k}": v for k, v in chosen.items() if k not in ["split", "selector"]},
                    **{f"external_{k}": v for k, v in ext.items() if k not in ["split", "selector"]},
                }
            )
    return pd.DataFrame(rows)


def make_plots(curves: pd.DataFrame, selected: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ext = curves.loc[curves["split"].eq("external")].copy()
    key = [
        "safe_by_learned_hard_gate",
        "safe_by_p2_main_confidence",
        "safe_by_p2_core_confidence",
        "safe_by_score_margin_agree",
        "safe_by_core_agreement",
    ]
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for selector in key:
        sub = ext.loc[ext["selector"].eq(selector)].sort_values("auto_rate")
        if sub.empty:
            continue
        ax.plot(sub["auto_rate"] * 100, sub["auto_accuracy"] * 100, marker="o", linewidth=1.7, label=selector)
    ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(95, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("External auto-pass coverage (%)")
    ax.set_ylabel("External auto-pass accuracy (%)")
    ax.set_title("Safe auto-pass selector comparison")
    ax.set_xlim(0, 102)
    ax.set_ylim(45, 101)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v45_safe_auto_pass_selector_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v45_safe_auto_pass_selector_curves.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    focus = selected.loc[selected["target_dev_auto_accuracy"].isin([0.90, 0.95])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for ax, target in zip(axes, [0.90, 0.95]):
        sub = focus.loc[focus["target_dev_auto_accuracy"].eq(target)].copy()
        ax.scatter(sub["external_auto_rate"] * 100, sub["external_auto_accuracy"] * 100, s=62, color="#117a65", edgecolor="white")
        for _, row in sub.iterrows():
            ax.text(row["external_auto_rate"] * 100 + 0.8, row["external_auto_accuracy"] * 100, row["selector"].replace("safe_by_", ""), fontsize=7)
        ax.axhline(target * 100, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"Dev auto-pass Acc target {int(target * 100)}%")
        ax.set_xlabel("External auto-pass coverage (%)")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("External auto-pass accuracy (%)")
    axes[0].set_ylim(45, 101)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v45_dev_selected_safe_selector_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v45_dev_selected_safe_selector_transfer.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = v30.load_development()
    ext = v30.load_external()
    dev_scores, ext_scores = safe_scores(dev, ext)
    dev_curve = build_curves(dev, dev_scores, "development")
    ext_curve = build_curves(ext, ext_scores, "external")
    curves = pd.concat([dev_curve, ext_curve], ignore_index=True)
    selected = select_by_dev(dev_curve, ext_curve)
    curves.to_csv(OUT_DIR / "v45_safe_auto_pass_curves.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v45_dev_selected_safe_auto_pass_external_eval.csv", index=False, encoding="utf-8-sig")
    make_plots(curves, selected)

    show = selected.loc[selected["target_dev_auto_accuracy"].isin([0.90, 0.95]), [
        "selector",
        "target_dev_auto_accuracy",
        "selected_coverage",
        "external_auto_rate",
        "external_auto_accuracy",
        "external_auto_sensitivity",
        "external_auto_specificity",
        "external_auto_fn",
        "external_auto_fp",
        "external_workflow_balanced_accuracy",
    ]].sort_values(["target_dev_auto_accuracy", "external_auto_accuracy", "external_auto_rate"], ascending=[True, False, False])
    print(show.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
