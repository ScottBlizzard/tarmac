from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v67_devfit_ood_quality_controller_20260527 as v67  # noqa: E402
import run_grosspath_rc_v73_pseudodomain_policy_search_20260527 as v73  # noqa: E402
import run_grosspath_rc_v79_lowrisk_protector_supervised_fp_20260527 as v79  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v80_tiered_lowrisk_guard_summary_20260527"
FIG_DIR = OUT_DIR / "figures"


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["domain"] = np.where(out["source_folder"].notna(), "third_batch", "old_data")
    return out


def lowrisk_guard_review(reference: pd.DataFrame, target: pd.DataFrame, scores: dict[str, np.ndarray], rate: float) -> tuple[np.ndarray, np.ndarray]:
    base = v79.v75_base_review(reference, target, scores)
    eligible = (~base) & (target["p2_pred"].to_numpy(int) == 1)
    score = v79.direct_scores(reference, target, scores)["direct_low_core_agree_count"]
    extra = v73.top_by_rate(v79.masked_score(score, eligible), rate) & (~base)
    return base | extra, extra


def evaluate(domain: str, policy: str, reference: pd.DataFrame, target: pd.DataFrame, scores: dict[str, np.ndarray], rate: float | None) -> tuple[dict[str, object], pd.DataFrame]:
    if policy == "v50_main":
        review = v73.v50_review(target, scores)
        extra = np.zeros(len(target), dtype=bool)
    elif policy == "v75_quality_lowconf":
        review = v79.v75_base_review(reference, target, scores)
        extra = np.zeros(len(target), dtype=bool)
    elif policy == "v79_light_lowrisk_guard":
        review, extra = lowrisk_guard_review(reference, target, scores, 0.025)
    elif policy == "v79_strict_lowrisk_guard":
        review, extra = lowrisk_guard_review(reference, target, scores, 0.075)
    else:
        raise ValueError(policy)
    row = v73.evaluate(domain, policy, target, review, extra)
    row["guard_rate"] = rate
    y = target["label_idx"].to_numpy(int)
    p2 = target["p2_pred"].to_numpy(int)
    final = p2.copy()
    final[review] = y[review]
    cols = [
        c
        for c in [
            "case_id",
            "original_case_id",
            "task_l6_label",
            "task_l7_label",
            "source_folder",
            "view_type_final",
            "image_name",
            "quality_score",
            "quality_status",
            "p2_pred",
            "main_prob",
            "robust_prob",
            "prob_mean_core",
            "core_agree_count",
        ]
        if c in target.columns
    ]
    cases = target[cols].copy()
    cases.insert(0, "domain", domain)
    cases.insert(1, "policy", policy)
    cases["guard_rate"] = rate
    cases["review_or_control"] = review.astype(int)
    cases["lowrisk_guard_extra"] = extra.astype(int)
    cases["label_idx"] = y
    cases["final_pred"] = final
    cases["final_correct"] = (final == y).astype(int)
    cases["p2_wrong"] = (p2 != y).astype(int)
    cases["error_direction"] = np.select(
        [(y == 1) & (p2 == 0), (y == 0) & (p2 == 1)],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )
    return row, cases


def make_plot(summary: pd.DataFrame) -> None:
    v67.configure_matplotlib_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    policies = ["v50_main", "v75_quality_lowconf", "v79_light_lowrisk_guard", "v79_strict_lowrisk_guard"]
    colors = {
        "v50_main": "#1f618d",
        "v75_quality_lowconf": "#117a65",
        "v79_light_lowrisk_guard": "#d68910",
        "v79_strict_lowrisk_guard": "#c0392b",
    }
    labels = {
        "v50_main": "v50 main",
        "v75_quality_lowconf": "v75 high-risk safety",
        "v79_light_lowrisk_guard": "v79 light low-risk guard",
        "v79_strict_lowrisk_guard": "v79 strict low-risk guard",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    for ax, domain in zip(axes, ["old_data", "third_batch", "strict_external"]):
        sub = summary.loc[summary["split"].eq(domain)]
        for _, row in sub.iterrows():
            ax.scatter(row["control_rate"] * 100, row["balanced_accuracy"] * 100, s=86, color=colors[row["policy"]], edgecolor="white")
            ax.text(row["control_rate"] * 100 + 0.3, row["balanced_accuracy"] * 100 + 0.05, labels[row["policy"]], fontsize=7)
        ax.axhline(95, color="#7d6608", linestyle="--", linewidth=1, alpha=0.65)
        ax.axhline(99, color="#7d6608", linestyle=":", linewidth=1, alpha=0.65)
        ax.set_title(domain)
        ax.set_xlabel("Control rate (%)")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_ylim(94, 101)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v80_tiered_lowrisk_guard_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v80_tiered_lowrisk_guard_tradeoff.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev = add_domain(dev)
    old_df, old_scores = v73.subset(dev, dev_scores, dev["domain"].eq("old_data").to_numpy())
    third_df, third_scores = v73.subset(dev, dev_scores, dev["domain"].eq("third_batch").to_numpy())
    specs = [
        ("old_data", third_df, old_df, old_scores),
        ("third_batch", old_df, third_df, third_scores),
        ("strict_external", dev, ext, ext_scores),
    ]
    policies = [
        ("v50_main", None),
        ("v75_quality_lowconf", None),
        ("v79_light_lowrisk_guard", 0.025),
        ("v79_strict_lowrisk_guard", 0.075),
    ]
    rows = []
    cases = []
    for domain, ref, target, scores in specs:
        for policy, rate in policies:
            row, case = evaluate(domain, policy, ref, target, scores, rate)
            rows.append(row)
            cases.append(case)
    summary = pd.DataFrame(rows)
    case_df = pd.concat(cases, ignore_index=True)
    summary.to_csv(OUT_DIR / "v80_tiered_lowrisk_guard_summary.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(OUT_DIR / "v80_tiered_lowrisk_guard_case_routes.csv", index=False, encoding="utf-8-sig")
    make_plot(summary)

    print("Tiered low-risk guard summary:")
    print(
        summary[
            [
                "split",
                "policy",
                "control_rate",
                "accuracy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "fn",
                "fp",
                "remaining_error_n",
                "extra_captured_wrong_n",
            ]
        ].to_string(index=False)
    )
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
