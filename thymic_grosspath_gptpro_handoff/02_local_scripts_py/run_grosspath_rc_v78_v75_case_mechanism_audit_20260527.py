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
import run_grosspath_rc_v74_dev_prespecified_quality_gate_20260527 as v74  # noqa: E402
import run_grosspath_rc_v75_joint_quality_risk_ood_search_20260527 as v75  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v78_v75_case_mechanism_audit_20260527"
FIG_DIR = OUT_DIR / "figures"


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["domain"] = np.where(out["source_folder"].notna(), "third_batch", "old_data")
    return out


def v75_components(reference: pd.DataFrame, target: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    q_raw = v74.quality_proxy_risk(reference, target)
    lowconf_raw = 1.0 - np.minimum(
        pd.to_numeric(target["main_margin_abs"], errors="coerce").to_numpy(float),
        pd.to_numeric(target["robust_margin_abs"], errors="coerce").to_numpy(float),
    )
    q_rank = v75.rank01(q_raw)
    lowconf_rank = v75.rank01(lowconf_raw)
    joint = (q_rank + lowconf_rank) / 2
    return pd.DataFrame(
        {
            "quality_proxy_raw": q_raw,
            "quality_proxy_rank": q_rank,
            "lowconf_raw": lowconf_raw,
            "lowconf_rank": lowconf_rank,
            "v75_joint_score": joint,
        }
    )


def build_case_table(domain: str, reference: pd.DataFrame, target: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    base = v73.v50_review(target, scores)
    comp = v75_components(reference, target, scores)
    extra = v73.top_by_rate(comp["v75_joint_score"].to_numpy(float), 0.30) & (~base)
    v75_review = base | extra

    y = target["label_idx"].to_numpy(int)
    p2 = target["p2_pred"].to_numpy(int)
    v50_final = p2.copy()
    v50_final[base] = y[base]
    v75_final = p2.copy()
    v75_final[v75_review] = y[v75_review]

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
            "manual_quality_status_v1",
            "p2_pred",
            "main_prob",
            "robust_prob",
            "prob_mean_core",
            "main_margin_abs",
            "robust_margin_abs",
        ]
        if c in target.columns
    ]
    out = target[cols].copy()
    out.insert(0, "domain", domain)
    out["label_idx"] = y
    out["p2_wrong"] = (p2 != y).astype(int)
    out["error_direction"] = np.select(
        [(y == 1) & (p2 == 0), (y == 0) & (p2 == 1)],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )
    out["v50_control"] = base.astype(int)
    out["v75_extra_control"] = extra.astype(int)
    out["v75_control"] = v75_review.astype(int)
    out["v50_final_pred"] = v50_final
    out["v75_final_pred"] = v75_final
    out["v50_final_correct"] = (v50_final == y).astype(int)
    out["v75_final_correct"] = (v75_final == y).astype(int)
    out["rescued_by_v75"] = ((v50_final != y) & (v75_final == y)).astype(int)
    out = pd.concat([out.reset_index(drop=True), comp.reset_index(drop=True)], axis=1)
    out["v75_extra_reason"] = np.select(
        [
            out["v75_extra_control"].eq(1) & out["quality_proxy_rank"].ge(out["lowconf_rank"] + 0.15),
            out["v75_extra_control"].eq(1) & out["lowconf_rank"].ge(out["quality_proxy_rank"] + 0.15),
            out["v75_extra_control"].eq(1),
        ],
        ["quality_shift_dominant", "low_confidence_dominant", "quality_lowconf_mixed"],
        default="not_extra",
    )
    return out


def summarize(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, sub in case_df.groupby("domain", sort=False):
        v50_err = sub["v50_final_correct"].eq(0)
        v75_err = sub["v75_final_correct"].eq(0)
        extra = sub["v75_extra_control"].eq(1)
        rows.append(
            {
                "domain": domain,
                "n": int(len(sub)),
                "v50_control_rate": float(sub["v50_control"].mean()),
                "v75_control_rate": float(sub["v75_control"].mean()),
                "v75_extra_n": int(extra.sum()),
                "v75_extra_rate": float(extra.mean()),
                "v50_remaining_error_n": int(v50_err.sum()),
                "v75_remaining_error_n": int(v75_err.sum()),
                "rescued_by_v75_n": int(sub["rescued_by_v75"].sum()),
                "rescued_fn_n": int((sub["rescued_by_v75"].eq(1) & sub["error_direction"].eq("FN_high_to_low")).sum()),
                "rescued_fp_n": int((sub["rescued_by_v75"].eq(1) & sub["error_direction"].eq("FP_low_to_high")).sum()),
                "v75_extra_wrong_capture_n": int((extra & sub["p2_wrong"].eq(1)).sum()),
                "v75_extra_wrong_capture_rate": float((extra & sub["p2_wrong"].eq(1)).sum() / max(extra.sum(), 1)),
                "v75_extra_quality_dominant_n": int((extra & sub["v75_extra_reason"].eq("quality_shift_dominant")).sum()),
                "v75_extra_lowconf_dominant_n": int((extra & sub["v75_extra_reason"].eq("low_confidence_dominant")).sum()),
                "v75_extra_mixed_n": int((extra & sub["v75_extra_reason"].eq("quality_lowconf_mixed")).sum()),
            }
        )
    return pd.DataFrame(rows)


def make_plot(case_df: pd.DataFrame) -> None:
    v67.configure_matplotlib_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    domains = ["old_data", "third_batch", "strict_external"]
    colors = {
        "correct": "#95a5a6",
        "v50_error_remaining": "#c0392b",
        "rescued_by_v75": "#117a65",
    }
    for ax, domain in zip(axes, domains):
        sub = case_df.loc[case_df["domain"].eq(domain)].copy()
        status = np.where(
            sub["rescued_by_v75"].eq(1),
            "rescued_by_v75",
            np.where(sub["v50_final_correct"].eq(0), "v50_error_remaining", "correct"),
        )
        for key in ["correct", "v50_error_remaining", "rescued_by_v75"]:
            ss = sub.loc[status == key]
            ax.scatter(ss["quality_proxy_rank"], ss["lowconf_rank"], s=32, alpha=0.78, color=colors[key], label=key)
        ax.set_title(domain)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel("Quality proxy rank")
    axes[0].set_ylabel("Low-confidence rank")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG_DIR / "v78_v75_case_mechanism_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v78_v75_case_mechanism_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev = add_domain(dev)
    old_df, old_scores = v73.subset(dev, dev_scores, dev["domain"].eq("old_data").to_numpy())
    third_df, third_scores = v73.subset(dev, dev_scores, dev["domain"].eq("third_batch").to_numpy())

    case_tables = [
        build_case_table("old_data", third_df, old_df, old_scores),
        build_case_table("third_batch", old_df, third_df, third_scores),
        build_case_table("strict_external", dev, ext, ext_scores),
    ]
    case_df = pd.concat(case_tables, ignore_index=True)
    summary = summarize(case_df)

    case_df.to_csv(OUT_DIR / "v78_v75_case_mechanism_all_cases.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v78_v75_case_mechanism_summary.csv", index=False, encoding="utf-8-sig")
    focus = case_df.loc[
        case_df["v50_final_correct"].eq(0)
        | case_df["rescued_by_v75"].eq(1)
        | case_df["v75_final_correct"].eq(0)
        | case_df["v75_extra_control"].eq(1)
    ].copy()
    focus.to_csv(OUT_DIR / "v78_v50_v75_residual_extra_and_rescued_cases.csv", index=False, encoding="utf-8-sig")
    make_plot(case_df)

    print("V75 case mechanism summary:")
    print(summary.to_string(index=False))
    print("\nErrors/rescues:")
    cols = [
        c
        for c in [
            "domain",
            "case_id",
            "original_case_id",
            "task_l6_label",
            "task_l7_label",
            "error_direction",
            "v50_control",
            "v75_extra_control",
            "rescued_by_v75",
            "v50_final_correct",
            "v75_final_correct",
            "quality_proxy_rank",
            "lowconf_rank",
            "v75_joint_score",
            "v75_extra_reason",
            "image_name",
        ]
        if c in focus.columns
    ]
    print(focus[cols].sort_values(["domain", "rescued_by_v75", "v75_joint_score"], ascending=[True, False, False]).to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
