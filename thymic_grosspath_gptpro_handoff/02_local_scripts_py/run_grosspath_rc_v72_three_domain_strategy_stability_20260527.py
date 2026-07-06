from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v48_directional_risk_controller_20260527 as v48  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v67_devfit_ood_quality_controller_20260527 as v67  # noqa: E402
import run_grosspath_rc_v68_rank_ood_overlay_20260527 as v68  # noqa: E402
import run_grosspath_rc_v70_pseudo_domain_ood_validation_20260527 as v70  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v72_three_domain_strategy_stability_20260527"
FIG_DIR = OUT_DIR / "figures"
V65_FULL = ROOT / "outputs" / "grosspath_rc_v65_rank_normalized_conformal_autopass_20260527" / "v65_full_dev_rank_calibrated_external_eval.csv"


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["domain"] = np.where(out["source_folder"].notna(), "third_batch", "old_data")
    return out


def subset(df: pd.DataFrame, scores: dict[str, np.ndarray], mask: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    return df.loc[mask].reset_index(drop=True), {k: np.asarray(v)[mask] for k, v in scores.items()}


def v50_review(df: pd.DataFrame, scores: dict[str, np.ndarray]) -> np.ndarray:
    base = v30.top_budget(scores["any"], 0.525)
    return v50.add_top_candidates(df, base, scores["direction"], 0.200, "all_direction")


def top_by_rate(score: np.ndarray, rate: float, high: bool = True) -> np.ndarray:
    out = np.zeros(len(score), dtype=bool)
    k = int(round(len(score) * rate))
    if k <= 0:
        return out
    order = np.argsort(-score if high else score, kind="mergesort")
    out[order[: min(k, len(score))]] = True
    return out


def evaluate(domain: str, policy: str, df: pd.DataFrame, review: np.ndarray, note: str, route_extra: np.ndarray | None = None) -> dict[str, object]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    final = p2.copy()
    final[review] = y[review]
    m = v30.metrics_binary(y, final)
    masks = v48.error_masks(df)
    auto = ~review
    row: dict[str, object] = {
        "domain": domain,
        "policy": policy,
        "note": note,
        "n": int(len(df)),
        "control_n": int(review.sum()),
        "control_rate": float(review.mean()),
        "auto_n": int(auto.sum()),
        "auto_rate": float(auto.mean()),
        "auto_wrong_n": int((auto & masks["any_wrong"]).sum()),
        "auto_fn_n": int((auto & masks["fn_high_to_low"]).sum()),
        "auto_fp_n": int((auto & masks["fp_low_to_high"]).sum()),
        "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
        "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
        "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
        "remaining_error_n": int((final != y).sum()),
    }
    row.update(m)
    if route_extra is not None:
        row["extra_control_n"] = int(route_extra.sum())
        row["extra_control_rate"] = float(route_extra.mean())
        row["extra_captured_wrong_n"] = int((route_extra & masks["any_wrong"]).sum())
        row["extra_captured_fn_n"] = int((route_extra & masks["fn_high_to_low"]).sum())
        row["extra_captured_fp_n"] = int((route_extra & masks["fp_low_to_high"]).sum())
    return row


def case_routes(domain: str, policy: str, df: pd.DataFrame, review: np.ndarray, extra: np.ndarray | None = None) -> pd.DataFrame:
    y = df["label_idx"].to_numpy(int)
    p2 = df["p2_pred"].to_numpy(int)
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
        ]
        if c in df.columns
    ]
    out = df[cols].copy()
    out.insert(0, "domain", domain)
    out.insert(1, "policy", policy)
    out["review_or_control"] = review.astype(int)
    if extra is not None:
        out["extra_control"] = extra.astype(int)
    out["label_idx"] = y
    out["final_pred"] = final
    out["p2_wrong"] = (p2 != y).astype(int)
    out["final_correct"] = (final == y).astype(int)
    out["error_direction"] = np.select(
        [(y == 1) & (p2 == 0), (y == 0) & (p2 == 1)],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )
    return out


def image_mahalanobis_score(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    cols = v67.common_features(train, target)["image_common5"]
    _train_score, target_score = v67.mahalanobis_scores(train, target, cols)
    return target_score


def image_isolation_score(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    cols = v67.common_features(train, target)["image_common5"]
    _train_score, target_score = v67.isolation_scores(train, target, cols)
    return target_score


def build_domain_specs() -> list[dict[str, object]]:
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev = add_domain(dev)
    domains: list[dict[str, object]] = []
    for name in ["old_data", "third_batch"]:
        mask = dev["domain"].eq(name).to_numpy()
        df, scores = subset(dev, dev_scores, mask)
        train_name = "third_batch" if name == "old_data" else "old_data"
        train_mask = dev["domain"].eq(train_name).to_numpy()
        train_df, _ = subset(dev, dev_scores, train_mask)
        domains.append({"name": name, "df": df, "scores": scores, "ood_train": train_df})
    domains.append({"name": "strict_external", "df": ext, "scores": ext_scores, "ood_train": dev})
    return domains


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    domains = build_domain_specs()
    v65_cov = float(
        pd.read_csv(V65_FULL).loc[
            lambda d: d["target_auto_error"].eq(0.05) & d["selector"].eq("safe_by_any_risk"),
            "cal_auto_rate",
        ].iloc[0]
    )

    rows = []
    cases = []
    for spec in domains:
        name = str(spec["name"])
        df: pd.DataFrame = spec["df"]  # type: ignore[assignment]
        scores: dict[str, np.ndarray] = spec["scores"]  # type: ignore[assignment]
        train_df: pd.DataFrame = spec["ood_train"]  # type: ignore[assignment]

        none = np.zeros(len(df), dtype=bool)
        rows.append(evaluate(name, "P2_pure_auto", df, none, "pure automatic reference"))
        cases.append(case_routes(name, "P2_pure_auto", df, none))

        base = v50_review(df, scores)
        rows.append(evaluate(name, "v50_main", df, base, "main deployable risk-control workflow"))
        cases.append(case_routes(name, "v50_main", df, base))

        safe_auto = top_by_rate(-scores["any"], v65_cov, high=True)
        safe_review = ~safe_auto
        rows.append(evaluate(name, "v65_safe_auto_only", df, safe_review, "rank-calibrated safe auto-pass; all others controlled", route_extra=safe_review))
        cases.append(case_routes(name, "v65_safe_auto_only", df, safe_review, safe_review))

        maha = image_mahalanobis_score(train_df, df)
        maha_extra = top_by_rate(maha, 0.20, high=True) & (~base)
        maha_review = base | maha_extra
        rows.append(evaluate(name, "v71_image_mahalanobis20", df, maha_review, "pseudo-domain selected image OOD high-safety candidate", route_extra=maha_extra))
        cases.append(case_routes(name, "v71_image_mahalanobis20", df, maha_review, maha_extra))

        if name == "strict_external":
            q = pd.to_numeric(df["quality_score"], errors="coerce")
            q82 = q.le(82).fillna(False).to_numpy()
            q88 = q.le(88).fillna(False).to_numpy()
            rows.append(evaluate(name, "v50_quality82_exploratory", df, base | q82, "external quality-control exploratory overlay", route_extra=q82 & (~base)))
            rows.append(evaluate(name, "v50_quality88_exploratory", df, base | q88, "external quality-control exploratory overlay", route_extra=q88 & (~base)))
            cases.append(case_routes(name, "v50_quality82_exploratory", df, base | q82, q82 & (~base)))
            cases.append(case_routes(name, "v50_quality88_exploratory", df, base | q88, q88 & (~base)))

            iso = image_isolation_score(train_df, df)
            iso_extra = top_by_rate(iso, 0.075, high=True) & (~base)
            q82_iso_review = base | q82 | iso_extra
            q82_iso_extra = (q82 | iso_extra) & (~base)
            rows.append(
                evaluate(
                    name,
                    "v69_quality82_plus_iso075_exploratory",
                    df,
                    q82_iso_review,
                    "external-exposed quality + OOD complementarity upper candidate",
                    route_extra=q82_iso_extra,
                )
            )
            cases.append(case_routes(name, "v69_quality82_plus_iso075_exploratory", df, q82_iso_review, q82_iso_extra))

    summary = pd.DataFrame(rows)
    case_df = pd.concat(cases, ignore_index=True)
    summary.to_csv(OUT_DIR / "v72_three_domain_strategy_summary.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(OUT_DIR / "v72_three_domain_strategy_case_routes.csv", index=False, encoding="utf-8-sig")

    make_plot(summary)

    show_cols = [
        "domain",
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
    print(summary[[c for c in show_cols if c in summary.columns]].sort_values(["domain", "balanced_accuracy", "control_rate"], ascending=[True, False, True]).to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


def make_plot(summary: pd.DataFrame) -> None:
    v67.configure_matplotlib_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    label_map = {
        "P2_pure_auto": "A",
        "v50_main": "B",
        "v65_safe_auto_only": "C",
        "v71_image_mahalanobis20": "D",
        "v50_quality82_exploratory": "E",
        "v69_quality82_plus_iso075_exploratory": "F",
    }
    legend_map = {
        "A": "P2 pure auto",
        "B": "v50 main",
        "C": "v65 safe auto only",
        "D": "v71 image Mahalanobis 20%",
        "E": "v50 + quality<=82 (external exploratory)",
        "F": "quality<=82 + image OOD 7.5% (external exploratory)",
    }
    plot_df = summary.loc[
        summary["policy"].isin(
            [
                "P2_pure_auto",
                "v50_main",
                "v65_safe_auto_only",
                "v71_image_mahalanobis20",
                "v50_quality82_exploratory",
                "v69_quality82_plus_iso075_exploratory",
            ]
        )
    ].copy()
    domains = ["old_data", "third_batch", "strict_external"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    colors = {
        "P2_pure_auto": "#7f8c8d",
        "v50_main": "#1f618d",
        "v65_safe_auto_only": "#c0392b",
        "v71_image_mahalanobis20": "#d68910",
        "v50_quality82_exploratory": "#8e44ad",
        "v69_quality82_plus_iso075_exploratory": "#117a65",
    }
    for ax, domain in zip(axes, domains):
        sub = plot_df.loc[plot_df["domain"].eq(domain)].copy()
        for _, row in sub.iterrows():
            ax.scatter(row["control_rate"] * 100, row["balanced_accuracy"] * 100, s=84, color=colors.get(row["policy"], "#34495e"), edgecolor="white")
            ax.text(
                row["control_rate"] * 100 + 0.35,
                row["balanced_accuracy"] * 100 + 0.12,
                label_map.get(row["policy"], "?"),
                fontsize=9,
                weight="bold",
            )
        ax.axhline(95, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(99, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(domain)
        ax.set_xlabel("Control rate (%)")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_ylim(65, 101)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"{key}: {legend_map[key]}", markerfacecolor=colors[policy], markeredgecolor="white", markersize=8)
        for policy, key in label_map.items()
        if policy in set(plot_df["policy"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.055))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v72_three_domain_strategy_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v72_three_domain_strategy_tradeoff.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
