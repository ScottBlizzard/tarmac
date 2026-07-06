from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v37_rank_normalized_risk_20260527"
FIG_DIR = OUT_DIR / "figures"
V35_DIR = ROOT / "outputs" / "grosspath_rc_v35_dev_targeted_review_budget_20260527"


def metric_under_oracle_review(df: pd.DataFrame, score: np.ndarray, budget: float) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
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
    return m, review, final


def make_cases(ext: pd.DataFrame, ext_score: np.ndarray, selected: pd.DataFrame) -> pd.DataFrame:
    y = ext["label_idx"].to_numpy(dtype=int)
    p2 = ext["p2_pred"].to_numpy(dtype=int)
    wrong = p2 != y
    frames = []
    for _, row in selected.iterrows():
        budget = float(row["dev_budget"])
        m, review, final = metric_under_oracle_review(ext, ext_score, budget)
        tmp = ext[
            [
                "case_id",
                "original_case_id",
                "source_folder",
                "task_l6_label",
                "task_l7_label",
                "label_idx",
                "image_name",
                "quality_status",
                "quality_score",
                "manual_quality_status_v1",
                "main_prob",
                "main_pred",
                "robust_prob",
                "robust_pred",
                "prob_mean_core",
                "p2_pred",
                "p2_wrong",
            ]
        ].copy()
        tmp.insert(0, "target_dev_bacc", row["target_dev_bacc"])
        tmp.insert(1, "rank_budget_from_dev", budget)
        tmp["external_review_rate_actual"] = m["review_rate"]
        tmp["hard_risk_score"] = ext_score
        tmp["hard_risk_rank_pct_external"] = pd.Series(ext_score).rank(method="first", ascending=False).to_numpy() / len(ext_score)
        tmp["review_flag"] = review.astype(int)
        tmp["final_pred_oracle_review"] = final
        tmp["final_correct_oracle_review"] = (final == y).astype(int)
        tmp["p2_error_direction"] = np.select(
            [wrong & (y == 1) & (p2 == 0), wrong & (y == 0) & (p2 == 1)],
            ["FN_high_to_low", "FP_low_to_high"],
            default="correct",
        )
        tmp["route_bucket"] = np.select(
            [review & wrong, review & ~wrong, ~review & wrong],
            ["captured_p2_error", "review_on_p2_correct", "missed_p2_error"],
            default="auto_correct",
        )
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


def subgroup_summary(cases: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for (target, value), g in cases.groupby(["target_dev_bacc", group_col], dropna=False):
        review = g["review_flag"].eq(1)
        wrong = g["p2_wrong"].astype(int).eq(1)
        rows.append(
            {
                "target_dev_bacc": target,
                "group_col": group_col,
                "group_value": value,
                "n": int(len(g)),
                "review_n": int(review.sum()),
                "review_rate": float(review.mean()),
                "p2_wrong_n": int(wrong.sum()),
                "captured_p2_wrong_n": int((review & wrong).sum()),
                "missed_p2_wrong_n": int((~review & wrong).sum()),
                "capture_rate": float((review & wrong).sum() / wrong.sum()) if wrong.sum() else np.nan,
                "review_precision_vs_p2_error": float((review & wrong).sum() / review.sum()) if review.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def score_distribution(dev_score: np.ndarray, ext_score: np.ndarray) -> pd.DataFrame:
    qs = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    rows = []
    for q in qs:
        rows.append(
            {
                "quantile": q,
                "dev_score": float(np.quantile(dev_score, q)),
                "external_score": float(np.quantile(ext_score, q)),
                "external_minus_dev": float(np.quantile(ext_score, q) - np.quantile(dev_score, q)),
            }
        )
    return pd.DataFrame(rows)


def make_plots(selected: pd.DataFrame, subgroup: pd.DataFrame, dist: pd.DataFrame, cases: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8.8, 5.2))
    x = selected["target_dev_bacc"] * 100
    ax1.plot(x, selected["external_balanced_accuracy"] * 100, marker="o", color="#117a65", linewidth=2, label="External BAcc")
    ax1.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Development target BAcc (%)")
    ax1.set_ylabel("External balanced accuracy (%)", color="#117a65")
    ax1.tick_params(axis="y", labelcolor="#117a65")
    ax1.set_ylim(68, 97)
    ax2 = ax1.twinx()
    ax2.plot(x, selected["external_review_rate"] * 100, marker="s", color="#c0392b", linewidth=2, label="External review rate")
    ax2.set_ylabel("External review rate (%)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(0, 80)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.set_title("Rank-normalized risk controller: target transfer")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v37_rank_controller_target_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v37_rank_controller_target_transfer.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(dist["quantile"], dist["dev_score"], marker="o", label="Development risk score", color="#2471a3")
    ax.plot(dist["quantile"], dist["external_score"], marker="s", label="External risk score", color="#c0392b")
    ax.set_xlabel("Score quantile")
    ax.set_ylabel("Hard-risk score")
    ax.set_title("Absolute hard-risk scores shift across domains")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v37_score_distribution_shift.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v37_score_distribution_shift.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    focus = cases.loc[cases["target_dev_bacc"].isin([0.95, 0.97])].copy()
    comp = (
        focus.groupby(["target_dev_bacc", "task_l6_label", "route_bucket"])
        .size()
        .reset_index(name="n")
        .pivot_table(index=["target_dev_bacc", "task_l6_label"], columns="route_bucket", values="n", fill_value=0)
        .reset_index()
    )
    for col in ["captured_p2_error", "review_on_p2_correct", "missed_p2_error", "auto_correct"]:
        if col not in comp.columns:
            comp[col] = 0
    labels = [f"{int(t * 100)}% {s}" for t, s in zip(comp["target_dev_bacc"], comp["task_l6_label"])]
    colors = {
        "captured_p2_error": "#117a65",
        "review_on_p2_correct": "#f5b041",
        "missed_p2_error": "#c0392b",
        "auto_correct": "#5dade2",
    }
    x_pos = np.arange(len(comp))
    bottom = np.zeros(len(comp))
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for col in ["captured_p2_error", "review_on_p2_correct", "missed_p2_error", "auto_correct"]:
        vals = comp[col].to_numpy(dtype=float)
        ax.bar(x_pos, vals, bottom=bottom, color=colors[col], label=col)
        bottom += vals
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Case count")
    ax.set_title("Route composition by subtype for high-safety targets")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v37_route_composition_by_subtype.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v37_route_composition_by_subtype.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = v30.load_development()
    ext = v30.load_external()
    selected_budget = pd.read_csv(V35_DIR / "v35_dev_target_selected_budgets_external_eval.csv")

    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical
    router = v30.make_models(numeric, categorical)["hard_logistic"]
    dev_score, ext_score = v30.oof_and_external_scores(dev, ext, features, router)

    rows = []
    for _, row in selected_budget.iterrows():
        m, _, _ = metric_under_oracle_review(ext, ext_score, float(row["dev_budget"]))
        rows.append(
            {
                "target_dev_bacc": float(row["target_dev_bacc"]),
                "rank_budget_from_dev": float(row["dev_budget"]),
                "dev_review_rate": float(row["dev_review_rate"]),
                "dev_balanced_accuracy": float(row["dev_balanced_accuracy"]),
                **{f"external_{k}": v for k, v in m.items()},
            }
        )
    selected = pd.DataFrame(rows)
    cases = make_cases(ext, ext_score, selected_budget)
    subgroup = pd.concat(
        [
            subgroup_summary(cases, "task_l6_label"),
            subgroup_summary(cases, "quality_status"),
            subgroup_summary(cases, "manual_quality_status_v1"),
            subgroup_summary(cases, "p2_error_direction"),
        ],
        ignore_index=True,
    )
    dist = score_distribution(dev_score, ext_score)

    selected.to_csv(OUT_DIR / "v37_rank_controller_selected_targets_external_eval.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v37_rank_controller_case_routes_external.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(OUT_DIR / "v37_rank_controller_subgroup_summary_external.csv", index=False, encoding="utf-8-sig")
    dist.to_csv(OUT_DIR / "v37_score_distribution_quantiles.csv", index=False, encoding="utf-8-sig")
    make_plots(selected, subgroup, dist, cases)

    show = selected[
        [
            "target_dev_bacc",
            "rank_budget_from_dev",
            "external_review_rate",
            "external_balanced_accuracy",
            "external_accuracy",
            "external_fn",
            "external_fp",
            "external_captured_p2_wrong_n",
            "external_missed_p2_wrong_n",
            "external_review_precision_vs_p2_error",
        ]
    ]
    print(show.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
