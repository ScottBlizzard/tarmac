from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v36_threshold_transfer_20260527"
FIG_DIR = OUT_DIR / "figures"
V35_DIR = ROOT / "outputs" / "grosspath_rc_v35_dev_targeted_review_budget_20260527"
TARGETS = [0.85, 0.88, 0.90, 0.92, 0.95, 0.97]


def review_by_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores >= threshold, dtype=bool)


def oracle_review_metrics(df: pd.DataFrame, scores: np.ndarray, review: np.ndarray) -> dict[str, float | int]:
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
            "risk_threshold_min_score": float(np.min(scores[review])) if review.any() else float("inf"),
            "captured_p2_wrong_n": int((review & wrong).sum()),
            "captured_p2_wrong_rate": float((review & wrong).sum() / wrong.sum()) if wrong.sum() else 0.0,
            "missed_p2_wrong_n": int((~review & wrong).sum()),
            "review_on_p2_correct_n": int((review & ~wrong).sum()),
            "review_precision_vs_p2_error": float((review & wrong).sum() / review.sum()) if review.sum() else 0.0,
        }
    )
    return m


def choose_thresholds_from_dev(dev: pd.DataFrame, dev_scores: np.ndarray) -> pd.DataFrame:
    candidate_thresholds = np.r_[np.inf, np.sort(np.unique(dev_scores))[::-1], -np.inf]
    rows = []
    for thr in candidate_thresholds:
        review = review_by_threshold(dev_scores, float(thr))
        m = oracle_review_metrics(dev, dev_scores, review)
        rows.append({"threshold": float(thr), **m})
    curve = pd.DataFrame(rows)
    selected = []
    for target in TARGETS:
        ok = curve.loc[curve["balanced_accuracy"].ge(target)].copy()
        if ok.empty:
            continue
        # Minimal review burden; tie-break by higher threshold.
        best = ok.sort_values(["review_rate", "threshold"], ascending=[True, False]).iloc[0].to_dict()
        best["target_dev_bacc"] = target
        selected.append(best)
    return curve, pd.DataFrame(selected)


def add_case_routes(ext: pd.DataFrame, ext_scores: np.ndarray, selected: pd.DataFrame) -> pd.DataFrame:
    frames = []
    y = ext["label_idx"].to_numpy(dtype=int)
    p2 = ext["p2_pred"].to_numpy(dtype=int)
    wrong = p2 != y
    for _, row in selected.iterrows():
        review = review_by_threshold(ext_scores, float(row["threshold"]))
        final = p2.copy()
        final[review] = y[review]
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
        tmp.insert(1, "threshold", row["threshold"])
        tmp["hard_risk_score"] = ext_scores
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
        total = len(g)
        review = g["review_flag"].eq(1)
        wrong = g["p2_wrong"].astype(int).eq(1)
        rows.append(
            {
                "target_dev_bacc": target,
                "group_col": group_col,
                "group_value": value,
                "n": total,
                "review_n": int(review.sum()),
                "review_rate": float(review.mean()),
                "p2_wrong_n": int(wrong.sum()),
                "captured_p2_wrong_n": int((review & wrong).sum()),
                "missed_p2_wrong_n": int((~review & wrong).sum()),
                "p2_wrong_capture_rate": float((review & wrong).sum() / wrong.sum()) if wrong.sum() else np.nan,
                "review_precision_vs_p2_error": float((review & wrong).sum() / review.sum()) if review.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_plots(selected_eval: pd.DataFrame, comparison: pd.DataFrame, cases: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(
        selected_eval["external_review_rate"] * 100,
        selected_eval["external_balanced_accuracy"] * 100,
        marker="o",
        color="#117a65",
        linewidth=2,
        label="Dev-threshold transfer",
    )
    ax.plot(
        comparison["external_review_rate_budget"] * 100,
        comparison["external_balanced_accuracy_budget"] * 100,
        marker="s",
        color="#c0392b",
        linewidth=2,
        label="Dev-budget transfer",
    )
    for _, row in selected_eval.iterrows():
        if row["target_dev_bacc"] in [0.90, 0.95, 0.97]:
            ax.text(
                row["external_review_rate"] * 100 + 0.8,
                row["external_balanced_accuracy"] * 100,
                f"{row['target_dev_bacc'] * 100:.0f}%",
                fontsize=8,
                color="#145a32",
            )
    ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("External review / risk-control rate (%)")
    ax.set_ylabel("External balanced accuracy (%)")
    ax.set_title("Threshold-based transfer vs fixed-budget transfer")
    ax.set_xlim(-2, 82)
    ax.set_ylim(68, 98)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v36_threshold_vs_budget_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v36_threshold_vs_budget_transfer.pdf", dpi=300, bbox_inches="tight")
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
    x = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(11, 5.6))
    bottom = np.zeros(len(comp))
    colors = {
        "captured_p2_error": "#117a65",
        "review_on_p2_correct": "#f5b041",
        "missed_p2_error": "#c0392b",
        "auto_correct": "#5dade2",
    }
    for col in ["captured_p2_error", "review_on_p2_correct", "missed_p2_error", "auto_correct"]:
        vals = comp[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=col, color=colors[col])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Case count")
    ax.set_title("External route composition by subtype under selected thresholds")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v36_external_route_composition_by_subtype.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v36_external_route_composition_by_subtype.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = v30.load_development()
    ext = v30.load_external()

    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical
    router = v30.make_models(numeric, categorical)["hard_logistic"]
    dev_scores, ext_scores = v30.oof_and_external_scores(dev, ext, features, router)

    dev_curve, selected_dev = choose_thresholds_from_dev(dev, dev_scores)
    eval_rows = []
    for _, row in selected_dev.iterrows():
        ext_review = review_by_threshold(ext_scores, float(row["threshold"]))
        ext_m = oracle_review_metrics(ext, ext_scores, ext_review)
        eval_rows.append({**{f"dev_{k}": v for k, v in row.items()}, **{f"external_{k}": v for k, v in ext_m.items()}})
    selected_eval = pd.DataFrame(eval_rows)

    cases = add_case_routes(ext, ext_scores, selected_dev)
    subgroup = pd.concat(
        [
            subgroup_summary(cases, "task_l6_label"),
            subgroup_summary(cases, "quality_status"),
            subgroup_summary(cases, "manual_quality_status_v1"),
            subgroup_summary(cases, "p2_error_direction"),
        ],
        ignore_index=True,
    )

    budget = pd.read_csv(V35_DIR / "v35_dev_target_selected_budgets_external_eval.csv")
    comparison = selected_eval.merge(
        budget[
            [
                "target_dev_bacc",
                "external_review_rate",
                "external_balanced_accuracy",
                "external_accuracy",
                "external_fn",
                "external_fp",
            ]
        ].rename(
            columns={
                "external_review_rate": "external_review_rate_budget",
                "external_balanced_accuracy": "external_balanced_accuracy_budget",
                "external_accuracy": "external_accuracy_budget",
                "external_fn": "external_fn_budget",
                "external_fp": "external_fp_budget",
            }
        ),
        left_on="dev_target_dev_bacc",
        right_on="target_dev_bacc",
        how="left",
    )
    comparison = comparison.rename(columns={"dev_target_dev_bacc": "target_dev_bacc"})

    dev_curve.to_csv(OUT_DIR / "v36_dev_threshold_curve.csv", index=False, encoding="utf-8-sig")
    selected_eval.to_csv(OUT_DIR / "v36_selected_thresholds_external_eval.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v36_threshold_case_routes_external.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(OUT_DIR / "v36_threshold_subgroup_summary_external.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "v36_threshold_vs_budget_comparison.csv", index=False, encoding="utf-8-sig")

    make_plots(
        selected_eval.rename(columns={"dev_target_dev_bacc": "target_dev_bacc"}),
        comparison,
        cases,
    )

    show = selected_eval[
        [
            "dev_target_dev_bacc",
            "dev_threshold",
            "dev_review_rate",
            "dev_balanced_accuracy",
            "external_review_rate",
            "external_balanced_accuracy",
            "external_accuracy",
            "external_fn",
            "external_fp",
            "external_captured_p2_wrong_n",
            "external_missed_p2_wrong_n",
        ]
    ].copy()
    print(show.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
