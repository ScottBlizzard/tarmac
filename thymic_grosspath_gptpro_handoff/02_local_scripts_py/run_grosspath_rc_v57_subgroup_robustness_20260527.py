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
import run_grosspath_rc_v51_workflow_validation_20260527 as v51  # noqa: E402
import run_grosspath_rc_v54_constrained_policy_search_20260527 as v54  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v57_subgroup_robustness_20260527"
FIG_DIR = OUT_DIR / "figures"


POLICY_ORDER = [
    "P2_pure_auto",
    "v54_low_control_highsens",
    "v50_sens98_spec90",
    "v52_quality_le82",
    "v52_quality_le88",
]


def quality_bucket(score: object) -> str:
    try:
        x = float(score)
    except Exception:
        return "missing"
    if x <= 74:
        return "<=74"
    if x <= 82:
        return "75-82"
    if x <= 88:
        return "83-88"
    if x < 100:
        return "89-99"
    return "100"


def make_review_and_pred(
    ext: pd.DataFrame,
    ext_scores: dict[str, np.ndarray],
    ext_reviews_v54: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    y = ext["label_idx"].to_numpy(dtype=int)
    p2 = ext["p2_pred"].to_numpy(dtype=int)

    out["P2_pure_auto"] = {"review": np.zeros(len(ext), dtype=bool), "final": p2.copy(), "retake": np.zeros(len(ext), dtype=bool)}

    v54_policy = "fusion_rank::dir_plus_pred_low_fn::budget=0.650"
    review = ext_reviews_v54[v54_policy]
    out["v54_low_control_highsens"] = {"review": review, "final": v51.final_prediction(ext, review), "retake": np.zeros(len(ext), dtype=bool)}

    v50_policy = [p for p in v51.POLICIES if p["policy"] == "v50_sens98_spec90"][0]
    review50 = v51.make_review(ext, ext_scores, v50_policy)
    out["v50_sens98_spec90"] = {"review": review50, "final": v51.final_prediction(ext, review50), "retake": np.zeros(len(ext), dtype=bool)}

    for name, threshold in [("v52_quality_le82", 82.0), ("v52_quality_le88", 88.0)]:
        retake = (~review50) & ext["quality_score"].fillna(-1).le(threshold).to_numpy()
        final = v51.final_prediction(ext, review50)
        final[retake] = y[retake]
        out[name] = {"review": review50 | retake, "final": final, "retake": retake}

    return out


def subgroup_metrics(df: pd.DataFrame, final: np.ndarray, review: np.ndarray, group_by: str, group_value: object) -> dict[str, object]:
    sub = df.loc[df[group_by].astype(str).eq(str(group_value))]
    idx = sub.index.to_numpy()
    y = df.loc[idx, "label_idx"].to_numpy(dtype=int)
    pred = final[idx]
    rev = review[idx]
    n = len(idx)
    correct = pred == y
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    n_low = int((y == 0).sum())
    n_high = int((y == 1).sum())
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    bacc = (sensitivity + specificity) / 2 if not np.isnan(sensitivity) and not np.isnan(specificity) else np.nan
    return {
        "group_by": group_by,
        "group_value": str(group_value),
        "n": n,
        "n_low": n_low,
        "n_high": n_high,
        "accuracy": float(correct.mean()) if n else np.nan,
        "balanced_accuracy": bacc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "review_or_control_n": int(rev.sum()),
        "review_or_control_rate": float(rev.mean()) if n else np.nan,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "error_n": int((~correct).sum()),
    }


def build_subgroup_tables(ext: pd.DataFrame, policies: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = ext.copy().reset_index(drop=True)
    data["quality_score_bucket"] = data["quality_score"].map(quality_bucket)
    data["pred_group_p2"] = np.where(data["p2_pred"].astype(int).eq(1), "p2_pred_high", "p2_pred_low")
    group_cols = ["task_l7_label", "task_l6_label", "quality_status", "quality_score_bucket", "pred_group_p2"]

    rows = []
    case_frames = []
    y = data["label_idx"].to_numpy(dtype=int)
    p2 = data["p2_pred"].to_numpy(dtype=int)
    p2_wrong = p2 != y
    p2_direction = np.select(
        [p2_wrong & (y == 1) & (p2 == 0), p2_wrong & (y == 0) & (p2 == 1)],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )
    for policy, pack in policies.items():
        final = pack["final"]
        review = pack["review"]
        for group_col in group_cols:
            for value in sorted(data[group_col].dropna().astype(str).unique()):
                row = subgroup_metrics(data, final, review, group_col, value)
                row.insert if False else None
                row["policy"] = policy
                rows.append(row)

        wrong = final != y
        cols = [
            "case_id",
            "original_case_id",
            "source_folder",
            "task_l6_label",
            "task_l7_label",
            "image_name",
            "quality_status",
            "quality_score",
            "p2_pred",
            "main_prob",
            "robust_prob",
            "prob_mean_core",
        ]
        tmp = data.loc[wrong, cols].copy()
        tmp.insert(0, "policy", policy)
        tmp["final_pred"] = final[wrong]
        tmp["p2_error_direction"] = p2_direction[wrong]
        tmp["review_or_control"] = review[wrong].astype(int)
        case_frames.append(tmp)
    metrics = pd.DataFrame(rows)
    cases = pd.concat(case_frames, ignore_index=True) if case_frames else pd.DataFrame()
    return metrics, cases


def make_plots(metrics: pd.DataFrame, cases: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    l6 = metrics.loc[metrics["group_by"].eq("task_l6_label")].copy()
    pivot = l6.pivot_table(index="group_value", columns="policy", values="accuracy", aggfunc="first")
    pivot = pivot[[p for p in POLICY_ORDER if p in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    im = ax.imshow(pivot.to_numpy(dtype=float) * 100, cmap="YlGnBu", vmin=60, vmax=100)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                txt = "-"
            else:
                txt = f"{val * 100:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Subtype-level accuracy by final workflow")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v57_subtype_accuracy_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v57_subtype_accuracy_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    err = cases.groupby(["policy", "task_l6_label"], dropna=False).size().reset_index(name="error_n")
    all_l6 = sorted(cases["task_l6_label"].dropna().astype(str).unique())
    x = np.arange(len(all_l6))
    width = 0.16
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for k, policy in enumerate(POLICY_ORDER):
        sub = err.loc[err["policy"].eq(policy)].set_index("task_l6_label")
        vals = [int(sub.loc[v, "error_n"]) if v in sub.index else 0 for v in all_l6]
        ax.bar(x + (k - 2) * width, vals, width=width, label=policy)
    ax.set_xticks(x)
    ax.set_xticklabels(all_l6)
    ax.set_ylabel("External residual error count")
    ax.set_title("Residual errors by subtype")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v57_residual_errors_by_subtype.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v57_residual_errors_by_subtype.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_summary(metrics: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in POLICY_ORDER:
        sub_cases = cases.loc[cases["policy"].eq(policy)]
        l6 = sub_cases["task_l6_label"].value_counts(dropna=False)
        q = sub_cases["quality_status"].value_counts(dropna=False)
        rows.append(
            {
                "policy": policy,
                "residual_error_n": int(len(sub_cases)),
                "residual_by_l6": "; ".join([f"{k}:{v}" for k, v in l6.items()]),
                "residual_by_quality": "; ".join([f"{k}:{v}" for k, v in q.items()]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, _dev_scores, ext_scores = v50.get_scores()
    _grid, ext_reviews = v54.build_policy_grid(dev, ext, _dev_scores, ext_scores)
    policies = make_review_and_pred(ext.reset_index(drop=True), ext_scores, ext_reviews)
    metrics, cases = build_subgroup_tables(ext.reset_index(drop=True), policies)
    summary = make_summary(metrics, cases)

    metrics.to_csv(OUT_DIR / "v57_subgroup_metrics.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v57_residual_error_cases_by_policy.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v57_residual_error_summary.csv", index=False, encoding="utf-8-sig")
    make_plots(metrics, cases)

    print(summary.to_string(index=False))
    focus = metrics.loc[
        metrics["group_by"].isin(["task_l6_label", "quality_status"])
        & metrics["policy"].isin(["v54_low_control_highsens", "v50_sens98_spec90", "v52_quality_le82"])
    ][
        [
            "policy",
            "group_by",
            "group_value",
            "n",
            "accuracy",
            "review_or_control_rate",
            "error_n",
            "fn",
            "fp",
        ]
    ]
    print("\nFocused subgroup metrics:")
    print(focus.sort_values(["policy", "group_by", "group_value"]).to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
