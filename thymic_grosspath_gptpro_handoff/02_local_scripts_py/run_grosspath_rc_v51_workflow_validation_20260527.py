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


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v51_workflow_validation_20260527"
FIG_DIR = OUT_DIR / "figures"
N_BOOT_EXTERNAL = 5000
N_BOOT_SELECTION = 2000
SEED = 20260527


POLICIES = [
    {
        "policy": "P2_pure_auto",
        "display_name": "P2 纯自动安全切换",
        "type": "pure_auto",
        "kind": "none",
        "clinical_position": "不建议作为最终部署输出",
    },
    {
        "policy": "v37_balanced_dev97",
        "display_name": "v37 均衡风险控制",
        "type": "risk_control_balanced",
        "kind": "single_score",
        "score": "any",
        "budget": 0.600,
        "clinical_position": "中等复核负担，均衡控制 FN/FP",
    },
    {
        "policy": "v48_direction_dev97",
        "display_name": "v48 方向感知风险控制",
        "type": "direction_aware",
        "kind": "single_score",
        "score": "direction",
        "budget": 0.675,
        "clinical_position": "更重视错误方向，进一步减少高危漏诊",
    },
    {
        "policy": "v48_fn_high_safety",
        "display_name": "v48 FN 高安全版本",
        "type": "direction_aware_high_safety",
        "kind": "single_score",
        "score": "fn",
        "budget": 0.725,
        "clinical_position": "更高复核比例，优先捕获高危漏诊",
    },
    {
        "policy": "v50_sens95_spec90",
        "display_name": "v50 敏感性95档",
        "type": "sensitivity_first",
        "kind": "buffer",
        "base_budget": 0.475,
        "addon_candidate": "pred_low_fn",
        "addon_score": "fn",
        "addon_rate": 0.100,
        "clinical_position": "较低安全增强档",
    },
    {
        "policy": "v50_sens97_spec90",
        "display_name": "v50 敏感性97档",
        "type": "sensitivity_first",
        "kind": "buffer",
        "base_budget": 0.525,
        "addon_candidate": "pred_low_fn",
        "addon_score": "fn",
        "addon_rate": 0.175,
        "clinical_position": "高危漏诊优先档",
    },
    {
        "policy": "v50_sens98_spec90",
        "display_name": "v50 敏感性98档",
        "type": "sensitivity_first",
        "kind": "buffer",
        "base_budget": 0.525,
        "addon_candidate": "all_direction",
        "addon_score": "direction",
        "addon_rate": 0.200,
        "clinical_position": "当前最高安全档，外部 FN 最少",
    },
]


def make_review(df: pd.DataFrame, scores: dict[str, np.ndarray], policy: dict[str, object]) -> np.ndarray:
    kind = str(policy["kind"])
    if kind == "none":
        return np.zeros(len(df), dtype=bool)
    if kind == "single_score":
        return v30.top_budget(scores[str(policy["score"])], float(policy["budget"]))
    if kind == "buffer":
        base = v30.top_budget(scores["any"], float(policy["base_budget"]))
        return v50.add_top_candidates(
            df,
            base,
            scores[str(policy["addon_score"])],
            float(policy["addon_rate"]),
            str(policy["addon_candidate"]),
        )
    raise ValueError(kind)


def final_prediction(df: pd.DataFrame, review: np.ndarray) -> np.ndarray:
    y = df["label_idx"].to_numpy(dtype=int)
    p = df["p2_pred"].to_numpy(dtype=int).copy()
    p[review] = y[review]
    return p


def metric_row(df: pd.DataFrame, review: np.ndarray) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    p = final_prediction(df, review)
    m = v30.metrics_binary(y, p)
    masks = v48.error_masks(df)
    m.update(
        {
            "n": int(len(df)),
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
            "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
            "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
            "remaining_error_n": int((p != y).sum()),
            "review_precision_vs_p2_error": float((review & masks["any_wrong"]).sum() / review.sum()) if review.sum() else 0.0,
        }
    )
    return m


def stratified_boot_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def metric_from_arrays(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return v30.metrics_binary(np.asarray(y, dtype=int), np.asarray(pred, dtype=int))


def external_bootstrap_ci(ext: pd.DataFrame, final_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y = ext["label_idx"].to_numpy(dtype=int)
    rows = []
    for policy, pred in final_preds.items():
        vals = []
        for _ in range(N_BOOT_EXTERNAL):
            idx = stratified_boot_indices(y, rng)
            m = metric_from_arrays(y[idx], pred[idx])
            vals.append(
                {
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "fn": m["fn"],
                    "fp": m["fp"],
                }
            )
        boot = pd.DataFrame(vals)
        row = {"policy": policy, "n_boot": N_BOOT_EXTERNAL}
        for key in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "fn", "fp"]:
            row[f"{key}_median"] = float(boot[key].median())
            row[f"{key}_ci025"] = float(boot[key].quantile(0.025))
            row[f"{key}_ci975"] = float(boot[key].quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_grid_reviews(dev: pd.DataFrame, ext: pd.DataFrame, dev_scores: dict[str, np.ndarray], ext_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    base_budgets = np.round(np.arange(0.45, 0.625, 0.025), 3)
    addon_rates = np.round(np.arange(0.0, 0.225, 0.025), 3)
    candidates = [
        ("pred_low_fn", "fn"),
        ("pred_high_fp", "fp"),
        ("all_direction", "direction"),
    ]
    y_dev = dev["label_idx"].to_numpy(dtype=int)
    y_ext = ext["label_idx"].to_numpy(dtype=int)
    for base_budget in base_budgets:
        dev_base = v30.top_budget(dev_scores["any"], float(base_budget))
        ext_base = v30.top_budget(ext_scores["any"], float(base_budget))
        for addon_candidate, score_name in candidates:
            for addon_rate in addon_rates:
                dev_review = v50.add_top_candidates(dev, dev_base, dev_scores[score_name], float(addon_rate), addon_candidate)
                ext_review = v50.add_top_candidates(ext, ext_base, ext_scores[score_name], float(addon_rate), addon_candidate)
                dev_pred = final_prediction(dev, dev_review)
                ext_pred = final_prediction(ext, ext_review)
                dev_m = v30.metrics_binary(y_dev, dev_pred)
                ext_m = v30.metrics_binary(y_ext, ext_pred)
                rows.append(
                    {
                        "base_budget": float(base_budget),
                        "addon_candidate": addon_candidate,
                        "addon_score": score_name,
                        "addon_rate": float(addon_rate),
                        "dev_review_rate": float(dev_review.mean()),
                        "ext_review_rate": float(ext_review.mean()),
                        "dev_pred": dev_pred,
                        "ext_pred": ext_pred,
                        **{f"dev_{k}": v for k, v in dev_m.items()},
                        **{f"external_{k}": v for k, v in ext_m.items()},
                    }
                )
    return pd.DataFrame(rows)


def sensitivity_selection_stability(dev: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + 51)
    y_dev = dev["label_idx"].to_numpy(dtype=int)
    targets = [0.95, 0.97, 0.98]
    spec_min = 0.90
    rows = []
    preds = np.vstack(candidates["dev_pred"].to_list())
    for boot_id in range(N_BOOT_SELECTION):
        idx = stratified_boot_indices(y_dev, rng)
        y_b = y_dev[idx]
        boot_metrics = []
        for i in range(len(candidates)):
            m = v30.metrics_binary(y_b, preds[i, idx])
            boot_metrics.append((m["sensitivity"], m["specificity"], m["balanced_accuracy"], m["accuracy"]))
        bm = np.asarray(boot_metrics, dtype=float)
        for target in targets:
            ok_mask = (bm[:, 0] >= target) & (bm[:, 1] >= spec_min)
            tmp = candidates.copy()
            tmp["boot_sensitivity"] = bm[:, 0]
            tmp["boot_specificity"] = bm[:, 1]
            tmp["boot_bacc"] = bm[:, 2]
            tmp["boot_acc"] = bm[:, 3]
            if ok_mask.any():
                chosen = tmp.loc[ok_mask].sort_values(
                    ["dev_review_rate", "base_budget", "addon_rate", "boot_bacc"],
                    ascending=[True, True, True, False],
                ).iloc[0]
            else:
                chosen = tmp.sort_values(["boot_sensitivity", "boot_specificity", "boot_bacc"], ascending=[False, False, False]).iloc[0]
            rows.append(
                {
                    "boot_id": boot_id,
                    "target_sensitivity": float(target),
                    "specificity_min": spec_min,
                    "selected_base_budget": float(chosen["base_budget"]),
                    "selected_addon_candidate": chosen["addon_candidate"],
                    "selected_addon_rate": float(chosen["addon_rate"]),
                    "selected_dev_review_rate": float(chosen["dev_review_rate"]),
                    "boot_dev_sensitivity": float(chosen["boot_sensitivity"]),
                    "boot_dev_specificity": float(chosen["boot_specificity"]),
                    "boot_dev_bacc": float(chosen["boot_bacc"]),
                    "external_review_rate": float(chosen["ext_review_rate"]),
                    "external_sensitivity": float(chosen["external_sensitivity"]),
                    "external_specificity": float(chosen["external_specificity"]),
                    "external_balanced_accuracy": float(chosen["external_balanced_accuracy"]),
                    "external_accuracy": float(chosen["external_accuracy"]),
                    "external_fn": int(chosen["external_fn"]),
                    "external_fp": int(chosen["external_fp"]),
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows = []
    for target, g in detail.groupby("target_sensitivity"):
        summary_rows.append(
            {
                "target_sensitivity": float(target),
                "n_boot": int(len(g)),
                "selected_review_rate_median": float(g["external_review_rate"].median()),
                "selected_review_rate_ci025": float(g["external_review_rate"].quantile(0.025)),
                "selected_review_rate_ci975": float(g["external_review_rate"].quantile(0.975)),
                "external_bacc_median": float(g["external_balanced_accuracy"].median()),
                "external_bacc_ci025": float(g["external_balanced_accuracy"].quantile(0.025)),
                "external_bacc_ci975": float(g["external_balanced_accuracy"].quantile(0.975)),
                "external_sensitivity_median": float(g["external_sensitivity"].median()),
                "external_fn_median": float(g["external_fn"].median()),
                "p_external_bacc_ge_95": float((g["external_balanced_accuracy"] >= 0.95).mean()),
                "p_external_fn_le_1": float((g["external_fn"] <= 1).mean()),
                "most_common_rule": g[["selected_base_budget", "selected_addon_candidate", "selected_addon_rate"]]
                .astype(str)
                .agg("|".join, axis=1)
                .mode()
                .iloc[0],
            }
        )
    return detail, pd.DataFrame(summary_rows)


def make_case_table(ext: pd.DataFrame, review: np.ndarray, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    y = ext["label_idx"].to_numpy(dtype=int)
    final = final_prediction(ext, review)
    p2 = ext["p2_pred"].to_numpy(dtype=int)
    wrong = final != y
    err_dir = np.select(
        [(p2 != y) & (y == 1) & (p2 == 0), (p2 != y) & (y == 0) & (p2 == 1)],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )
    cols = [
        "case_id",
        "original_case_id",
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "image_name",
        "quality_status",
        "quality_score",
        "p2_pred",
        "main_prob",
        "robust_prob",
        "prob_mean_core",
    ]
    out = ext[[c for c in cols if c in ext.columns]].copy()
    out["v50_review"] = review.astype(int)
    out["v50_final_pred"] = final
    out["v50_final_correct"] = (final == y).astype(int)
    out["p2_error_direction"] = err_dir
    out["risk_any"] = scores["any"]
    out["risk_fn"] = scores["fn"]
    out["risk_fp"] = scores["fp"]
    out["risk_direction"] = scores["direction"]
    return out.loc[wrong].sort_values(["p2_error_direction", "task_l6_label", "original_case_id"]).reset_index(drop=True)


def make_plots(summary: pd.DataFrame, selection_summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = [
        "P2_pure_auto",
        "v37_balanced_dev97",
        "v48_direction_dev97",
        "v48_fn_high_safety",
        "v50_sens95_spec90",
        "v50_sens97_spec90",
        "v50_sens98_spec90",
    ]
    sub = summary.loc[summary["split"].eq("external")].set_index("policy").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(sub["review_rate"] * 100, sub["balanced_accuracy"] * 100, marker="o", linewidth=2.0, color="#117a65")
    for _, row in sub.iterrows():
        ax.text(row["review_rate"] * 100 + 0.7, row["balanced_accuracy"] * 100, row["policy"].replace("_", "\n"), fontsize=7)
    ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(95, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("External review / risk-control rate (%)")
    ax.set_ylabel("External balanced accuracy (%)")
    ax.set_title("Tiered risk-controlled workflow")
    ax.set_xlim(-2, 82)
    ax.set_ylim(66, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v51_tiered_workflow_external_bacc.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v51_tiered_workflow_external_bacc.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not selection_summary.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        x = np.arange(len(selection_summary))
        ax.errorbar(
            x,
            selection_summary["external_bacc_median"] * 100,
            yerr=[
                (selection_summary["external_bacc_median"] - selection_summary["external_bacc_ci025"]) * 100,
                (selection_summary["external_bacc_ci975"] - selection_summary["external_bacc_median"]) * 100,
            ],
            fmt="o",
            capsize=4,
            color="#1f618d",
        )
        ax.axhline(95, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Sens {int(v*100)}%" for v in selection_summary["target_sensitivity"]])
        ax.set_ylabel("External BAcc after bootstrapped dev selection (%)")
        ax.set_title("Sensitivity-first rule selection stability")
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "v51_sensitivity_selection_stability.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / "v51_sensitivity_selection_stability.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    summary_rows = []
    final_preds_ext: dict[str, np.ndarray] = {}
    reviews_ext: dict[str, np.ndarray] = {}

    for policy in POLICIES:
        for split, df, scores in [("development", dev, dev_scores), ("external", ext, ext_scores)]:
            review = make_review(df, scores, policy)
            if split == "external":
                final_preds_ext[str(policy["policy"])] = final_prediction(df, review)
                reviews_ext[str(policy["policy"])] = review
            row = {
                "policy": policy["policy"],
                "display_name": policy["display_name"],
                "type": policy["type"],
                "clinical_position": policy["clinical_position"],
                "split": split,
                **metric_row(df, review),
            }
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    ci = external_bootstrap_ci(ext, final_preds_ext)
    candidates = candidate_grid_reviews(dev, ext, dev_scores, ext_scores)
    # Store candidate metadata and metrics without array-valued columns.
    candidate_export = candidates.drop(columns=["dev_pred", "ext_pred"])
    selection_detail, selection_summary = sensitivity_selection_stability(dev, candidates)
    v50_errors = make_case_table(ext, reviews_ext["v50_sens98_spec90"], ext_scores)

    summary.to_csv(OUT_DIR / "v51_tiered_workflow_summary.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(OUT_DIR / "v51_external_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    candidate_export.to_csv(OUT_DIR / "v51_v50_candidate_grid_metrics.csv", index=False, encoding="utf-8-sig")
    selection_detail.to_csv(OUT_DIR / "v51_sensitivity_selection_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    selection_summary.to_csv(OUT_DIR / "v51_sensitivity_selection_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    v50_errors.to_csv(OUT_DIR / "v51_v50_sens98_remaining_errors.csv", index=False, encoding="utf-8-sig")
    make_plots(summary, selection_summary)

    show = summary.loc[summary["split"].eq("external")][
        [
            "policy",
            "review_rate",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "fn",
            "fp",
            "captured_fn_n",
            "captured_fp_n",
        ]
    ]
    print("External tiered workflow:")
    print(show.to_string(index=False))
    print("\nSensitivity-first selection bootstrap:")
    print(selection_summary.to_string(index=False))
    print("\nV50 sens98 remaining errors:")
    print(v50_errors[["original_case_id", "task_l6_label", "quality_status", "quality_score", "p2_error_direction", "p2_pred", "main_prob", "robust_prob", "image_name"]].to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
