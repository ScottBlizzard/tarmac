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


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v54_constrained_policy_search_20260527"
FIG_DIR = OUT_DIR / "figures"

BUDGETS = np.round(np.arange(0.35, 0.805, 0.025), 3)
BASE_BUDGETS = np.round(np.arange(0.35, 0.655, 0.025), 3)
ADDON_RATES = np.round(np.arange(0.0, 0.255, 0.025), 3)
SENS_TARGETS = [0.95, 0.97, 0.98, 0.985, 0.99]
SPEC_MINS = [0.90, 0.93, 0.95, 0.97]


def rank01(score: np.ndarray, group: np.ndarray | None = None) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    out = np.zeros(len(score), dtype=float)
    if group is None:
        group = np.zeros(len(score), dtype=int)
    for value in np.unique(group):
        idx = np.flatnonzero(group == value)
        if len(idx) <= 1:
            out[idx] = 0.0
            continue
        order = idx[np.argsort(score[idx], kind="mergesort")]
        vals = np.linspace(0.0, 1.0, len(idx))
        out[order] = vals
    return out


def metric_with_review(df: pd.DataFrame, review: np.ndarray) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    final = v51.final_prediction(df, review)
    m = v30.metrics_binary(y, final)
    masks = v48.error_masks(df)
    m.update(
        {
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
            "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
            "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
            "remaining_error_n": int((final != y).sum()),
            "review_precision_vs_p2_error": float((review & masks["any_wrong"]).sum() / review.sum()) if review.sum() else 0.0,
        }
    )
    return m


def top_by_score(score: np.ndarray, budget: float) -> np.ndarray:
    return v30.top_budget(np.asarray(score, dtype=float), float(budget))


def feature_scores(df: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pred = df["p2_pred"].to_numpy(dtype=int)
    out = {
        "rank_any": rank01(scores["any"]),
        "rank_fn": rank01(scores["fn"]),
        "rank_fp": rank01(scores["fp"]),
        "rank_direction": rank01(scores["direction"]),
        "rank_direction_by_pred": rank01(scores["direction"], pred),
        "pred_low_rank_fn": rank01(scores["fn"]) * (pred == 0),
        "pred_high_rank_fp": rank01(scores["fp"]) * (pred == 1),
    }
    return out


def build_score_candidates(dev: pd.DataFrame, ext: pd.DataFrame, dev_scores: dict[str, np.ndarray], ext_scores: dict[str, np.ndarray]) -> list[dict[str, object]]:
    dev_fs = feature_scores(dev, dev_scores)
    ext_fs = feature_scores(ext, ext_scores)
    candidates: list[dict[str, object]] = []

    single_names = ["rank_any", "rank_fn", "rank_fp", "rank_direction", "rank_direction_by_pred", "pred_low_rank_fn", "pred_high_rank_fp"]
    for name in single_names:
        candidates.append({"family": "single_rank", "name": name, "dev_score": dev_fs[name], "ext_score": ext_fs[name]})

    weight_grid = [
        ("any_plus_dir", {"rank_any": 0.5, "rank_direction": 0.5}),
        ("any_plus_pred_low_fn", {"rank_any": 0.6, "pred_low_rank_fn": 0.4}),
        ("any_plus_pred_low_fn_heavy", {"rank_any": 0.4, "pred_low_rank_fn": 0.6}),
        ("dir_plus_pred_low_fn", {"rank_direction": 0.5, "pred_low_rank_fn": 0.5}),
        ("dir_plus_lowfn_heavy", {"rank_direction": 0.3, "pred_low_rank_fn": 0.7}),
        ("lowfn_highfp", {"pred_low_rank_fn": 0.65, "pred_high_rank_fp": 0.35}),
        ("any_lowfn_highfp", {"rank_any": 0.4, "pred_low_rank_fn": 0.4, "pred_high_rank_fp": 0.2}),
        ("direction_lowfn_highfp", {"rank_direction": 0.4, "pred_low_rank_fn": 0.4, "pred_high_rank_fp": 0.2}),
        ("lowfn_very_heavy", {"rank_any": 0.2, "rank_direction": 0.2, "pred_low_rank_fn": 0.6}),
    ]
    for name, weights in weight_grid:
        dev_score = sum(float(w) * dev_fs[k] for k, w in weights.items())
        ext_score = sum(float(w) * ext_fs[k] for k, w in weights.items())
        candidates.append({"family": "fusion_rank", "name": name, "dev_score": dev_score, "ext_score": ext_score})

    return candidates


def build_policy_grid(dev: pd.DataFrame, ext: pd.DataFrame, dev_scores: dict[str, np.ndarray], ext_scores: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    reviews_ext: dict[str, np.ndarray] = {}

    for cand in build_score_candidates(dev, ext, dev_scores, ext_scores):
        for budget in BUDGETS:
            policy = f"{cand['family']}::{cand['name']}::budget={budget:.3f}"
            dev_review = top_by_score(cand["dev_score"], float(budget))
            ext_review = top_by_score(cand["ext_score"], float(budget))
            rows.append(
                {
                    "policy": policy,
                    "family": cand["family"],
                    "name": cand["name"],
                    "budget": float(budget),
                    "base_budget": np.nan,
                    "addon_candidate": "",
                    "addon_rate": np.nan,
                    **{f"dev_{k}": v for k, v in metric_with_review(dev, dev_review).items()},
                    **{f"external_{k}": v for k, v in metric_with_review(ext, ext_review).items()},
                }
            )
            reviews_ext[policy] = ext_review

    addon_defs = [
        ("pred_low_fn", "fn"),
        ("pred_high_fp", "fp"),
        ("all_direction", "direction"),
        ("all_any", "any"),
    ]
    for base_budget in BASE_BUDGETS:
        dev_base = top_by_score(dev_scores["any"], float(base_budget))
        ext_base = top_by_score(ext_scores["any"], float(base_budget))
        for addon_candidate, score_name in addon_defs:
            for addon_rate in ADDON_RATES:
                policy = f"buffer::{addon_candidate}::{base_budget:.3f}+{addon_rate:.3f}"
                dev_review = v50.add_top_candidates(dev, dev_base, dev_scores[score_name], float(addon_rate), "all_direction" if addon_candidate == "all_any" else addon_candidate)
                if addon_candidate == "all_any":
                    # all_any intentionally adds by the original hard-gate score, not by direction score.
                    dev_review = v50.add_top_candidates(dev, dev_base, dev_scores["any"], float(addon_rate), "all_direction")
                    ext_review = v50.add_top_candidates(ext, ext_base, ext_scores["any"], float(addon_rate), "all_direction")
                else:
                    ext_review = v50.add_top_candidates(ext, ext_base, ext_scores[score_name], float(addon_rate), addon_candidate)
                rows.append(
                    {
                        "policy": policy,
                        "family": "buffer",
                        "name": addon_candidate,
                        "budget": np.nan,
                        "base_budget": float(base_budget),
                        "addon_candidate": addon_candidate,
                        "addon_rate": float(addon_rate),
                        **{f"dev_{k}": v for k, v in metric_with_review(dev, dev_review).items()},
                        **{f"external_{k}": v for k, v in metric_with_review(ext, ext_review).items()},
                    }
                )
                reviews_ext[policy] = ext_review

    return pd.DataFrame(rows), reviews_ext


def select_by_dev_constraints(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sens_target in SENS_TARGETS:
        for spec_min in SPEC_MINS:
            ok = grid.loc[(grid["dev_sensitivity"] >= sens_target) & (grid["dev_specificity"] >= spec_min)].copy()
            if ok.empty:
                chosen = grid.sort_values(["dev_sensitivity", "dev_specificity", "dev_balanced_accuracy"], ascending=[False, False, False]).iloc[0]
                met = "no"
            else:
                chosen = ok.sort_values(
                    ["dev_review_rate", "dev_balanced_accuracy", "dev_specificity", "external_review_rate"],
                    ascending=[True, False, False, True],
                ).iloc[0]
                met = "yes"
            row = chosen.to_dict()
            row.update({"target_sensitivity": sens_target, "specificity_min": spec_min, "dev_constraints_met": met})
            rows.append(row)
    return pd.DataFrame(rows)


def external_oracle_frontier(grid: pd.DataFrame) -> pd.DataFrame:
    # Post-hoc external frontier is for upper-bound analysis only; never use it for selecting the proposed rule.
    rows = []
    for sens_target in SENS_TARGETS:
        for spec_min in SPEC_MINS:
            ok = grid.loc[(grid["external_sensitivity"] >= sens_target) & (grid["external_specificity"] >= spec_min)].copy()
            if ok.empty:
                continue
            chosen = ok.sort_values(["external_review_rate", "external_balanced_accuracy"], ascending=[True, False]).iloc[0]
            row = chosen.to_dict()
            row.update({"target_sensitivity": sens_target, "specificity_min": spec_min, "selection_type": "external_oracle_upper_bound"})
            rows.append(row)
    return pd.DataFrame(rows)


def case_table(ext: pd.DataFrame, policy: str, review: np.ndarray) -> pd.DataFrame:
    y = ext["label_idx"].to_numpy(dtype=int)
    p2 = ext["p2_pred"].to_numpy(dtype=int)
    final = v51.final_prediction(ext, review)
    wrong = final != y
    p2_wrong = p2 != y
    direction = np.select(
        [p2_wrong & (y == 1) & (p2 == 0), p2_wrong & (y == 0) & (p2 == 1)],
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
    out.insert(0, "policy", policy)
    out["review"] = review.astype(int)
    out["final_pred"] = final
    out["final_correct"] = (final == y).astype(int)
    out["p2_error_direction"] = direction
    return out.loc[wrong].sort_values(["p2_error_direction", "task_l6_label", "original_case_id"]).reset_index(drop=True)


def make_plots(grid: pd.DataFrame, selected: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for family, color in [("single_rank", "#566573"), ("fusion_rank", "#1f618d"), ("buffer", "#117a65")]:
        sub = grid.loc[grid["family"].eq(family)]
        ax.scatter(sub["external_review_rate"] * 100, sub["external_balanced_accuracy"] * 100, s=18, alpha=0.35, label=family, color=color)
    focus = selected.loc[(selected["target_sensitivity"].isin([0.98, 0.99])) & (selected["specificity_min"].eq(0.95))]
    ax.scatter(focus["external_review_rate"] * 100, focus["external_balanced_accuracy"] * 100, s=88, color="#c0392b", marker="*", label="dev-selected high sensitivity")
    ax.axhline(95, color="#7d6608", linestyle="--", linewidth=1)
    ax.set_xlabel("External review/control rate (%)")
    ax.set_ylabel("External BAcc (%)")
    ax.set_title("v54 constrained policy search")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(30, 85)
    ax.set_ylim(86, 100)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v54_policy_search_external_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v54_policy_search_external_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    grid, ext_reviews = build_policy_grid(dev, ext, dev_scores, ext_scores)
    selected = select_by_dev_constraints(grid)
    oracle = external_oracle_frontier(grid)

    grid.to_csv(OUT_DIR / "v54_policy_grid_metrics.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v54_dev_constraint_selected_policies.csv", index=False, encoding="utf-8-sig")
    oracle.to_csv(OUT_DIR / "v54_external_oracle_upper_bound.csv", index=False, encoding="utf-8-sig")
    make_plots(grid, selected)

    # Export residual cases for the most clinically relevant selected rule.
    focus = selected.loc[(selected["target_sensitivity"].eq(0.98)) & (selected["specificity_min"].eq(0.95))]
    if not focus.empty:
        policy = str(focus.iloc[0]["policy"])
        cases = case_table(ext, policy, ext_reviews[policy])
        cases.to_csv(OUT_DIR / "v54_selected_sens98_spec95_remaining_errors.csv", index=False, encoding="utf-8-sig")

    show = selected.loc[selected["specificity_min"].isin([0.95, 0.97])][
        [
            "target_sensitivity",
            "specificity_min",
            "policy",
            "family",
            "dev_review_rate",
            "dev_sensitivity",
            "dev_specificity",
            "external_review_rate",
            "external_balanced_accuracy",
            "external_sensitivity",
            "external_specificity",
            "external_fn",
            "external_fp",
        ]
    ].sort_values(["target_sensitivity", "specificity_min"])
    print("Dev-constrained selected policies:")
    print(show.to_string(index=False))
    if not oracle.empty:
        oracle_show = oracle.loc[oracle["specificity_min"].isin([0.95, 0.97])][
            [
                "target_sensitivity",
                "specificity_min",
                "policy",
                "family",
                "external_review_rate",
                "external_balanced_accuracy",
                "external_sensitivity",
                "external_specificity",
                "external_fn",
                "external_fp",
            ]
        ].sort_values(["target_sensitivity", "specificity_min"])
        print("\nExternal oracle upper bound (not for model selection):")
        print(oracle_show.to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
