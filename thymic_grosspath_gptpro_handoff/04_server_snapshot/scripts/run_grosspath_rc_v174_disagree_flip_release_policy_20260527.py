from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v174_disagree_flip_release_policy_20260527"
V173_CASES = ROOT / "outputs" / "grosspath_rc_v173_image_only_review_corrector_20260527" / "v173_corrector_case_outputs.csv"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_cases() -> pd.DataFrame:
    df = pd.read_csv(V173_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["label_idx", "final_pred", "corrector_pred", "fold_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    for col in ["corrector_prob_high", "corrector_confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["v118_review_or_control", "v161_final_review_or_reject", "base_wrong", "corrector_correct"]:
        if col in df.columns:
            df[col] = as_bool(df[col])
    df["base_wrong"] = df["final_pred"].ne(df["label_idx"])
    df["corrector_disagrees_base"] = df["corrector_pred"].ne(df["final_pred"])
    return df


def policy_masks(g: pd.DataFrame, mode: str, flip_t: float, release_t: float) -> tuple[np.ndarray, np.ndarray]:
    review_policy = str(g["review_policy"].iloc[0])
    review = g[review_policy].to_numpy(bool)
    conf = g["corrector_confidence"].to_numpy(float)
    disagrees = g["corrector_disagrees_base"].to_numpy(bool)
    flip = review & disagrees & (conf >= flip_t)
    release = review & (~disagrees) & (conf >= release_t)
    if mode == "disagree_flip_only":
        release[:] = False
    elif mode == "agree_release_only":
        flip[:] = False
    elif mode == "flip_then_agree_release":
        pass
    else:
        raise ValueError(mode)
    return flip, release


def summarize_one(g: pd.DataFrame, mode: str, flip_t: float, release_t: float) -> list[dict[str, object]]:
    review_policy = str(g["review_policy"].iloc[0])
    model = str(g["model"].iloc[0])
    feature_set = str(g["feature_set"].iloc[0])
    y = g["label_idx"].to_numpy(int)
    base = g["final_pred"].to_numpy(int)
    corr = g["corrector_pred"].to_numpy(int)
    review = g[review_policy].to_numpy(bool)
    flip, release = policy_masks(g, mode, flip_t, release_t)
    auto_action = flip | release
    remaining_review = review & ~auto_action

    pred = base.copy()
    pred[flip] = corr[flip]
    pred[release] = base[release]
    pred[remaining_review] = y[remaining_review]

    wrong_auto = auto_action & pred.ne(y) if isinstance(pred, pd.Series) else auto_action & (pred != y)
    rescued = auto_action & (base != y) & (pred == y)
    hurt = auto_action & (base == y) & (pred != y)

    rows = []
    for scope, mask in [
        ("old_data", g["domain"].eq("old_data").to_numpy()),
        ("third_batch", g["domain"].eq("third_batch").to_numpy()),
        ("strict_external", g["domain"].eq("strict_external").to_numpy()),
        ("all_domains", g["domain"].isin(["old_data", "third_batch", "strict_external"]).to_numpy()),
    ]:
        m = metrics(y[mask], pred[mask], g.loc[mask, "corrector_prob_high"].to_numpy(float))
        rows.append(
            {
                "review_policy": review_policy,
                "model": model,
                "feature_set": feature_set,
                "mode": mode,
                "flip_threshold": float(flip_t),
                "release_threshold": float(release_t),
                "scope": scope,
                "n": int(mask.sum()),
                "original_review_n": int(review[mask].sum()),
                "original_review_rate": float(review[mask].mean()),
                "auto_flip_n": int(flip[mask].sum()),
                "auto_release_n": int(release[mask].sum()),
                "auto_action_n": int(auto_action[mask].sum()),
                "auto_action_rate": float(auto_action[mask].mean()),
                "remaining_review_n": int(remaining_review[mask].sum()),
                "remaining_review_rate": float(remaining_review[mask].mean()),
                "auto_action_error_n": int(wrong_auto[mask].sum()),
                "auto_action_error_rate": float(wrong_auto[mask].sum() / max(1, auto_action[mask].sum())),
                "rescued_n": int(rescued[mask].sum()),
                "hurt_n": int(hurt[mask].sum()),
                "accuracy": float(m["accuracy"]),
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "f1": float(m["f1"]),
                "auc": float(m["auc"]),
                "fn": int(m["fn"]),
                "fp": int(m["fp"]),
            }
        )
    return rows


def scan() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_cases()
    rows: list[dict[str, object]] = []
    modes = ["disagree_flip_only", "agree_release_only", "flip_then_agree_release"]
    for (review_policy, model), g in df.groupby(["review_policy", "model"], sort=False):
        for mode in modes:
            if mode == "disagree_flip_only":
                grid = [(t, 0.95) for t in THRESHOLDS]
            elif mode == "agree_release_only":
                grid = [(0.95, t) for t in THRESHOLDS]
            else:
                grid = list(product(THRESHOLDS, THRESHOLDS))
            for flip_t, release_t in grid:
                rows += summarize_one(g.copy(), mode, float(flip_t), float(release_t))
    summary = pd.DataFrame(rows)
    selected_rows = []
    for keys, sub in summary.groupby(["review_policy", "model", "mode"], sort=False):
        internal = sub.loc[sub["scope"].isin(["old_data", "third_batch"])].copy()
        agg = (
            internal.groupby(["flip_threshold", "release_threshold"], as_index=False)
            .agg(
                internal_auto_action_n=("auto_action_n", "sum"),
                internal_auto_flip_n=("auto_flip_n", "sum"),
                internal_auto_release_n=("auto_release_n", "sum"),
                internal_remaining_review_n=("remaining_review_n", "sum"),
                internal_auto_action_error_n=("auto_action_error_n", "sum"),
                internal_rescued_n=("rescued_n", "sum"),
                internal_hurt_n=("hurt_n", "sum"),
            )
        )
        zero = agg.loc[(agg["internal_auto_action_error_n"].eq(0)) & (agg["internal_auto_action_n"].ge(5))].copy()
        if not zero.empty:
            chosen = zero.sort_values(
                ["internal_remaining_review_n", "internal_rescued_n", "internal_auto_action_n"],
                ascending=[True, False, False],
            ).iloc[0]
            selection_status = "internal_zero_auto_error"
        else:
            agg["internal_auto_error_rate"] = agg["internal_auto_action_error_n"] / agg["internal_auto_action_n"].clip(lower=1)
            viable = agg.loc[agg["internal_auto_action_n"].ge(10)].copy()
            if viable.empty:
                continue
            chosen = viable.sort_values(
                ["internal_auto_error_rate", "internal_remaining_review_n", "internal_rescued_n"],
                ascending=[True, True, False],
            ).iloc[0]
            selection_status = "lowest_internal_auto_error_rate"
        selected_rows.append(
            {
                "review_policy": keys[0],
                "model": keys[1],
                "mode": keys[2],
                "flip_threshold": float(chosen["flip_threshold"]),
                "release_threshold": float(chosen["release_threshold"]),
                "selection_status": selection_status,
                "internal_auto_action_n": int(chosen["internal_auto_action_n"]),
                "internal_auto_flip_n": int(chosen["internal_auto_flip_n"]),
                "internal_auto_release_n": int(chosen["internal_auto_release_n"]),
                "internal_remaining_review_n": int(chosen["internal_remaining_review_n"]),
                "internal_auto_action_error_n": int(chosen["internal_auto_action_error_n"]),
                "internal_rescued_n": int(chosen["internal_rescued_n"]),
                "internal_hurt_n": int(chosen["internal_hurt_n"]),
            }
        )
    selected = pd.DataFrame(selected_rows)
    selected_detail = summary.merge(
        selected[["review_policy", "model", "mode", "flip_threshold", "release_threshold", "selection_status"]],
        on=["review_policy", "model", "mode", "flip_threshold", "release_threshold"],
        how="inner",
    )
    return summary, selected_detail


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, selected_detail = scan()
    summary.to_csv(OUT_DIR / "v174_flip_release_threshold_grid.csv", index=False, encoding="utf-8-sig")
    selected_detail.to_csv(OUT_DIR / "v174_selected_flip_release_summary.csv", index=False, encoding="utf-8-sig")

    # Rank selected policies by internal safety first, then by review reduction. External is reported, not used in selection.
    all_rows = selected_detail.loc[selected_detail["scope"].eq("all_domains")].copy()
    if not all_rows.empty:
        best = all_rows.sort_values(
            ["balanced_accuracy", "remaining_review_rate", "auto_action_error_n", "rescued_n"],
            ascending=[False, True, True, False],
        ).iloc[0]
    else:
        best = pd.Series(dtype=object)

    flip_focus = selected_detail.loc[
        selected_detail["scope"].eq("all_domains") & selected_detail["mode"].eq("disagree_flip_only")
    ].copy()
    if not flip_focus.empty:
        best_flip = flip_focus.sort_values(
            ["rescued_n", "hurt_n", "balanced_accuracy", "remaining_review_rate"],
            ascending=[False, True, False, True],
        ).iloc[0]
    else:
        best_flip = pd.Series(dtype=object)

    md = [
        "# v174 Disagreement Flip + Agreement Release Policy",
        "",
        "## Purpose",
        "",
        "v173 showed that a label corrector can release many reviewed cases but does not automatically rescue the primary model's wrong cases. v174 therefore separates two actions: disagreement-based flipping for true correction, and agreement-based release for review reduction.",
        "",
        "## Selection Boundary",
        "",
        "Thresholds are selected on old+third OOF rows only. Strict external rows are evaluated after selection and are not used for threshold choice.",
        "",
        "## Best Selected Policy",
        "",
    ]
    if not best.empty:
        md.append(
            f"- Best all-domain selected policy: `{best['mode']}` / `{best['model']}` / `{best['review_policy']}`, "
            f"flip_t={float(best['flip_threshold']):.2f}, release_t={float(best['release_threshold']):.2f}; "
            f"BAcc {100 * float(best['balanced_accuracy']):.2f}%, remaining review {100 * float(best['remaining_review_rate']):.2f}%, "
            f"rescued {int(best['rescued_n'])}, hurt {int(best['hurt_n'])}, auto-action errors {int(best['auto_action_error_n'])}."
        )
    if not best_flip.empty:
        md.append(
            f"- Best true flip-only policy: `{best_flip['model']}` / `{best_flip['review_policy']}`, "
            f"flip_t={float(best_flip['flip_threshold']):.2f}; rescued {int(best_flip['rescued_n'])}, "
            f"hurt {int(best_flip['hurt_n'])}, remaining review {100 * float(best_flip['remaining_review_rate']):.2f}%."
        )
    md += [
        "",
        "## Files",
        "",
        "- v174_flip_release_threshold_grid.csv",
        "- v174_selected_flip_release_summary.csv",
    ]
    (OUT_DIR / "v174_disagree_flip_release_policy.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "selected_rows": int(len(selected_detail)),
        "best_all_domain": best.to_dict() if not best.empty else None,
        "best_flip_only": best_flip.to_dict() if not best_flip.empty else None,
    }
    (OUT_DIR / "v174_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v174] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
