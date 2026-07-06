from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v34_branch_switch_reviewer_20260527"
BUDGETS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75]


PROB_COLS = [
    "prob_base162",
    "prob103_vitl",
    "prob107_qkvb",
    "prob_mean_core",
    "prob_stack_plain",
    "prob_stack_balanced",
]


def ensure_preds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    thresholds = {
        "prob_base162": 0.595,
        "prob103_vitl": 0.5,
        "prob107_qkvb": 0.5,
        "prob_mean_core": 0.5,
        "prob_stack_plain": 0.5,
        "prob_stack_balanced": 0.5,
    }
    for col, thr in thresholds.items():
        if col in out.columns:
            out[f"pred_from_{col}"] = (out[col].astype(float) >= thr).astype(int)
    return out


def committee_pred(df: pd.DataFrame, mode: str) -> np.ndarray:
    pred_cols = [f"pred_from_{c}" for c in PROB_COLS if f"pred_from_{c}" in df.columns]
    votes = df[pred_cols].to_numpy(dtype=int)
    if mode == "any_high":
        return (votes.sum(axis=1) >= 1).astype(int)
    if mode == "majority_high":
        return (votes.sum(axis=1) >= int(np.ceil(votes.shape[1] / 2))).astype(int)
    if mode == "all_high":
        return (votes.sum(axis=1) == votes.shape[1]).astype(int)
    if mode == "mean_prob_055":
        return (df[[c for c in PROB_COLS if c in df.columns]].mean(axis=1).to_numpy() >= 0.55).astype(int)
    if mode == "mean_prob_060":
        return (df[[c for c in PROB_COLS if c in df.columns]].mean(axis=1).to_numpy() >= 0.60).astype(int)
    raise ValueError(mode)


def metrics_with_switch(y: np.ndarray, p2: np.ndarray, review: np.ndarray, alt: np.ndarray) -> dict[str, float | int]:
    final = p2.copy()
    final[review] = alt[review]
    before_wrong = p2 != y
    after_wrong = final != y
    changed = final != p2
    rescued = before_wrong & ~after_wrong & changed
    harmed = ~before_wrong & after_wrong & changed
    m = v30.metrics_binary(y, final)
    m.update(
        {
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "changed_n": int(changed.sum()),
            "rescued_n": int(rescued.sum()),
            "harmed_n": int(harmed.sum()),
            "net_rescue_n": int(rescued.sum() - harmed.sum()),
            "p2_errors_in_review_n": int((before_wrong & review).sum()),
            "p2_errors_missed_by_review_n": int((before_wrong & ~review).sum()),
        }
    )
    return m


def build_alternatives(df: pd.DataFrame) -> dict[str, np.ndarray]:
    alts: dict[str, np.ndarray] = {}
    for col in PROB_COLS:
        pred_col = f"pred_from_{col}"
        if pred_col in df.columns:
            alts[pred_col] = df[pred_col].to_numpy(dtype=int)
    for mode in ["any_high", "majority_high", "all_high", "mean_prob_055", "mean_prob_060"]:
        alts[f"committee_{mode}"] = committee_pred(df, mode)
    # One-sided switches preserve the P2 output unless the alternative moves in
    # the specified direction. These are included because external errors are
    # clinically asymmetric.
    p2 = df["p2_pred"].to_numpy(dtype=int)
    for name, pred in list(alts.items()):
        high_only = p2.copy()
        high_only[(p2 == 0) & (pred == 1)] = 1
        alts[f"{name}__high_only"] = high_only
        low_only = p2.copy()
        low_only[(p2 == 1) & (pred == 0)] = 0
        alts[f"{name}__low_only"] = low_only
    return alts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dev = ensure_preds(v30.load_development())
    ext = ensure_preds(v30.load_external())

    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical
    router = v30.make_models(numeric, categorical)["hard_logistic"]
    router_oof, router_ext = v30.oof_and_external_scores(dev, ext, features, router)

    y_dev = dev["label_idx"].to_numpy(dtype=int)
    p2_dev = dev["p2_pred"].to_numpy(dtype=int)
    y_ext = ext["label_idx"].to_numpy(dtype=int)
    p2_ext = ext["p2_pred"].to_numpy(dtype=int)
    dev_alts = build_alternatives(dev)
    ext_alts = build_alternatives(ext)

    rows = []
    selected_rows = []
    case_frames = []
    for budget in BUDGETS:
        dev_review = v30.top_budget(router_oof, budget)
        ext_review = v30.top_budget(router_ext, budget)
        candidates = []
        for alt_name, dev_alt in dev_alts.items():
            dev_m = metrics_with_switch(y_dev, p2_dev, dev_review, dev_alt)
            row = {
                "budget": budget,
                "alt_policy": alt_name,
                **{f"dev_{k}": v for k, v in dev_m.items()},
            }
            row["dev_objective"] = (
                float(dev_m["balanced_accuracy"])
                + 0.001 * float(dev_m["net_rescue_n"])
                - 0.0003 * float(dev_m["changed_n"])
            )
            candidates.append(row)
            rows.append(row)

        cand = pd.DataFrame(candidates)
        best = cand.sort_values(
            ["dev_objective", "dev_balanced_accuracy", "dev_net_rescue_n", "dev_changed_n"],
            ascending=[False, False, False, True],
        ).iloc[0].to_dict()
        ext_alt = ext_alts[str(best["alt_policy"])]
        ext_m = metrics_with_switch(y_ext, p2_ext, ext_review, ext_alt)
        selected = {**best, **{f"external_{k}": v for k, v in ext_m.items()}}
        selected_rows.append(selected)

        final = p2_ext.copy()
        final[ext_review] = ext_alt[ext_review]
        before_wrong = p2_ext != y_ext
        after_wrong = final != y_ext
        changed = final != p2_ext
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
                "p2_pred",
                "p2_wrong",
                "main_prob",
                "robust_prob",
                "prob_mean_core",
            ]
        ].copy()
        tmp.insert(0, "budget", budget)
        tmp.insert(1, "alt_policy", best["alt_policy"])
        tmp["router_hard_score"] = router_ext
        tmp["review_flag"] = ext_review.astype(int)
        tmp["alt_pred"] = ext_alt
        tmp["final_pred"] = final
        tmp["changed_flag"] = changed.astype(int)
        tmp["bucket"] = np.select(
            [
                changed & before_wrong & ~after_wrong,
                changed & ~before_wrong & after_wrong,
                ext_review & before_wrong & after_wrong,
                ext_review & ~before_wrong,
                ~ext_review & before_wrong,
            ],
            ["rescued", "harmed", "review_still_wrong", "review_kept_correct", "missed_p2_error"],
            default="auto_correct",
        )
        case_frames.append(tmp)

    all_df = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected_rows)
    cases = pd.concat(case_frames, ignore_index=True)
    all_df.to_csv(OUT_DIR / "v34_all_dev_branch_switch_candidates.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(OUT_DIR / "v34_selected_dev_branch_switch_external_eval.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v34_selected_branch_switch_case_routes_external.csv", index=False, encoding="utf-8-sig")

    show_cols = [
        "budget",
        "alt_policy",
        "dev_balanced_accuracy",
        "dev_net_rescue_n",
        "dev_changed_n",
        "external_balanced_accuracy",
        "external_accuracy",
        "external_fn",
        "external_fp",
        "external_net_rescue_n",
        "external_changed_n",
        "external_review_rate",
    ]
    print(selected_df[show_cols].sort_values(["external_balanced_accuracy", "external_changed_n"], ascending=[False, True]).to_string(index=False))
    p2 = v30.metrics_binary(y_ext, p2_ext)
    print(f"\nP2 external BAcc={p2['balanced_accuracy']:.4f}, Acc={p2['accuracy']:.4f}, FN={p2['fn']}, FP={p2['fp']}")
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
