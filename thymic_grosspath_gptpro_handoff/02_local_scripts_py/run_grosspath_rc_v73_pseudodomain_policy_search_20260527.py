from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v48_directional_risk_controller_20260527 as v48  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v67_devfit_ood_quality_controller_20260527 as v67  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v73_pseudodomain_policy_search_20260527"
RATES = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]


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
    score = np.asarray(score, dtype=float)
    safe_score = np.nan_to_num(score, nan=-np.inf if high else np.inf)
    out = np.zeros(len(safe_score), dtype=bool)
    k = int(round(len(safe_score) * rate))
    if k <= 0:
        return out
    order = np.argsort(-safe_score if high else safe_score, kind="mergesort")
    out[order[: min(k, len(safe_score))]] = True
    return out


def evaluate(split: str, policy: str, df: pd.DataFrame, review: np.ndarray, extra: np.ndarray | None = None) -> dict[str, object]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    final = p2.copy()
    final[review] = y[review]
    m = v30.metrics_binary(y, final)
    masks = v48.error_masks(df)
    row: dict[str, object] = {
        "split": split,
        "policy": policy,
        "n": int(len(df)),
        "control_n": int(review.sum()),
        "control_rate": float(review.mean()),
        "auto_n": int((~review).sum()),
        "auto_rate": float((~review).mean()),
        "remaining_error_n": int((final != y).sum()),
        "auto_wrong_n": int(((~review) & masks["any_wrong"]).sum()),
        "auto_fn_n": int(((~review) & masks["fn_high_to_low"]).sum()),
        "auto_fp_n": int(((~review) & masks["fp_low_to_high"]).sum()),
        "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
        "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
        "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
    }
    row.update(m)
    if extra is not None:
        row.update(
            {
                "extra_control_n": int(extra.sum()),
                "extra_control_rate": float(extra.mean()),
                "extra_captured_wrong_n": int((extra & masks["any_wrong"]).sum()),
                "extra_captured_fn_n": int((extra & masks["fn_high_to_low"]).sum()),
                "extra_captured_fp_n": int((extra & masks["fp_low_to_high"]).sum()),
            }
        )
    return row


def candidate_scores(train_df: pd.DataFrame, target_df: pd.DataFrame, target_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {
        "risk_any": np.asarray(target_scores["any"], dtype=float),
        "risk_direction": np.asarray(target_scores["direction"], dtype=float),
    }
    if {"main_margin_abs", "robust_margin_abs"}.issubset(target_df.columns):
        main = pd.to_numeric(target_df["main_margin_abs"], errors="coerce").to_numpy(float)
        robust = pd.to_numeric(target_df["robust_margin_abs"], errors="coerce").to_numpy(float)
        out["low_confidence_min_margin"] = 1.0 - np.minimum(main, robust)
    if "main_robust_abs_diff" in target_df.columns:
        out["main_robust_disagreement"] = pd.to_numeric(target_df["main_robust_abs_diff"], errors="coerce").to_numpy(float)
    if "core_prob_range" in target_df.columns:
        out["core_model_range"] = pd.to_numeric(target_df["core_prob_range"], errors="coerce").to_numpy(float)
    if "quality_score" in target_df.columns:
        out["low_quality_score"] = -pd.to_numeric(target_df["quality_score"], errors="coerce").to_numpy(float)

    groups = v67.common_features(train_df, target_df)
    for group in ["image_common5", "model_prob_compact", "image_plus_model"]:
        cols = groups.get(group, [])
        if len(cols) < 2:
            continue
        try:
            _train_score, target_score = v67.mahalanobis_scores(train_df, target_df, cols)
            out[f"ood_{group}_mahalanobis"] = target_score
        except Exception as exc:
            print(f"[skip] {group}/mahalanobis: {exc}")
        try:
            _train_score, target_score = v67.isolation_scores(train_df, target_df, cols)
            out[f"ood_{group}_isolation"] = target_score
        except Exception as exc:
            print(f"[skip] {group}/isolation: {exc}")
    return out


def run_split(split: str, train_df: pd.DataFrame, target_df: pd.DataFrame, target_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    base = v50_review(target_df, target_scores)
    rows = [evaluate(split, "v50_base", target_df, base)]
    scores = candidate_scores(train_df, target_df, target_scores)
    for name, score in scores.items():
        for rate in RATES:
            extra = top_by_rate(score, rate) & (~base)
            review = base | extra
            row = evaluate(split, f"v50_plus_{name}_r{int(rate * 1000):03d}", target_df, review, extra)
            row["candidate"] = name
            row["rate"] = rate
            rows.append(row)
    out = pd.DataFrame(rows)
    out["candidate"] = out["candidate"].fillna("none")
    out["rate"] = out["rate"].fillna(0.0)
    return out


def pseudo_aggregate(pseudo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, rate), sub in pseudo.groupby(["candidate", "rate"], dropna=False):
        rows.append(
            {
                "candidate": candidate,
                "rate": float(rate),
                "pseudo_mean_control": float(sub["control_rate"].mean()),
                "pseudo_max_control": float(sub["control_rate"].max()),
                "pseudo_mean_bacc": float(sub["balanced_accuracy"].mean()),
                "pseudo_min_bacc": float(sub["balanced_accuracy"].min()),
                "pseudo_mean_acc": float(sub["accuracy"].mean()),
                "pseudo_min_sensitivity": float(sub["sensitivity"].min()),
                "pseudo_min_specificity": float(sub["specificity"].min()),
                "pseudo_total_fn": int(sub["fn"].sum()),
                "pseudo_total_fp": int(sub["fp"].sum()),
                "pseudo_total_remaining_error": int(sub["remaining_error_n"].sum()),
                "pseudo_total_extra_captured_wrong": int(sub.get("extra_captured_wrong_n", pd.Series(0, index=sub.index)).fillna(0).sum()),
            }
        )
    agg = pd.DataFrame(rows)
    agg["pseudo_utility"] = agg["pseudo_min_bacc"] - 0.035 * agg["pseudo_max_control"] - 0.005 * agg["pseudo_total_fn"]
    return agg.sort_values(["pseudo_min_bacc", "pseudo_mean_control"], ascending=[False, True]).reset_index(drop=True)


def mark_selected(agg: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    rules = [
        ("best_control_le080", agg["pseudo_max_control"].le(0.80)),
        ("best_control_le085", agg["pseudo_max_control"].le(0.85)),
        ("best_control_le090", agg["pseudo_max_control"].le(0.90)),
        ("best_utility_le085", agg["pseudo_max_control"].le(0.85)),
        ("best_fn_safety_le090", agg["pseudo_max_control"].le(0.90)),
    ]
    for tag, mask in rules:
        pool = agg.loc[mask].copy()
        if pool.empty:
            continue
        if tag == "best_utility_le085":
            pick = pool.sort_values(["pseudo_utility", "pseudo_mean_control"], ascending=[False, True]).head(1)
        elif tag == "best_fn_safety_le090":
            pick = pool.sort_values(["pseudo_total_fn", "pseudo_total_remaining_error", "pseudo_min_bacc", "pseudo_mean_control"], ascending=[True, True, False, True]).head(1)
        else:
            pick = pool.sort_values(["pseudo_min_bacc", "pseudo_mean_control"], ascending=[False, True]).head(1)
        tmp = pick.copy()
        tmp.insert(0, "selection_rule", tag)
        selected_rows.append(tmp)
    return pd.concat(selected_rows, ignore_index=True).drop_duplicates(["selection_rule", "candidate", "rate"])


def case_routes(split: str, policy: str, candidate: str, rate: float, df: pd.DataFrame, review: np.ndarray, extra: np.ndarray) -> pd.DataFrame:
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
    out.insert(0, "split", split)
    out.insert(1, "policy", policy)
    out["candidate"] = candidate
    out["rate"] = rate
    out["review_or_control"] = review.astype(int)
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


def run_selected_external(ext: pd.DataFrame, ext_scores: dict[str, np.ndarray], dev: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = v50_review(ext, ext_scores)
    scores = candidate_scores(dev, ext, ext_scores)
    rows = [evaluate("strict_external", "v50_base", ext, base)]
    cases = [case_routes("strict_external", "v50_base", "none", 0.0, ext, base, np.zeros(len(ext), dtype=bool))]
    for _, row in selected.iterrows():
        candidate = str(row["candidate"])
        rate = float(row["rate"])
        if candidate == "none":
            extra = np.zeros(len(ext), dtype=bool)
        elif candidate in scores:
            extra = top_by_rate(scores[candidate], rate) & (~base)
        else:
            continue
        review = base | extra
        policy = f"selected_{row['selection_rule']}__{candidate}_r{int(rate * 1000):03d}"
        metric = evaluate("strict_external", policy, ext, review, extra)
        metric["selection_rule"] = row["selection_rule"]
        metric["candidate"] = candidate
        metric["rate"] = rate
        rows.append(metric)
        cases.append(case_routes("strict_external", policy, candidate, rate, ext, review, extra))
    return pd.DataFrame(rows), pd.concat(cases, ignore_index=True)


def run_external_all_for_diagnosis(ext: pd.DataFrame, ext_scores: dict[str, np.ndarray], dev: pd.DataFrame) -> pd.DataFrame:
    base = v50_review(ext, ext_scores)
    rows = [evaluate("strict_external_exploratory", "v50_base", ext, base)]
    scores = candidate_scores(dev, ext, ext_scores)
    for name, score in scores.items():
        for rate in RATES:
            extra = top_by_rate(score, rate) & (~base)
            review = base | extra
            row = evaluate("strict_external_exploratory", f"v50_plus_{name}_r{int(rate * 1000):03d}", ext, review, extra)
            row["candidate"] = name
            row["rate"] = rate
            rows.append(row)
    out = pd.DataFrame(rows)
    out["candidate"] = out["candidate"].fillna("none")
    out["rate"] = out["rate"].fillna(0.0)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev = add_domain(dev)
    old_mask = dev["domain"].eq("old_data").to_numpy()
    third_mask = dev["domain"].eq("third_batch").to_numpy()
    old_df, old_scores = subset(dev, dev_scores, old_mask)
    third_df, third_scores = subset(dev, dev_scores, third_mask)

    pseudo_old = run_split("pseudo_third_to_old", third_df, old_df, old_scores)
    pseudo_third = run_split("pseudo_old_to_third", old_df, third_df, third_scores)
    pseudo = pd.concat([pseudo_old, pseudo_third], ignore_index=True)
    agg = pseudo_aggregate(pseudo)
    selected = mark_selected(agg)
    ext_eval, ext_cases = run_selected_external(ext, ext_scores, dev, selected)
    ext_all = run_external_all_for_diagnosis(ext, ext_scores, dev)

    pseudo.to_csv(OUT_DIR / "v73_pseudodomain_all_candidate_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "v73_pseudodomain_candidate_aggregate.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v73_pseudodomain_selected_policies.csv", index=False, encoding="utf-8-sig")
    ext_eval.to_csv(OUT_DIR / "v73_selected_policies_strict_external_eval.csv", index=False, encoding="utf-8-sig")
    ext_all.to_csv(OUT_DIR / "v73_all_candidates_strict_external_exploratory.csv", index=False, encoding="utf-8-sig")
    ext_cases.to_csv(OUT_DIR / "v73_selected_policies_strict_external_case_routes.csv", index=False, encoding="utf-8-sig")

    print("Pseudo-domain top 15:")
    print(
        agg[
            [
                "candidate",
                "rate",
                "pseudo_mean_control",
                "pseudo_max_control",
                "pseudo_min_bacc",
                "pseudo_mean_bacc",
                "pseudo_total_fn",
                "pseudo_total_fp",
                "pseudo_total_remaining_error",
                "pseudo_total_extra_captured_wrong",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )
    print("\nSelected external evaluation:")
    print(
        ext_eval[
            [
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
    print("\nStrict external exploratory top 12:")
    print(
        ext_all[
            [
                "candidate",
                "rate",
                "control_rate",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "fn",
                "fp",
                "remaining_error_n",
                "extra_captured_wrong_n",
            ]
        ]
        .sort_values(["balanced_accuracy", "control_rate"], ascending=[False, True])
        .head(12)
        .to_string(index=False)
    )
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
