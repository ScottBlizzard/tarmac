from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics, safe_ap, safe_auc
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v199_directional_error_flip_corrector_20260527"
V185_CASES = ROOT / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527" / "v185_unlabeled_shift_adaptive_cases.csv"
V182_CASES = ROOT / "outputs" / "grosspath_rc_v182_stable_fixed_image_agreement_release_20260527" / "v182_stable_fixed_case_outputs.csv"
V145_WIDE = ROOT / "outputs" / "grosspath_rc_v145_fusion_router_20260527" / "v145_fusion_router_scores_wide.csv"
V173_CASES = ROOT / "outputs" / "grosspath_rc_v173_image_only_review_corrector_20260527" / "v173_corrector_case_outputs.csv"


DIRECTIONS = {
    "fn_high_to_low": {"base_pred": 0, "flip_pred": 1},
    "fp_low_to_high": {"base_pred": 1, "flip_pred": 0},
}
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.985, 0.99, 0.995]


def slug(x: object) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(x)).strip("_")


def load_base() -> pd.DataFrame:
    base = pd.read_csv(V185_CASES, dtype={"case_id": str, "original_case_id": str})
    fold = pd.read_csv(V182_CASES, dtype={"case_id": str, "original_case_id": str})[
        ["case_id", "fold_id"]
    ].copy()
    base = base.merge(fold, on="case_id", how="left", validate="one_to_one")
    for col in ["fixed_v118_review", "fixed_v182_review", "adaptive_review", "adaptive_auto_decision"]:
        base[col] = as_bool(base[col])
    for col in ["label_idx", "final_pred", "fold_id"]:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(-1).astype(int)
    base["prob_mean_core"] = pd.to_numeric(base["prob_mean_core"], errors="coerce")
    base["base_wrong"] = base["final_pred"].ne(base["label_idx"])
    base["base_confidence"] = np.where(base["final_pred"].eq(1), base["prob_mean_core"], 1.0 - base["prob_mean_core"])
    base["base_uncertainty"] = 1.0 - base["base_confidence"]
    p = np.clip(base["prob_mean_core"].astype(float), 1e-6, 1 - 1e-6)
    base["base_entropy"] = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return base


def attach_v145_features(base: pd.DataFrame) -> pd.DataFrame:
    wide = pd.read_csv(V145_WIDE, dtype={"case_id": str})
    feature_cols = [
        "base_pred",
        "base_prob",
        "single:low_conf",
        "single:v143_pca_directional",
        "single:v143_pca_any",
        "single:v143_extra_directional",
        "single:v144_concept_directional",
        "single:v144_concept_plus_base_directional",
        "single:v144_concept_any",
        "fusion:rank_mean_lowconf_v143_v144",
        "fusion:selected_internal_budget03",
    ]
    out = base.copy()
    for model, g in wide.groupby("base_model", sort=False):
        keep = ["case_id"] + [c for c in feature_cols if c in g.columns]
        renamed = g[keep].copy()
        renamed = renamed.rename(columns={c: f"v145_{slug(model)}_{slug(c)}" for c in keep if c != "case_id"})
        out = out.merge(renamed, on="case_id", how="left", validate="one_to_one")
    return out


def attach_v173_features(base: pd.DataFrame) -> pd.DataFrame:
    corr = pd.read_csv(V173_CASES, dtype={"case_id": str})
    corr["corrector_prob_high"] = pd.to_numeric(corr["corrector_prob_high"], errors="coerce")
    corr["corrector_confidence"] = pd.to_numeric(corr["corrector_confidence"], errors="coerce")
    corr["corrector_pred"] = pd.to_numeric(corr["corrector_pred"], errors="coerce")
    pieces = []
    for (review_policy, model), g in corr.groupby(["review_policy", "model"], sort=False):
        key = f"v173_{slug(review_policy)}_{slug(model)}"
        tmp = g[["case_id", "corrector_prob_high", "corrector_confidence", "corrector_pred"]].copy()
        tmp = tmp.rename(
            columns={
                "corrector_prob_high": f"{key}_prob_high",
                "corrector_confidence": f"{key}_confidence",
                "corrector_pred": f"{key}_pred",
            }
        )
        pieces.append(tmp)
    out = base.copy()
    for tmp in pieces:
        out = out.merge(tmp, on="case_id", how="left", validate="one_to_one")
    pred_cols = [c for c in out.columns if c.startswith("v173_") and c.endswith("_pred")]
    for c in pred_cols:
        out[f"{c}_disagree_v185"] = pd.to_numeric(out[c], errors="coerce").ne(out["final_pred"]).astype(float)
    return out


def make_dataset() -> tuple[pd.DataFrame, list[str]]:
    df = attach_v173_features(attach_v145_features(load_base()))
    exclude = {
        "domain",
        "case_id",
        "original_case_id",
        "task_l6_label",
        "label_idx",
        "final_pred",
        "fold_id",
        "image_name",
        "gate_domain_decision",
        "adaptive_policy_branch",
        "base_wrong",
    }
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return df, feature_cols


def make_models(seed: int) -> dict[str, object]:
    return {
        "logreg_c03": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        ),
        "logreg_c1": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        ),
        "extra_d3": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=320,
                max_depth=3,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        "extra_d5": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=420,
                max_depth=5,
                min_samples_leaf=6,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed + 1,
                n_jobs=-1,
            ),
        ),
    }


def fit_direction_scores(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df[["domain", "case_id", "label_idx", "final_pred", "fold_id", "adaptive_review"]].copy()
    quality_rows = []
    x = df[feature_cols].to_numpy(float)
    internal = df["domain"].isin(["old_data", "third_batch"]).to_numpy()
    strict = df["domain"].eq("strict_external").to_numpy()
    train_pool = internal & df["adaptive_review"].to_numpy(bool)
    folds = df["fold_id"].to_numpy(int)
    models = make_models(20260527)
    for direction, spec in DIRECTIONS.items():
        candidate_mask = df["final_pred"].eq(spec["base_pred"]).to_numpy()
        target = (
            df["adaptive_review"].to_numpy(bool)
            & candidate_mask
            & df["label_idx"].eq(spec["flip_pred"]).to_numpy()
        ).astype(int)
        direction_train = train_pool & candidate_mask
        for model_name, model in models.items():
            score = np.zeros(len(df), dtype=float)
            if direction_train.any():
                fallback = float(target[direction_train].mean())
            else:
                fallback = 0.0
            score[:] = fallback
            for fold in sorted(np.unique(folds[internal])):
                test = internal & df["adaptive_review"].to_numpy(bool) & candidate_mask & (folds == fold)
                train = direction_train & (folds != fold)
                if not test.any():
                    continue
                if len(np.unique(target[train])) < 2:
                    score[test] = fallback
                    continue
                clf = clone(model)
                clf.fit(x[train], target[train])
                score[test] = clf.predict_proba(x[test])[:, 1]
            if strict.any() and len(np.unique(target[direction_train])) >= 2:
                clf = clone(model)
                clf.fit(x[direction_train], target[direction_train])
                ext_mask = strict & df["adaptive_review"].to_numpy(bool) & candidate_mask
                if ext_mask.any():
                    score[ext_mask] = clf.predict_proba(x[ext_mask])[:, 1]
            out[f"{direction}__{model_name}"] = score

            for scope, mask in [
                ("internal_review_candidates_oof", direction_train),
                ("old_review_candidates_oof", direction_train & df["domain"].eq("old_data").to_numpy()),
                ("third_review_candidates_oof", direction_train & df["domain"].eq("third_batch").to_numpy()),
                (
                    "strict_external_review_candidates_locked",
                    strict & df["adaptive_review"].to_numpy(bool) & candidate_mask,
                ),
            ]:
                y = target[mask]
                s = score[mask]
                quality_rows.append(
                    {
                        "direction": direction,
                        "model": model_name,
                        "scope": scope,
                        "candidate_n": int(mask.sum()),
                        "positive_n": int(y.sum()) if len(y) else 0,
                        "positive_rate": float(y.mean()) if len(y) else np.nan,
                        "auroc": safe_auc(y, s) if len(y) else np.nan,
                        "average_precision": safe_ap(y, s) if len(y) else np.nan,
                    }
                )
    return out, pd.DataFrame(quality_rows)


def apply_rule(df: pd.DataFrame, scores: pd.DataFrame, direction: str, model: str, threshold: float) -> pd.DataFrame:
    spec = DIRECTIONS[direction]
    out = df.copy()
    col = f"{direction}__{model}"
    risk = scores[col].to_numpy(float)
    trigger = out["adaptive_review"].to_numpy(bool) & out["final_pred"].eq(spec["base_pred"]).to_numpy() & (risk >= threshold)
    out["flip_trigger"] = trigger
    out["flip_direction"] = direction
    out["flip_model"] = model
    out["flip_threshold"] = float(threshold)
    out["flip_score"] = risk
    out["flip_pred"] = out["final_pred"].astype(int)
    out.loc[out["flip_trigger"], "flip_pred"] = int(spec["flip_pred"])
    y = out["label_idx"].astype(int)
    base = out["final_pred"].astype(int)
    pred = out["flip_pred"].astype(int)
    out["flip_error"] = out["flip_trigger"] & pred.ne(y)
    out["rescued_by_flip"] = out["flip_trigger"] & base.ne(y) & pred.eq(y)
    out["hurt_by_flip"] = out["flip_trigger"] & base.eq(y) & pred.ne(y)
    out["remaining_review"] = out["adaptive_review"] & ~out["flip_trigger"]
    out["system_pred"] = pred
    out.loc[out["remaining_review"], "system_pred"] = y[out["remaining_review"]]
    return out


def summarize_subset(sub: pd.DataFrame, scope: str, candidate_id: str) -> dict[str, object]:
    sub = sub.copy()
    y = sub["label_idx"].astype(int).to_numpy()
    pred = sub["system_pred"].astype(int).to_numpy()
    base = sub["final_pred"].astype(int).to_numpy()
    prob = sub["prob_mean_core"].astype(float).to_numpy()
    m = metrics(y, pred, prob)
    base_m = metrics(y, base, prob)
    row = {
        "candidate_id": candidate_id,
        "scope": scope,
        "n": int(len(sub)),
        "baseline_adaptive_review_n": int(sub["adaptive_review"].sum()),
        "baseline_adaptive_review_rate": float(sub["adaptive_review"].mean()) if len(sub) else np.nan,
        "flip_n": int(sub["flip_trigger"].sum()),
        "flip_rate": float(sub["flip_trigger"].mean()) if len(sub) else np.nan,
        "remaining_review_n": int(sub["remaining_review"].sum()),
        "remaining_review_rate": float(sub["remaining_review"].mean()) if len(sub) else np.nan,
        "flip_error_n": int(sub["flip_error"].sum()),
        "rescued_n": int(sub["rescued_by_flip"].sum()),
        "hurt_n": int(sub["hurt_by_flip"].sum()),
        "base_raw_error_n": int((base != y).sum()),
        "base_raw_balanced_accuracy": float(base_m["balanced_accuracy"]),
    }
    row.update({k: float(v) if isinstance(v, (float, np.floating)) else int(v) for k, v in m.items()})
    return row


def train_select_candidate(df: pd.DataFrame, scores: pd.DataFrame, train_mask: pd.Series) -> dict[str, object]:
    rows = []
    for direction in DIRECTIONS:
        model_names = [c.split("__", 1)[1] for c in scores.columns if c.startswith(f"{direction}__")]
        for model in model_names:
            train_scores = scores.loc[train_mask, f"{direction}__{model}"].dropna().to_numpy(float)
            dynamic_thresholds = list(THRESHOLDS)
            if len(train_scores):
                dynamic_thresholds += [float(x) for x in np.quantile(train_scores, [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99])]
            for threshold in sorted(set(x for x in dynamic_thresholds if np.isfinite(x))):
                cid = f"{direction}||{model}||{threshold:.6f}"
                applied = apply_rule(df, scores, direction, model, threshold)
                sub = applied.loc[train_mask].copy()
                rows.append(
                    {
                        "candidate_id": cid,
                        "direction": direction,
                        "model": model,
                        "threshold": float(threshold),
                        "train_flip_n": int(sub["flip_trigger"].sum()),
                        "train_flip_error_n": int(sub["flip_error"].sum()),
                        "train_rescued_n": int(sub["rescued_by_flip"].sum()),
                        "train_hurt_n": int(sub["hurt_by_flip"].sum()),
                        "train_remaining_review_rate": float(sub["remaining_review"].mean()) if len(sub) else np.nan,
                    }
                )
    rank = pd.DataFrame(rows)
    safe = rank.loc[
        rank["train_flip_error_n"].eq(0)
        & rank["train_rescued_n"].gt(0)
        & rank["train_hurt_n"].eq(0)
        & rank["train_flip_n"].gt(0)
    ].copy()
    if safe.empty:
        selected = rank.sort_values(
            ["train_flip_error_n", "train_rescued_n", "train_flip_n"],
            ascending=[True, False, False],
        ).iloc[0]
        status = "no_train_safe_rescue_candidate"
    else:
        selected = safe.sort_values(
            ["train_rescued_n", "train_flip_n", "train_remaining_review_rate"],
            ascending=[False, False, True],
        ).iloc[0]
        status = "train_safe_rescue_candidate"
    return {**selected.to_dict(), "selection_status": status}


def nested_validation(df: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal = df["domain"].isin(["old_data", "third_batch"])
    folds = sorted(int(x) for x in df.loc[internal, "fold_id"].dropna().unique() if int(x) >= 0)
    case_frames = []
    selection_rows = []
    for fold in folds:
        train_mask = internal & df["adaptive_review"] & df["fold_id"].ne(fold)
        held_mask = internal & df["adaptive_review"] & df["fold_id"].eq(fold)
        chosen = train_select_candidate(df, scores, train_mask)
        applied = apply_rule(df, scores, str(chosen["direction"]), str(chosen["model"]), float(chosen["threshold"]))
        held = applied.loc[df["fold_id"].eq(fold) & internal].copy()
        held["selection_source"] = "nested_internal"
        held["selected_candidate_id"] = chosen["candidate_id"]
        case_frames.append(held)
        held_review = applied.loc[held_mask].copy()
        selection_rows.append(
            {
                **chosen,
                "heldout_fold": int(fold),
                "heldout_review_n": int(held_review.shape[0]),
                "heldout_flip_n": int(held_review["flip_trigger"].sum()),
                "heldout_flip_error_n": int(held_review["flip_error"].sum()),
                "heldout_rescued_n": int(held_review["rescued_by_flip"].sum()),
                "heldout_hurt_n": int(held_review["hurt_by_flip"].sum()),
            }
        )

    train_mask = internal & df["adaptive_review"]
    chosen = train_select_candidate(df, scores, train_mask)
    applied = apply_rule(df, scores, str(chosen["direction"]), str(chosen["model"]), float(chosen["threshold"]))
    strict = applied.loc[df["domain"].eq("strict_external")].copy()
    strict["selection_source"] = "locked_external_from_full_internal"
    strict["selected_candidate_id"] = chosen["candidate_id"]
    case_frames.append(strict)
    strict_review = applied.loc[df["domain"].eq("strict_external") & df["adaptive_review"]].copy()
    selection_rows.append(
        {
            **chosen,
            "heldout_fold": -1,
            "heldout_review_n": int(strict_review.shape[0]),
            "heldout_flip_n": int(strict_review["flip_trigger"].sum()),
            "heldout_flip_error_n": int(strict_review["flip_error"].sum()),
            "heldout_rescued_n": int(strict_review["rescued_by_flip"].sum()),
            "heldout_hurt_n": int(strict_review["hurt_by_flip"].sum()),
        }
    )
    return pd.concat(case_frames, ignore_index=True), pd.DataFrame(selection_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_cols = make_dataset()
    scores, quality = fit_direction_scores(df, feature_cols)
    nested_cases, selections = nested_validation(df, scores)

    summary_rows = []
    for scope, mask in [
        ("old_data_nested", nested_cases["domain"].eq("old_data")),
        ("third_batch_nested", nested_cases["domain"].eq("third_batch")),
        ("internal_nested_old_third", nested_cases["domain"].isin(["old_data", "third_batch"])),
        ("strict_external_locked", nested_cases["domain"].eq("strict_external")),
        ("all_domains_nested_plus_locked_external", nested_cases["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        sub = nested_cases.loc[mask].copy()
        cid = "nested_or_locked_selected"
        summary_rows.append(summarize_subset(sub, scope, cid))
    summary = pd.DataFrame(summary_rows)

    scores.to_csv(OUT_DIR / "v199_directional_error_scores.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUT_DIR / "v199_directional_error_model_quality.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(OUT_DIR / "v199_nested_selected_flip_rules.csv", index=False, encoding="utf-8-sig")
    nested_cases.to_csv(OUT_DIR / "v199_nested_flip_cases.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v199_nested_flip_summary.csv", index=False, encoding="utf-8-sig")

    all_row = summary.loc[summary["scope"].eq("all_domains_nested_plus_locked_external")].iloc[0]
    internal_row = summary.loc[summary["scope"].eq("internal_nested_old_third")].iloc[0]
    strict_row = summary.loc[summary["scope"].eq("strict_external_locked")].iloc[0]
    report = {
        "feature_count": int(len(feature_cols)),
        "all_domain_bacc": float(all_row["balanced_accuracy"]),
        "all_domain_remaining_review_rate": float(all_row["remaining_review_rate"]),
        "all_domain_flip_n": int(all_row["flip_n"]),
        "all_domain_flip_error_n": int(all_row["flip_error_n"]),
        "all_domain_rescued_n": int(all_row["rescued_n"]),
        "all_domain_hurt_n": int(all_row["hurt_n"]),
        "internal_bacc": float(internal_row["balanced_accuracy"]),
        "internal_remaining_review_rate": float(internal_row["remaining_review_rate"]),
        "internal_flip_error_n": int(internal_row["flip_error_n"]),
        "strict_external_bacc": float(strict_row["balanced_accuracy"]),
        "strict_external_remaining_review_rate": float(strict_row["remaining_review_rate"]),
        "strict_external_flip_error_n": int(strict_row["flip_error_n"]),
    }
    (OUT_DIR / "v199_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# v199 Directional Error Flip Corrector",
        "",
        "## Purpose",
        "",
        "v195 showed that high-confidence second-stage agreement can safely release cases, but true correction was not established. v199 trains direction-specific error-risk models for FN_high_to_low and FP_low_to_high inside the adaptive review pool, then flips only when an internally selected no-harm threshold is triggered.",
        "",
        "## Result",
        "",
        f"- All-domain BAcc: {100 * report['all_domain_bacc']:.2f}%; remaining review/reject: {100 * report['all_domain_remaining_review_rate']:.2f}%.",
        f"- All-domain flips: {report['all_domain_flip_n']}; rescued: {report['all_domain_rescued_n']}; hurt/action-errors: {report['all_domain_hurt_n']} / {report['all_domain_flip_error_n']}.",
        f"- Internal nested BAcc: {100 * report['internal_bacc']:.2f}%; remaining review/reject: {100 * report['internal_remaining_review_rate']:.2f}%; flip errors: {report['internal_flip_error_n']}.",
        f"- Strict external locked BAcc: {100 * report['strict_external_bacc']:.2f}%; remaining review/reject: {100 * report['strict_external_remaining_review_rate']:.2f}%; flip errors: {report['strict_external_flip_error_n']}.",
        "",
        "## Interpretation",
        "",
        "If rescued cases exceed hurt cases under nested validation, this becomes the first credible automatic correction candidate. If the model selects no safe rescue or hurts held-out cases, the evidence again supports safe release and rejection rather than autonomous flipping.",
    ]
    (OUT_DIR / "v199_directional_error_flip_corrector.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[v199] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
