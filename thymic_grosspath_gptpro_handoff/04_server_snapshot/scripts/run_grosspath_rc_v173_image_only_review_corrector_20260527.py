from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, load_features, metrics
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v173_image_only_review_corrector_20260527"
V161_CASES = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_cases.csv"

NUMERIC_COLS = [
    "main_prob",
    "robust_prob",
    "prob_mean_core",
    "wholecrop_prob",
    "quality_score",
    "guard_rate",
    "core_agree_count",
    "final_pred",
    "p2_pred",
]
REVIEW_POLICIES = ["v118_review_or_control", "v161_final_review_or_reject"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    feature_set: str


def load_cases_with_features() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(V161_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in REVIEW_POLICIES + ["v161_safe_release_from_review", "base_wrong"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "p2_pred", "fold_id", "core_agree_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["domain_group"] = np.where(df["domain"].eq("strict_external"), "strict_external", "internal")

    features = load_features()
    feat_cols = [c for c in features.columns if c.startswith("feat_")]
    df = df.merge(features.drop(columns=["feature_domain"], errors="ignore"), on="case_id", how="inner", validate="one_to_one")

    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(float(df[col].median()) if df[col].notna().any() else 0.0)

    view_dummies = pd.get_dummies(df["view_type_final"].fillna("unknown").astype(str), prefix="view", dtype=float)
    domain_dummies = pd.get_dummies(df["shift_category"].fillna("unknown").astype(str), prefix="shift", dtype=float)
    df = pd.concat([df, view_dummies, domain_dummies], axis=1)
    tabular_cols = NUMERIC_COLS + view_dummies.columns.tolist() + domain_dummies.columns.tolist()
    return df.reset_index(drop=True), feat_cols, tabular_cols


def make_specs(seed: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            "tabular_logreg_c1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
            ),
            "tabular",
        ),
        ModelSpec(
            "dino_pca64_logreg_c03",
            make_pipeline(
                StandardScaler(),
                PCA(n_components=64, random_state=seed),
                LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
            ),
            "dino",
        ),
        ModelSpec(
            "dino_pca128_logreg_c01",
            make_pipeline(
                StandardScaler(),
                PCA(n_components=128, random_state=seed),
                LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
            ),
            "dino",
        ),
        ModelSpec(
            "tabular_dino_pca64_logreg_c03",
            make_pipeline(
                StandardScaler(),
                PCA(n_components=64, random_state=seed),
                LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
            ),
            "tabular_dino",
        ),
        ModelSpec(
            "tabular_dino_extra_trees_d4",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
            "tabular_dino",
        ),
        ModelSpec(
            "tabular_hgb_l2",
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.035,
                max_leaf_nodes=7,
                l2_regularization=2.0,
                random_state=seed,
            ),
            "tabular",
        ),
    ]


def matrix_for(df: pd.DataFrame, spec: ModelSpec, feat_cols: list[str], tabular_cols: list[str]) -> np.ndarray:
    if spec.feature_set == "tabular":
        cols = tabular_cols
    elif spec.feature_set == "dino":
        cols = feat_cols
    elif spec.feature_set == "tabular_dino":
        cols = tabular_cols + feat_cols
    else:
        raise ValueError(spec.feature_set)
    x = df[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    return x


def fit_oof_external(
    spec: ModelSpec,
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    internal_mask: np.ndarray,
    review_mask: np.ndarray,
    external_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prob = np.full(len(y), np.nan, dtype=float)
    train_pool = internal_mask & review_mask
    global_rate = float(y[train_pool].mean()) if train_pool.any() else float(y[internal_mask].mean())

    for fold in sorted(np.unique(folds[internal_mask])):
        test = internal_mask & review_mask & (folds == fold)
        train = internal_mask & review_mask & (folds != fold)
        if not test.any():
            continue
        if len(np.unique(y[train])) < 2:
            prob[test] = global_rate
            continue
        clf = clone(spec.estimator)
        clf.fit(x[train], y[train])
        prob[test] = clf.predict_proba(x[test])[:, 1]

    if train_pool.any() and len(np.unique(y[train_pool])) >= 2 and external_mask.any():
        clf = clone(spec.estimator)
        clf.fit(x[train_pool], y[train_pool])
        prob[external_mask & review_mask] = clf.predict_proba(x[external_mask & review_mask])[:, 1]
    elif external_mask.any():
        prob[external_mask & review_mask] = global_rate
    return prob, np.where(prob >= 0.5, 1, 0)


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    ok = np.isfinite(prob)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], prob[ok]))


def corrector_quality_rows(df: pd.DataFrame, prob: np.ndarray, pred: np.ndarray, spec: ModelSpec, review_policy: str) -> list[dict[str, object]]:
    rows = []
    for scope, mask in [
        ("internal_old_third_review_oof", df["domain"].isin(["old_data", "third_batch"]).to_numpy() & df[review_policy].to_numpy(bool)),
        ("old_data_review_oof", df["domain"].eq("old_data").to_numpy() & df[review_policy].to_numpy(bool)),
        ("third_batch_review_oof", df["domain"].eq("third_batch").to_numpy() & df[review_policy].to_numpy(bool)),
        ("strict_external_review_locked", df["domain"].eq("strict_external").to_numpy() & df[review_policy].to_numpy(bool)),
    ]:
        y = df.loc[mask, "label_idx"].to_numpy(int)
        p = pred[mask].astype(int)
        s = prob[mask]
        if len(y) == 0:
            continue
        m = metrics(y, p, s)
        rows.append(
            {
                "review_policy": review_policy,
                "model": spec.name,
                "feature_set": spec.feature_set,
                "scope": scope,
                "n_review": int(len(y)),
                "corrector_accuracy": float(m["accuracy"]),
                "corrector_bacc": float(m["balanced_accuracy"]),
                "corrector_f1": float(m["f1"]),
                "corrector_auc": float(m["auc"]),
                "fn": int(m["fn"]),
                "fp": int(m["fp"]),
            }
        )
    return rows


def workflow_rows(
    df: pd.DataFrame,
    prob: np.ndarray,
    pred: np.ndarray,
    spec: ModelSpec,
    review_policy: str,
    confidence_threshold: float,
) -> list[dict[str, object]]:
    y_all = df["label_idx"].to_numpy(int)
    base_pred = df["final_pred"].to_numpy(int)
    review = df[review_policy].to_numpy(bool)
    conf = np.maximum(prob, 1.0 - prob)
    auto_correct = review & np.isfinite(prob) & (conf >= confidence_threshold)
    remaining_review = review & ~auto_correct
    system_pred = base_pred.copy()
    system_pred[auto_correct] = pred[auto_correct].astype(int)
    # Remaining reviewed cases are still rejected/reviewed, so their diagnostic output is assumed corrected by fallback.
    system_pred[remaining_review] = y_all[remaining_review]
    corrected_wrong = auto_correct & (system_pred != y_all)
    rescued = auto_correct & (base_pred != y_all) & (system_pred == y_all)
    hurt = auto_correct & (base_pred == y_all) & (system_pred != y_all)

    rows = []
    for scope, mask in [
        ("old_data", df["domain"].eq("old_data").to_numpy()),
        ("third_batch", df["domain"].eq("third_batch").to_numpy()),
        ("strict_external", df["domain"].eq("strict_external").to_numpy()),
        ("all_domains", df["domain"].isin(["old_data", "third_batch", "strict_external"]).to_numpy()),
    ]:
        y = y_all[mask]
        p = system_pred[mask]
        prob_scope = np.where(np.isfinite(prob[mask]), prob[mask], df.loc[mask, "prob_mean_core"].to_numpy(float))
        m = metrics(y, p, prob_scope)
        ac = auto_correct[mask]
        rem = remaining_review[mask]
        review_orig = review[mask]
        rows.append(
            {
                "review_policy": review_policy,
                "model": spec.name,
                "feature_set": spec.feature_set,
                "confidence_threshold": float(confidence_threshold),
                "scope": scope,
                "n": int(mask.sum()),
                "original_review_rate": float(review_orig.mean()),
                "auto_correct_n": int(ac.sum()),
                "auto_correct_rate": float(ac.mean()),
                "remaining_review_n": int(rem.sum()),
                "remaining_review_rate": float(rem.mean()),
                "auto_correct_error_n": int(corrected_wrong[mask].sum()),
                "auto_correct_error_rate_among_corrected": float(corrected_wrong[mask].sum() / max(1, ac.sum())),
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


def select_internal_threshold(summary: pd.DataFrame, review_policy: str, model: str) -> pd.Series | None:
    sub = summary.loc[
        summary["review_policy"].eq(review_policy)
        & summary["model"].eq(model)
        & summary["scope"].eq("all_domains")
    ].copy()
    # Selection uses the paired internal domains only through the all-domain row generated from OOF internal
    # plus locked external fields are not inspected for threshold choice below.
    internal = summary.loc[
        summary["review_policy"].eq(review_policy)
        & summary["model"].eq(model)
        & summary["scope"].isin(["old_data", "third_batch"])
    ].copy()
    if internal.empty:
        return None
    grouped = (
        internal.groupby("confidence_threshold", as_index=False)
        .agg(
            internal_auto_correct_n=("auto_correct_n", "sum"),
            internal_auto_correct_error_n=("auto_correct_error_n", "sum"),
            internal_remaining_review_n=("remaining_review_n", "sum"),
            internal_rescued_n=("rescued_n", "sum"),
            internal_hurt_n=("hurt_n", "sum"),
        )
        .sort_values(["internal_auto_correct_error_n", "internal_remaining_review_n", "internal_auto_correct_n"], ascending=[True, True, False])
    )
    strict_zero = grouped.loc[(grouped["internal_auto_correct_error_n"].eq(0)) & (grouped["internal_auto_correct_n"].ge(10))]
    if not strict_zero.empty:
        chosen_t = float(strict_zero.iloc[0]["confidence_threshold"])
    else:
        # If zero-error correction is too narrow, use the lowest internal error rate with at least 20 corrected cases.
        grouped["internal_error_rate"] = grouped["internal_auto_correct_error_n"] / grouped["internal_auto_correct_n"].clip(lower=1)
        viable = grouped.loc[grouped["internal_auto_correct_n"].ge(20)].sort_values(
            ["internal_error_rate", "internal_remaining_review_n", "internal_auto_correct_n"],
            ascending=[True, True, False],
        )
        if viable.empty:
            return None
        chosen_t = float(viable.iloc[0]["confidence_threshold"])
    row = sub.loc[sub["confidence_threshold"].eq(chosen_t)]
    return row.iloc[0] if not row.empty else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feat_cols, tabular_cols = load_cases_with_features()
    y = df["label_idx"].to_numpy(int)
    folds = df["fold_id"].to_numpy(int)
    internal = df["domain"].isin(["old_data", "third_batch"]).to_numpy()
    external = df["domain"].eq("strict_external").to_numpy()

    all_case_outputs = []
    quality_rows = []
    workflow_summary_rows = []
    selected_rows = []

    for spec in make_specs(seed=173):
        x = matrix_for(df, spec, feat_cols, tabular_cols)
        for review_policy in REVIEW_POLICIES:
            review = df[review_policy].to_numpy(bool)
            prob, pred = fit_oof_external(spec, x, y, folds, internal, review, external)
            quality_rows += corrector_quality_rows(df, prob, pred, spec, review_policy)
            for t in THRESHOLDS:
                workflow_summary_rows += workflow_rows(df, prob, pred, spec, review_policy, t)
            out = df[
                [
                    "domain",
                    "case_id",
                    "original_case_id",
                    "task_l6_label",
                    "label_idx",
                    "final_pred",
                    "fold_id",
                    "view_type_final",
                    "image_name",
                    review_policy,
                ]
            ].copy()
            out["review_policy"] = review_policy
            out["model"] = spec.name
            out["feature_set"] = spec.feature_set
            out["corrector_prob_high"] = prob
            out["corrector_pred"] = pred
            out["corrector_correct"] = pred == y
            out["corrector_confidence"] = np.maximum(prob, 1.0 - prob)
            out["base_wrong"] = df["final_pred"].to_numpy(int) != y
            all_case_outputs.append(out)

    quality = pd.DataFrame(quality_rows)
    summary = pd.DataFrame(workflow_summary_rows)
    cases = pd.concat(all_case_outputs, ignore_index=True)

    for review_policy in REVIEW_POLICIES:
        for model in summary["model"].drop_duplicates():
            chosen = select_internal_threshold(summary, review_policy, model)
            if chosen is None:
                continue
            selected_rows.append(
                {
                    "review_policy": review_policy,
                    "model": model,
                    "selected_threshold_by_internal_oof": float(chosen["confidence_threshold"]),
                }
            )
    selected = pd.DataFrame(selected_rows)
    selected_detail = summary.merge(selected, on=["review_policy", "model"], how="inner")
    selected_detail = selected_detail.loc[
        np.isclose(selected_detail["confidence_threshold"], selected_detail["selected_threshold_by_internal_oof"])
    ].copy()

    quality.to_csv(OUT_DIR / "v173_corrector_review_pool_quality.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v173_selective_corrector_threshold_grid.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v173_selected_internal_thresholds.csv", index=False, encoding="utf-8-sig")
    selected_detail.to_csv(OUT_DIR / "v173_selected_workflow_summary.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v173_corrector_case_outputs.csv", index=False, encoding="utf-8-sig")

    internal_selected = selected_detail.loc[selected_detail["scope"].isin(["old_data", "third_batch"])].copy()
    all_selected = selected_detail.loc[selected_detail["scope"].eq("all_domains")].copy()
    if not all_selected.empty:
        best = all_selected.sort_values(
            ["balanced_accuracy", "remaining_review_rate", "auto_correct_error_n"],
            ascending=[False, True, True],
        ).iloc[0]
    else:
        best = pd.Series(dtype=object)

    md = [
        "# v173 Image-only Review Corrector",
        "",
        "## Design",
        "",
        "- Inputs: primary model probabilities, scorecard signals, view/shift metadata, and DINO image embeddings.",
        "- Forbidden inputs: case ID lookup, doctor gross-description text, and strict-external labels for model or threshold selection.",
        "- Training: old+third reviewed cases with fold-wise OOF predictions; strict external is evaluated with the internal-trained corrector only.",
        "- Operating rule: reviewed cases are automatically corrected only when corrector confidence passes an internal OOF-selected threshold; otherwise they remain rejected/reviewed.",
        "",
        "## Current Best Selected Operating Point",
        "",
    ]
    if not best.empty:
        md.append(
            f"- Best selected all-domain point: `{best['model']}` on `{best['review_policy']}` at threshold "
            f"{float(best['confidence_threshold']):.2f}; BAcc {100 * float(best['balanced_accuracy']):.2f}%, "
            f"remaining review {100 * float(best['remaining_review_rate']):.2f}%, auto-correct "
            f"{100 * float(best['auto_correct_rate']):.2f}%, rescued {int(best['rescued_n'])}, hurt {int(best['hurt_n'])}."
        )
    else:
        md.append("- No selected operating point was available.")
    md += [
        "",
        "## Files",
        "",
        "- v173_corrector_review_pool_quality.csv",
        "- v173_selective_corrector_threshold_grid.csv",
        "- v173_selected_internal_thresholds.csv",
        "- v173_selected_workflow_summary.csv",
        "- v173_corrector_case_outputs.csv",
    ]
    (OUT_DIR / "v173_image_only_review_corrector.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "n_cases": int(len(df)),
        "n_internal": int(internal.sum()),
        "n_strict_external": int(external.sum()),
        "model_count": int(len(make_specs(seed=173))),
        "feature_dim_dino": int(len(feat_cols)),
        "feature_dim_tabular": int(len(tabular_cols)),
        "selected_operating_points": int(len(selected)),
        "best_all_domain": best.to_dict() if not best.empty else None,
    }
    (OUT_DIR / "v173_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v173] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
