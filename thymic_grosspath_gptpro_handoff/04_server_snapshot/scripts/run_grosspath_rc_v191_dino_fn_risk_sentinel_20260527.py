from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, load_features, metrics
from run_grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527 import as_bool


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v191_dino_fn_risk_sentinel_20260527"
V118_CASES = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527" / "v118_global_two_signal_cases.csv"
V185_CASES = ROOT / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527" / "v185_unlabeled_shift_adaptive_cases.csv"
TAB_COLS = [
    "main_prob",
    "robust_prob",
    "prob_mean_core",
    "wholecrop_prob",
    "v105_crop_prob",
    "core_agree_count",
    "pm_minus_whole",
    "pm_minus_main",
    "robust_minus_main",
    "crop_minus_whole",
]
THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def load_cases() -> tuple[pd.DataFrame, list[str]]:
    v118 = pd.read_csv(V118_CASES, dtype={"case_id": str, "original_case_id": str})
    v185 = pd.read_csv(V185_CASES, dtype={"case_id": str, "original_case_id": str})
    keep_v118 = [
        "case_id",
        "main_prob",
        "robust_prob",
        "wholecrop_prob",
        "v105_crop_prob",
        "core_agree_count",
        "fold_id",
        "view_type_final",
    ]
    df = v185.merge(v118[keep_v118], on="case_id", how="inner", validate="one_to_one")
    for col in ["fixed_v118_review", "fixed_v182_review", "adaptive_review", "adaptive_auto_decision"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "fold_id", "core_agree_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    for col in ["prob_mean_core", "main_prob", "robust_prob", "wholecrop_prob", "v105_crop_prob"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["base_wrong"] = df["final_pred"].ne(df["label_idx"])
    df["fn_target"] = df["label_idx"].eq(1) & df["final_pred"].eq(0)
    df["error_direction"] = "correct"
    df.loc[df["fn_target"], "error_direction"] = "FN_high_to_low"
    df.loc[df["label_idx"].eq(0) & df["final_pred"].eq(1), "error_direction"] = "FP_low_to_high"
    df["auto_lowrisk"] = df["adaptive_auto_decision"] & df["final_pred"].eq(0)
    df["lowrisk_pred"] = df["final_pred"].eq(0)
    df["pm_minus_whole"] = df["prob_mean_core"] - df["wholecrop_prob"]
    df["pm_minus_main"] = df["prob_mean_core"] - df["main_prob"]
    df["robust_minus_main"] = df["robust_prob"] - df["main_prob"]
    df["crop_minus_whole"] = df["v105_crop_prob"] - df["wholecrop_prob"]

    features = load_features()
    feat_cols = [c for c in features.columns if c.startswith("feat_")]
    df = df.merge(features.drop(columns=["feature_domain"], errors="ignore"), on="case_id", how="inner", validate="one_to_one")
    return df.reset_index(drop=True), feat_cols


def build_feature_matrix(df: pd.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    internal = df["domain"].isin(["old_data", "third_batch"])
    pca = PCA(n_components=48, random_state=191)
    pca.fit(df.loc[internal, feat_cols].astype(float).fillna(0.0).to_numpy(float))
    pcs = pca.transform(df[feat_cols].astype(float).fillna(0.0).to_numpy(float))
    pc_cols = [f"pc_{i:02d}" for i in range(pcs.shape[1])]
    pc_df = pd.DataFrame(pcs, columns=pc_cols)
    view = pd.get_dummies(df["view_type_final"].fillna("unknown").astype(str), prefix="view", dtype=float)
    xdf = pd.concat([df[TAB_COLS].reset_index(drop=True), view.reset_index(drop=True), pc_df], axis=1)
    return xdf.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float), xdf.columns.tolist()


def models() -> dict[str, object]:
    return {
        "tab_dino_logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=191),
        ),
        "tab_dino_extra_d3": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=12,
            max_features="sqrt",
            class_weight="balanced",
            random_state=191,
            n_jobs=-1,
        ),
        "tab_hgb_l2": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.035,
            max_leaf_nodes=5,
            l2_regularization=2.0,
            random_state=191,
        ),
    }


def fit_predict_foldwise(df: pd.DataFrame, x: np.ndarray, model: object) -> np.ndarray:
    risk = np.full(len(df), np.nan, dtype=float)
    internal = df["domain"].isin(["old_data", "third_batch"]).to_numpy()
    low = df["lowrisk_pred"].to_numpy(bool)
    y = df["fn_target"].astype(int).to_numpy()
    folds = df["fold_id"].to_numpy(int)
    for fold in sorted(np.unique(folds[internal])):
        train = internal & low & (folds != int(fold))
        test = internal & low & (folds == int(fold))
        if not test.any():
            continue
        if len(np.unique(y[train])) < 2:
            risk[test] = float(y[train].mean()) if train.any() else 0.0
            continue
        clf = clone(model)
        clf.fit(x[train], y[train])
        risk[test] = clf.predict_proba(x[test])[:, 1]
    ext = df["domain"].eq("strict_external").to_numpy() & low
    train_all = internal & low
    if ext.any() and len(np.unique(y[train_all])) >= 2:
        clf = clone(model)
        clf.fit(x[train_all], y[train_all])
        risk[ext] = clf.predict_proba(x[ext])[:, 1]
    elif ext.any():
        risk[ext] = float(y[train_all].mean()) if train_all.any() else 0.0
    return risk


def choose_threshold(train_risk: np.ndarray, train_fn: np.ndarray, train_candidate: np.ndarray) -> float | None:
    rows = []
    for t in THRESHOLDS:
        m = train_candidate & np.isfinite(train_risk) & (train_risk >= t)
        rescued = int((m & train_fn).sum())
        if rescued <= 0:
            continue
        clean = int((m & ~train_fn).sum())
        rows.append((rescued, clean, int(m.sum()), float(t)))
    if not rows:
        return None
    rows.sort(key=lambda z: (-z[0], z[1], z[2], -z[3]))
    return rows[0][3]


def nested_sentinel(df: pd.DataFrame, risk: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    internal = df["domain"].isin(["old_data", "third_batch"]).to_numpy()
    folds = df["fold_id"].to_numpy(int)
    fn = df["fn_target"].to_numpy(bool)
    candidate = df["auto_lowrisk"].to_numpy(bool)
    review = np.zeros(len(df), dtype=bool)
    rows = []
    for fold in sorted(np.unique(folds[internal])):
        train = internal & (folds != int(fold))
        held = internal & (folds == int(fold))
        t = choose_threshold(risk, fn, train & candidate)
        if t is None:
            rows.append({"fold_id": int(fold), "threshold": np.nan, "selection_status": "no_train_fn_candidate"})
            continue
        fire = candidate & np.isfinite(risk) & (risk >= t)
        review |= fire & held
        rows.append(
            {
                "fold_id": int(fold),
                "threshold": float(t),
                "selection_status": "selected_by_train_fn_risk",
                "train_review_n": int((fire & train).sum()),
                "train_rescued_fn_n": int((fire & train & fn).sum()),
                "train_clean_review_n": int((fire & train & ~fn).sum()),
                "heldout_review_n": int((fire & held).sum()),
                "heldout_rescued_fn_n": int((fire & held & fn).sum()),
                "heldout_clean_review_n": int((fire & held & ~fn).sum()),
            }
        )
    thresholds = [float(r["threshold"]) for r in rows if pd.notna(r.get("threshold", np.nan))]
    if thresholds:
        ext_t = max(thresholds)
        ext = df["domain"].eq("strict_external").to_numpy()
        review |= candidate & ext & np.isfinite(risk) & (risk >= ext_t)
    return pd.Series(review, index=df.index), pd.DataFrame(rows)


def summarize(df: pd.DataFrame, sentinel_review: pd.Series, workflow: str) -> list[dict[str, object]]:
    review = df["adaptive_review"] | sentinel_review
    y = df["label_idx"].to_numpy(int)
    pred = df["final_pred"].to_numpy(int).copy()
    pred[review.to_numpy(bool)] = y[review.to_numpy(bool)]
    rows = []
    add = sentinel_review & ~df["adaptive_review"]
    for scope, mask in [
        ("old_data", df["domain"].eq("old_data")),
        ("third_batch", df["domain"].eq("third_batch")),
        ("strict_external", df["domain"].eq("strict_external")),
        ("all_domains", df["domain"].isin(["old_data", "third_batch", "strict_external"])),
    ]:
        m = metrics(y[mask.to_numpy(bool)], pred[mask.to_numpy(bool)], df.loc[mask, "prob_mean_core"].to_numpy(float))
        rows.append(
            {
                "workflow": workflow,
                "scope": scope,
                "additional_sentinel_review_n": int((add & mask).sum()),
                "sentinel_rescued_fn_n": int((add & mask & df["fn_target"]).sum()),
                "sentinel_clean_review_n": int((add & mask & ~df["fn_target"]).sum()),
                "review_or_reject_n": int((review & mask).sum()),
                "review_or_reject_rate": float((review & mask).mean()),
                "auto_error_n": int((~review & mask & df["base_wrong"]).sum()),
                "accuracy": float(m["accuracy"]),
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "f1": float(m["f1"]),
                "auc": float(m["auc"]),
                "fn": int(m["fn"]),
                "fp": int(m["fp"]),
            }
        )
    return rows


def safe_auc(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(s)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y[ok], s[ok])), float(average_precision_score(y[ok], s[ok]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feat_cols = load_cases()
    x, x_cols = build_feature_matrix(df, feat_cols)
    all_summary = []
    model_quality = []
    case_frames = []
    selection_frames = []
    best_row = None
    best_review = None
    best_risk = None
    best_model = None

    baseline_review = pd.Series(False, index=df.index)
    all_summary += summarize(df, baseline_review, "v185_adaptive_baseline")

    for name, model in models().items():
        risk = fit_predict_foldwise(df, x, model)
        sentinel_review, selections = nested_sentinel(df, risk)
        selections.insert(0, "model", name)
        selection_frames.append(selections)
        all_summary += summarize(df, sentinel_review, f"v191_{name}")
        internal_low = df["domain"].isin(["old_data", "third_batch"]).to_numpy() & df["lowrisk_pred"].to_numpy(bool)
        auto_low = df["domain"].isin(["old_data", "third_batch"]).to_numpy() & df["auto_lowrisk"].to_numpy(bool)
        auc_low, ap_low = safe_auc(df["fn_target"].astype(int).to_numpy()[internal_low], risk[internal_low])
        auc_auto, ap_auto = safe_auc(df["fn_target"].astype(int).to_numpy()[auto_low], risk[auto_low])
        model_quality.append(
            {
                "model": name,
                "internal_lowrisk_auc": auc_low,
                "internal_lowrisk_ap": ap_low,
                "internal_auto_lowrisk_auc": auc_auto,
                "internal_auto_lowrisk_ap": ap_auto,
            }
        )
        tmp = df[["domain", "case_id", "original_case_id", "task_l6_label", "label_idx", "final_pred", "fn_target", "fold_id", "image_name", "adaptive_review", "adaptive_auto_decision", "auto_lowrisk", "base_wrong"]].copy()
        tmp["model"] = name
        tmp["fn_risk"] = risk
        tmp["v191_sentinel_review"] = sentinel_review
        tmp["v191_additional_review"] = sentinel_review & ~df["adaptive_review"]
        case_frames.append(tmp)

    summary = pd.DataFrame(all_summary)
    summary.to_csv(OUT_DIR / "v191_fn_risk_sentinel_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(model_quality).to_csv(OUT_DIR / "v191_fn_risk_model_quality.csv", index=False, encoding="utf-8-sig")
    pd.concat(selection_frames, ignore_index=True).to_csv(OUT_DIR / "v191_fold_selected_thresholds.csv", index=False, encoding="utf-8-sig")
    cases = pd.concat(case_frames, ignore_index=True)
    cases.to_csv(OUT_DIR / "v191_fn_risk_case_outputs.csv", index=False, encoding="utf-8-sig")

    all_rows = summary.loc[summary["scope"].eq("all_domains") & ~summary["workflow"].eq("v185_adaptive_baseline")].copy()
    if not all_rows.empty:
        best = all_rows.sort_values(
            ["auto_error_n", "sentinel_rescued_fn_n", "additional_sentinel_review_n"],
            ascending=[True, False, True],
        ).iloc[0]
    else:
        best = pd.Series(dtype=object)
    base = summary.loc[summary["workflow"].eq("v185_adaptive_baseline") & summary["scope"].eq("all_domains")].iloc[0]
    case251 = cases.loc[cases["original_case_id"].astype(str).eq("2516531")].copy()
    case251.to_csv(OUT_DIR / "v191_case_2516531_model_outputs.csv", index=False, encoding="utf-8-sig")

    report = {
        "model_count": int(len(models())),
        "baseline_auto_error_n": int(base["auto_error_n"]),
        "best_workflow": str(best["workflow"]) if not best.empty else None,
        "best_auto_error_n": int(best["auto_error_n"]) if not best.empty else None,
        "best_additional_review_n": int(best["additional_sentinel_review_n"]) if not best.empty else None,
        "best_rescued_fn_n": int(best["sentinel_rescued_fn_n"]) if not best.empty else None,
        "best_review_rate": float(best["review_or_reject_rate"]) if not best.empty else None,
        "case_2516531_reviewed_by_models": {
            str(r["model"]): bool(r["v191_sentinel_review"]) for _, r in case251.iterrows()
        },
        "case_2516531_risks": {str(r["model"]): float(r["fn_risk"]) for _, r in case251.iterrows()},
    }
    (OUT_DIR / "v191_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# v191 DINO FN-risk Sentinel",
        "",
        "## Result",
        "",
        (
            f"- Baseline auto errors: {report['baseline_auto_error_n']}; best nested FN-risk workflow: "
            f"{report['best_workflow']} with auto errors {report['best_auto_error_n']}, additional review "
            f"{report['best_additional_review_n']}, rescued FN {report['best_rescued_fn_n']}."
        ),
        f"- Case 2516531 reviewed by models: {report['case_2516531_reviewed_by_models']}.",
        "",
        "## Boundary",
        "",
        "This tests whether a learned image/probability FN-risk sentinel can rescue the last automatic FN without case-specific tuning.",
    ]
    (OUT_DIR / "v191_dino_fn_risk_sentinel.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[v191] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
