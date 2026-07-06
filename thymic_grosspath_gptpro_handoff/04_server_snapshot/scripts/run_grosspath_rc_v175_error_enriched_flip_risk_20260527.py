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


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v175_error_enriched_flip_risk_20260527"
V135 = ROOT / "outputs" / "grosspath_rc_v135_stage1_base_candidate_scan_20260527" / "v135_stage1_candidate_predictions.csv"
V161_CASES = ROOT / "outputs" / "grosspath_rc_v161_safe_release_from_high_safety_review_pool_20260527" / "v161_safe_release_cases.csv"

REVIEW_POLICIES = ["v118_review_or_control", "v161_final_review_or_reject"]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
TABULAR_COLS = [
    "main_prob",
    "robust_prob",
    "prob_mean_core",
    "wholecrop_prob",
    "quality_score",
    "guard_rate",
    "core_agree_count",
]


def load_base_cases() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(V161_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in REVIEW_POLICIES + ["base_wrong"]:
        df[col] = as_bool(df[col])
    for col in ["label_idx", "final_pred", "fold_id", "core_agree_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    for col in TABULAR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(float(df[col].median()) if df[col].notna().any() else 0.0)
    features = load_features()
    feat_cols = [c for c in features.columns if c.startswith("feat_")]
    df = df.merge(features.drop(columns=["feature_domain"], errors="ignore"), on="case_id", how="inner", validate="one_to_one")

    internal = df["domain"].isin(["old_data", "third_batch"])
    pca = PCA(n_components=64, random_state=175)
    pca.fit(df.loc[internal, feat_cols].astype(float).fillna(0.0).to_numpy(float))
    pcs = pca.transform(df[feat_cols].astype(float).fillna(0.0).to_numpy(float))
    pc_cols = [f"dino_pca_{i:02d}" for i in range(pcs.shape[1])]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(pcs, columns=pc_cols)], axis=1)

    view = pd.get_dummies(df["view_type_final"].fillna("unknown").astype(str), prefix="view", dtype=float)
    df = pd.concat([df, view], axis=1)
    case_feature_cols = TABULAR_COLS + pc_cols + view.columns.tolist()
    for col in case_feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df, case_feature_cols


def load_error_enriched_rows(cases: pd.DataFrame, case_feature_cols: list[str]) -> pd.DataFrame:
    pred = pd.read_csv(V135, dtype={"case_id": str, "original_case_id": str})
    pred = pred.loc[pred["scope"].eq("internal_oof_old_third") & pred["objective"].eq("balanced_accuracy")].copy()
    pred["candidate_prob_high"] = pd.to_numeric(pred["prob_high"], errors="coerce").fillna(0.5)
    pred["candidate_pred"] = pd.to_numeric(pred["pred_idx"], errors="coerce").fillna(0).astype(int)
    pred["candidate_confidence"] = np.maximum(pred["candidate_prob_high"], 1.0 - pred["candidate_prob_high"])
    pred["candidate_wrong"] = pred["candidate_pred"].ne(pd.to_numeric(pred["label_idx"], errors="coerce").astype(int)).astype(int)
    keep = ["case_id", "candidate", "candidate_prob_high", "candidate_pred", "candidate_confidence", "candidate_wrong"]
    rows = pred[keep].merge(
        cases.loc[cases["domain"].isin(["old_data", "third_batch"]), ["case_id", "fold_id"] + case_feature_cols],
        on="case_id",
        how="inner",
        validate="many_to_one",
    )
    cand = pd.get_dummies(rows["candidate"].astype(str), prefix="cand", dtype=float)
    rows = pd.concat([rows.reset_index(drop=True), cand], axis=1)
    return rows


def current_rows(cases: pd.DataFrame, case_feature_cols: list[str], candidate_cols: list[str], review_policy: str) -> pd.DataFrame:
    cur = cases[[
        "domain",
        "case_id",
        "original_case_id",
        "task_l6_label",
        "label_idx",
        "final_pred",
        "fold_id",
        "image_name",
        review_policy,
    ] + case_feature_cols].copy()
    cur["candidate_prob_high"] = pd.to_numeric(cases["prob_mean_core"], errors="coerce").fillna(0.5)
    cur["candidate_pred"] = pd.to_numeric(cases["final_pred"], errors="coerce").fillna(0).astype(int)
    cur["candidate_confidence"] = np.maximum(cur["candidate_prob_high"], 1.0 - cur["candidate_prob_high"])
    cur["candidate_wrong"] = cur["candidate_pred"].ne(cur["label_idx"]).astype(int)
    cur["review_policy"] = review_policy
    cur["review_flag"] = cur[review_policy].to_numpy(bool)
    for col in candidate_cols:
        cur[col] = 0.0
    return cur


def make_models(seed: int) -> dict[str, object]:
    return {
        "enriched_logreg_c1": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        ),
        "enriched_logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        ),
        "enriched_extra_trees_d4": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "enriched_hgb_l2": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.035,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            random_state=seed,
        ),
    }


def feature_cols(train_rows: pd.DataFrame, case_feature_cols: list[str]) -> list[str]:
    cand_cols = [c for c in train_rows.columns if c.startswith("cand_")]
    return ["candidate_prob_high", "candidate_pred", "candidate_confidence"] + case_feature_cols + cand_cols


def predict_current_risk(
    model: object,
    train_rows: pd.DataFrame,
    current: pd.DataFrame,
    x_cols: list[str],
) -> np.ndarray:
    risk = np.full(len(current), np.nan, dtype=float)
    y_train = train_rows["candidate_wrong"].to_numpy(int)
    folds = train_rows["fold_id"].to_numpy(int)
    cur_folds = current["fold_id"].to_numpy(int)
    internal_current = current["domain"].isin(["old_data", "third_batch"]).to_numpy()
    external_current = current["domain"].eq("strict_external").to_numpy()
    x_train = train_rows[x_cols].astype(float).fillna(0.0).to_numpy(float)
    x_cur = current[x_cols].astype(float).fillna(0.0).to_numpy(float)
    for fold in sorted(np.unique(cur_folds[internal_current])):
        test = internal_current & (cur_folds == fold)
        train = folds != fold
        if not test.any():
            continue
        if len(np.unique(y_train[train])) < 2:
            risk[test] = float(y_train[train].mean())
            continue
        clf = clone(model)
        clf.fit(x_train[train], y_train[train])
        risk[test] = clf.predict_proba(x_cur[test])[:, 1]
    if external_current.any():
        clf = clone(model)
        clf.fit(x_train, y_train)
        risk[external_current] = clf.predict_proba(x_cur[external_current])[:, 1]
    return risk


def risk_quality(current: pd.DataFrame, risk: np.ndarray, model_name: str, review_policy: str) -> list[dict[str, object]]:
    rows = []
    for scope, mask in [
        ("internal_review_oof", current["domain"].isin(["old_data", "third_batch"]).to_numpy() & current["review_flag"].to_numpy(bool)),
        ("old_data_review_oof", current["domain"].eq("old_data").to_numpy() & current["review_flag"].to_numpy(bool)),
        ("third_batch_review_oof", current["domain"].eq("third_batch").to_numpy() & current["review_flag"].to_numpy(bool)),
        ("strict_external_review_locked", current["domain"].eq("strict_external").to_numpy() & current["review_flag"].to_numpy(bool)),
    ]:
        y = current.loc[mask, "candidate_wrong"].to_numpy(int)
        s = risk[mask]
        ok = np.isfinite(s)
        if ok.sum() == 0:
            continue
        if len(np.unique(y[ok])) == 2:
            auc = float(roc_auc_score(y[ok], s[ok]))
            ap = float(average_precision_score(y[ok], s[ok]))
        else:
            auc = float("nan")
            ap = float("nan")
        rows.append(
            {
                "review_policy": review_policy,
                "model": model_name,
                "scope": scope,
                "n_review": int(ok.sum()),
                "wrong_n": int(y[ok].sum()),
                "wrong_rate": float(y[ok].mean()) if ok.sum() else float("nan"),
                "error_risk_auc": auc,
                "error_risk_ap": ap,
                "risk_mean": float(np.nanmean(s[ok])),
            }
        )
    return rows


def workflow_grid(current: pd.DataFrame, risk: np.ndarray, model_name: str, review_policy: str) -> list[dict[str, object]]:
    rows = []
    y = current["label_idx"].to_numpy(int)
    base = current["final_pred"].to_numpy(int)
    review = current["review_flag"].to_numpy(bool)
    for t in THRESHOLDS:
        flip = review & np.isfinite(risk) & (risk >= t)
        remaining_review = review & ~flip
        pred = base.copy()
        pred[flip] = 1 - pred[flip]
        pred[remaining_review] = y[remaining_review]
        rescued = flip & (base != y) & (pred == y)
        hurt = flip & (base == y) & (pred != y)
        for scope, mask in [
            ("old_data", current["domain"].eq("old_data").to_numpy()),
            ("third_batch", current["domain"].eq("third_batch").to_numpy()),
            ("strict_external", current["domain"].eq("strict_external").to_numpy()),
            ("all_domains", current["domain"].isin(["old_data", "third_batch", "strict_external"]).to_numpy()),
        ]:
            m = metrics(y[mask], pred[mask], risk[mask])
            rows.append(
                {
                    "review_policy": review_policy,
                    "model": model_name,
                    "flip_risk_threshold": float(t),
                    "scope": scope,
                    "n": int(mask.sum()),
                    "original_review_rate": float(review[mask].mean()),
                    "auto_flip_n": int(flip[mask].sum()),
                    "auto_flip_rate": float(flip[mask].mean()),
                    "remaining_review_n": int(remaining_review[mask].sum()),
                    "remaining_review_rate": float(remaining_review[mask].mean()),
                    "rescued_n": int(rescued[mask].sum()),
                    "hurt_n": int(hurt[mask].sum()),
                    "auto_flip_error_n": int(hurt[mask].sum()),
                    "accuracy": float(m["accuracy"]),
                    "balanced_accuracy": float(m["balanced_accuracy"]),
                    "f1": float(m["f1"]),
                    "auc": float(m["auc"]),
                    "fn": int(m["fn"]),
                    "fp": int(m["fp"]),
                }
            )
    return rows


def select_threshold(summary: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (policy, model), sub in summary.groupby(["review_policy", "model"], sort=False):
        internal = sub.loc[sub["scope"].isin(["old_data", "third_batch"])]
        agg = (
            internal.groupby("flip_risk_threshold", as_index=False)
            .agg(
                internal_auto_flip_n=("auto_flip_n", "sum"),
                internal_remaining_review_n=("remaining_review_n", "sum"),
                internal_rescued_n=("rescued_n", "sum"),
                internal_hurt_n=("hurt_n", "sum"),
            )
        )
        zero_hurt = agg.loc[(agg["internal_hurt_n"].eq(0)) & (agg["internal_rescued_n"].ge(1))].copy()
        if not zero_hurt.empty:
            chosen = zero_hurt.sort_values(
                ["internal_rescued_n", "internal_remaining_review_n", "internal_auto_flip_n"],
                ascending=[False, True, False],
            ).iloc[0]
            status = "internal_zero_hurt_with_rescue"
        else:
            viable = agg.loc[agg["internal_auto_flip_n"].ge(1)].copy()
            if viable.empty:
                continue
            viable["net_rescue"] = viable["internal_rescued_n"] - viable["internal_hurt_n"]
            chosen = viable.sort_values(
                ["net_rescue", "internal_hurt_n", "internal_rescued_n"],
                ascending=[False, True, False],
            ).iloc[0]
            status = "best_internal_net_rescue"
        selected.append(
            {
                "review_policy": policy,
                "model": model,
                "flip_risk_threshold": float(chosen["flip_risk_threshold"]),
                "selection_status": status,
                "internal_auto_flip_n": int(chosen["internal_auto_flip_n"]),
                "internal_rescued_n": int(chosen["internal_rescued_n"]),
                "internal_hurt_n": int(chosen["internal_hurt_n"]),
                "internal_remaining_review_n": int(chosen["internal_remaining_review_n"]),
            }
        )
    selected_df = pd.DataFrame(selected)
    return summary.merge(selected_df, on=["review_policy", "model", "flip_risk_threshold"], how="inner")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases, case_feature_cols = load_base_cases()
    train_rows = load_error_enriched_rows(cases, case_feature_cols)
    cand_cols = [c for c in train_rows.columns if c.startswith("cand_")]
    x_cols = feature_cols(train_rows, case_feature_cols)

    quality_rows = []
    grid_rows = []
    case_outputs = []
    for review_policy in REVIEW_POLICIES:
        current = current_rows(cases, case_feature_cols, cand_cols, review_policy)
        for name, model in make_models(seed=175).items():
            risk = predict_current_risk(model, train_rows, current, x_cols)
            quality_rows += risk_quality(current, risk, name, review_policy)
            grid_rows += workflow_grid(current, risk, name, review_policy)
            out = current[["domain", "case_id", "original_case_id", "task_l6_label", "label_idx", "final_pred", "fold_id", "review_policy", "review_flag", "candidate_wrong"]].copy()
            out["model"] = name
            out["flip_risk"] = risk
            case_outputs.append(out)

    quality = pd.DataFrame(quality_rows)
    grid = pd.DataFrame(grid_rows)
    selected = select_threshold(grid)
    cases_out = pd.concat(case_outputs, ignore_index=True)

    quality.to_csv(OUT_DIR / "v175_error_risk_quality.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(OUT_DIR / "v175_flip_risk_threshold_grid.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v175_selected_flip_risk_summary.csv", index=False, encoding="utf-8-sig")
    cases_out.to_csv(OUT_DIR / "v175_flip_risk_case_outputs.csv", index=False, encoding="utf-8-sig")

    all_sel = selected.loc[selected["scope"].eq("all_domains")].copy()
    if not all_sel.empty:
        best = all_sel.sort_values(
            ["rescued_n", "hurt_n", "balanced_accuracy", "remaining_review_rate"],
            ascending=[False, True, False, True],
        ).iloc[0]
    else:
        best = pd.Series(dtype=object)

    md = [
        "# v175 Error-enriched Flip-risk Corrector",
        "",
        "## Design",
        "",
        "- Training samples come from v135 old+third OOF predictions across 22 candidate models, so the flip-risk learner sees many model errors rather than only the 9 current reviewed errors.",
        "- Current old+third cases are predicted fold-wise with their case fold held out. Strict external is evaluated with the internal-trained learner only.",
        "- Action: only cases already in the review pool can be automatically flipped; unflipped reviewed cases remain rejected/reviewed.",
        "",
        "## Best Selected Point",
        "",
    ]
    if not best.empty:
        md.append(
            f"- `{best['model']}` / `{best['review_policy']}` at threshold {float(best['flip_risk_threshold']):.2f}: "
            f"rescued {int(best['rescued_n'])}, hurt {int(best['hurt_n'])}, BAcc {100 * float(best['balanced_accuracy']):.2f}%, "
            f"remaining review {100 * float(best['remaining_review_rate']):.2f}%."
        )
    else:
        md.append("- No selected point.")
    md += [
        "",
        "## Files",
        "",
        "- v175_error_risk_quality.csv",
        "- v175_flip_risk_threshold_grid.csv",
        "- v175_selected_flip_risk_summary.csv",
        "- v175_flip_risk_case_outputs.csv",
    ]
    (OUT_DIR / "v175_error_enriched_flip_risk.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "train_rows": int(len(train_rows)),
        "train_cases": int(train_rows["case_id"].nunique()),
        "candidate_count": int(train_rows["candidate"].nunique()),
        "train_wrong_rows": int(train_rows["candidate_wrong"].sum()),
        "selected_rows": int(len(selected)),
        "best_all_domain": best.to_dict() if not best.empty else None,
    }
    (OUT_DIR / "v175_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v175] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
