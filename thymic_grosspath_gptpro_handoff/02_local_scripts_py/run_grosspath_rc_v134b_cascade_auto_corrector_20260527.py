from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v134_cascade_auto_corrector_20260527"

ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
MM_INTERNAL = ROOT / "outputs" / "grosspath_rc_v101_multimodel_oof_fn_sentinel_20260527" / "v101_internal_cases_with_multimodel_features.csv"
WHOLE_INTERNAL = ROOT / "outputs" / "grosspath_rc_v106_external_compatible_wholecrop_scorecard_20260527" / "v106_internal_cases_with_flags.csv"
WHOLE_EXTERNAL = ROOT / "outputs" / "grosspath_rc_v106_external_compatible_wholecrop_scorecard_20260527" / "v106_strict_external_cases_with_flags.csv"
SELECTED_OOF = (
    ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_adaptation_runs"
    / "44_old_third_unified_feature_cv_20260523"
    / "selected_unified_feature_oof_predictions.csv"
)
LOCKED_EXTERNAL_MM = (
    ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_external_runs"
    / "70_locked_536567_fullprob_external_eval_20260523"
    / "67_old_third_no64_meta_stack_plus_dinov3vitl_ft_20260523_external_predictions.csv"
)

TARGET_ACCEPT_ACCS = [0.88, 0.90, 0.92, 0.95]
CORRECTOR_CONF_THRESHOLDS = [0.00, 0.55, 0.60, 0.65, 0.70]
MIN_ACCEPT_TRAIN = 30


def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.2f}%"


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float | int]:
    if len(y) == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "sensitivity_high": np.nan,
            "specificity_low": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "auc": np.nan,
        }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": np.nan,
    }
    if prob is not None and len(np.unique(y)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y, prob))
        except ValueError:
            out["auc"] = np.nan
    return out


def fit_prob(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    clf = clone(model)
    clf.fit(x_train, y_train)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x_test)[:, 1]
    score = clf.decision_function(x_test)
    return 1.0 / (1.0 + np.exp(-score))


def models(seed: int) -> dict[str, object]:
    return {
        "logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        ),
        "extra_depth3": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gb_depth2": GradientBoostingClassifier(max_depth=2, learning_rate=0.035, n_estimators=120, random_state=seed),
    }


def select_threshold(train_score: np.ndarray, train_correct: np.ndarray, target_acc: float, min_accept: int) -> dict[str, float | int]:
    order = np.argsort(-train_score, kind="mergesort")
    score = train_score[order]
    correct = train_correct[order].astype(float)
    cumsum = np.cumsum(correct)
    ks = np.arange(1, len(score) + 1)
    acc = cumsum / ks
    ok = (acc >= target_acc) & (ks >= min_accept)
    if not ok.any():
        return {"threshold": np.inf, "train_accept_n": 0, "train_accept_acc": np.nan}
    idx = np.where(ok)[0][-1]
    return {"threshold": float(score[idx]), "train_accept_n": int(idx + 1), "train_accept_acc": float(acc[idx])}


def base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in [
        "main_prob",
        "robust_prob",
        "prob_mean_core",
        "wholecrop_prob",
        "selected_unified_prob",
        "selected_dinov3_prob",
        "mm_prob_mean",
        "mm_prob_median",
        "mm_prob_max",
        "mm_prob_p75",
        "mm_prob_std",
        "quality_score",
    ]:
        out[col] = pd.to_numeric(df.get(col, 0.5), errors="coerce").fillna(0.5).astype(float)
    probs = out[["main_prob", "robust_prob", "prob_mean_core", "wholecrop_prob", "selected_unified_prob", "selected_dinov3_prob"]]
    out["prob_avg6"] = probs.mean(axis=1)
    out["prob_std6"] = probs.std(axis=1)
    out["prob_range6"] = probs.max(axis=1) - probs.min(axis=1)
    out["core_agree_count"] = pd.to_numeric(df.get("core_agree_count", 0), errors="coerce").fillna(0).astype(float)
    out["base_pred"] = pd.to_numeric(df["base_pred"], errors="coerce").fillna(0).astype(float)
    out["base_conf_core"] = np.where(out["base_pred"].eq(1), out["prob_mean_core"], 1.0 - out["prob_mean_core"])
    out["base_conf_main"] = np.where(out["base_pred"].eq(1), out["main_prob"], 1.0 - out["main_prob"])
    out["base_entropy_core"] = entropy(out["prob_mean_core"].to_numpy())
    out["main_robust_gap"] = np.abs(out["main_prob"] - out["robust_prob"])
    out["whole_core_gap"] = np.abs(out["wholecrop_prob"] - out["prob_mean_core"])
    out["selected_core_gap"] = np.abs(out["selected_unified_prob"] - out["prob_mean_core"])
    out["domain_is_third"] = df["domain"].eq("third_batch").astype(float).to_numpy()
    out["domain_is_external"] = df["domain"].eq("strict_external").astype(float).to_numpy()
    if "task_l6_label" in df:
        subtype = pd.get_dummies(df["task_l6_label"].fillna("unknown"), prefix="subtype", dtype=float)
        out = pd.concat([out, subtype], axis=1)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_internal_external() -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = pd.read_csv(ROUTES, dtype={"case_id": str, "original_case_id": str})
    pure = routes.loc[routes["policy"].eq("pure_auto")].copy()
    pure = pure.rename(columns={"final_pred": "base_pred", "final_correct": "base_correct"})
    pure["base_wrong"] = pure["base_pred"].astype(int).ne(pure["label_idx"].astype(int))
    internal = pure.loc[pure["domain"].isin(["old_data", "third_batch"])].copy()
    external = pure.loc[pure["domain"].eq("strict_external")].copy()

    folds = pd.read_csv(SELECTED_OOF, dtype={"case_id": str, "original_case_id": str})[["case_id", "fold_id"]]
    internal = internal.merge(folds, on="case_id", how="left", validate="one_to_one")
    if internal["fold_id"].isna().any():
        missing = internal.loc[internal["fold_id"].isna(), "case_id"].head().tolist()
        raise RuntimeError(f"Missing fold ids: {missing}")
    internal["fold_id"] = internal["fold_id"].astype(int)

    mm = pd.read_csv(MM_INTERNAL, dtype={"case_id": str})
    mm_cols = [
        "case_id",
        "mm_prob_mean",
        "mm_prob_median",
        "mm_prob_max",
        "mm_prob_p75",
        "mm_prob_std",
        "selected_unified_prob",
        "selected_dinov3_prob",
    ]
    internal = internal.merge(mm[mm_cols].drop_duplicates("case_id"), on="case_id", how="left", validate="one_to_one")

    whole_i = pd.read_csv(WHOLE_INTERNAL, dtype={"case_id": str})[["case_id", "wholecrop_prob"]].drop_duplicates("case_id")
    internal = internal.merge(whole_i, on="case_id", how="left", validate="one_to_one")

    whole_e = pd.read_csv(WHOLE_EXTERNAL, dtype={"case_id": str})
    ext_cols = ["case_id", "wholecrop_refit_prob", "strict_task7_eval"]
    ext_cols = [c for c in ext_cols if c in whole_e.columns]
    external = external.merge(whole_e[ext_cols].drop_duplicates("case_id"), on="case_id", how="left", validate="one_to_one")
    external["wholecrop_prob"] = pd.to_numeric(external.get("wholecrop_refit_prob", np.nan), errors="coerce")
    if "strict_task7_eval" not in external:
        external["strict_task7_eval"] = 1

    if LOCKED_EXTERNAL_MM.exists():
        locked = pd.read_csv(LOCKED_EXTERNAL_MM, dtype={"case_id": str})
        rename = {}
        if "selected_base_prob" in locked:
            rename["selected_base_prob"] = "selected_unified_prob"
        if "dinov3vitl_ft_prob" in locked:
            rename["dinov3vitl_ft_prob"] = "selected_dinov3_prob"
        keep = ["case_id"] + list(rename)
        external = external.merge(locked[keep].rename(columns=rename).drop_duplicates("case_id"), on="case_id", how="left")

    for df in [internal, external]:
        for col in ["selected_unified_prob", "selected_dinov3_prob", "mm_prob_mean", "mm_prob_median", "mm_prob_max", "mm_prob_p75", "mm_prob_std"]:
            if col not in df:
                df[col] = np.nan
        df["selected_unified_prob"] = pd.to_numeric(df["selected_unified_prob"], errors="coerce").fillna(df["prob_mean_core"])
        df["selected_dinov3_prob"] = pd.to_numeric(df["selected_dinov3_prob"], errors="coerce").fillna(df["prob_mean_core"])
        for col in ["mm_prob_mean", "mm_prob_median", "mm_prob_max", "mm_prob_p75"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df["prob_mean_core"])
        df["mm_prob_std"] = pd.to_numeric(df["mm_prob_std"], errors="coerce").fillna(0.0)
        df["wholecrop_prob"] = pd.to_numeric(df["wholecrop_prob"], errors="coerce").fillna(df["prob_mean_core"])

    return internal.reset_index(drop=True), external.reset_index(drop=True)


def apply_pipeline(
    df: pd.DataFrame,
    accept_score: np.ndarray,
    accept_threshold: float,
    corr_prob: np.ndarray,
    corr_conf_min: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    y = df["label_idx"].astype(int).to_numpy()
    base_pred = df["base_pred"].astype(int).to_numpy()
    base_prob = pd.to_numeric(df["prob_mean_core"], errors="coerce").fillna(0.5).to_numpy()
    stage1_accept = accept_score >= accept_threshold
    review_pool = ~stage1_accept
    corr_pred = (corr_prob >= 0.5).astype(int)
    corr_conf = np.maximum(corr_prob, 1.0 - corr_prob)
    stage2_accept = review_pool & (corr_conf >= corr_conf_min)
    reject = review_pool & (~stage2_accept)
    final_pred_auto = base_pred.copy()
    final_prob_auto = base_prob.copy()
    final_pred_auto[stage2_accept] = corr_pred[stage2_accept]
    final_prob_auto[stage2_accept] = corr_prob[stage2_accept]
    automated = stage1_accept | stage2_accept
    base_wrong = base_pred != y
    final_wrong = final_pred_auto != y
    rescued = review_pool & base_wrong & (~final_wrong)
    hurt = review_pool & (~base_wrong) & final_wrong

    case = df[["domain", "case_id", "original_case_id", "task_l6_label", "label_idx", "base_pred", "base_correct"]].copy()
    case["accept_score"] = accept_score
    case["stage1_accept"] = stage1_accept
    case["stage2_prob_high"] = corr_prob
    case["stage2_conf"] = corr_conf
    case["stage2_accept"] = stage2_accept
    case["reject"] = reject
    case["final_auto_pred"] = final_pred_auto
    case["automated"] = automated
    case["rescued"] = rescued
    case["hurt"] = hurt
    case["final_auto_correct"] = final_pred_auto == y

    auto_metrics = metrics(y[automated], final_pred_auto[automated], final_prob_auto[automated]) if automated.any() else metrics(np.array([], dtype=int), np.array([], dtype=int))
    stage1_metrics = metrics(y[stage1_accept], base_pred[stage1_accept], base_prob[stage1_accept]) if stage1_accept.any() else metrics(np.array([], dtype=int), np.array([], dtype=int))
    stage2_metrics = metrics(y[stage2_accept], final_pred_auto[stage2_accept], final_prob_auto[stage2_accept]) if stage2_accept.any() else metrics(np.array([], dtype=int), np.array([], dtype=int))
    full_no_reject_pred = final_pred_auto.copy()
    full_no_reject_pred[reject] = y[reject]  # deployment interpretation if unresolved cases go to doctor.
    system_metrics = metrics(y, full_no_reject_pred, final_prob_auto)

    row: dict[str, object] = {
        "n": int(len(df)),
        "stage1_accept_n": int(stage1_accept.sum()),
        "stage1_accept_rate": float(stage1_accept.mean()),
        "stage2_accept_n": int(stage2_accept.sum()),
        "stage2_accept_rate": float(stage2_accept.mean()),
        "reject_n": int(reject.sum()),
        "reject_rate": float(reject.mean()),
        "automated_n": int(automated.sum()),
        "automated_rate": float(automated.mean()),
        "rescued_n": int(rescued.sum()),
        "hurt_n": int(hurt.sum()),
        "rescued_fn_n": int((rescued & df["label_idx"].eq(1).to_numpy() & (base_pred == 0)).sum()),
        "rescued_fp_n": int((rescued & df["label_idx"].eq(0).to_numpy() & (base_pred == 1)).sum()),
        "hurt_fn_n": int((hurt & df["label_idx"].eq(1).to_numpy() & (final_pred_auto == 0)).sum()),
        "hurt_fp_n": int((hurt & df["label_idx"].eq(0).to_numpy() & (final_pred_auto == 1)).sum()),
    }
    row.update({f"stage1_{k}": v for k, v in stage1_metrics.items()})
    row.update({f"stage2_{k}": v for k, v in stage2_metrics.items()})
    row.update({f"automated_{k}": v for k, v in auto_metrics.items()})
    row.update({f"system_if_reject_corrected_{k}": v for k, v in system_metrics.items()})
    return case, row


def evaluate_config(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    feature_set: str,
    gate_name: str,
    gate_model: object | None,
    corr_name: str,
    corr_model: object,
    target_accept_acc: float,
    corr_conf_min: float,
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object], pd.DataFrame]:
    x_all = base_feature_frame(internal)
    x_ext = base_feature_frame(external).reindex(columns=x_all.columns, fill_value=0.0)
    if feature_set == "core":
        cols = [c for c in x_all.columns if c in {"main_prob", "robust_prob", "prob_mean_core", "core_agree_count", "base_conf_core", "base_entropy_core", "main_robust_gap", "base_pred"}]
    elif feature_set == "multiview":
        cols = [c for c in x_all.columns if not c.startswith("subtype_") and c not in {"quality_score", "domain_is_external"}]
    elif feature_set == "multiview_subtype":
        cols = [c for c in x_all.columns if c != "domain_is_external"]
    else:
        raise ValueError(feature_set)
    x = x_all[cols].to_numpy(float)
    xe = x_ext[cols].to_numpy(float)
    y = internal["label_idx"].astype(int).to_numpy()
    base_pred = internal["base_pred"].astype(int).to_numpy()
    base_correct = (base_pred == y).astype(int)
    folds = internal["fold_id"].astype(int).to_numpy()

    accept_score = np.zeros(len(internal), dtype=float)
    corr_prob = np.zeros(len(internal), dtype=float)
    fold_rows: list[dict[str, object]] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        if gate_name == "base_conf_core":
            train_score = x_all.loc[train, "base_conf_core"].to_numpy(float)
            test_score = x_all.loc[test, "base_conf_core"].to_numpy(float)
        else:
            train_score = fit_prob(gate_model, x[train], base_correct[train], x[train])
            test_score = fit_prob(gate_model, x[train], base_correct[train], x[test])
        choice = select_threshold(train_score, base_correct[train], target_accept_acc, MIN_ACCEPT_TRAIN)
        accept_score[test] = test_score

        train_accept = train_score >= float(choice["threshold"])
        corr_train = train.copy()
        # Train the corrector on difficult/unaccepted training cases when possible, otherwise all training cases.
        if (~train_accept).sum() >= 40 and len(np.unique(y[train][~train_accept])) == 2:
            train_idx = np.where(train)[0][~train_accept]
        else:
            train_idx = np.where(train)[0]
        corr_prob[test] = fit_prob(corr_model, x[train_idx], y[train_idx], x[test])
        fold_rows.append({"fold_id": int(fold), **choice})

    threshold = np.nanmedian([r["threshold"] for r in fold_rows if np.isfinite(r["threshold"])])
    # The internal threshold is fold-specific. For case decisions, use the actual scores and fold thresholds by reapplying per fold.
    stage1_accept = np.zeros(len(internal), dtype=bool)
    for r in fold_rows:
        test = folds == int(r["fold_id"])
        stage1_accept[test] = accept_score[test] >= float(r["threshold"])
    # Use a synthetic threshold of 0.5 here and overwrite stage1 according to fold-specific decisions below.
    int_cases, int_row = apply_pipeline(internal, accept_score, 0.5, corr_prob, corr_conf_min)
    int_cases["stage1_accept"] = stage1_accept
    int_cases["stage2_accept"] = (~stage1_accept) & (int_cases["stage2_conf"].to_numpy(float) >= corr_conf_min)
    int_cases["reject"] = (~stage1_accept) & (~int_cases["stage2_accept"])
    int_cases["automated"] = int_cases["stage1_accept"] | int_cases["stage2_accept"]
    # Recompute metrics after fold-specific stage1 overwrite.
    int_cases, int_row = recompute_from_cases(internal, int_cases)

    if gate_name == "base_conf_core":
        ext_score = x_ext["base_conf_core"].to_numpy(float)
        full_train_score = x_all["base_conf_core"].to_numpy(float)
    else:
        full_train_score = fit_prob(gate_model, x, base_correct, x)
        ext_score = fit_prob(gate_model, x, base_correct, xe)
    ext_choice = select_threshold(full_train_score, base_correct, target_accept_acc, MIN_ACCEPT_TRAIN)
    ext_corr_prob = fit_prob(corr_model, x, y, xe)
    ext_cases, ext_row = apply_pipeline(external, ext_score, float(ext_choice["threshold"]), ext_corr_prob, corr_conf_min)

    meta = {
        "feature_set": feature_set,
        "gate": gate_name,
        "corrector": corr_name,
        "target_accept_acc": target_accept_acc,
        "corrector_conf_min": corr_conf_min,
    }
    int_row.update(meta)
    int_row["scope"] = "internal_nested_old_third"
    ext_row.update(meta)
    ext_row["scope"] = "strict_external_locked"
    int_cases = int_cases.assign(**meta, scope="internal_nested_old_third")
    ext_cases = ext_cases.assign(**meta, scope="strict_external_locked")
    return int_row, int_cases, ext_row, ext_cases


def recompute_from_cases(df: pd.DataFrame, cases: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    y = df["label_idx"].astype(int).to_numpy()
    base_pred = df["base_pred"].astype(int).to_numpy()
    base_prob = pd.to_numeric(df["prob_mean_core"], errors="coerce").fillna(0.5).to_numpy()
    final_pred = cases["final_auto_pred"].astype(int).to_numpy()
    final_prob = cases["stage2_prob_high"].astype(float).to_numpy()
    stage1 = cases["stage1_accept"].astype(bool).to_numpy()
    stage2 = cases["stage2_accept"].astype(bool).to_numpy()
    reject = cases["reject"].astype(bool).to_numpy()
    automated = stage1 | stage2
    base_wrong = base_pred != y
    final_wrong = final_pred != y
    rescued = (~stage1) & base_wrong & (~final_wrong)
    hurt = (~stage1) & (~base_wrong) & final_wrong
    cases["automated"] = automated
    cases["rescued"] = rescued
    cases["hurt"] = hurt
    cases["final_auto_correct"] = final_pred == y
    system_pred = final_pred.copy()
    system_prob = final_prob.copy()
    system_pred[reject] = y[reject]
    system_prob[stage1] = base_prob[stage1]
    row = {
        "n": int(len(df)),
        "stage1_accept_n": int(stage1.sum()),
        "stage1_accept_rate": float(stage1.mean()),
        "stage2_accept_n": int(stage2.sum()),
        "stage2_accept_rate": float(stage2.mean()),
        "reject_n": int(reject.sum()),
        "reject_rate": float(reject.mean()),
        "automated_n": int(automated.sum()),
        "automated_rate": float(automated.mean()),
        "rescued_n": int(rescued.sum()),
        "hurt_n": int(hurt.sum()),
        "rescued_fn_n": int((rescued & (y == 1) & (base_pred == 0)).sum()),
        "rescued_fp_n": int((rescued & (y == 0) & (base_pred == 1)).sum()),
        "hurt_fn_n": int((hurt & (y == 1) & (final_pred == 0)).sum()),
        "hurt_fp_n": int((hurt & (y == 0) & (final_pred == 1)).sum()),
    }
    row.update({f"stage1_{k}": v for k, v in metrics(y[stage1], base_pred[stage1], base_prob[stage1]).items()})
    row.update({f"stage2_{k}": v for k, v in metrics(y[stage2], final_pred[stage2], final_prob[stage2]).items()})
    row.update({f"automated_{k}": v for k, v in metrics(y[automated], final_pred[automated], system_prob[automated]).items()})
    row.update({f"system_if_reject_corrected_{k}": v for k, v in metrics(y, system_pred, system_prob).items()})
    return cases, row


def format_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col.endswith("_accuracy") or col in [
            "stage1_sensitivity_high",
            "stage1_specificity_low",
            "stage2_sensitivity_high",
            "stage2_specificity_low",
            "automated_sensitivity_high",
            "automated_specificity_low",
            "system_if_reject_corrected_sensitivity_high",
            "system_if_reject_corrected_specificity_low",
            "automated_auc",
        ]:
            out[col] = out[col].map(lambda x: pct(float(x)) if pd.notna(x) else "")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    internal, external = build_internal_external()
    external_strict = external.loc[external["strict_task7_eval"].astype(int).eq(1)].copy()
    ms = models(20260527)
    gate_models = {"base_conf_core": None, "gate_logreg": ms["logreg_c03"], "gate_extra": ms["extra_depth3"]}
    corr_models = {"corr_logreg": ms["logreg_c03"], "corr_extra": ms["extra_depth3"], "corr_gb": ms["gb_depth2"]}

    rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    for feature_set in ["core", "multiview", "multiview_subtype"]:
        for gate_name, gate_model in gate_models.items():
            for corr_name, corr_model in corr_models.items():
                for target in TARGET_ACCEPT_ACCS:
                    for conf in CORRECTOR_CONF_THRESHOLDS:
                        int_row, int_cases, ext_row, ext_cases = evaluate_config(
                            internal,
                            external_strict,
                            feature_set,
                            gate_name,
                            gate_model,
                            corr_name,
                            corr_model,
                            target,
                            conf,
                        )
                        rows.extend([int_row, ext_row])
                        # Keep full case outputs for promising or interpretable settings only to limit file size.
                        if target in {0.90, 0.95} and conf in {0.0, 0.65} and feature_set in {"core", "multiview"}:
                            case_frames.extend([int_cases, ext_cases])

    summary = pd.DataFrame(rows)
    sort_cols = ["scope", "automated_rate", "automated_balanced_accuracy", "system_if_reject_corrected_balanced_accuracy"]
    summary = summary.sort_values(sort_cols, ascending=[True, False, False, False])
    summary.to_csv(OUT_DIR / "v134_cascade_summary.csv", index=False, encoding="utf-8-sig")
    format_summary(summary).to_csv(OUT_DIR / "v134_cascade_summary_formatted.csv", index=False, encoding="utf-8-sig")
    pd.concat(case_frames, ignore_index=True).to_csv(OUT_DIR / "v134_selected_case_outputs.csv", index=False, encoding="utf-8-sig")

    internal_base = metrics(internal["label_idx"].astype(int).to_numpy(), internal["base_pred"].astype(int).to_numpy(), internal["prob_mean_core"].astype(float).to_numpy())
    external_base = metrics(external_strict["label_idx"].astype(int).to_numpy(), external_strict["base_pred"].astype(int).to_numpy(), external_strict["prob_mean_core"].astype(float).to_numpy())
    baseline = pd.DataFrame(
        [
            {"scope": "internal_old_third_pure_auto", **internal_base},
            {"scope": "strict_external_pure_auto", **external_base},
        ]
    )
    baseline.to_csv(OUT_DIR / "v134_pure_auto_baseline.csv", index=False, encoding="utf-8-sig")

    focus = summary.loc[
        (summary["scope"].eq("internal_nested_old_third"))
        & (summary["automated_rate"].between(0.70, 0.90))
        & (summary["automated_balanced_accuracy"].ge(0.90))
    ].sort_values(["automated_balanced_accuracy", "automated_rate", "reject_rate"], ascending=[False, False, True])
    focus.to_csv(OUT_DIR / "v134_internal_promising_70_90cov.csv", index=False, encoding="utf-8-sig")

    top = summary.sort_values(["scope", "system_if_reject_corrected_balanced_accuracy", "automated_rate"], ascending=[True, False, False]).groupby("scope").head(10)
    top.to_csv(OUT_DIR / "v134_top_by_scope.csv", index=False, encoding="utf-8-sig")
    format_summary(top).to_csv(OUT_DIR / "v134_top_by_scope_formatted.csv", index=False, encoding="utf-8-sig")

    report = {
        "boundary": {
            "development_data": "old_data + third_batch only; nested by existing fold_id",
            "strict_external": "胸腺瘤+癌 strict_task7_eval=1 only; no threshold tuning on external",
            "base_model": "pure_auto rows from v91, not manual-review-corrected workflow",
        },
        "n_internal": int(len(internal)),
        "n_external_strict": int(len(external_strict)),
        "pure_auto_internal": internal_base,
        "pure_auto_external_strict": external_base,
        "n_configs": int(len(summary) // 2),
    }
    (OUT_DIR / "v134_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote", OUT_DIR)
    print("Baseline")
    print(baseline.to_string(index=False))
    print("\nTop by scope")
    print(format_summary(top).to_string(index=False))
    print("\nPromising internal 70-90% automated coverage")
    print(format_summary(focus.head(20)).to_string(index=False))


if __name__ == "__main__":
    main()
