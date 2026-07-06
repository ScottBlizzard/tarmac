from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_grosspath_rc_v134c_cascade_auto_corrector_20260527 import build_internal_external


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v143_image_feature_error_router_20260527"
V135_PREDS = ROOT / "outputs" / "grosspath_rc_v135_stage1_base_candidate_scan_20260527" / "v135_stage1_candidate_predictions.csv"

OLD_FEAT_DIR = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_gross_feature_runs" / "75_wpc_plus_image_stats_20260521"
THIRD_FEAT_DIR = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_external_runs" / "13_third_batch_wpc_plus_image_stats_64style_20260521"
EXTERNAL_FEAT_DIR = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_external_runs" / "20_external_thymoma_carcinoma_64style_wpc_20260522"

BUDGETS = np.array([0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
BASE_CANDIDATES = ["robust_prob", "prob_mean_core"]


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
    out: dict[str, float | int] = {
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
            pass
    return out


def choose_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        score = balanced_accuracy_score(y, (prob >= t).astype(int))
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, score))


def load_feature_dir(feat_dir: Path, table_name: str) -> pd.DataFrame:
    table = pd.read_csv(feat_dir / table_name, dtype={"case_id": str})
    x = np.load(feat_dir / ("case_dino_concat_features.npy" if table_name.startswith("case_") else "third_batch_dino_concat_features.npy"))
    feat = pd.DataFrame(x, columns=[f"feat_{i:04d}" for i in range(x.shape[1])])
    feat["case_id"] = table["case_id"].astype(str).to_numpy()
    return feat


def load_features() -> pd.DataFrame:
    old = load_feature_dir(OLD_FEAT_DIR, "case_dino_concat_feature_table.csv")
    third = load_feature_dir(THIRD_FEAT_DIR, "third_batch_dino_concat_feature_table.csv")
    external = load_feature_dir(EXTERNAL_FEAT_DIR, "third_batch_dino_concat_feature_table.csv")
    old["feature_domain"] = "old"
    third["feature_domain"] = "third"
    external["feature_domain"] = "strict_external"
    return pd.concat([old, third, external], ignore_index=True)


def load_probs(scope: str) -> pd.DataFrame:
    preds = pd.read_csv(V135_PREDS, dtype={"case_id": str, "original_case_id": str})
    preds = preds.loc[preds["scope"].eq(scope) & preds["objective"].eq("balanced_accuracy")].copy()
    preds = preds.loc[preds["candidate"].isin(BASE_CANDIDATES)].copy()
    return preds.pivot_table(index="case_id", columns="candidate", values="prob_high", aggfunc="first").reset_index()


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    internal, external = build_internal_external()
    external = external.loc[external["strict_task7_eval"].astype(int).eq(1)].copy()
    external = external.loc[~external["task_l6_label"].astype(str).eq("MNT_assumed_low")].copy()
    internal = internal.merge(load_probs("internal_oof_old_third"), on="case_id", how="left", validate="one_to_one", suffixes=("", "_v135"))
    external = external.merge(load_probs("strict_external_locked"), on="case_id", how="left", validate="one_to_one", suffixes=("", "_v135"))

    features = load_features()
    feat_cols = [c for c in features.columns if c.startswith("feat_")]
    internal = internal.merge(features.drop(columns=["feature_domain"]), on="case_id", how="inner", validate="one_to_one")
    external = external.merge(features.drop(columns=["feature_domain"]), on="case_id", how="inner", validate="one_to_one")
    for col in BASE_CANDIDATES:
        if col not in internal and f"{col}_v135" in internal:
            internal[col] = internal[f"{col}_v135"]
        if col not in external and f"{col}_v135" in external:
            external[col] = external[f"{col}_v135"]
        internal[col] = pd.to_numeric(internal[col], errors="coerce").fillna(0.5)
        external[col] = pd.to_numeric(external[col], errors="coerce").fillna(0.5)
    return internal.reset_index(drop=True), external.reset_index(drop=True), feat_cols


def make_models(seed: int) -> dict[str, object]:
    return {
        "pca64_logreg_c03": make_pipeline(
            StandardScaler(),
            PCA(n_components=64, random_state=seed),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        ),
        "extra_d3": ExtraTreesClassifier(
            n_estimators=320,
            max_depth=3,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def oof_and_external_scores(
    model: object,
    x: np.ndarray,
    y_target: np.ndarray,
    folds: np.ndarray,
    x_ext: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y_target), dtype=float)
    global_rate = float(np.mean(y_target))
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        if len(np.unique(y_target[train])) < 2:
            oof[test] = global_rate
            continue
        clf = clone(model)
        clf.fit(x[train], y_target[train])
        oof[test] = clf.predict_proba(x[test])[:, 1]
    if len(np.unique(y_target)) < 2:
        ext = np.full(x_ext.shape[0], global_rate, dtype=float)
    else:
        clf = clone(model)
        clf.fit(x, y_target)
        ext = clf.predict_proba(x_ext)[:, 1]
    return oof, ext


def base_confidence(prob: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.where(pred == 1, prob, 1.0 - prob)


def review_budget_rows(
    scope: str,
    base_name: str,
    router_name: str,
    y: np.ndarray,
    base_pred: np.ndarray,
    base_prob: np.ndarray,
    risk: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_wrong = base_pred != y
    order = np.argsort(-risk)
    n = len(y)
    for budget in BUDGETS:
        review = np.zeros(n, dtype=bool)
        k = int(round(float(budget) * n))
        if k > 0:
            review[order[:k]] = True
        auto = ~review
        system_pred = base_pred.copy()
        system_pred[review] = y[review]
        row = {
            "scope": scope,
            "base_model": base_name,
            "router": router_name,
            "review_budget": float(budget),
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "auto_n": int(auto.sum()),
            "auto_rate": float(auto.mean()),
            "captured_errors": int((review & base_wrong).sum()),
            "total_base_errors": int(base_wrong.sum()),
            "error_capture_rate": float((review & base_wrong).sum() / max(1, base_wrong.sum())),
            "review_clean_n": int((review & ~base_wrong).sum()),
        }
        row.update({f"auto_{k2}": v for k2, v in metrics(y[auto], base_pred[auto], base_prob[auto]).items()})
        row.update({f"system_if_review_corrected_{k2}": v for k2, v in metrics(y, system_pred, base_prob).items()})
        rows.append(row)
    return rows


def risk_summary_row(scope: str, base_name: str, target: str, router: str, y_target: np.ndarray, score: np.ndarray) -> dict[str, object]:
    return {
        "scope": scope,
        "base_model": base_name,
        "target": target,
        "router": router,
        "positive_n": int(y_target.sum()),
        "positive_rate": float(np.mean(y_target)),
        "auroc": safe_auc(y_target, score),
        "average_precision": safe_ap(y_target, score),
    }


def run_for_base(
    base_name: str,
    internal: pd.DataFrame,
    external: pd.DataFrame,
    feat_cols: list[str],
    models: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    y = internal["label_idx"].astype(int).to_numpy()
    y_ext = external["label_idx"].astype(int).to_numpy()
    folds = internal["fold_id"].astype(int).to_numpy()
    p = internal[base_name].to_numpy(float)
    p_ext = external[base_name].to_numpy(float)
    threshold = choose_threshold(y, p)
    pred = (p >= threshold).astype(int)
    pred_ext = (p_ext >= threshold).astype(int)

    targets = {
        "any_error": (pred != y).astype(int),
        "fn_error": ((y == 1) & (pred == 0)).astype(int),
        "fp_error": ((y == 0) & (pred == 1)).astype(int),
    }
    targets_ext = {
        "any_error": (pred_ext != y_ext).astype(int),
        "fn_error": ((y_ext == 1) & (pred_ext == 0)).astype(int),
        "fp_error": ((y_ext == 0) & (pred_ext == 1)).astype(int),
    }
    x = np.nan_to_num(internal[feat_cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    x_ext = np.nan_to_num(external[feat_cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)

    risk_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    case_scores = internal[["case_id", "original_case_id", "domain", "fold_id", "task_l6_label", "label_idx"]].copy()
    case_scores["scope"] = "internal_oof"
    case_scores["base_model"] = base_name
    case_scores["base_threshold"] = threshold
    case_scores["base_prob"] = p
    case_scores["base_pred"] = pred
    case_scores["base_correct"] = pred == y

    case_scores_ext = external[["case_id", "original_case_id", "domain", "task_l6_label", "label_idx"]].copy()
    case_scores_ext["fold_id"] = -1
    case_scores_ext["scope"] = "strict_external_locked"
    case_scores_ext["base_model"] = base_name
    case_scores_ext["base_threshold"] = threshold
    case_scores_ext["base_prob"] = p_ext
    case_scores_ext["base_pred"] = pred_ext
    case_scores_ext["base_correct"] = pred_ext == y_ext

    low_conf = 1.0 - base_confidence(p, pred)
    low_conf_ext = 1.0 - base_confidence(p_ext, pred_ext)
    for scope, yy, ppred, pprob, yy_target, score in [
        ("internal_oof", y, pred, p, targets["any_error"], low_conf),
        ("strict_external_locked", y_ext, pred_ext, p_ext, targets_ext["any_error"], low_conf_ext),
    ]:
        risk_rows.append(risk_summary_row(scope, base_name, "any_error", "heuristic_low_conf", yy_target, score))
        budget_rows.extend(review_budget_rows(scope, base_name, "heuristic_low_conf", yy, ppred, pprob, score))

    directional_internal: dict[str, np.ndarray] = {}
    directional_external: dict[str, np.ndarray] = {}
    for target_name, target in targets.items():
        for model_name, model in models.items():
            oof, ext = oof_and_external_scores(model, x, target, folds, x_ext)
            risk_rows.append(risk_summary_row("internal_oof", base_name, target_name, model_name, target, oof))
            risk_rows.append(risk_summary_row("strict_external_locked", base_name, target_name, model_name, targets_ext[target_name], ext))
            if target_name == "any_error":
                budget_rows.extend(review_budget_rows("internal_oof", base_name, f"{model_name}:any", y, pred, p, oof))
                budget_rows.extend(review_budget_rows("strict_external_locked", base_name, f"{model_name}:any", y_ext, pred_ext, p_ext, ext))
                case_scores[f"{model_name}_any_risk"] = oof
                case_scores_ext[f"{model_name}_any_risk"] = ext
            else:
                directional_internal[f"{model_name}:{target_name}"] = oof
                directional_external[f"{model_name}:{target_name}"] = ext

    for model_name in models:
        fn_oof = directional_internal[f"{model_name}:fn_error"]
        fp_oof = directional_internal[f"{model_name}:fp_error"]
        fn_ext = directional_external[f"{model_name}:fn_error"]
        fp_ext = directional_external[f"{model_name}:fp_error"]
        dir_score = np.where(pred == 0, fn_oof, fp_oof)
        dir_score_ext = np.where(pred_ext == 0, fn_ext, fp_ext)
        router = f"{model_name}:directional"
        risk_rows.append(risk_summary_row("internal_oof", base_name, "any_error", router, targets["any_error"], dir_score))
        risk_rows.append(risk_summary_row("strict_external_locked", base_name, "any_error", router, targets_ext["any_error"], dir_score_ext))
        budget_rows.extend(review_budget_rows("internal_oof", base_name, router, y, pred, p, dir_score))
        budget_rows.extend(review_budget_rows("strict_external_locked", base_name, router, y_ext, pred_ext, p_ext, dir_score_ext))
        case_scores[f"{model_name}_directional_risk"] = dir_score
        case_scores_ext[f"{model_name}_directional_risk"] = dir_score_ext

    return risk_rows, budget_rows, pd.concat([case_scores, case_scores_ext], ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    internal, external, feat_cols = prepare()
    models = make_models(20260527)
    all_risk: list[dict[str, object]] = []
    all_budget: list[dict[str, object]] = []
    case_tables: list[pd.DataFrame] = []
    for base_name in BASE_CANDIDATES:
        rows, budgets, cases = run_for_base(base_name, internal, external, feat_cols, models)
        all_risk.extend(rows)
        all_budget.extend(budgets)
        case_tables.append(cases)

    risk = pd.DataFrame(all_risk).sort_values(["scope", "base_model", "target", "auroc"], ascending=[True, True, True, False])
    budgets = pd.DataFrame(all_budget).sort_values(
        ["scope", "base_model", "review_budget", "system_if_review_corrected_balanced_accuracy"],
        ascending=[True, True, True, False],
    )
    cases = pd.concat(case_tables, ignore_index=True)
    risk.to_csv(OUT_DIR / "v143_image_feature_error_risk_summary.csv", index=False, encoding="utf-8-sig")
    budgets.to_csv(OUT_DIR / "v143_image_feature_review_budget_curve.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v143_image_feature_case_risks.csv", index=False, encoding="utf-8-sig")
    report = {
        "internal_n": int(len(internal)),
        "strict_external_n": int(len(external)),
        "n_features": int(len(feat_cols)),
        "base_candidates": BASE_CANDIDATES,
        "feature_source": "75_wpc_plus_image_stats / matched third and strict external WPC features",
    }
    (OUT_DIR / "v143_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v143] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
