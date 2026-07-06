from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
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


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v142_concept_guided_error_router_20260527"
V141_PREDS = ROOT / "outputs" / "grosspath_rc_v141_image_concept_fusion_probe_20260527" / "image_concept_fusion_oof_predictions.csv"
CONCEPTS = ROOT / "outputs" / "grosspath_rc_v0_20260526" / "gross_concepts_v1.csv"

BUDGETS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60])


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


def flatten_prediction_table() -> pd.DataFrame:
    preds = pd.read_csv(V141_PREDS, dtype={"case_id": str, "original_case_id": str})
    meta_cols = ["case_id", "original_case_id", "domain", "fold_id", "task_l6_label", "label_idx"]
    meta = preds[meta_cols].drop_duplicates("case_id").copy()
    piv = preds.pivot_table(index="case_id", columns=["feature_set", "model"], values="prob_high", aggfunc="first")
    piv.columns = [f"p_{a}_{b}" for a, b in piv.columns]
    return meta.merge(piv.reset_index(), on="case_id", how="left", validate="one_to_one")


def load_concepts() -> tuple[pd.DataFrame, list[str]]:
    concepts = pd.read_csv(CONCEPTS, dtype={"original_case_id": str})
    skip = {"original_case_id", "sex", "gross_text", "diagnosis_text"}
    concept_cols = []
    for col in concepts.columns:
        if col in skip:
            continue
        values = pd.to_numeric(concepts[col], errors="coerce")
        if values.notna().any():
            concepts[col] = values
            concept_cols.append(col)
    return concepts[["original_case_id"] + concept_cols], concept_cols


def prepare() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = flatten_prediction_table()
    concepts, concept_cols = load_concepts()
    df = df.merge(concepts, on="original_case_id", how="left", validate="many_to_one")
    for col in concept_cols:
        if df[col].isna().all():
            df[col] = 0.0
        elif df[col].isna().any():
            df[col] = df[col].fillna(float(df[col].median()))

    image_cols = [c for c in df.columns if c.startswith("p_image_probs_only_")]
    concept_fusion_prob_cols = [c for c in df.columns if c.startswith("p_image_plus_concept_")]
    feature_sets = {
        "confidence_only": [],
        "image_prob_augmented": image_cols,
        "concept_augmented": concept_cols,
        "image_concept_augmented": image_cols + concept_cols,
        "fusion_concept_augmented": image_cols + concept_fusion_prob_cols + concept_cols,
    }
    return df, feature_sets


def make_models(seed: int) -> dict[str, object]:
    return {
        "logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        ),
        "extra_d3": ExtraTreesClassifier(
            n_estimators=220,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def oof_risk_scores(x: pd.DataFrame, y: np.ndarray, folds: np.ndarray, model: object) -> np.ndarray:
    out = np.zeros(len(y), dtype=float)
    values = x.to_numpy(float)
    global_rate = float(np.mean(y))
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        if len(np.unique(y[train])) < 2:
            out[test] = global_rate
            continue
        clf = clone(model)
        clf.fit(values[train], y[train])
        out[test] = clf.predict_proba(values[test])[:, 1]
    return out


def add_base_features(df: pd.DataFrame, base_prob: np.ndarray, base_pred: np.ndarray) -> pd.DataFrame:
    base_conf = np.where(base_pred == 1, base_prob, 1.0 - base_prob)
    entropy = -(base_prob * np.log(np.clip(base_prob, 1e-6, 1.0)) + (1.0 - base_prob) * np.log(np.clip(1.0 - base_prob, 1e-6, 1.0)))
    out = pd.DataFrame(
        {
            "base_prob": base_prob,
            "base_pred": base_pred.astype(float),
            "base_conf": base_conf,
            "base_uncertainty": 1.0 - base_conf,
            "base_margin_to_half": np.abs(base_prob - 0.5),
            "base_entropy": entropy,
        },
        index=df.index,
    )
    return out


def risk_eval_rows(y_target: np.ndarray, score: np.ndarray, base_name: str, target_name: str, feature_set: str, model_name: str) -> dict[str, object]:
    return {
        "base_model": base_name,
        "target": target_name,
        "feature_set": feature_set,
        "router_model": model_name,
        "positive_n": int(y_target.sum()),
        "positive_rate": float(np.mean(y_target)),
        "auroc": safe_auc(y_target, score),
        "average_precision": safe_ap(y_target, score),
    }


def evaluate_review_budget(
    df: pd.DataFrame,
    y: np.ndarray,
    base_pred: np.ndarray,
    base_prob: np.ndarray,
    base_name: str,
    router_name: str,
    risk_score: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_wrong = base_pred != y
    order = np.argsort(-risk_score)
    n = len(y)
    for budget in BUDGETS:
        k = int(round(float(budget) * n))
        review = np.zeros(n, dtype=bool)
        if k > 0:
            review[order[:k]] = True
        auto = ~review
        system_pred = base_pred.copy()
        system_pred[review] = y[review]
        row = {
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


def fit_full_extra_importance(x: pd.DataFrame, target: np.ndarray, target_name: str, feature_set: str) -> pd.DataFrame:
    if len(np.unique(target)) < 2 or x.shape[1] == 0:
        return pd.DataFrame()
    model = ExtraTreesClassifier(
        n_estimators=320,
        max_depth=3,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=20260527,
        n_jobs=-1,
    )
    model.fit(x.to_numpy(float), target)
    out = pd.DataFrame(
        {
            "target": target_name,
            "feature_set": feature_set,
            "feature": x.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return out.head(30)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_sets = prepare()
    y = df["label_idx"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()

    base_models = {
        "image_probs_only_extra_d3": "p_image_probs_only_extra_d3",
    }
    models = make_models(20260527)
    all_auc_rows: list[dict[str, object]] = []
    all_budget_rows: list[dict[str, object]] = []
    all_case_rows: list[pd.DataFrame] = []
    importance_rows: list[pd.DataFrame] = []

    for base_name, prob_col in base_models.items():
        if prob_col not in df:
            continue
        base_prob = df[prob_col].to_numpy(float)
        base_t = choose_threshold(y, base_prob)
        base_pred = (base_prob >= base_t).astype(int)
        base_wrong = (base_pred != y).astype(int)
        fn_error = ((y == 1) & (base_pred == 0)).astype(int)
        fp_error = ((y == 0) & (base_pred == 1)).astype(int)
        base_feat = add_base_features(df, base_prob, base_pred)

        baseline_risks = {
            "heuristic_low_conf_any": base_feat["base_uncertainty"].to_numpy(float),
            "heuristic_direction": np.where(base_pred == 0, base_prob, 1.0 - base_prob),
        }
        for router_name, risk in baseline_risks.items():
            all_auc_rows.append(risk_eval_rows(base_wrong, risk, base_name, "any_error", "heuristic", router_name))
            all_budget_rows.extend(evaluate_review_budget(df, y, base_pred, base_prob, base_name, router_name, risk))

        target_map = {
            "any_error": base_wrong,
            "fn_error": fn_error,
            "fp_error": fp_error,
        }
        per_target_scores: dict[tuple[str, str, str], np.ndarray] = {}
        for fs_name, cols in feature_sets.items():
            if fs_name == "confidence_only":
                x_base = base_feat
            else:
                x_base = pd.concat([base_feat, df[cols]], axis=1)
            x_base = x_base.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if fs_name in {"concept_augmented", "image_concept_augmented", "fusion_concept_augmented"}:
                importance_rows.append(fit_full_extra_importance(x_base, base_wrong, "any_error", f"{base_name}:{fs_name}"))

            for target_name, target in target_map.items():
                for model_name, model in models.items():
                    score = oof_risk_scores(x_base, target, folds, model)
                    per_target_scores[(fs_name, model_name, target_name)] = score
                    all_auc_rows.append(risk_eval_rows(target, score, base_name, target_name, fs_name, model_name))
                    if target_name == "any_error":
                        router_name = f"{fs_name}:{model_name}:any"
                        all_budget_rows.extend(evaluate_review_budget(df, y, base_pred, base_prob, base_name, router_name, score))

            for model_name in models:
                fn_score = per_target_scores[(fs_name, model_name, "fn_error")]
                fp_score = per_target_scores[(fs_name, model_name, "fp_error")]
                directional_score = np.where(base_pred == 0, fn_score, fp_score)
                router_name = f"{fs_name}:{model_name}:directional"
                all_auc_rows.append(risk_eval_rows(base_wrong, directional_score, base_name, "any_error", fs_name, f"{model_name}_directional"))
                all_budget_rows.extend(evaluate_review_budget(df, y, base_pred, base_prob, base_name, router_name, directional_score))

        case_scores = df[["case_id", "original_case_id", "domain", "fold_id", "task_l6_label", "label_idx"]].copy()
        case_scores["base_model"] = base_name
        case_scores["base_threshold"] = base_t
        case_scores["base_prob"] = base_prob
        case_scores["base_pred"] = base_pred
        case_scores["base_correct"] = base_pred == y
        case_scores["base_error_type"] = np.where(fn_error == 1, "FN_high_as_low", np.where(fp_error == 1, "FP_low_as_high", "correct"))
        case_scores["low_conf_risk"] = baseline_risks["heuristic_low_conf_any"]
        for key, score in per_target_scores.items():
            fs_name, model_name, target_name = key
            if fs_name == "image_concept_augmented" and model_name == "extra_d3":
                case_scores[f"{fs_name}_{model_name}_{target_name}_risk"] = score
        all_case_rows.append(case_scores)

    auc_summary = pd.DataFrame(all_auc_rows).sort_values(["base_model", "target", "auroc", "average_precision"], ascending=[True, True, False, False])
    budget_curve = pd.DataFrame(all_budget_rows).sort_values(["base_model", "review_budget", "system_if_review_corrected_balanced_accuracy", "error_capture_rate"], ascending=[True, True, False, False])
    case_scores = pd.concat(all_case_rows, ignore_index=True)
    importances = pd.concat([x for x in importance_rows if not x.empty], ignore_index=True) if importance_rows else pd.DataFrame()

    auc_summary.to_csv(OUT_DIR / "v142_error_risk_auc_summary.csv", index=False, encoding="utf-8-sig")
    budget_curve.to_csv(OUT_DIR / "v142_review_budget_curve.csv", index=False, encoding="utf-8-sig")
    case_scores.to_csv(OUT_DIR / "v142_case_risk_scores.csv", index=False, encoding="utf-8-sig")
    importances.to_csv(OUT_DIR / "v142_top_error_risk_features.csv", index=False, encoding="utf-8-sig")

    report = {
        "n_cases": int(len(df)),
        "concept_matched": int(df["concept_has_gross_text"].fillna(0).sum()) if "concept_has_gross_text" in df else None,
        "base_models": list(base_models),
        "budgets": [float(x) for x in BUDGETS],
    }
    (OUT_DIR / "v142_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v142] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
