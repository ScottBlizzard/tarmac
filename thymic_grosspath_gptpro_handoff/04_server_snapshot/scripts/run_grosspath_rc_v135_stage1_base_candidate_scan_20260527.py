from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_grosspath_rc_v134c_cascade_auto_corrector_20260527 import build_internal_external, base_feature_frame


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v135_stage1_base_candidate_scan_20260527"


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
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


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def choose_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, float]:
    best_t = 0.5
    best_s = -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (prob >= t).astype(int)
        if objective == "accuracy":
            score = accuracy_score(y, pred)
        elif objective == "high_sensitivity":
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            score = sens - 0.20 * max(0.0, 0.60 - spec)
        else:
            score = balanced_accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t = float(t)
            best_s = float(score)
    return best_t, best_s


def fit_prob(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    clf = clone(model)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def make_models(seed: int) -> dict[str, object]:
    return {
        "stack_logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        ),
        "stack_extra_d3": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "stack_gb_d2": GradientBoostingClassifier(max_depth=2, learning_rate=0.035, n_estimators=120, random_state=seed),
    }


def add_handcrafted_probs(df: pd.DataFrame, x: pd.DataFrame) -> dict[str, np.ndarray]:
    cols = ["main_prob", "robust_prob", "prob_mean_core", "wholecrop_prob", "selected_unified_prob", "selected_dinov3_prob"]
    probs = {c: x[c].to_numpy(float) for c in cols if c in x}
    arr = np.vstack([probs[c] for c in cols if c in probs]).T
    probs["avg_all6"] = arr.mean(axis=1)
    probs["median_all6"] = np.median(arr, axis=1)
    probs["max_all6"] = arr.max(axis=1)
    probs["min_all6"] = arr.min(axis=1)
    probs["avg_main_core_whole"] = x[["main_prob", "prob_mean_core", "wholecrop_prob"]].mean(axis=1).to_numpy(float)
    probs["avg_core_selected"] = x[["prob_mean_core", "selected_unified_prob", "selected_dinov3_prob"]].mean(axis=1).to_numpy(float)
    probs["pure_auto_pred_as_prob"] = df["base_pred"].astype(float).to_numpy()
    return probs


def nested_model_probs(x: pd.DataFrame, y: np.ndarray, folds: np.ndarray, external_x: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    feature_sets = {
        "core": [c for c in x.columns if c in {"main_prob", "robust_prob", "prob_mean_core", "wholecrop_prob", "base_conf_core", "base_entropy_core", "core_agree_count", "main_robust_gap", "whole_core_gap"}],
        "multiview": [c for c in x.columns if not c.startswith("subtype_") and c != "domain_is_external"],
        "multiview_subtype": [c for c in x.columns if c != "domain_is_external"],
    }
    models = make_models(20260527)
    internal_probs: dict[str, np.ndarray] = {}
    external_probs: dict[str, np.ndarray] = {}
    for fs_name, cols in feature_sets.items():
        xi = x[cols].to_numpy(float)
        xe = external_x.reindex(columns=x.columns, fill_value=0.0)[cols].to_numpy(float)
        for model_name, model in models.items():
            name = f"{model_name}_{fs_name}"
            oof = np.zeros(len(x), dtype=float)
            for fold in sorted(np.unique(folds)):
                train = folds != fold
                test = folds == fold
                oof[test] = fit_prob(model, xi[train], y[train], xi[test])
            internal_probs[name] = oof
            external_probs[name] = fit_prob(model, xi, y, xe)
    return internal_probs, external_probs


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in {"accuracy", "balanced_accuracy", "f1", "sensitivity_high", "specificity_low", "auc"}:
            out[col] = out[col].map(lambda v: pct(float(v)) if pd.notna(v) else "")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    internal, external = build_internal_external()
    external = external.loc[external["strict_task7_eval"].astype(int).eq(1)].copy()
    y = internal["label_idx"].astype(int).to_numpy()
    y_ext = external["label_idx"].astype(int).to_numpy()
    folds = internal["fold_id"].astype(int).to_numpy()
    x = base_feature_frame(internal)
    x_ext = base_feature_frame(external).reindex(columns=x.columns, fill_value=0.0)

    probs = add_handcrafted_probs(internal, x)
    ext_probs = add_handcrafted_probs(external, x_ext)
    nested, nested_ext = nested_model_probs(x, y, folds, x_ext)
    probs.update(nested)
    ext_probs.update(nested_ext)

    rows = []
    pred_frames = []
    for name, p in probs.items():
        pe = ext_probs[name]
        for obj in ["balanced_accuracy", "accuracy", "high_sensitivity"]:
            t, train_score = choose_threshold(y, p, obj)
            pred = (p >= t).astype(int)
            pred_e = (pe >= t).astype(int)
            row_i = {
                "candidate": name,
                "objective": obj,
                "threshold": t,
                "scope": "internal_oof_old_third",
                "selection_score": train_score,
                **metrics(y, pred, p),
            }
            row_e = {
                "candidate": name,
                "objective": obj,
                "threshold": t,
                "scope": "strict_external_locked",
                "selection_score": train_score,
                **metrics(y_ext, pred_e, pe),
            }
            rows.extend([row_i, row_e])
            pred_frames.append(
                pd.DataFrame(
                    {
                        "scope": "internal_oof_old_third",
                        "candidate": name,
                        "objective": obj,
                        "case_id": internal["case_id"],
                        "original_case_id": internal["original_case_id"],
                        "domain": internal["domain"],
                        "task_l6_label": internal["task_l6_label"],
                        "label_idx": y,
                        "prob_high": p,
                        "threshold": t,
                        "pred_idx": pred,
                        "correct": pred == y,
                    }
                )
            )
            pred_frames.append(
                pd.DataFrame(
                    {
                        "scope": "strict_external_locked",
                        "candidate": name,
                        "objective": obj,
                        "case_id": external["case_id"],
                        "original_case_id": external["original_case_id"],
                        "domain": external["domain"],
                        "task_l6_label": external["task_l6_label"],
                        "label_idx": y_ext,
                        "prob_high": pe,
                        "threshold": t,
                        "pred_idx": pred_e,
                        "correct": pred_e == y_ext,
                    }
                )
            )

    summary = pd.DataFrame(rows).sort_values(["scope", "balanced_accuracy", "accuracy"], ascending=[True, False, False])
    summary.to_csv(OUT_DIR / "v135_stage1_candidate_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v135_stage1_candidate_summary_formatted.csv", index=False, encoding="utf-8-sig")
    all_preds = pd.concat(pred_frames, ignore_index=True)
    all_preds.to_csv(OUT_DIR / "v135_stage1_candidate_predictions.csv", index=False, encoding="utf-8-sig")
    top = summary.groupby("scope").head(20)
    top.to_csv(OUT_DIR / "v135_top_by_scope.csv", index=False, encoding="utf-8-sig")
    format_table(top).to_csv(OUT_DIR / "v135_top_by_scope_formatted.csv", index=False, encoding="utf-8-sig")
    report = {
        "boundary": "Thresholds selected on internal old+third OOF only; strict external locked for evaluation.",
        "n_internal": int(len(internal)),
        "n_external_strict": int(len(external)),
        "n_candidates": int(len(probs)),
    }
    (OUT_DIR / "v135_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT_DIR)
    print(format_table(top).to_string(index=False))


if __name__ == "__main__":
    main()
