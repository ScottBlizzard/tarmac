from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "grosspath_rc_v2_20260526"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v23_probability_meta_20260527"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    use_sample_weight: bool = False


def logit(x: pd.Series) -> pd.Series:
    value = pd.to_numeric(x, errors="coerce").clip(1e-5, 1 - 1e-5)
    return np.log(value / (1 - value))


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    prob_cols = [
        "prob_base162",
        "prob103_vitl",
        "prob107_qkvb",
        "prob_mean_core",
        "prob_stack_plain",
        "prob_stack_balanced",
    ]
    for col in prob_cols:
        if col in out.columns:
            out[f"logit_{col}"] = logit(out[col])
    if {"prob_base162", "prob103_vitl", "prob_mean_core"}.issubset(out.columns):
        out["robust_mean_162_103_core"] = out[["prob_base162", "prob103_vitl", "prob_mean_core"]].mean(axis=1)
        out["diff_162_robust"] = out["prob_base162"] - out["robust_mean_162_103_core"]
        out["abs_diff_162_robust"] = out["diff_162_robust"].abs()
    if {"prob_base162", "prob103_vitl", "prob107_qkvb"}.issubset(out.columns):
        out["prob_min_core3"] = out[["prob_base162", "prob103_vitl", "prob107_qkvb"]].min(axis=1)
        out["prob_max_core3"] = out[["prob_base162", "prob103_vitl", "prob107_qkvb"]].max(axis=1)
        out["prob_range_core3"] = out["prob_max_core3"] - out["prob_min_core3"]
    return out


def feature_columns(dev: pd.DataFrame, external: pd.DataFrame) -> list[str]:
    candidates = [
        "prob_base162",
        "prob103_vitl",
        "prob107_qkvb",
        "prob_mean_core",
        "prob_stack_plain",
        "prob_stack_balanced",
        "logit_prob_base162",
        "logit_prob103_vitl",
        "logit_prob107_qkvb",
        "logit_prob_mean_core",
        "logit_prob_stack_plain",
        "logit_prob_stack_balanced",
        "core_prob_std",
        "core_prob_range",
        "abs_162_103",
        "abs_162_107",
        "abs_103_107",
        "margin162",
        "score_margin_agree",
        "score_v0_simple",
        "core_agree_count",
        "robust_mean_162_103_core",
        "diff_162_robust",
        "abs_diff_162_robust",
        "prob_min_core3",
        "prob_max_core3",
        "prob_range_core3",
    ]
    return [col for col in candidates if col in dev.columns and col in external.columns]


def auc(y: np.ndarray, prob: np.ndarray) -> float:
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (prob >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "auc": auc(y, prob),
        "accuracy": (tn + tp) / len(y),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def choose_threshold(frame: pd.DataFrame, prob: np.ndarray, objective: str) -> float:
    y = frame["label_idx"].to_numpy(dtype=int)
    best_key: tuple[float, ...] | None = None
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.951, 0.005):
        overall = metrics(y, prob, float(threshold))
        if objective == "bacc":
            key = (float(overall["balanced_accuracy"]), float(overall["accuracy"]), -abs(float(threshold) - 0.5))
        elif objective == "sensitivity90":
            penalty = 0.0 if float(overall["sensitivity"]) >= 0.90 else -1.0
            key = (penalty, float(overall["specificity"]), float(overall["balanced_accuracy"]), -abs(float(threshold) - 0.5))
        elif objective == "domain_worst_bacc":
            domain_scores = []
            for _, group in frame.groupby("domain"):
                idx = group.index.to_numpy()
                domain_scores.append(float(metrics(group["label_idx"].to_numpy(dtype=int), prob[idx], float(threshold))["balanced_accuracy"]))
            key = (min(domain_scores), float(overall["balanced_accuracy"]), float(overall["accuracy"]), -abs(float(threshold) - 0.5))
        else:
            raise ValueError(objective)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    key = frame["domain"].astype(str) + "::" + frame["label_idx"].astype(str)
    counts = key.value_counts()
    weights = key.map(lambda item: 1.0 / counts[item]).to_numpy(dtype=float)
    return weights / weights.mean()


def fit_predict_oof(dev: pd.DataFrame, feature_cols: list[str], spec: ModelSpec) -> np.ndarray:
    prob = np.zeros(len(dev), dtype=float)
    X = dev[feature_cols]
    y = dev["label_idx"].to_numpy(dtype=int)
    folds = sorted(dev["fold_id"].astype(int).unique().tolist())
    for fold in folds:
        train_mask = dev["fold_id"].astype(int).to_numpy() != fold
        test_mask = ~train_mask
        estimator = spec.estimator
        fit_kwargs = {}
        if spec.use_sample_weight:
            fit_kwargs["clf__sample_weight"] = sample_weights(dev.loc[train_mask])
        estimator.fit(X.loc[train_mask], y[train_mask], **fit_kwargs)
        prob[test_mask] = estimator.predict_proba(X.loc[test_mask])[:, 1]
    return prob


def fit_predict_external(dev: pd.DataFrame, external: pd.DataFrame, feature_cols: list[str], spec: ModelSpec) -> np.ndarray:
    estimator = spec.estimator
    fit_kwargs = {}
    if spec.use_sample_weight:
        fit_kwargs["clf__sample_weight"] = sample_weights(dev)
    estimator.fit(dev[feature_cols], dev["label_idx"].to_numpy(dtype=int), **fit_kwargs)
    return estimator.predict_proba(external[feature_cols])[:, 1]


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "logistic_balanced",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=0.35, class_weight="balanced", max_iter=3000, random_state=20260527)),
                ]
            ),
            False,
        ),
        ModelSpec(
            "logistic_domain_weighted",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=0.25, max_iter=3000, random_state=20260527)),
                ]
            ),
            True,
        ),
        ModelSpec(
            "extra_trees_domain_weighted",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("clf", ExtraTreesClassifier(n_estimators=600, max_depth=3, min_samples_leaf=12, random_state=20260527, n_jobs=-1)),
                ]
            ),
            True,
        ),
        ModelSpec(
            "rf_domain_weighted",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("clf", RandomForestClassifier(n_estimators=500, max_depth=4, min_samples_leaf=10, random_state=20260527, n_jobs=-1)),
                ]
            ),
            True,
        ),
        ModelSpec(
            "grad_boost",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.035, max_depth=2, min_samples_leaf=10, random_state=20260527)),
                ]
            ),
            False,
        ),
    ]


def add_eval(rows: list[dict[str, object]], model_name: str, objective: str, threshold: float, frame_name: str, frame: pd.DataFrame, prob: np.ndarray) -> None:
    row = {"model": model_name, "objective": objective, "threshold": threshold, "split": frame_name}
    row.update(metrics(frame["label_idx"].to_numpy(dtype=int), prob, threshold))
    rows.append(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = add_features(pd.read_csv(V2_DIR / "v2_development_diagnostic_table.csv"))
    external = add_features(pd.read_csv(V2_DIR / "v2_external_diagnostic_table.csv"))
    cols = feature_columns(dev, external)

    eval_rows: list[dict[str, object]] = []
    pred_exports: list[pd.DataFrame] = []
    for spec in model_specs():
        oof = fit_predict_oof(dev, cols, spec)
        external_prob = fit_predict_external(dev, external, cols, spec)
        for objective in ["bacc", "domain_worst_bacc", "sensitivity90"]:
            threshold = choose_threshold(dev, oof, objective)
            add_eval(eval_rows, spec.name, objective, threshold, "development_oof", dev, oof)
            add_eval(eval_rows, spec.name, objective, threshold, "external_strict_frozen", external, external_prob)
            for domain, group in dev.groupby("domain"):
                idx = group.index.to_numpy()
                add_eval(eval_rows, spec.name, objective, threshold, f"development_oof:{domain}", group, oof[idx])
        pred_exports.append(
            pd.DataFrame(
                {
                    "case_id": dev["case_id"].astype(str),
                    "domain": dev["domain"].astype(str),
                    "fold_id": dev["fold_id"].astype(int),
                    "label_idx": dev["label_idx"].astype(int),
                    f"prob_{spec.name}": oof,
                }
            )
        )

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(OUT_DIR / "v23_probability_meta_metrics.csv", index=False, encoding="utf-8-sig")
    merged_pred = pred_exports[0]
    for item in pred_exports[1:]:
        merged_pred = merged_pred.merge(item, on=["case_id", "domain", "fold_id", "label_idx"], how="left")
    merged_pred.to_csv(OUT_DIR / "v23_probability_meta_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": cols}).to_csv(OUT_DIR / "v23_probability_meta_features.csv", index=False, encoding="utf-8-sig")

    print(f"[done] {OUT_DIR}")
    print("\nMetrics sorted by external balanced accuracy:")
    print(eval_df[eval_df["split"].eq("external_strict_frozen")].sort_values(["balanced_accuracy", "accuracy"], ascending=False).to_string(index=False))
    print("\nDevelopment metrics:")
    print(eval_df[eval_df["split"].eq("development_oof")].sort_values(["balanced_accuracy", "accuracy"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
