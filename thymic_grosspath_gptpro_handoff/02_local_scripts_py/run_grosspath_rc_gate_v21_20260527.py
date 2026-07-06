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
OUT_DIR = ROOT / "outputs" / "grosspath_rc_gate_v21_20260527"

MAIN_THRESHOLD = 0.595
ROBUST_THRESHOLD = 0.57
ROBUST_SOURCES = ("prob_base162", "prob103_vitl", "prob_mean_core")


@dataclass(frozen=True)
class GateModelSpec:
    name: str
    estimator: object


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def metric_counts(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "n": int(len(y)),
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
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


def build_base_columns(dev: pd.DataFrame, external: pd.DataFrame) -> list[str]:
    numeric_candidates = [
        "prob_base162",
        "prob103_vitl",
        "prob107_qkvb",
        "prob_mean_core",
        "prob_stack_plain",
        "prob_stack_balanced",
        "core_prob_std",
        "core_prob_range",
        "abs_162_103",
        "abs_162_107",
        "abs_103_107",
        "margin162",
        "score_margin_agree",
        "score_v0_simple",
        "core_agree_count",
    ]
    return [col for col in numeric_candidates if col in dev.columns and col in external.columns]


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["main_prob"] = out["prob_base162"].astype(float)
    out["main_margin_abs"] = (out["main_prob"] - MAIN_THRESHOLD).abs()
    out["main_pred_idx"] = (out["main_prob"] >= MAIN_THRESHOLD).astype(int)
    out["robust_prob"] = out[list(ROBUST_SOURCES)].mean(axis=1).astype(float)
    out["robust_margin_abs"] = (out["robust_prob"] - ROBUST_THRESHOLD).abs()
    out["robust_pred_idx"] = (out["robust_prob"] >= ROBUST_THRESHOLD).astype(int)
    out["main_robust_abs_diff"] = (out["main_prob"] - out["robust_prob"]).abs()
    out["main_robust_disagree"] = (out["main_pred_idx"] != out["robust_pred_idx"]).astype(int)
    out["low_main_high_robust"] = ((out["main_pred_idx"] == 0) & (out["robust_pred_idx"] == 1)).astype(int)
    out["high_main_low_robust"] = ((out["main_pred_idx"] == 1) & (out["robust_pred_idx"] == 0)).astype(int)
    if "quality_group" in out.columns:
        text = out["quality_group"].fillna("").astype(str)
        out["quality_pass_readable"] = text.str.contains("pass", case=False, na=False).astype(int)
        out["quality_not_labeled"] = text.eq("not_labeled").astype(int)
    return out


def make_feature_matrix(dev: pd.DataFrame, external: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dev2 = add_derived_features(dev)
    ext2 = add_derived_features(external)
    base_cols = build_base_columns(dev2, ext2)
    derived_cols = [
        "main_margin_abs",
        "main_pred_idx",
        "robust_prob",
        "robust_margin_abs",
        "robust_pred_idx",
        "main_robust_abs_diff",
        "main_robust_disagree",
        "low_main_high_robust",
        "high_main_low_robust",
        "quality_pass_readable",
        "quality_not_labeled",
    ]
    feature_cols = [col for col in base_cols + derived_cols if col in dev2.columns and col in ext2.columns]
    return dev2, ext2, feature_cols


def model_specs() -> list[GateModelSpec]:
    return [
        GateModelSpec(
            "logistic_l2",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.5,
                            class_weight="balanced",
                            max_iter=2000,
                            solver="lbfgs",
                            random_state=20260527,
                        ),
                    ),
                ]
            ),
        ),
        GateModelSpec(
            "random_forest",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=400,
                            max_depth=4,
                            min_samples_leaf=8,
                            class_weight="balanced_subsample",
                            random_state=20260527,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        GateModelSpec(
            "extra_trees",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            max_depth=4,
                            min_samples_leaf=8,
                            class_weight="balanced",
                            random_state=20260527,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        GateModelSpec(
            "grad_boost",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        GradientBoostingClassifier(
                            n_estimators=120,
                            learning_rate=0.035,
                            max_depth=2,
                            min_samples_leaf=10,
                            random_state=20260527,
                        ),
                    ),
                ]
            ),
        ),
    ]


def oof_gate_risk(dev: pd.DataFrame, feature_cols: list[str], spec: GateModelSpec) -> np.ndarray:
    risk = np.zeros(len(dev), dtype=float)
    X = dev[feature_cols]
    y = dev["main_wrong"].astype(int).to_numpy()
    for fold in sorted(dev["fold_id"].dropna().astype(int).unique().tolist()):
        train_mask = dev["fold_id"].astype(int).to_numpy() != fold
        test_mask = ~train_mask
        model = spec.estimator
        model.fit(X.loc[train_mask], y[train_mask])
        risk[test_mask] = model.predict_proba(X.loc[test_mask])[:, 1]
    return risk


def full_gate_risk(dev: pd.DataFrame, external: pd.DataFrame, feature_cols: list[str], spec: GateModelSpec) -> np.ndarray:
    model = spec.estimator
    model.fit(dev[feature_cols], dev["main_wrong"].astype(int))
    return model.predict_proba(external[feature_cols])[:, 1]


def choose_switch_threshold(dev: pd.DataFrame, risk: np.ndarray) -> float:
    y = dev["label_idx"].astype(int).to_numpy()
    main_pred = dev["main_pred_idx"].astype(int).to_numpy()
    robust_pred = dev["robust_pred_idx"].astype(int).to_numpy()
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    for threshold in np.arange(0.05, 0.951, 0.01):
        gate = risk >= threshold
        pred = np.where(gate, robust_pred, main_pred)
        metrics = metric_counts(y, pred)
        key = (
            float(metrics["balanced_accuracy"]),
            float(metrics["accuracy"]),
            -float(gate.mean()),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def choose_pass_threshold(dev: pd.DataFrame, risk: np.ndarray, min_accuracy: float) -> float:
    y = dev["label_idx"].astype(int).to_numpy()
    main_pred = dev["main_pred_idx"].astype(int).to_numpy()
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.0
    for threshold in np.arange(0.05, 0.951, 0.01):
        pass_mask = risk < threshold
        if int(pass_mask.sum()) < 30:
            continue
        metrics = metric_counts(y[pass_mask], main_pred[pass_mask])
        if float(metrics["accuracy"]) < min_accuracy:
            continue
        key = (float(pass_mask.mean()), float(metrics["balanced_accuracy"]), -abs(float(threshold) - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def switch_metrics(frame: pd.DataFrame, risk: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = frame["label_idx"].astype(int).to_numpy()
    main_pred = frame["main_pred_idx"].astype(int).to_numpy()
    robust_pred = frame["robust_pred_idx"].astype(int).to_numpy()
    gate = risk >= threshold
    pred = np.where(gate, robust_pred, main_pred)
    metrics = metric_counts(y, pred)
    metrics.update(
        {
            "gate_threshold": threshold,
            "routed_to_robust_n": int(gate.sum()),
            "routed_to_robust_rate": float(gate.mean()),
            "main_wrong_routed_n": int(((frame["main_wrong"].astype(int).to_numpy() == 1) & gate).sum())
            if "main_wrong" in frame.columns
            else float("nan"),
            "main_wrong_routed_rate": float(gate[frame["main_wrong"].astype(int).to_numpy() == 1].mean())
            if "main_wrong" in frame.columns and int((frame["main_wrong"].astype(int) == 1).sum()) > 0
            else float("nan"),
        }
    )
    return metrics


def review_metrics(frame: pd.DataFrame, risk: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = frame["label_idx"].astype(int).to_numpy()
    main_pred = frame["main_pred_idx"].astype(int).to_numpy()
    robust_pred = frame["robust_pred_idx"].astype(int).to_numpy()
    pass_mask = risk < threshold
    review_mask = ~pass_mask
    pass_metrics = metric_counts(y[pass_mask], main_pred[pass_mask]) if int(pass_mask.sum()) else {}
    review_main = metric_counts(y[review_mask], main_pred[review_mask]) if int(review_mask.sum()) else {}
    review_robust = metric_counts(y[review_mask], robust_pred[review_mask]) if int(review_mask.sum()) else {}
    rescued = int(((main_pred != y) & (robust_pred == y) & review_mask).sum())
    harmed = int(((main_pred == y) & (robust_pred != y) & review_mask).sum())
    return {
        "pass_threshold": threshold,
        "pass_n": int(pass_mask.sum()),
        "pass_rate": float(pass_mask.mean()),
        "pass_accuracy": float(pass_metrics.get("accuracy", float("nan"))),
        "pass_balanced_accuracy": float(pass_metrics.get("balanced_accuracy", float("nan"))),
        "pass_fn": int(pass_metrics.get("fn", 0)),
        "pass_fp": int(pass_metrics.get("fp", 0)),
        "review_n": int(review_mask.sum()),
        "review_rate": float(review_mask.mean()),
        "review_main_accuracy": float(review_main.get("accuracy", float("nan"))),
        "review_main_balanced_accuracy": float(review_main.get("balanced_accuracy", float("nan"))),
        "review_robust_accuracy": float(review_robust.get("accuracy", float("nan"))),
        "review_robust_balanced_accuracy": float(review_robust.get("balanced_accuracy", float("nan"))),
        "review_main_fn": int(review_main.get("fn", 0)),
        "review_robust_fn": int(review_robust.get("fn", 0)),
        "review_rescued_by_robust_n": rescued,
        "review_harmed_by_robust_n": harmed,
    }


def add_baselines(rows: list[dict[str, object]], frame_name: str, frame: pd.DataFrame) -> None:
    y = frame["label_idx"].astype(int).to_numpy()
    for policy, pred_col in [("main_only_base162", "main_pred_idx"), ("robust_all_diversity", "robust_pred_idx")]:
        row = {"model": "none", "policy": policy, "split": frame_name}
        row.update(metric_counts(y, frame[pred_col].astype(int).to_numpy()))
        rows.append(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev_raw = pd.read_csv(V2_DIR / "v2_development_diagnostic_table.csv")
    external_raw = pd.read_csv(V2_DIR / "v2_external_diagnostic_table.csv")
    dev, external, feature_cols = make_feature_matrix(dev_raw, external_raw)

    baseline_rows: list[dict[str, object]] = []
    add_baselines(baseline_rows, "development_oof", dev)
    add_baselines(baseline_rows, "external_strict_frozen", external)
    for domain, group in dev.groupby("domain"):
        add_baselines(baseline_rows, f"development_oof:{domain}", group)
    pd.DataFrame(baseline_rows).to_csv(OUT_DIR / "gate_v21_baselines.csv", index=False, encoding="utf-8-sig")

    switch_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    risk_frames: list[pd.DataFrame] = []
    gate_auc_rows: list[dict[str, object]] = []

    for spec in model_specs():
        dev_risk = oof_gate_risk(dev, feature_cols, spec)
        external_risk = full_gate_risk(dev, external, feature_cols, spec)
        gate_auc_rows.append(
            {
                "model": spec.name,
                "split": "development_oof",
                "main_wrong_auc": safe_auc(dev["main_wrong"].astype(int).to_numpy(), dev_risk),
                "mean_risk": float(np.mean(dev_risk)),
            }
        )
        gate_auc_rows.append(
            {
                "model": spec.name,
                "split": "external_strict_frozen",
                "main_wrong_auc": safe_auc(external["main_wrong"].astype(int).to_numpy(), external_risk),
                "mean_risk": float(np.mean(external_risk)),
            }
        )

        switch_threshold = choose_switch_threshold(dev, dev_risk)
        for split_name, frame, risk in [
            ("development_oof", dev, dev_risk),
            ("external_strict_frozen", external, external_risk),
        ]:
            row = {"model": spec.name, "policy": "gate_switch_to_robust", "split": split_name}
            row.update(switch_metrics(frame, risk, switch_threshold))
            switch_rows.append(row)
        for domain, group in dev.groupby("domain"):
            idx = group.index.to_numpy()
            row = {"model": spec.name, "policy": "gate_switch_to_robust", "split": f"development_oof:{domain}"}
            row.update(switch_metrics(group, dev_risk[idx], switch_threshold))
            switch_rows.append(row)

        for target_acc in [0.90, 0.95]:
            pass_threshold = choose_pass_threshold(dev, dev_risk, target_acc)
            for split_name, frame, risk in [
                ("development_oof", dev, dev_risk),
                ("external_strict_frozen", external, external_risk),
            ]:
                row = {
                    "model": spec.name,
                    "policy": f"auto_pass_ge_{int(target_acc * 100)}",
                    "split": split_name,
                    "target_development_pass_accuracy": target_acc,
                }
                row.update(review_metrics(frame, risk, pass_threshold))
                review_rows.append(row)

        risk_frame = pd.DataFrame(
            {
                "case_id": dev["case_id"].astype(str),
                "split": "development_oof",
                "model": spec.name,
                "gate_risk": dev_risk,
                "main_wrong": dev["main_wrong"].astype(int),
                "label_idx": dev["label_idx"].astype(int),
                "main_pred_idx": dev["main_pred_idx"].astype(int),
                "robust_pred_idx": dev["robust_pred_idx"].astype(int),
                "domain": dev.get("domain", ""),
            }
        )
        risk_frames.append(risk_frame)

    pd.DataFrame(gate_auc_rows).to_csv(OUT_DIR / "gate_v21_main_wrong_auc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(switch_rows).to_csv(OUT_DIR / "gate_v21_switch_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(review_rows).to_csv(OUT_DIR / "gate_v21_review_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(risk_frames, ignore_index=True).to_csv(OUT_DIR / "gate_v21_development_oof_risks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols}).to_csv(OUT_DIR / "gate_v21_feature_columns.csv", index=False, encoding="utf-8-sig")

    print(f"[done] {OUT_DIR}")
    print("\nFeature columns:")
    print(pd.DataFrame({"feature": feature_cols}).to_string(index=False))
    print("\nGate AUC:")
    print(pd.DataFrame(gate_auc_rows).to_string(index=False))
    print("\nSwitch metrics:")
    print(pd.DataFrame(switch_rows).to_string(index=False))
    print("\nReview metrics:")
    print(pd.DataFrame(review_rows).to_string(index=False))


if __name__ == "__main__":
    main()
