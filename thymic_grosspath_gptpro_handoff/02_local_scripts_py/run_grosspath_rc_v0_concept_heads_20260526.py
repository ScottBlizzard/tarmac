from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "grosspath_rc_v0_20260526"
FEATURE_DIR = (
    ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_gross_feature_runs"
    / "73_flip_lr_whole_plus_crop_embedding_probe_20260521"
)


CONCEPTS_TO_TEST = [
    "boundary_clear",
    "boundary_unclear",
    "capsule_any",
    "capsule_complete",
    "capsule_absent",
    "invasion",
    "fat_involved_or_attached",
    "hemorrhage",
    "necrosis",
    "cystic_change",
    "nodular_lobulated",
    "septum",
    "gray_white",
    "gray_yellow",
    "gray_red",
    "texture_soft",
    "texture_medium",
    "texture_tough",
    "cut_surface_mentioned",
]


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, prob))


def best_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_s = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        s = balanced_accuracy_score(y, pred)
        if (s, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t = float(t)
            best_s = float(s)
    return best_t, best_s


def metric_row(target: str, y: np.ndarray, prob: np.ndarray) -> dict[str, object]:
    t, _ = best_threshold(y, prob)
    pred = (prob >= t).astype(int)
    return {
        "target": target,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "threshold": t,
        "auc": safe_auc(y, prob),
        "ap": float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(FEATURE_DIR / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    feats = np.load(FEATURE_DIR / "case_dino_concat_features.npy")
    if len(table) != len(feats):
        raise RuntimeError(f"feature table/features length mismatch: {len(table)} vs {len(feats)}")
    folds = pd.read_csv(
        ROOT / "outputs" / "batch1_batch2_task567_20260514" / "frozen_inputs" / "combined_5fold_assignments.csv",
        dtype={"case_id": str, "original_case_id": str},
    )
    concepts = pd.read_csv(OUT / "gross_concepts_v1.csv", dtype={"original_case_id": str})
    df = table.merge(folds[["case_id", "original_case_id", "master_fold_id", "task_l7_label"]], on="case_id", how="left")
    df["original_case_id"] = df["original_case_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df = df.merge(concepts, on="original_case_id", how="left")
    df["label_idx"] = df["task_l7_label"].map({"low_risk_group": 0, "high_risk_group": 1}).astype(int)
    fold_ids = sorted(df["master_fold_id"].dropna().astype(int).unique())

    rows = []
    pred_cols: dict[str, np.ndarray] = {}
    for concept in CONCEPTS_TO_TEST:
        if concept not in df.columns:
            continue
        y = df[concept].fillna(0).astype(int).to_numpy()
        prevalence = y.mean()
        if prevalence < 0.03 or prevalence > 0.97:
            continue
        prob = np.zeros(len(df), dtype=float)
        for fold in fold_ids:
            train = df["master_fold_id"].astype(int).to_numpy() != fold
            test = ~train
            # Skip concepts that collapse in a training fold.
            if len(np.unique(y[train])) < 2:
                prob[test] = prevalence
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", C=0.1),
            )
            clf.fit(feats[train], y[train])
            prob[test] = clf.predict_proba(feats[test])[:, 1]
        pred_cols[f"pred_concept_{concept}"] = prob
        rows.append(metric_row(concept, y, prob))

    concept_metrics = pd.DataFrame(rows).sort_values("auc", ascending=False)
    concept_metrics.to_csv(OUT / "image_to_concept_head_metrics_old_oof.csv", index=False, encoding="utf-8-sig")

    pred_df = df[
        [
            "case_id",
            "original_case_id",
            "master_fold_id",
            "task_l7_label",
            "label_idx",
        ]
        + [c for c in CONCEPTS_TO_TEST if c in df.columns]
    ].copy()
    for col, values in pred_cols.items():
        pred_df[col] = values
    pred_df.to_csv(OUT / "image_to_concept_head_oof_predictions_old.csv", index=False, encoding="utf-8-sig")

    # Lightweight risk test: image embedding only vs image embedding + OOF predicted concept probabilities.
    y_risk = df["label_idx"].to_numpy(int)
    concept_prob_cols = list(pred_cols.keys())
    risk_sets: dict[str, np.ndarray] = {
        "embedding_only": feats,
    }
    if concept_prob_cols:
        risk_sets["embedding_plus_predicted_concepts"] = np.hstack([feats, pred_df[concept_prob_cols].to_numpy(float)])
        risk_sets["predicted_concepts_only"] = pred_df[concept_prob_cols].to_numpy(float)

    risk_rows = []
    risk_pred = pd.DataFrame(
        {
            "case_id": df["case_id"],
            "original_case_id": df["original_case_id"],
            "label_idx": y_risk,
            "master_fold_id": df["master_fold_id"],
        }
    )
    for name, X in risk_sets.items():
        prob = np.zeros(len(df), dtype=float)
        for fold in fold_ids:
            train = df["master_fold_id"].astype(int).to_numpy() != fold
            test = ~train
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", C=0.1),
            )
            clf.fit(X[train], y_risk[train])
            prob[test] = clf.predict_proba(X[test])[:, 1]
        risk_pred[f"prob_{name}"] = prob
        risk_rows.append(metric_row(name, y_risk, prob))
    risk_metrics = pd.DataFrame(risk_rows).sort_values("balanced_accuracy", ascending=False)
    risk_metrics.to_csv(OUT / "image_concept_risk_cv_metrics_old.csv", index=False, encoding="utf-8-sig")
    risk_pred.to_csv(OUT / "image_concept_risk_oof_predictions_old.csv", index=False, encoding="utf-8-sig")

    summary = {
        "feature_dir": str(FEATURE_DIR),
        "n_cases": int(len(df)),
        "n_concepts_tested": int(len(concept_metrics)),
        "best_concepts": concept_metrics.head(10).to_dict("records"),
        "risk_metrics": risk_metrics.to_dict("records"),
    }
    (OUT / "image_to_concept_head_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# 图像到医生概念头实验",
        "",
        f"旧数据病例数：{len(df)}",
        f"测试概念数：{len(concept_metrics)}",
        "",
        "## 概念头 OOF 最好结果",
        "",
        concept_metrics.head(12).to_string(index=False),
        "",
        "## 风险分类轻量对照",
        "",
        risk_metrics.to_string(index=False),
        "",
        "## 解释",
        "",
        "- 该实验只使用旧数据 DINO whole+crop embedding，因此是 concept head 可行性验证，不是最终全数据模型。",
        "- 如果某些概念 AUC 明显高于 0.70，说明这些医生概念可以从图像中学习出来。",
        "- 如果 `embedding_plus_predicted_concepts` 不能超过 `embedding_only`，说明概念头当前更适合解释和风险控制，不适合直接提分。",
    ]
    (OUT / "image_to_concept_head_summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
