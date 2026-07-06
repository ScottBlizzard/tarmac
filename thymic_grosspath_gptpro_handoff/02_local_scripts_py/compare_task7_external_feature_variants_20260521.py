from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
OUT = BASE / "task7_external_runs" / "05_feature_variant_fast_compare_20260521"
OUT.mkdir(parents=True, exist_ok=True)

OLD_FRAME = BASE / "task7_curriculum_runs" / "09_case_mlp_schemeB_m060_salvagehard_full5fold" / "curriculum_case_table.csv"
THIRD_REG = BASE / "task7_external_runs" / "01_third_batch_64style_image_only_20260521" / "third_batch_task7_registry.csv"


VARIANTS = {
    "whole": {
        "old_table": BASE / "task7_gross_feature_runs" / "10_review_router_embedding_probe_20260520" / "case_dino_concat_feature_table.csv",
        "old_npy": BASE / "task7_gross_feature_runs" / "10_review_router_embedding_probe_20260520" / "case_dino_concat_features.npy",
        "third_table": BASE / "task7_external_runs" / "01_third_batch_64style_image_only_20260521" / "third_batch_dino_concat_feature_table.csv",
        "third_npy": BASE / "task7_external_runs" / "01_third_batch_64style_image_only_20260521" / "third_batch_dino_concat_features.npy",
    },
    "crop": {
        "old_table": BASE / "task7_gross_feature_runs" / "67_roi_crop_embedding_probe_20260521" / "case_dino_concat_feature_table.csv",
        "old_npy": BASE / "task7_gross_feature_runs" / "67_roi_crop_embedding_probe_20260521" / "case_dino_concat_features.npy",
        "third_table": BASE / "task7_external_runs" / "03_third_batch_crop_64style_20260521" / "third_batch_dino_concat_feature_table.csv",
        "third_npy": BASE / "task7_external_runs" / "03_third_batch_crop_64style_20260521" / "third_batch_dino_concat_features.npy",
    },
    "whole_plus_crop": {
        "old_table": BASE / "task7_gross_feature_runs" / "68_roi_whole_plus_crop_embedding_probe_20260521" / "case_dino_concat_feature_table.csv",
        "old_npy": BASE / "task7_gross_feature_runs" / "68_roi_whole_plus_crop_embedding_probe_20260521" / "case_dino_concat_features.npy",
        "third_table": BASE / "task7_external_runs" / "04_third_batch_whole_plus_crop_64style_20260521" / "third_batch_dino_concat_feature_table.csv",
        "third_npy": BASE / "task7_external_runs" / "04_third_batch_whole_plus_crop_64style_20260521" / "third_batch_dino_concat_features.npy",
    },
}


def metric_row(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    row: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }
    if prob is not None and len(np.unique(y)) == 2:
        row["auc"] = float(roc_auc_score(y, prob))
    return row


def choose_threshold(y: np.ndarray, prob: np.ndarray, objective: str = "balanced_accuracy") -> tuple[float, dict[str, float | int]]:
    best_t = 0.5
    best_key: tuple[float, ...] | None = None
    best_row: dict[str, float | int] = {}
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        row = metric_row(y, pred, prob)
        if objective == "accuracy":
            key = (float(row["accuracy"]), float(row["balanced_accuracy"]), float(row["f1"]))
        elif objective == "sens90":
            penalty = min(0.0, float(row["sensitivity"]) - 0.90)
            key = (float(row["balanced_accuracy"]) + penalty, float(row["accuracy"]), float(row["f1"]))
        else:
            key = (float(row["balanced_accuracy"]), float(row["accuracy"]), float(row["f1"]))
        if best_key is None or key > best_key:
            best_key = key
            best_t = float(t)
            best_row = row
    return best_t, best_row


def align_features(frame: pd.DataFrame, table_path: Path, npy_path: Path) -> np.ndarray:
    table = pd.read_csv(table_path, dtype={"case_id": str})
    arr = np.load(npy_path).astype(np.float32)
    order = frame[["case_id"]].merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if order["feature_idx"].isna().any():
        missing = order.loc[order["feature_idx"].isna(), "case_id"].head().tolist()
        raise KeyError(f"Missing features from {table_path}: {missing}")
    return arr[order["feature_idx"].astype(int).to_numpy()]


def make_models() -> dict[str, object]:
    models: dict[str, object] = {}
    for c in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        models[f"logreg_c{c:g}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=5000),
        )
    return models


def fit_predict_prob(model: object, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model.fit(train_x, train_y)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(test_x)[:, 1]
    scores = model.decision_function(test_x)
    return 1.0 / (1.0 + np.exp(-scores))


def evaluate_variant(name: str, paths: dict[str, Path], old: pd.DataFrame, third: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[variant] {name}", flush=True)
    old_x = align_features(old, paths["old_table"], paths["old_npy"])
    third_x = align_features(third, paths["third_table"], paths["third_npy"])
    print(f"  old_x={old_x.shape} third_x={third_x.shape}", flush=True)
    y = old["label_idx"].astype(int).to_numpy()
    folds = old["fold_id"].astype(int).to_numpy()
    y3 = third["label_idx"].astype(int).to_numpy()
    rows = []
    pred_rows = []
    for model_name, model_template in make_models().items():
        print(f"  [model] {model_name}", flush=True)
        oof = np.zeros(len(old), dtype=float)
        for fold in sorted(np.unique(folds)):
            tr = folds != fold
            va = folds == fold
            # Recreate model each fold through sklearn clone where possible.
            import sklearn.base

            model = sklearn.base.clone(model_template)
            oof[va] = fit_predict_prob(model, old_x[tr], y[tr], old_x[va])
        for objective in ["balanced_accuracy", "accuracy", "sens90"]:
            threshold, old_row = choose_threshold(y, oof, objective)
            import sklearn.base

            final_model = sklearn.base.clone(model_template)
            third_prob = fit_predict_prob(final_model, old_x, y, third_x)
            third_pred = (third_prob >= threshold).astype(int)
            third_row = metric_row(y3, third_pred, third_prob)
            rows.append(
                {
                    "variant": name,
                    "model": model_name,
                    "objective": objective,
                    "threshold": threshold,
                    **{f"old_{k}": v for k, v in old_row.items()},
                    **{f"third_{k}": v for k, v in third_row.items()},
                }
            )
            for idx, case_id in enumerate(third["case_id"].astype(str).tolist()):
                pred_rows.append(
                    {
                        "variant": name,
                        "model": model_name,
                        "objective": objective,
                        "case_id": case_id,
                        "original_case_id": third.loc[idx, "original_case_id"],
                        "task_l6_label": third.loc[idx, "task_l6_label"],
                        "label_idx": int(y3[idx]),
                        "prob_high": float(third_prob[idx]),
                        "pred_idx": int(third_pred[idx]),
                        "correct": int(third_pred[idx] == y3[idx]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def pass_curve(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for (variant, model, objective), g in df.groupby(["variant", "model", "objective"]):
        conf = np.maximum(g["prob_high"].to_numpy(), 1.0 - g["prob_high"].to_numpy())
        correct = g["correct"].to_numpy().astype(int)
        order = np.argsort(-conf)
        for target in [0.90, 0.95]:
            best_n = 0
            best_acc = np.nan
            for n in range(1, len(order) + 1):
                acc = correct[order[:n]].mean()
                if acc >= target:
                    best_n = n
                    best_acc = acc
            rows.append(
                {
                    "variant": variant,
                    "model": model,
                    "objective": objective,
                    "target_acc": target,
                    "max_pass_n": best_n,
                    "max_pass_frac": best_n / len(order),
                    "pass_acc": best_acc,
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def main() -> None:
    old = pd.read_csv(OLD_FRAME, dtype={"case_id": str})
    old = old[["case_id", "fold_id", "label_idx", "difficulty_fine"]].copy()
    third = pd.read_csv(THIRD_REG, dtype={"case_id": str, "original_case_id": str})
    all_rows = []
    all_preds = []
    for name, paths in VARIANTS.items():
        missing = [p for p in paths.values() if not p.exists()]
        if missing:
            print(f"[skip] {name}: missing {missing}")
            continue
        rows, preds = evaluate_variant(name, paths, old, third)
        all_rows.append(rows)
        all_preds.append(preds)
    summary = pd.concat(all_rows, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    summary.to_csv(OUT / "feature_variant_fast_compare_summary.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT / "feature_variant_fast_compare_predictions.csv", index=False, encoding="utf-8-sig")
    pass_curve(preds, OUT / "feature_variant_third_confidence_pass_curve.csv")
    best = summary.sort_values(["third_balanced_accuracy", "third_accuracy", "third_f1"], ascending=False).head(30)
    best.to_csv(OUT / "feature_variant_fast_compare_top30.csv", index=False, encoding="utf-8-sig")
    print(best[["variant", "model", "objective", "threshold", "old_balanced_accuracy", "third_accuracy", "third_balanced_accuracy", "third_f1", "third_tn", "third_fp", "third_fn", "third_tp", "third_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
