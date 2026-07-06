from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from run_task7_external_folder_unsup_domain_sweep_20260522 import read_external, read_old


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast focused diag-SVC sweep for Task7 external folder.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--old-feature-dir", required=True)
    parser.add_argument("--external-feature-dir", required=True)
    parser.add_argument(
        "--old-label-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def metric_dict(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
    return out


def best_threshold(y: np.ndarray, score: np.ndarray, objective: str) -> tuple[float, float]:
    lo, hi = float(np.quantile(score, 0.02)), float(np.quantile(score, 0.98))
    best_t, best_s = 0.0, -1.0
    for t in np.linspace(lo, hi, 101):
        pred = (score >= t).astype(int)
        if objective == "accuracy":
            val = accuracy_score(y, pred)
        elif objective == "f1":
            val = f1_score(y, pred, zero_division=0)
        else:
            val = balanced_accuracy_score(y, pred)
        if (val, -abs(t)) > (best_s, -abs(best_t)):
            best_t, best_s = float(t), float(val)
    return best_t, best_s


def diag_align(x_source: np.ndarray, x_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ms = x_source.mean(axis=0, keepdims=True)
    mt = x_target.mean(axis=0, keepdims=True)
    ss = x_source.std(axis=0, keepdims=True) + 1e-5
    st = x_target.std(axis=0, keepdims=True) + 1e-5
    return ((x_source - ms) / ss * st + mt).astype(np.float32), x_target.astype(np.float32)


def eval_external(frame: pd.DataFrame, score: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (score >= threshold).astype(int)
    y = frame["label_idx"].to_numpy(dtype=int)
    strict = frame["strict_task7_eval"].astype(int).to_numpy() == 1
    out = {f"all_{k}": v for k, v in metric_dict(y, pred, score).items()}
    out.update({f"strict_{k}": v for k, v in metric_dict(y[strict], pred[strict], score[strict]).items()})
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    old, x_old_raw = read_old(root, args.old_feature_dir, args.old_label_csv)
    external, x_ext_raw = read_external(root, args.external_feature_dir)
    y_old = old["label_idx"].to_numpy(dtype=int)

    rows: list[dict[str, object]] = []
    best_external: tuple[float, float] = (-1.0, -1.0)
    best_external_payload: tuple[dict[str, object], np.ndarray] | None = None
    for pca_components in [16, 24, 32, 40, 48, 64]:
        scaler = StandardScaler()
        x_old_scaled = scaler.fit_transform(x_old_raw)
        x_ext_scaled = scaler.transform(x_ext_raw)
        pca = PCA(n_components=min(pca_components, x_old_scaled.shape[0] - 1), random_state=args.seed, svd_solver="randomized")
        x_old = pca.fit_transform(x_old_scaled).astype(np.float32)
        x_ext = pca.transform(x_ext_scaled).astype(np.float32)
        x_old, x_ext = diag_align(x_old, x_ext)
        for c in [0.1, 0.3, 1.0, 3.0, 10.0]:
            for gamma in ["scale", 0.001, 0.003, 0.01]:
                for objective in ["balanced_accuracy", "accuracy"]:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
                    oof = np.zeros(len(y_old), dtype=np.float32)
                    for fold_id, (tr, va) in enumerate(skf.split(x_old, y_old)):
                        model = make_pipeline(
                            StandardScaler(),
                            SVC(C=c, kernel="rbf", gamma=gamma, class_weight="balanced", probability=False, random_state=args.seed + fold_id),
                        )
                        model.fit(x_old[tr], y_old[tr])
                        oof[va] = model.decision_function(x_old[va]).astype(np.float32)
                    threshold, threshold_score = best_threshold(y_old, oof, objective)
                    old_pred = (oof >= threshold).astype(int)
                    final = make_pipeline(
                        StandardScaler(),
                        SVC(C=c, kernel="rbf", gamma=gamma, class_weight="balanced", probability=False, random_state=args.seed + 999),
                    )
                    final.fit(x_old, y_old)
                    ext_score = final.decision_function(x_ext).astype(np.float32)
                    row: dict[str, object] = {
                        "align": "diag",
                        "pca_components": int(pca_components),
                        "pca_explained": float(pca.explained_variance_ratio_.sum()),
                        "c": float(c),
                        "gamma": str(gamma),
                        "threshold_objective": objective,
                        "old_cv_threshold": float(threshold),
                        "old_cv_threshold_score": float(threshold_score),
                    }
                    row.update({f"old_{k}": v for k, v in metric_dict(y_old, old_pred, oof).items()})
                    row.update(eval_external(external, ext_score, threshold))
                    rows.append(row)
                    key = (float(row["strict_balanced_accuracy"]), float(row["strict_accuracy"]))
                    if key > best_external:
                        best_external = key
                        best_external_payload = (row, ext_score)
        print(f"[pca {pca_components}] current best strict_bacc={best_external[0]:.4f} strict_acc={best_external[1]:.4f}", flush=True)

    summary = pd.DataFrame(rows).sort_values(["strict_balanced_accuracy", "strict_accuracy", "old_balanced_accuracy"], ascending=False)
    summary.to_csv(out / "diag_svc_focus_fast_summary.csv", index=False, encoding="utf-8-sig")
    if best_external_payload:
        row, score = best_external_payload
        pred = external[["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "strict_task7_eval", "image_name"]].copy()
        pred["score_high"] = score
        pred["prob_high_sigmoid"] = sigmoid(score)
        pred["pred_idx"] = (score >= float(row["old_cv_threshold"])).astype(int)
        pred["correct"] = (pred["pred_idx"].astype(int) == pred["label_idx"].astype(int)).astype(int)
        pred.to_csv(out / "best_external_case_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "boundary": {
            "uses_external_labels_for_training": False,
            "selection_note": "Rows are trained and thresholded with old OOF only; sorting by external metrics is exploratory diagnostics.",
        },
        "best_external": best_external_payload[0] if best_external_payload else None,
    }
    (out / "diag_svc_focus_fast_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
