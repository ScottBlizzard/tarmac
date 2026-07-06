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
    parser = argparse.ArgumentParser(description="Focused diag-aligned SVC sweep for Task7 external folder.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--old-feature-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521",
    )
    parser.add_argument(
        "--external-feature-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/20_external_thymoma_carcinoma_64style_wpc_20260522",
    )
    parser.add_argument(
        "--old-label-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/30_external_thymoma_carcinoma_diag_svc_focus_20260522",
    )
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float | int]:
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
        out["auc"] = float(roc_auc_score(y, prob))
    return out


def best_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        if objective == "accuracy":
            score = accuracy_score(y, pred)
        elif objective == "f1":
            score = f1_score(y, pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def diag_align(x_source: np.ndarray, x_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ms = x_source.mean(axis=0, keepdims=True)
    mt = x_target.mean(axis=0, keepdims=True)
    ss = x_source.std(axis=0, keepdims=True) + 1e-5
    st = x_target.std(axis=0, keepdims=True) + 1e-5
    return ((x_source - ms) / ss * st + mt).astype(np.float32), x_target


def make_model(c: float, gamma: str | float, seed: int):
    return make_pipeline(
        StandardScaler(),
        SVC(C=c, kernel="rbf", gamma=gamma, class_weight="balanced", probability=True, random_state=seed),
    )


def eval_external(frame: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (prob >= threshold).astype(int)
    y = frame["label_idx"].to_numpy(dtype=int)
    strict = frame["strict_task7_eval"].astype(int).to_numpy() == 1
    out = {f"all_{k}": v for k, v in metric_dict(y, pred, prob).items()}
    out.update({f"strict_{k}": v for k, v in metric_dict(y[strict], pred[strict], prob[strict]).items()})
    return out


def run_one(
    x_old_raw: np.ndarray,
    y_old: np.ndarray,
    x_ext_raw: np.ndarray,
    external: pd.DataFrame,
    pca_components: int,
    c: float,
    gamma: str | float,
    threshold_objective: str,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    scaler = StandardScaler()
    x_old_scaled = scaler.fit_transform(x_old_raw)
    x_ext_scaled = scaler.transform(x_ext_raw)
    n_components = min(pca_components, x_old_scaled.shape[0] - 1, x_old_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    x_old = pca.fit_transform(x_old_scaled).astype(np.float32)
    x_ext = pca.transform(x_ext_scaled).astype(np.float32)
    x_old, x_ext = diag_align(x_old, x_ext)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_old), dtype=np.float32)
    for fold_id, (tr, va) in enumerate(skf.split(x_old, y_old)):
        model = make_model(c, gamma, seed + fold_id * 19)
        model.fit(x_old[tr], y_old[tr])
        oof[va] = model.predict_proba(x_old[va])[:, 1]
    threshold, threshold_score = best_threshold(y_old, oof, threshold_objective)
    old_pred = (oof >= threshold).astype(int)

    final = make_model(c, gamma, seed + 999)
    final.fit(x_old, y_old)
    prob = final.predict_proba(x_ext)[:, 1]
    row: dict[str, object] = {
        "align": "diag",
        "pca_components": int(n_components),
        "pca_explained": float(pca.explained_variance_ratio_.sum()),
        "c": float(c),
        "gamma": str(gamma),
        "threshold_objective": threshold_objective,
        "old_cv_threshold": float(threshold),
        "old_cv_threshold_score": float(threshold_score),
    }
    row.update({f"old_{k}": v for k, v in metric_dict(y_old, old_pred, oof).items()})
    row.update(eval_external(external, prob, threshold))
    return row, prob


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    old, x_old = read_old(root, args.old_feature_dir, args.old_label_csv)
    external, x_ext = read_external(root, args.external_feature_dir)
    y_old = old["label_idx"].to_numpy(dtype=int)

    rows: list[dict[str, object]] = []
    best_external = (-1.0, -1.0)
    best_old = (-1.0, -1.0)
    best_external_payload: tuple[dict[str, object], np.ndarray] | None = None
    best_old_payload: tuple[dict[str, object], np.ndarray] | None = None
    gammas: list[str | float] = ["scale", 0.0005, 0.001, 0.003, 0.01, 0.03]
    for idx, pca_components in enumerate([12, 16, 24, 32, 40, 48, 64, 96], start=1):
        for c in [0.1, 0.3, 1.0, 3.0, 10.0]:
            for gamma in gammas:
                for objective in ["balanced_accuracy", "accuracy"]:
                    row, prob = run_one(x_old, y_old, x_ext, external, pca_components, c, gamma, objective, args.seed + idx * 101)
                    rows.append(row)
                    ext_key = (float(row["strict_balanced_accuracy"]), float(row["strict_accuracy"]))
                    old_key = (float(row["old_balanced_accuracy"]), float(row["old_accuracy"]))
                    if ext_key > best_external:
                        best_external = ext_key
                        best_external_payload = (row, prob)
                    if old_key > best_old:
                        best_old = old_key
                        best_old_payload = (row, prob)
    summary = pd.DataFrame(rows).sort_values(["strict_balanced_accuracy", "strict_accuracy", "old_balanced_accuracy"], ascending=False)
    summary.to_csv(out / "diag_svc_focus_summary.csv", index=False, encoding="utf-8-sig")

    for name, payload in [("best_external", best_external_payload), ("best_old_oof", best_old_payload)]:
        if payload is None:
            continue
        row, prob = payload
        pred = external[["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "strict_task7_eval", "image_name"]].copy()
        pred["prob_high"] = prob
        pred["pred_idx"] = (prob >= float(row["old_cv_threshold"])).astype(int)
        pred["correct"] = (pred["pred_idx"].astype(int) == pred["label_idx"].astype(int)).astype(int)
        pred.to_csv(out / f"{name}_case_predictions.csv", index=False, encoding="utf-8-sig")

    report = {
        "boundary": {
            "uses_external_labels_for_training": False,
            "selection_note": "best_external is exploratory; best_old_oof is selected without external labels.",
        },
        "best_external": best_external_payload[0] if best_external_payload else None,
        "best_old_oof": best_old_payload[0] if best_old_payload else None,
    }
    (out / "diag_svc_focus_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
