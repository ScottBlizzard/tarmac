from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised CORAL probe for a free-form Task7 external folder.")
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
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/25_external_thymoma_carcinoma_coral_wpc_20260522",
    )
    parser.add_argument(
        "--old-label-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument("--pca-components", type=int, default=128)
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


def best_threshold(y: np.ndarray, prob: np.ndarray, objective: str = "balanced_accuracy") -> tuple[float, float]:
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


def read_old(root: Path, feature_dir: str, label_csv: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / feature_dir
    table = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    feat = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    x = feat[table["feature_idx"].astype(int).to_numpy()]
    if "label_idx" not in table.columns:
        labels = pd.read_csv(root / label_csv, dtype={"case_id": str})[["case_id", "label_idx"]]
        table = table.merge(labels, on="case_id", how="left")
    if table["label_idx"].isna().any():
        missing = table.loc[table["label_idx"].isna(), "case_id"].head(10).tolist()
        raise KeyError(f"Missing old labels for feature table rows: {missing}")
    table["label_idx"] = table["label_idx"].astype(int)
    return table.reset_index(drop=True), x.astype(np.float32)


def read_external(root: Path, feature_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / feature_dir
    registry = pd.read_csv(d / "external_folder_task7_registry.csv", dtype={"case_id": str, "original_case_id": str})
    table = pd.read_csv(d / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    feat = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    merged = registry.merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if merged["feature_idx"].isna().any():
        missing = merged.loc[merged["feature_idx"].isna(), "case_id"].head(10).tolist()
        raise KeyError(f"Missing external feature rows: {missing}")
    x = feat[merged["feature_idx"].astype(int).to_numpy()]
    merged["label_idx"] = merged["label_idx"].astype(int)
    if "strict_task7_eval" not in merged.columns:
        merged["strict_task7_eval"] = 1
    return merged.reset_index(drop=True), x.astype(np.float32)


def _sym_inv_sqrt(cov: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov + eps * np.eye(cov.shape[0], dtype=cov.dtype))
    vals = np.clip(vals, eps, None)
    return (vecs * (1.0 / np.sqrt(vals))).dot(vecs.T)


def _sym_sqrt(cov: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov + eps * np.eye(cov.shape[0], dtype=cov.dtype))
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)).dot(vecs.T)


def coral_source_to_target(xs: np.ndarray, xt: np.ndarray) -> np.ndarray:
    ms = xs.mean(axis=0, keepdims=True)
    mt = xt.mean(axis=0, keepdims=True)
    xs0 = xs - ms
    xt0 = xt - mt
    cs = np.cov(xs0, rowvar=False)
    ct = np.cov(xt0, rowvar=False)
    return (xs0 @ _sym_inv_sqrt(cs) @ _sym_sqrt(ct) + mt).astype(np.float32)


def make_model(name: str, seed: int):
    if name == "logreg_c001":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced", random_state=seed))
    if name == "logreg_c01":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced", random_state=seed))
    if name == "logreg_c1":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed))
    if name == "svc_c1":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=seed))
    if name == "rf_d5":
        return RandomForestClassifier(n_estimators=600, max_depth=5, min_samples_leaf=4, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    if name == "extra_d5":
        return ExtraTreesClassifier(n_estimators=800, max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=-1)
    raise ValueError(name)


def prob_high(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    score = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-score))


def evaluate_external(frame: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (prob >= threshold).astype(int)
    y = frame["label_idx"].to_numpy(dtype=int)
    out = {f"all_{k}": v for k, v in metric_dict(y, pred, prob).items()}
    strict = frame["strict_task7_eval"].astype(int).to_numpy() == 1
    out.update({f"strict_{k}": v for k, v in metric_dict(y[strict], pred[strict], prob[strict]).items()})
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    old, x_old_raw = read_old(root, args.old_feature_dir, args.old_label_csv)
    external, x_ext_raw = read_external(root, args.external_feature_dir)
    y_old = old["label_idx"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_old_scaled = scaler.fit_transform(x_old_raw)
    x_ext_scaled = scaler.transform(x_ext_raw)
    n_components = min(args.pca_components, x_old_scaled.shape[0] - 1, x_old_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed, svd_solver="randomized")
    x_old = pca.fit_transform(x_old_scaled).astype(np.float32)
    x_ext = pca.transform(x_ext_scaled).astype(np.float32)
    x_old_coral = coral_source_to_target(x_old, x_ext)

    names = ["logreg_c001", "logreg_c01", "logreg_c1", "svc_c1", "rf_d5", "extra_d5"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    rows: list[dict[str, object]] = []
    pred_cols = external[
        [
            "case_id",
            "original_case_id",
            "task_l6_label",
            "task_l7_label",
            "label_idx",
            "strict_task7_eval",
            "image_name",
        ]
    ].copy()

    for i, name in enumerate(names):
        oof = np.zeros(len(y_old), dtype=np.float32)
        for fold, (tr, va) in enumerate(skf.split(x_old, y_old)):
            model = make_model(name, args.seed + i * 31 + fold)
            model.fit(x_old[tr], y_old[tr])
            oof[va] = prob_high(model, x_old[va])
        threshold, _ = best_threshold(y_old, oof)
        old_pred = (oof >= threshold).astype(int)
        old_metric = metric_dict(y_old, old_pred, oof)

        model = make_model(name, args.seed + i * 31 + 1000)
        model.fit(x_old_coral, y_old)
        ext_prob = prob_high(model, x_ext)
        row: dict[str, object] = {
            "model": name,
            "old_cv_threshold": threshold,
            "pca_components": n_components,
            "pca_explained": float(pca.explained_variance_ratio_.sum()),
        }
        row.update({f"old_{k}": v for k, v in old_metric.items()})
        row.update(evaluate_external(external, ext_prob, threshold))
        rows.append(row)
        pred_cols[f"{name}_prob_high"] = ext_prob
        pred_cols[f"{name}_pred"] = (ext_prob >= threshold).astype(int)

    summary = pd.DataFrame(rows).sort_values(["strict_balanced_accuracy", "strict_accuracy"], ascending=False)
    summary.to_csv(out / "coral_probe_summary.csv", index=False, encoding="utf-8-sig")
    pred_cols.to_csv(out / "coral_probe_case_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "boundary": {
            "uses_external_labels_for_training": False,
            "uses_external_images_unlabeled_for_coral_alignment": True,
            "note": "CORAL is unsupervised target-domain adaptation. External labels are used only after inference for metric reporting.",
        },
        "old_n": int(len(old)),
        "external_n": int(len(external)),
        "strict_n": int(external["strict_task7_eval"].astype(int).sum()),
        "rows": rows,
    }
    (out / "coral_probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
