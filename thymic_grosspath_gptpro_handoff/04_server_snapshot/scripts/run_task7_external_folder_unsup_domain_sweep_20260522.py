from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class MethodSpec:
    align: str
    pca_components: int
    model: str
    weighting: str = "none"
    pseudo: str = "none"
    threshold_objective: str = "balanced_accuracy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised domain adaptation sweep for a Task7 external image folder.")
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
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/28_external_thymoma_carcinoma_unsup_domain_sweep_20260522",
    )
    parser.add_argument(
        "--old-label-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
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


def read_old(root: Path, feature_dir: str, label_csv: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / feature_dir
    table = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    feat = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    if "label_idx" not in table.columns:
        labels = pd.read_csv(root / label_csv, dtype={"case_id": str})[["case_id", "label_idx"]]
        table = table.merge(labels, on="case_id", how="left")
    if table["label_idx"].isna().any():
        missing = table.loc[table["label_idx"].isna(), "case_id"].head(10).tolist()
        raise KeyError(f"Missing old labels for feature rows: {missing}")
    x = feat[table["feature_idx"].astype(int).to_numpy()]
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
    if "strict_task7_eval" not in merged.columns:
        merged["strict_task7_eval"] = 1
    x = feat[merged["feature_idx"].astype(int).to_numpy()]
    merged["label_idx"] = merged["label_idx"].astype(int)
    merged["strict_task7_eval"] = merged["strict_task7_eval"].astype(int)
    return merged.reset_index(drop=True), x.astype(np.float32)


def sym_inv_sqrt(cov: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov + eps * np.eye(cov.shape[0], dtype=cov.dtype))
    vals = np.clip(vals, eps, None)
    return (vecs * (1.0 / np.sqrt(vals))).dot(vecs.T)


def sym_sqrt(cov: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov + eps * np.eye(cov.shape[0], dtype=cov.dtype))
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)).dot(vecs.T)


def align_source_to_target(x_source: np.ndarray, x_target: np.ndarray, align: str) -> tuple[np.ndarray, np.ndarray]:
    if align == "none":
        return x_source, x_target
    ms = x_source.mean(axis=0, keepdims=True)
    mt = x_target.mean(axis=0, keepdims=True)
    if align == "mean":
        return (x_source - ms + mt).astype(np.float32), x_target
    ss = x_source.std(axis=0, keepdims=True) + 1e-5
    st = x_target.std(axis=0, keepdims=True) + 1e-5
    if align == "diag":
        return ((x_source - ms) / ss * st + mt).astype(np.float32), x_target
    if align == "coral":
        xs0 = x_source - ms
        xt0 = x_target - mt
        cs = np.cov(xs0, rowvar=False)
        ct = np.cov(xt0, rowvar=False)
        return (xs0 @ sym_inv_sqrt(cs) @ sym_sqrt(ct) + mt).astype(np.float32), x_target
    raise ValueError(align)


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


def fit_model(model, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
    if sample_weight is None:
        model.fit(x, y)
        return model
    try:
        model.fit(x, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        step_name = list(model.named_steps.keys())[-1] if hasattr(model, "named_steps") else None
        if step_name is None:
            model.fit(x, y)
        else:
            model.fit(x, y, **{f"{step_name}__sample_weight": sample_weight})
    return model


def target_similarity_weights(x_old: np.ndarray, x_target: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    old_norm = x_old / np.maximum(np.linalg.norm(x_old, axis=1, keepdims=True), 1e-8)
    target_norm = x_target / np.maximum(np.linalg.norm(x_target, axis=1, keepdims=True), 1e-8)
    sim = old_norm @ target_norm.T
    if mode == "maxsim":
        score = sim.max(axis=1)
    elif mode == "top5sim":
        score = np.sort(sim, axis=1)[:, -min(5, sim.shape[1]) :].mean(axis=1)
    else:
        raise ValueError(mode)
    score = score - score.min()
    score = score / max(float(score.max()), 1e-8)
    weights = 0.25 + 1.75 * score
    return weights.astype(np.float32)


def apply_pseudo_training(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_target: np.ndarray,
    seed: int,
    pseudo: str,
    sample_weight: np.ndarray | None,
) -> object:
    base = make_model(model_name, seed)
    fit_model(base, x_train, y_train, sample_weight)
    if pseudo == "none":
        return base
    prob = prob_high(base, x_target)
    if pseudo == "p90":
        mask = (prob <= 0.10) | (prob >= 0.90)
        pseudo_w = 0.5
    elif pseudo == "p85":
        mask = (prob <= 0.15) | (prob >= 0.85)
        pseudo_w = 0.35
    elif pseudo == "p80":
        mask = (prob <= 0.20) | (prob >= 0.80)
        pseudo_w = 0.25
    else:
        raise ValueError(pseudo)
    if int(mask.sum()) < 4:
        return base
    x_aug = np.concatenate([x_train, x_target[mask]], axis=0)
    y_aug = np.concatenate([y_train, (prob[mask] >= 0.5).astype(int)], axis=0)
    if sample_weight is None:
        w_aug = np.concatenate([np.ones(len(y_train), dtype=np.float32), np.full(mask.sum(), pseudo_w, dtype=np.float32)])
    else:
        w_aug = np.concatenate([sample_weight.astype(np.float32), np.full(mask.sum(), pseudo_w, dtype=np.float32)])
    final = make_model(model_name, seed + 777)
    fit_model(final, x_aug, y_aug, w_aug)
    return final


def evaluate_external(frame: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (prob >= threshold).astype(int)
    y = frame["label_idx"].to_numpy(dtype=int)
    strict = frame["strict_task7_eval"].to_numpy(dtype=int) == 1
    out = {f"all_{k}": v for k, v in metric_dict(y, pred, prob).items()}
    out.update({f"strict_{k}": v for k, v in metric_dict(y[strict], pred[strict], prob[strict]).items()})
    return out


def method_grid() -> list[MethodSpec]:
    models = ["logreg_c001", "logreg_c01", "logreg_c1", "svc_c1", "rf_d5", "extra_d5"]
    specs: list[MethodSpec] = []
    for pca_components in [32, 64, 128, 200]:
        for align in ["none", "mean", "diag", "coral"]:
            for model in models:
                specs.append(MethodSpec(align=align, pca_components=pca_components, model=model))
        for align in ["none", "coral"]:
            for weighting in ["maxsim", "top5sim"]:
                for model in ["logreg_c001", "logreg_c01", "rf_d5", "extra_d5"]:
                    specs.append(MethodSpec(align=align, pca_components=pca_components, model=model, weighting=weighting))
        for align in ["none", "coral"]:
            for pseudo in ["p90", "p85", "p80"]:
                for model in ["logreg_c001", "logreg_c01", "rf_d5", "extra_d5"]:
                    specs.append(MethodSpec(align=align, pca_components=pca_components, model=model, pseudo=pseudo))
    return specs


def run_spec(
    spec: MethodSpec,
    x_old_raw: np.ndarray,
    y_old: np.ndarray,
    x_ext_raw: np.ndarray,
    external: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    scaler = StandardScaler()
    x_old_scaled = scaler.fit_transform(x_old_raw)
    x_ext_scaled = scaler.transform(x_ext_raw)
    n_components = min(spec.pca_components, x_old_scaled.shape[0] - 1, x_old_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    x_old = pca.fit_transform(x_old_scaled).astype(np.float32)
    x_ext = pca.transform(x_ext_scaled).astype(np.float32)
    x_old_aligned, x_ext_aligned = align_source_to_target(x_old, x_ext, spec.align)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_old), dtype=np.float32)
    for fold_id, (tr, va) in enumerate(folds.split(x_old_aligned, y_old)):
        sample_weight = target_similarity_weights(x_old_aligned[tr], x_ext_aligned, spec.weighting)
        model = apply_pseudo_training(
            spec.model,
            x_old_aligned[tr],
            y_old[tr],
            x_ext_aligned,
            seed + fold_id * 37,
            spec.pseudo,
            sample_weight,
        )
        oof[va] = prob_high(model, x_old_aligned[va])

    threshold, threshold_score = best_threshold(y_old, oof, spec.threshold_objective)
    old_pred = (oof >= threshold).astype(int)
    old_metrics = metric_dict(y_old, old_pred, oof)

    sample_weight = target_similarity_weights(x_old_aligned, x_ext_aligned, spec.weighting)
    final_model = apply_pseudo_training(spec.model, x_old_aligned, y_old, x_ext_aligned, seed + 999, spec.pseudo, sample_weight)
    ext_prob = prob_high(final_model, x_ext_aligned)
    row: dict[str, object] = {
        "align": spec.align,
        "pca_components": int(n_components),
        "pca_explained": float(pca.explained_variance_ratio_.sum()),
        "model": spec.model,
        "weighting": spec.weighting,
        "pseudo": spec.pseudo,
        "threshold_objective": spec.threshold_objective,
        "old_cv_threshold": float(threshold),
        "old_cv_threshold_score": float(threshold_score),
    }
    row.update({f"old_{k}": v for k, v in old_metrics.items()})
    row.update(evaluate_external(external, ext_prob, threshold))
    return row, ext_prob


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    old, x_old = read_old(root, args.old_feature_dir, args.old_label_csv)
    external, x_ext = read_external(root, args.external_feature_dir)
    y_old = old["label_idx"].to_numpy(dtype=int)

    rows: list[dict[str, object]] = []
    best_key = (-1.0, -1.0)
    best_prob: np.ndarray | None = None
    best_row: dict[str, object] | None = None
    for idx, spec in enumerate(method_grid(), start=1):
        row, prob = run_spec(spec, x_old, y_old, x_ext, external, args.seed + idx * 13)
        rows.append(row)
        key = (float(row["strict_balanced_accuracy"]), float(row["strict_accuracy"]))
        if key > best_key:
            best_key = key
            best_prob = prob
            best_row = row
        if idx % 25 == 0:
            print(
                f"[{idx:03d}] best strict_bacc={best_key[0]:.4f} strict_acc={best_key[1]:.4f} "
                f"align={best_row['align'] if best_row else ''} model={best_row['model'] if best_row else ''}",
                flush=True,
            )

    summary = pd.DataFrame(rows).sort_values(["strict_balanced_accuracy", "strict_accuracy", "strict_f1"], ascending=False)
    summary.to_csv(out / "unsup_domain_sweep_summary.csv", index=False, encoding="utf-8-sig")

    if best_prob is not None and best_row is not None:
        pred = external[
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
        pred["prob_high"] = best_prob
        pred["pred_idx"] = (best_prob >= float(best_row["old_cv_threshold"])).astype(int)
        pred["correct"] = (pred["pred_idx"].astype(int) == pred["label_idx"].astype(int)).astype(int)
        pred.to_csv(out / "best_unsup_domain_case_predictions.csv", index=False, encoding="utf-8-sig")

    report = {
        "boundary": {
            "uses_external_labels_for_training": False,
            "uses_external_images_unlabeled_for_alignment_weighting_or_pseudo_labels": True,
            "note": "External labels are used only for metric reporting and experiment diagnostics.",
        },
        "old_n": int(len(old)),
        "external_n": int(len(external)),
        "strict_n": int(external["strict_task7_eval"].sum()),
        "best_row": best_row,
    }
    (out / "unsup_domain_sweep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
