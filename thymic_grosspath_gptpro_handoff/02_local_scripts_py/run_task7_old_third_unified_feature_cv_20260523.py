from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
OUT = BASE / "task7_adaptation_runs" / "44_old_third_unified_feature_cv_20260523"

OLD_LABELS = BASE / "task7_curriculum_runs" / "09_case_mlp_schemeB_m060_salvagehard_full5fold" / "curriculum_case_table.csv"
THIRD_REGISTRY = BASE / "task7_external_runs" / "01_third_batch_64style_image_only_20260521" / "third_batch_task7_registry.csv"

VARIANTS = {
    "whole": (
        BASE / "task7_gross_feature_runs" / "10_review_router_embedding_probe_20260520",
        BASE / "task7_external_runs" / "01_third_batch_64style_image_only_20260521",
    ),
    "crop": (
        BASE / "task7_gross_feature_runs" / "67_roi_crop_embedding_probe_20260521",
        BASE / "task7_external_runs" / "03_third_batch_crop_64style_20260521",
    ),
    "whole_crop": (
        BASE / "task7_gross_feature_runs" / "68_roi_whole_plus_crop_embedding_probe_20260521",
        BASE / "task7_external_runs" / "04_third_batch_whole_plus_crop_64style_20260521",
    ),
    "cropm40": (
        BASE / "task7_gross_feature_runs" / "69_roi_cropm40_embedding_probe_20260521",
        BASE / "task7_external_runs" / "07_third_batch_cropm40_64style_20260521",
    ),
    "bgnorm085_whole_crop": (
        BASE / "task7_gross_feature_runs" / "70_bgnorm085_whole_plus_crop_embedding_probe_20260521",
        BASE / "task7_external_runs" / "08_third_batch_bgnorm085_whole_plus_crop_64style_20260521",
    ),
    "flip_lr_whole_crop": (
        BASE / "task7_gross_feature_runs" / "73_flip_lr_whole_plus_crop_embedding_probe_20260521",
        BASE / "task7_external_runs" / "11_third_batch_flip_lr_whole_plus_crop_64style_20260521",
    ),
    "tta_avg_whole_crop": (
        BASE / "task7_gross_feature_runs" / "74_tta_avg_orig_flip_lr_whole_plus_crop_20260521",
        BASE / "task7_external_runs" / "12_third_batch_tta_avg_orig_flip_lr_whole_plus_crop_64style_20260521",
    ),
    "wpc_stats": (
        BASE / "task7_gross_feature_runs" / "75_wpc_plus_image_stats_20260521",
        BASE / "task7_external_runs" / "13_third_batch_wpc_plus_image_stats_64style_20260521",
    ),
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})


def load_feature_dir(directory: Path, third: bool) -> tuple[pd.DataFrame, np.ndarray]:
    table_name = "third_batch_dino_concat_feature_table.csv" if third else "case_dino_concat_feature_table.csv"
    npy_name = "third_batch_dino_concat_features.npy" if third else "case_dino_concat_features.npy"
    table = read_csv(directory / table_name)
    features = np.load(directory / npy_name).astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    return table, features


def align_features(order: pd.Series, table: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    idx = order.astype(str).to_frame("case_id").merge(table[["case_id", "feature_idx"]], on="case_id", how="left")["feature_idx"]
    if idx.isna().any():
        missing = order[idx.isna()].head(10).tolist()
        raise KeyError(f"Missing feature rows: {missing}")
    return features[idx.astype(int).to_numpy()].astype(np.float32)


def metric_row(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    row: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }
    row["auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")
    return row


def choose_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, dict[str, float | int]]:
    best_t = 0.5
    best_key: tuple[float, ...] | None = None
    best_row: dict[str, float | int] = {}
    for t in np.linspace(0.05, 0.95, 181):
        row = metric_row(y, prob, float(t))
        if objective == "accuracy":
            key = (float(row["accuracy"]), float(row["balanced_accuracy"]), float(row["f1"]))
        elif objective == "balanced_accuracy":
            key = (float(row["balanced_accuracy"]), float(row["accuracy"]), float(row["f1"]))
        elif objective == "high_sens80":
            penalty = min(0.0, float(row["sensitivity_high"]) - 0.80)
            key = (float(row["balanced_accuracy"]) + 0.4 * penalty, float(row["accuracy"]), float(row["f1"]))
        else:
            raise ValueError(objective)
        if best_key is None or key > best_key:
            best_key = key
            best_t = float(t)
            best_row = row
    return best_t, best_row


def make_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    old = read_csv(OLD_LABELS)
    old = old[["case_id", "fold_id", "label_idx", "difficulty", "difficulty_fine"]].copy()
    old["original_case_id"] = old["case_id"].astype(str).str.extract(r"_(.+)$", expand=False).fillna(old["case_id"].astype(str))
    old["task_l6_label"] = ""
    old["task_l7_label"] = np.where(old["label_idx"].astype(int).eq(1), "high_risk_group", "low_risk_group")
    old["domain"] = "old"
    old["source_folder"] = "old"

    third = read_csv(THIRD_REGISTRY)
    third = third[["case_id", "original_case_id", "label_idx", "task_l6_label", "task_l7_label", "source_folder", "image_name", "image_path"]].copy()
    third["domain"] = "third"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260523)
    third["fold_id"] = -1
    for fold, (_, idx) in enumerate(skf.split(third, third["label_idx"].astype(int)), start=1):
        third.loc[idx, "fold_id"] = fold
    return old.reset_index(drop=True), third.reset_index(drop=True)


def model_grid() -> dict[str, object]:
    models: dict[str, object] = {}
    for c in [0.003, 0.01, 0.03]:
        models[f"logreg_bal_c{c:g}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=5000),
        )
        models[f"logreg_plain_c{c:g}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, solver="liblinear", max_iter=5000),
        )
    return models


def sample_weights(frame: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    y = frame["label_idx"].astype(int).to_numpy()
    weights = np.ones(len(frame), dtype=float)
    if mode in {"label", "domain_label"}:
        for label in sorted(np.unique(y)):
            mask = y == label
            weights[mask] *= len(y) / (2.0 * mask.sum())
    if mode in {"domain", "domain_label"}:
        domains = frame["domain"].astype(str).to_numpy()
        for domain in sorted(np.unique(domains)):
            mask = domains == domain
            weights[mask] *= len(domains) / (2.0 * mask.sum())
    return weights


def fit_model(model: object, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> object:
    fitted = clone(model)
    if weights is None:
        fitted.fit(x, y)
    elif hasattr(fitted, "steps"):
        last_name = fitted.steps[-1][0]
        fitted.fit(x, y, **{f"{last_name}__sample_weight": weights})
    else:
        fitted.fit(x, y, sample_weight=weights)
    return fitted


def predict_prob(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    score = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-score))


def evaluate_variant(name: str, old: pd.DataFrame, third: pd.DataFrame, old_dir: Path, third_dir: Path) -> tuple[list[dict[str, object]], pd.DataFrame]:
    old_table, old_features = load_feature_dir(old_dir, third=False)
    third_table, third_features = load_feature_dir(third_dir, third=True)
    old_x = align_features(old["case_id"], old_table, old_features)
    third_x = align_features(third["case_id"], third_table, third_features)
    frame = pd.concat([old, third], ignore_index=True, sort=False)
    x = np.vstack([old_x, third_x]).astype(np.float32)
    y = frame["label_idx"].astype(int).to_numpy()
    folds = frame["fold_id"].astype(int).to_numpy()

    if x.shape[1] > 512:
        # Unsupervised compression is used only to make this development sweep tractable.
        # The final locked-external run should refit the same preprocessing on development data only.
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        n_components = min(128, x_scaled.shape[0] - 1, x_scaled.shape[1])
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=20260523)
        x = pca.fit_transform(x_scaled).astype(np.float32)
        print(
            f"[variant] {name} pca_x={x.shape} explained={float(pca.explained_variance_ratio_.sum()):.4f} old={len(old)} third={len(third)}",
            flush=True,
        )
    else:
        print(f"[variant] {name} x={x.shape} old={len(old)} third={len(third)}", flush=True)
    rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    for model_name, model in model_grid().items():
        for weight_mode in ["none", "domain_label"]:
            oof = np.zeros(len(frame), dtype=float)
            for fold in sorted(np.unique(folds)):
                tr = folds != fold
                va = folds == fold
                w = sample_weights(frame.loc[tr].reset_index(drop=True), weight_mode)
                fitted = fit_model(model, x[tr], y[tr], w)
                oof[va] = predict_prob(fitted, x[va])
            for objective in ["balanced_accuracy", "accuracy", "high_sens80"]:
                threshold, all_row = choose_threshold(y, oof, objective)
                old_mask = frame["domain"].eq("old").to_numpy()
                third_mask = frame["domain"].eq("third").to_numpy()
                old_row = metric_row(y[old_mask], oof[old_mask], threshold)
                third_row = metric_row(y[third_mask], oof[third_mask], threshold)
                row = {
                    "variant": name,
                    "model": model_name,
                    "weight_mode": weight_mode,
                    "objective": objective,
                    **{f"all_{k}": v for k, v in all_row.items()},
                    **{f"old_{k}": v for k, v in old_row.items()},
                    **{f"third_{k}": v for k, v in third_row.items()},
                }
                row["selection_score"] = (
                    0.30 * float(row["old_balanced_accuracy"])
                    + 0.30 * float(row["third_balanced_accuracy"])
                    + 0.20 * float(row["old_accuracy"])
                    + 0.20 * float(row["third_accuracy"])
                    - max(0.0, 0.88 - float(row["old_accuracy"]))
                    - max(0.0, 0.70 - float(row["third_sensitivity_high"]))
                )
                row["old_guard_090"] = bool(row["old_accuracy"] >= 0.90 and row["old_balanced_accuracy"] >= 0.90)
                rows.append(row)

            pred = frame[
                [
                    "case_id",
                    "original_case_id",
                    "domain",
                    "source_folder",
                    "fold_id",
                    "task_l6_label",
                    "task_l7_label",
                    "label_idx",
                ]
            ].copy()
            pred["variant"] = name
            pred["model"] = model_name
            pred["weight_mode"] = weight_mode
            pred["oof_prob_high"] = oof
            pred_frames.append(pred)

    return rows, pd.concat(pred_frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old, third = make_frames()
    old.to_csv(OUT / "old_frame_with_folds.csv", index=False, encoding="utf-8-sig")
    third.to_csv(OUT / "third_frame_with_folds.csv", index=False, encoding="utf-8-sig")

    all_rows: list[dict[str, object]] = []
    all_preds: list[pd.DataFrame] = []
    active_variants = ["whole_crop", "wpc_stats", "bgnorm085_whole_crop", "crop"]
    for name in active_variants:
        old_dir, third_dir = VARIANTS[name]
        required = [
            old_dir / "case_dino_concat_feature_table.csv",
            old_dir / "case_dino_concat_features.npy",
            third_dir / "third_batch_dino_concat_feature_table.csv",
            third_dir / "third_batch_dino_concat_features.npy",
        ]
        if not all(p.exists() for p in required):
            print(f"[skip] {name}: missing feature cache", flush=True)
            continue
        rows, preds = evaluate_variant(name, old, third, old_dir, third_dir)
        all_rows.extend(rows)
        all_preds.append(preds)

    summary = pd.DataFrame(all_rows)
    predictions = pd.concat(all_preds, ignore_index=True)
    summary = summary.sort_values(["selection_score", "third_accuracy", "third_balanced_accuracy"], ascending=False).reset_index(drop=True)
    summary.to_csv(OUT / "unified_feature_cv_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "unified_feature_cv_all_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.head(100).to_csv(OUT / "top100_unified_feature_cv.csv", index=False, encoding="utf-8-sig")

    guarded = summary[summary["old_guard_090"]].copy()
    selected_row = (guarded if not guarded.empty else summary).sort_values(
        ["selection_score", "third_accuracy", "third_balanced_accuracy"], ascending=False
    ).iloc[0]
    selected_pred = predictions[
        (predictions["variant"] == selected_row["variant"])
        & (predictions["model"] == selected_row["model"])
        & (predictions["weight_mode"] == selected_row["weight_mode"])
    ].copy()
    selected_pred["threshold"] = float(selected_row["all_threshold"])
    selected_pred["oof_pred_idx"] = (selected_pred["oof_prob_high"].astype(float) >= float(selected_row["all_threshold"])).astype(int)
    selected_pred["oof_correct"] = (selected_pred["oof_pred_idx"] == selected_pred["label_idx"].astype(int)).astype(int)
    selected_pred.to_csv(OUT / "selected_unified_feature_oof_predictions.csv", index=False, encoding="utf-8-sig")

    # Train a small final frozen-feature model on all development data for later locked external testing.
    old_dir, third_dir = VARIANTS[str(selected_row["variant"])]
    old_table, old_features = load_feature_dir(old_dir, third=False)
    third_table, third_features = load_feature_dir(third_dir, third=True)
    x_all = np.vstack([
        align_features(old["case_id"], old_table, old_features),
        align_features(third["case_id"], third_table, third_features),
    ]).astype(np.float32)
    preprocessor = None
    if x_all.shape[1] > 512:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_all)
        n_components = min(128, x_scaled.shape[0] - 1, x_scaled.shape[1])
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=20260523)
        x_all = pca.fit_transform(x_scaled).astype(np.float32)
        preprocessor = {"scaler": scaler, "pca": pca}
    frame_all = pd.concat([old, third], ignore_index=True, sort=False)
    y_all = frame_all["label_idx"].astype(int).to_numpy()
    model = model_grid()[str(selected_row["model"])]
    w = sample_weights(frame_all.reset_index(drop=True), str(selected_row["weight_mode"]))
    final_model = fit_model(model, x_all, y_all, w)
    joblib.dump(
        {
            "model": final_model,
            "preprocessor": preprocessor,
            "variant": str(selected_row["variant"]),
            "threshold": float(selected_row["all_threshold"]),
            "feature_old_dir": str(old_dir.relative_to(ROOT)),
            "feature_third_dir": str(third_dir.relative_to(ROOT)),
            "selected_row": selected_row.to_dict(),
            "data_boundary": "trained on old batch1+batch2 plus third-batch development cohort only; locked external 胸腺瘤+癌 not used",
        },
        OUT / "selected_final_model_old_plus_third.joblib",
    )

    report = {
        "experiment": "Task7 unified old+third frozen-feature development CV",
        "data_boundary": "Old data and third batch are used as development data. Locked external folder 胸腺瘤+癌 is not read or used.",
        "n_old": int(len(old)),
        "n_third": int(len(third)),
        "n_variants": int(summary["variant"].nunique()),
        "n_candidates": int(len(summary)),
        "selected": selected_row.to_dict(),
        "top10": summary.head(10).to_dict(orient="records"),
    }
    (OUT / "unified_feature_cv_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    show_cols = [
        "variant",
        "model",
        "weight_mode",
        "objective",
        "selection_score",
        "all_threshold",
        "all_accuracy",
        "all_balanced_accuracy",
        "old_accuracy",
        "old_balanced_accuracy",
        "old_tn",
        "old_fp",
        "old_fn",
        "old_tp",
        "third_accuracy",
        "third_balanced_accuracy",
        "third_f1",
        "third_sensitivity_high",
        "third_specificity_low",
        "third_tn",
        "third_fp",
        "third_fn",
        "third_tp",
        "third_auc",
    ]
    print(summary[show_cols].head(20).to_string(index=False), flush=True)
    print(f"[done] out={OUT}", flush=True)


if __name__ == "__main__":
    main()
