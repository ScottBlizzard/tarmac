from __future__ import annotations

import json
import sys
from pathlib import Path
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_task7_no64_plus_allthird_headonly_rescue_20260523 import load_base_frames  # noqa: E402


BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
OUT = BASE / "task7_adaptation_runs" / os.environ.get("TASK7_META_OUT_NAME", "52_old_third_no64_meta_stack_20260523")

OLD_WPC = BASE / "task7_gross_feature_runs" / "68_roi_whole_plus_crop_embedding_probe_20260521"
THIRD_WPC = BASE / "task7_external_runs" / "04_third_batch_whole_plus_crop_64style_20260521"
DINO_OLD_DIR = Path(os.environ.get("TASK7_META_DINO_OLD_DIR", str(OLD_WPC)))
DINO_THIRD_DIR = Path(os.environ.get("TASK7_META_DINO_THIRD_DIR", str(THIRD_WPC)))
DINO_TAG = os.environ.get("TASK7_META_DINO_TAG", "dino")
EXTRA_OOF_CSV = os.environ.get("TASK7_META_EXTRA_OOF_CSV", "").strip()
EXTRA_OOF_TAG = os.environ.get("TASK7_META_EXTRA_OOF_TAG", "extra_oof").strip()
EXTRA_OOF_CSVS_RAW = os.environ.get("TASK7_META_EXTRA_OOF_CSVS", "").strip()
EXTRA_OOF_TAGS_RAW = os.environ.get("TASK7_META_EXTRA_OOF_TAGS", "").strip()
NO_DOMAIN_FEATURES = os.environ.get("TASK7_META_NO_DOMAIN_FEATURES", "0") == "1"
NO_DOMAIN_WEIGHTS = os.environ.get("TASK7_META_NO_DOMAIN_WEIGHTS", "0") == "1"


def configured_extra_oofs() -> list[tuple[str, str]]:
    if EXTRA_OOF_CSVS_RAW:
        csvs = [item.strip() for item in EXTRA_OOF_CSVS_RAW.split(";") if item.strip()]
        tags = [item.strip() for item in EXTRA_OOF_TAGS_RAW.split(";") if item.strip()]
        if tags and len(tags) != len(csvs):
            raise ValueError("TASK7_META_EXTRA_OOF_TAGS must match TASK7_META_EXTRA_OOF_CSVS length.")
        if not tags:
            tags = [f"extra_oof_{idx + 1}" for idx in range(len(csvs))]
        return list(zip(csvs, tags))
    if EXTRA_OOF_CSV:
        return [(EXTRA_OOF_CSV, EXTRA_OOF_TAG)]
    return []


def metric_dict(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
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
    out["auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")
    return out


def choose_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> tuple[float, dict[str, Any]]:
    best_t = 0.5
    best_key: tuple[float, ...] | None = None
    best_row: dict[str, Any] | None = None
    for t in np.linspace(0.05, 0.95, 181):
        row = metric_dict(y, prob, float(t))
        if objective == "old_third_balanced":
            key = (float(row["balanced_accuracy"]), float(row["accuracy"]), float(row["f1"]), -abs(float(t) - 0.5))
        elif objective == "accuracy":
            key = (float(row["accuracy"]), float(row["balanced_accuracy"]), float(row["f1"]), -abs(float(t) - 0.5))
        elif objective == "high_sensitivity":
            sens = float(row["sensitivity_high"])
            key = (min(sens, 0.80), float(row["balanced_accuracy"]), float(row["accuracy"]), -abs(float(t) - 0.5))
        else:
            raise ValueError(objective)
        if best_key is None or key > best_key:
            best_key = key
            best_t = float(t)
            best_row = row
    assert best_row is not None
    return best_t, best_row


def read_feature_dir(directory: Path, third: bool) -> tuple[pd.DataFrame, np.ndarray]:
    table_name = "third_batch_dino_concat_feature_table.csv" if third else "case_dino_concat_feature_table.csv"
    npy_name = "third_batch_dino_concat_features.npy" if third else "case_dino_concat_features.npy"
    table = pd.read_csv(directory / table_name, dtype={"case_id": str})
    features = np.load(directory / npy_name).astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    return table, features


def align_features(case_ids: pd.Series, table: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    idx = case_ids.astype(str).to_frame("case_id").merge(table[["case_id", "feature_idx"]], on="case_id", how="left")["feature_idx"]
    if idx.isna().any():
        missing = case_ids[idx.isna()].head(20).tolist()
        raise KeyError(f"Missing feature rows: {missing}")
    return features[idx.astype(int).to_numpy()].astype(np.float32)


def add_third_folds(third: pd.DataFrame) -> pd.DataFrame:
    third = third.copy()
    y = third["label_idx"].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260523)
    third["fold_id"] = -1
    for fold, (_, idx) in enumerate(skf.split(third, y), start=1):
        third.iloc[idx, third.columns.get_loc("fold_id")] = fold
    return third


def make_frame() -> pd.DataFrame:
    old, adapt, hold, _ = load_base_frames()
    old = old.copy()
    adapt = adapt.copy()
    hold = hold.copy()
    old["domain"] = "old"
    old["third_split"] = "old"
    adapt["domain"] = "third"
    adapt["third_split"] = "adapt72"
    hold["domain"] = "third"
    hold["third_split"] = "holdout234"
    third = add_third_folds(pd.concat([adapt, hold], ignore_index=True, sort=False))
    frame = pd.concat([old, third], ignore_index=True, sort=False)
    frame["label_idx"] = frame["label_idx"].astype(int)
    frame["fold_id"] = frame["fold_id"].astype(int)
    return frame


def base_probability_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    candidate_cols = [
        c
        for c in frame.columns
        if c.startswith("oldonly_") or c.startswith("adapt_r")
    ]
    out = frame[["selected_base_prob", "selected_base_pred", "selected_base_routed"] + candidate_cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["base_margin"] = np.abs(out["selected_base_prob"].astype(float) - 0.5)
    out["base_logit"] = np.log(np.clip(out["selected_base_prob"], 1e-5, 1 - 1e-5) / np.clip(1 - out["selected_base_prob"], 1e-5, 1 - 1e-5))
    adapt_cols = [c for c in candidate_cols if c.startswith("adapt_")]
    old_cols = [c for c in candidate_cols if c.startswith("oldonly_")]
    out["adapt_mean"] = out[adapt_cols].mean(axis=1)
    out["adapt_std"] = out[adapt_cols].std(axis=1).fillna(0.0)
    out["oldonly_mean"] = out[old_cols].mean(axis=1)
    out["candidate_max"] = out[candidate_cols].max(axis=1)
    out["candidate_min"] = out[candidate_cols].min(axis=1)
    out["candidate_range"] = out["candidate_max"] - out["candidate_min"]
    out["candidate_mean_minus_base"] = out[candidate_cols].mean(axis=1) - out["selected_base_prob"]
    for extra_csv, extra_tag in configured_extra_oofs():
        extra = pd.read_csv(extra_csv, dtype={"case_id": str})
        if "prob_high_risk_group" in extra.columns:
            prob_col = "prob_high_risk_group"
        elif "prob_high" in extra.columns:
            prob_col = "prob_high"
        elif "oof_prob_high" in extra.columns:
            prob_col = "oof_prob_high"
        else:
            raise KeyError(f"Extra OOF file must contain prob_high_risk_group, prob_high, or oof_prob_high: {extra_csv}")
        aligned = frame[["case_id"]].merge(extra[["case_id", prob_col]], on="case_id", how="left")[prob_col]
        if aligned.isna().any():
            missing = frame.loc[aligned.isna(), "case_id"].head(20).tolist()
            raise KeyError(f"Missing extra OOF probabilities for cases: {missing}")
        tag = extra_tag.replace("-", "_")
        extra_prob = pd.to_numeric(aligned, errors="coerce").fillna(0.5).astype(float)
        out[f"{tag}_prob"] = extra_prob
        out[f"{tag}_margin"] = np.abs(extra_prob - 0.5)
        out[f"{tag}_logit"] = np.log(np.clip(extra_prob, 1e-5, 1 - 1e-5) / np.clip(1 - extra_prob, 1e-5, 1 - 1e-5))
        out[f"{tag}_agree_base"] = ((extra_prob >= 0.5).astype(int) == frame["selected_base_pred"].astype(int)).astype(float)
        out[f"{tag}_minus_base"] = extra_prob - frame["selected_base_prob"].astype(float)
    if not NO_DOMAIN_FEATURES:
        out = pd.concat([out, pd.get_dummies(frame[["domain", "third_split"]].fillna(""), dtype=float)], axis=1)
    return out, candidate_cols


def dino_features(frame: pd.DataFrame) -> np.ndarray:
    old_table, old_feat = read_feature_dir(DINO_OLD_DIR, third=False)
    third_table, third_feat = read_feature_dir(DINO_THIRD_DIR, third=True)
    old_mask = frame["domain"].eq("old").to_numpy()
    third_mask = ~old_mask
    x = np.zeros((len(frame), old_feat.shape[1]), dtype=np.float32)
    x[old_mask] = align_features(frame.loc[old_mask, "case_id"], old_table, old_feat)
    x[third_mask] = align_features(frame.loc[third_mask, "case_id"], third_table, third_feat)
    return x


def compress_dino(x: np.ndarray, n_components: int) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=20260523)
    z = pca.fit_transform(x_scaled).astype(np.float32)
    return z, {"scaler": scaler, "pca": pca, "explained": float(pca.explained_variance_ratio_.sum())}


def model_grid() -> dict[str, Any]:
    models: dict[str, Any] = {}
    fast = os.environ.get("TASK7_META_FULL", "0") != "1"
    c_values = [0.003, 0.01, 0.03] if fast else [0.01, 0.03, 0.1, 0.3]
    for c in c_values:
        models[f"logreg_bal_c{c:g}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=20260523),
        )
        if not fast:
            models[f"logreg_plain_c{c:g}"] = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, solver="liblinear", max_iter=5000, random_state=20260523),
            )
    if fast:
        return models
    depth_values = [2, 3] if fast else [2, 3, 4]
    for depth in depth_values:
        models[f"extra_d{depth}"] = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=depth,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=20260523 + depth,
            n_jobs=-1,
        )
        models[f"rf_d{depth}"] = RandomForestClassifier(
            n_estimators=600,
            max_depth=depth,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=20260623 + depth,
            n_jobs=-1,
        )
    lr_values = [0.05] if fast else [0.03, 0.05]
    for lr in lr_values:
        models[f"hgb_l2_lr{str(lr).replace('.', '')}"] = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=lr,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            random_state=20260523,
        )
    return models


def sample_weights(frame: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    y = frame["label_idx"].astype(int).to_numpy()
    weights = np.ones(len(frame), dtype=float)
    if "label" in mode:
        for label in sorted(np.unique(y)):
            mask = y == label
            weights[mask] *= len(y) / (len(np.unique(y)) * mask.sum())
    if "domain" in mode and not NO_DOMAIN_WEIGHTS:
        domains = frame["domain"].astype(str).to_numpy()
        for domain in sorted(np.unique(domains)):
            mask = domains == domain
            weights[mask] *= len(domains) / (len(np.unique(domains)) * mask.sum())
    if "third_up" in mode:
        weights[frame["domain"].eq("third").to_numpy()] *= 1.6
    return weights


def fit_model(model: Any, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> Any:
    fitted = clone(model)
    if weights is None:
        fitted.fit(x, y)
    elif hasattr(fitted, "steps"):
        fitted.fit(x, y, **{f"{fitted.steps[-1][0]}__sample_weight": weights})
    else:
        fitted.fit(x, y, sample_weight=weights)
    return fitted


def predict_prob(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    score = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-score))


def evaluate_group(frame: pd.DataFrame, prob: np.ndarray, threshold: float, group_mask: np.ndarray) -> dict[str, Any]:
    y = frame.loc[group_mask, "label_idx"].astype(int).to_numpy()
    return metric_dict(y, prob[group_mask], threshold)


def run_cv(frame: pd.DataFrame, feature_sets: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame["label_idx"].astype(int).to_numpy()
    folds = frame["fold_id"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    models = model_grid()
    weight_modes = ["none", "domain_label"] if os.environ.get("TASK7_META_FULL", "0") != "1" else [
        "none",
        "label",
        "domain_label",
        "domain_label_third_up",
    ]
    for feature_name, x in feature_sets.items():
        print(f"[feature] {feature_name} x={x.shape}", flush=True)
        for model_name, model in models.items():
            for weight_mode in weight_modes:
                oof = np.zeros(len(frame), dtype=float)
                for fold in sorted(np.unique(folds)):
                    tr = folds != fold
                    va = folds == fold
                    weights = sample_weights(frame.loc[tr].reset_index(drop=True), weight_mode)
                    fitted = fit_model(model, x[tr], y[tr], weights)
                    oof[va] = predict_prob(fitted, x[va])
                for objective in ["old_third_balanced", "accuracy", "high_sensitivity"]:
                    threshold, all_row = choose_threshold(y, oof, objective)
                    old_mask = frame["domain"].eq("old").to_numpy()
                    third_mask = frame["domain"].eq("third").to_numpy()
                    adapt_mask = frame["third_split"].eq("adapt72").to_numpy()
                    hold_mask = frame["third_split"].eq("holdout234").to_numpy()
                    old_row = evaluate_group(frame, oof, threshold, old_mask)
                    third_row = evaluate_group(frame, oof, threshold, third_mask)
                    adapt_row = evaluate_group(frame, oof, threshold, adapt_mask)
                    hold_row = evaluate_group(frame, oof, threshold, hold_mask)
                    row = {
                        "feature_set": feature_name,
                        "model": model_name,
                        "weight_mode": weight_mode,
                        "objective": objective,
                        **{f"all_{k}": v for k, v in all_row.items()},
                        **{f"old_{k}": v for k, v in old_row.items()},
                        **{f"third_{k}": v for k, v in third_row.items()},
                        **{f"adapt_{k}": v for k, v in adapt_row.items()},
                        **{f"holdout_{k}": v for k, v in hold_row.items()},
                    }
                    row["old_guard_092"] = bool(row["old_accuracy"] >= 0.92 and row["old_balanced_accuracy"] >= 0.92)
                    row["old_guard_090"] = bool(row["old_accuracy"] >= 0.90 and row["old_balanced_accuracy"] >= 0.90)
                    row["selection_score"] = (
                        0.36 * float(row["old_balanced_accuracy"])
                        + 0.34 * float(row["third_balanced_accuracy"])
                        + 0.15 * float(row["old_accuracy"])
                        + 0.15 * float(row["third_accuracy"])
                        - 1.5 * max(0.0, 0.92 - float(row["old_accuracy"]))
                        - 1.5 * max(0.0, 0.92 - float(row["old_balanced_accuracy"]))
                    )
                    rows.append(row)
                    if len(rows) % 6 == 0:
                        pd.DataFrame(rows).to_csv(OUT / "no64_meta_stack_cv_summary.partial.csv", index=False, encoding="utf-8-sig")

                pred = frame[
                    ["case_id", "original_case_id", "domain", "third_split", "fold_id", "task_l6_label", "task_l7_label", "label_idx"]
                ].copy()
                pred["feature_set"] = feature_name
                pred["model"] = model_name
                pred["weight_mode"] = weight_mode
                pred["oof_prob_high"] = oof
                pred_rows.append(pred)
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True)


def save_selected_predictions(frame: pd.DataFrame, predictions: pd.DataFrame, selected: pd.Series, name: str) -> None:
    pred = predictions[
        (predictions["feature_set"].eq(selected["feature_set"]))
        & (predictions["model"].eq(selected["model"]))
        & (predictions["weight_mode"].eq(selected["weight_mode"]))
    ].copy()
    threshold = float(selected["all_threshold"])
    pred["threshold"] = threshold
    pred["oof_pred_idx"] = (pred["oof_prob_high"].astype(float) >= threshold).astype(int)
    pred["oof_correct"] = (pred["oof_pred_idx"] == pred["label_idx"].astype(int)).astype(int)
    pred = pred.merge(
        frame[["case_id", "selected_base_prob", "selected_base_pred", "selected_base_correct"]],
        on="case_id",
        how="left",
    )
    pred.to_csv(OUT / f"{name}_oof_predictions.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = make_frame()
    frame.to_csv(OUT / "development_frame_old_plus_third.csv", index=False, encoding="utf-8-sig")

    prob_df, _ = base_probability_features(frame)
    x_prob = prob_df.to_numpy(dtype=np.float32)
    feature_sets = {"prob_only": x_prob}
    prep64: dict[str, Any] = {"explained": None}
    prep128: dict[str, Any] = {"explained": None}
    if os.environ.get("TASK7_META_INCLUDE_DINO", "0") == "1":
        print("[load] DINO feature cache", flush=True)
        dino = dino_features(frame)
        print("[pca] DINO 64/128", flush=True)
        dino64, prep64 = compress_dino(dino, n_components=64)
        dino128, prep128 = compress_dino(dino, n_components=128)
        feature_sets.update(
            {
                f"prob_{DINO_TAG}64": np.hstack([x_prob, dino64]).astype(np.float32),
                f"prob_{DINO_TAG}128": np.hstack([x_prob, dino128]).astype(np.float32),
            }
        )

    summary, predictions = run_cv(frame, feature_sets)
    summary = summary.sort_values(["selection_score", "old_guard_092", "third_balanced_accuracy", "third_accuracy"], ascending=False).reset_index(drop=True)
    summary.to_csv(OUT / "no64_meta_stack_cv_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "no64_meta_stack_all_oof_probabilities.csv", index=False, encoding="utf-8-sig")

    guarded92 = summary[summary["old_guard_092"]].copy()
    guarded90 = summary[summary["old_guard_090"]].copy()
    selected92 = (guarded92 if not guarded92.empty else summary).sort_values(
        ["selection_score", "third_balanced_accuracy", "third_accuracy"], ascending=False
    ).iloc[0]
    selected90 = (guarded90 if not guarded90.empty else summary).sort_values(
        ["selection_score", "third_balanced_accuracy", "third_accuracy"], ascending=False
    ).iloc[0]
    top_third = (guarded92 if not guarded92.empty else summary).sort_values(
        ["third_accuracy", "third_balanced_accuracy", "old_accuracy"], ascending=False
    ).iloc[0]

    for label, row in [("selected_guard92", selected92), ("selected_guard90", selected90), ("top_third_acc_guard92", top_third)]:
        save_selected_predictions(frame, predictions, row, label)

    # Store a final development model artifact for later locked-external scoring.
    selected = selected92
    selected_feature = str(selected["feature_set"])
    x_all = feature_sets[selected_feature]
    y_all = frame["label_idx"].astype(int).to_numpy()
    model = model_grid()[str(selected["model"])]
    weights = sample_weights(frame.reset_index(drop=True), str(selected["weight_mode"]))
    final_model = fit_model(model, x_all, y_all, weights)
    joblib.dump(
        {
            "model": final_model,
            "threshold": float(selected["all_threshold"]),
            "selected": selected.to_dict(),
            "prob_columns": prob_df.columns.tolist(),
            "dino_prep64": prep64,
            "dino_prep128": prep128,
            "feature_set": selected_feature,
            "data_boundary": "trained only on old batch1+batch2 and third-batch development cohort; locked external folder not used",
        },
        OUT / "selected_guard92_final_model.joblib",
    )

    report = {
        "experiment": "Task7 No.64-base old+third meta stack CV",
        "data_boundary": "Old data + third batch are development data. Locked external folder is not used.",
        "extra_oof_csv": EXTRA_OOF_CSV,
        "extra_oof_tag": EXTRA_OOF_TAG,
        "n_old": int(frame["domain"].eq("old").sum()),
        "n_third": int(frame["domain"].eq("third").sum()),
        "base_selected_metrics": {
            "old": metric_dict(
                frame.loc[frame["domain"].eq("old"), "label_idx"].to_numpy(int),
                frame.loc[frame["domain"].eq("old"), "selected_base_prob"].to_numpy(float),
                0.5,
            ),
            "third": metric_dict(
                frame.loc[frame["domain"].eq("third"), "label_idx"].to_numpy(int),
                frame.loc[frame["domain"].eq("third"), "selected_base_prob"].to_numpy(float),
                0.5,
            ),
        },
        "extra_oofs": [{"csv": csv, "tag": tag} for csv, tag in configured_extra_oofs()],
        "no_domain_features": NO_DOMAIN_FEATURES,
        "no_domain_weights": NO_DOMAIN_WEIGHTS,
        "dino_pca64_explained": prep64["explained"],
        "dino_pca128_explained": prep128["explained"],
        "selected_guard92": selected92.to_dict(),
        "selected_guard90": selected90.to_dict(),
        "top_third_acc_guard92": top_third.to_dict(),
        "top20": summary.head(20).to_dict(orient="records"),
    }
    (OUT / "no64_meta_stack_cv_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    show_cols = [
        "feature_set",
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
    ]
    print(summary[show_cols].head(30).to_string(index=False), flush=True)
    print(f"[done] out={OUT}", flush=True)


if __name__ == "__main__":
    main()

