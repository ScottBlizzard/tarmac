from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
ADAPT = BASE / "task7_adaptation_runs"
EXT = BASE / "task7_external_runs"
OUT = EXT / "69_dev_locked_no64_dinov3_meta_fusion_20260523"

DEV_FRAME = ADAPT / "53_old_third_no64_meta_stack_full_prob_20260523" / "development_frame_old_plus_third.csv"
DINO_B_DEV = ADAPT / "64_dinov3_vitb16_task7_whole352_lastblock_lowlr_5fold_20260523" / "oof_case_predictions_mean.csv"
DINO_L_DEV = ADAPT / "66_dinov3_vitl16_task7_whole352_lastblock_lowlr_5fold_20260523" / "oof_case_predictions_mean.csv"

EXT_NO64 = EXT / "20_external_thymoma_carcinoma_64style_wpc_20260522" / "external_folder_case_predictions.csv"
DINO_B_EXT = EXT / "68_dinov3_locked_external_eval_20260523" / "64_dinov3_vitb16_task7_whole352_lastblock_lowlr_5fold_20260523_external_locked_predictions.csv"
DINO_L_EXT = EXT / "68_dinov3_locked_external_eval_20260523" / "66_dinov3_vitl16_task7_whole352_lastblock_lowlr_5fold_20260523_external_locked_predictions.csv"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_dict(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity_high": float(tp / max(tp + fn, 1)),
        "specificity_low": float(tn / max(tn + fp, 1)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def choose_threshold(y: np.ndarray, prob: np.ndarray, objective: str = "bacc") -> tuple[float, dict[str, Any]]:
    best_key: tuple[float, ...] | None = None
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    for threshold in np.linspace(0.05, 0.95, 181):
        row = metric_dict(y, prob, float(threshold))
        if objective == "acc":
            key = (row["accuracy"], row["balanced_accuracy"], row["sensitivity_high"], -abs(threshold - 0.5))
        elif objective == "sens80":
            sens_penalty = min(float(row["sensitivity_high"]), 0.80)
            key = (sens_penalty, row["balanced_accuracy"], row["accuracy"], -abs(threshold - 0.5))
        else:
            key = (row["balanced_accuracy"], row["accuracy"], row["sensitivity_high"], -abs(threshold - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = row
    assert best_metrics is not None
    return best_threshold, best_metrics


def add_dino_prob(frame: pd.DataFrame, csv_path: Path, tag: str) -> pd.DataFrame:
    extra = pd.read_csv(csv_path, dtype={"case_id": str})
    cols = ["case_id", "prob_high_risk_group"]
    if not set(cols).issubset(extra.columns):
        raise KeyError(f"{csv_path} missing columns {cols}")
    out = frame.merge(extra[cols].rename(columns={"prob_high_risk_group": f"{tag}_prob"}), on="case_id", how="left")
    if out[f"{tag}_prob"].isna().any():
        missing = out.loc[out[f"{tag}_prob"].isna(), "case_id"].head(20).tolist()
        raise KeyError(f"Missing {tag} probs for cases: {missing}")
    return out


def add_external_dino_prob(frame: pd.DataFrame, csv_path: Path, tag: str) -> pd.DataFrame:
    extra = pd.read_csv(csv_path, dtype={"case_id": str})
    out = frame.merge(
        extra[["case_id", "prob_high_risk_group"]].rename(columns={"prob_high_risk_group": f"{tag}_prob"}),
        on="case_id",
        how="left",
    )
    if out[f"{tag}_prob"].isna().any():
        missing = out.loc[out[f"{tag}_prob"].isna(), "case_id"].head(20).tolist()
        raise KeyError(f"Missing external {tag} probs for cases: {missing}")
    return out


def build_features(frame: pd.DataFrame, external: bool) -> tuple[np.ndarray, list[str]]:
    if external:
        rename = {
            "base_prob_high": "no64_base_prob_high",
            "base_pred_idx": "no64_base_pred_idx",
            "route_score": "no64_route_score",
            "routed_to_reviewer": "no64_routed",
            "reviewer_prob_high": "reviewer_prob_high",
            "reviewer_pred_idx": "reviewer_pred_idx",
            "final_prob_high": "no64_final_prob_high",
            "final_pred_idx": "no64_final_pred_idx",
        }
        source = frame.rename(columns=rename).copy()
    else:
        source = frame.copy()
    cols = [
        "no64_base_prob_high",
        "no64_route_score",
        "no64_routed",
        "reviewer_prob_high",
        "no64_final_prob_high",
        "dino_vitb_prob",
        "dino_vitl_prob",
    ]
    x = source[cols].apply(pd.to_numeric, errors="coerce").fillna(0.5).copy()
    x["dino_mean"] = x[["dino_vitb_prob", "dino_vitl_prob"]].mean(axis=1)
    x["dino_max"] = x[["dino_vitb_prob", "dino_vitl_prob"]].max(axis=1)
    x["dino_min"] = x[["dino_vitb_prob", "dino_vitl_prob"]].min(axis=1)
    x["dino_range"] = x["dino_max"] - x["dino_min"]
    x["dino_mean_minus_no64_final"] = x["dino_mean"] - x["no64_final_prob_high"]
    x["dino_mean_minus_no64_base"] = x["dino_mean"] - x["no64_base_prob_high"]
    x["no64_final_margin"] = np.abs(x["no64_final_prob_high"] - 0.5)
    x["dino_mean_margin"] = np.abs(x["dino_mean"] - 0.5)
    x["agreement_final_dino"] = ((x["no64_final_prob_high"] >= 0.5).astype(int) == (x["dino_mean"] >= 0.5).astype(int)).astype(float)
    x["no64_final_logit"] = np.log(np.clip(x["no64_final_prob_high"], 1e-5, 1 - 1e-5) / np.clip(1 - x["no64_final_prob_high"], 1e-5, 1 - 1e-5))
    x["dino_mean_logit"] = np.log(np.clip(x["dino_mean"], 1e-5, 1 - 1e-5) / np.clip(1 - x["dino_mean"], 1e-5, 1 - 1e-5))
    return x.to_numpy(dtype=np.float32), x.columns.tolist()


def model_grid() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logreg_c003_bal": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.03, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=20260523),
        ),
        "logreg_c01_bal": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=20260523),
        ),
        "logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, solver="liblinear", max_iter=5000, random_state=20260523),
        ),
        "extra_d2": ExtraTreesClassifier(
            n_estimators=800,
            max_depth=2,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=20260523,
            n_jobs=-1,
        ),
        "extra_d3": ExtraTreesClassifier(
            n_estimators=800,
            max_depth=3,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=20260524,
            n_jobs=-1,
        ),
        "rf_d3": RandomForestClassifier(
            n_estimators=800,
            max_depth=3,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=20260525,
            n_jobs=-1,
        ),
        "hgb_l2": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.04,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            random_state=20260523,
        ),
    }
    return models


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
        return model.predict_proba(x)[:, 1].astype(float)
    score = model.decision_function(x)
    return (1 / (1 + np.exp(-score))).astype(float)


def sample_weights(frame: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    weights = np.ones(len(frame), dtype=float)
    y = frame["label_idx"].astype(int).to_numpy()
    if "label" in mode:
        for label in sorted(np.unique(y)):
            mask = y == label
            weights[mask] *= len(y) / (len(np.unique(y)) * mask.sum())
    if "domain" in mode:
        domain = frame["domain"].astype(str).to_numpy()
        for item in sorted(np.unique(domain)):
            mask = domain == item
            weights[mask] *= len(domain) / (len(np.unique(domain)) * mask.sum())
    if "third_up" in mode:
        weights[frame["domain"].eq("third").to_numpy()] *= 1.5
    return weights


def evaluate_by_domain(frame: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y = frame["label_idx"].astype(int).to_numpy()
    row: dict[str, Any] = {}
    all_metrics = metric_dict(y, prob, threshold)
    row.update({f"dev_{k}": v for k, v in all_metrics.items()})
    for domain in ["old", "third"]:
        mask = frame["domain"].eq(domain).to_numpy()
        metrics = metric_dict(y[mask], prob[mask], threshold)
        row.update({f"{domain}_{k}": v for k, v in metrics.items()})
    return row


def run_oof_search(frame: pd.DataFrame, x: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y = frame["label_idx"].astype(int).to_numpy()
    folds = sorted(int(v) for v in frame["fold_id"].dropna().unique().tolist())
    models = model_grid()
    weight_modes = ["none", "label", "domain_label", "domain_label_third_up"]
    rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    for model_name, model in models.items():
        for weight_mode in weight_modes:
            prob = np.zeros(len(frame), dtype=float)
            for fold in folds:
                train_mask = frame["fold_id"].astype(int).to_numpy() != fold
                val_mask = ~train_mask
                weights = sample_weights(frame.loc[train_mask], weight_mode)
                fitted = fit_model(model, x[train_mask], y[train_mask], weights)
                prob[val_mask] = predict_prob(fitted, x[val_mask])
            for objective in ["bacc", "acc", "sens80"]:
                threshold, _ = choose_threshold(y, prob, objective)
                row = evaluate_by_domain(frame, prob, threshold)
                row.update({"model": model_name, "weight_mode": weight_mode, "objective": objective})
                old_guard = min(float(row["old_accuracy"]), float(row["old_balanced_accuracy"]))
                third_quality = 0.50 * float(row["third_balanced_accuracy"]) + 0.25 * float(row["third_accuracy"]) + 0.25 * float(row["third_sensitivity_high"])
                row["selection_score_guard90"] = third_quality if old_guard >= 0.90 else third_quality - (0.90 - old_guard) * 3.0
                row["selection_score_guard92"] = third_quality if old_guard >= 0.92 else third_quality - (0.92 - old_guard) * 3.0
                rows.append(row)
                pred_df = frame[["case_id", "original_case_id", "domain", "third_split", "fold_id", "task_l6_label", "task_l7_label", "label_idx"]].copy()
                pred_df["model"] = model_name
                pred_df["weight_mode"] = weight_mode
                pred_df["objective"] = objective
                pred_df["oof_prob_high"] = prob
                pred_df["threshold"] = threshold
                pred_df["oof_pred_idx"] = (prob >= threshold).astype(int)
                pred_frames.append(pred_df)
    summary = pd.DataFrame(rows).sort_values(["selection_score_guard92", "third_balanced_accuracy", "old_accuracy"], ascending=False)
    selected = summary.iloc[0].to_dict()
    all_preds = pd.concat(pred_frames, ignore_index=True)
    return summary, all_preds, selected


def baseline_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, prob_col in [
        ("no64_final", "no64_final_prob_high"),
        ("dino_vitb", "dino_vitb_prob"),
        ("dino_vitl", "dino_vitl_prob"),
    ]:
        prob = pd.to_numeric(frame[prob_col], errors="coerce").fillna(0.5).to_numpy(float)
        y = frame["label_idx"].astype(int).to_numpy()
        threshold, _ = choose_threshold(y, prob, "bacc")
        row = evaluate_by_domain(frame, prob, threshold)
        row.update({"model": name, "weight_mode": "none", "objective": "dev_bacc_baseline"})
        rows.append(row)
    return pd.DataFrame(rows)


def apply_selected_model(
    selected: dict[str, Any],
    frame: pd.DataFrame,
    x: np.ndarray,
    external_frame: pd.DataFrame,
    x_external: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    y = frame["label_idx"].astype(int).to_numpy()
    model = model_grid()[str(selected["model"])]
    weights = sample_weights(frame, str(selected["weight_mode"]))
    fitted = fit_model(model, x, y, weights)
    threshold = float(selected["dev_threshold"])
    prob_external = predict_prob(fitted, x_external)
    external_pred = external_frame.copy()
    external_pred["fusion_prob_high"] = prob_external
    external_pred["fusion_pred_idx"] = (prob_external >= threshold).astype(int)
    external_pred["fusion_correct"] = (external_pred["fusion_pred_idx"].astype(int) == external_pred["label_idx"].astype(int)).astype(int)

    rows: list[dict[str, Any]] = []
    for subset_name, mask in [
        ("all", np.ones(len(external_pred), dtype=bool)),
        ("strict", external_pred["strict_task7_eval"].astype(int).to_numpy() == 1),
    ]:
        group = external_pred.loc[mask]
        metrics = metric_dict(group["label_idx"].astype(int).to_numpy(), group["fusion_prob_high"].astype(float).to_numpy(), threshold)
        metrics.update({"subset": subset_name, "model": selected["model"], "weight_mode": selected["weight_mode"], "objective": selected["objective"]})
        rows.append(metrics)
    return external_pred, pd.DataFrame(rows), fitted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DEV_FRAME, dtype={"case_id": str})
    frame = add_dino_prob(frame, DINO_B_DEV, "dino_vitb")
    frame = add_dino_prob(frame, DINO_L_DEV, "dino_vitl")
    x, feature_names = build_features(frame, external=False)
    frame.to_csv(OUT / "development_frame_with_dinov3_probs.csv", index=False, encoding="utf-8-sig")

    baseline = baseline_rows(frame)
    baseline.to_csv(OUT / "dev_single_branch_baselines.csv", index=False, encoding="utf-8-sig")
    summary, all_preds, selected = run_oof_search(frame, x)
    summary.to_csv(OUT / "dev_meta_fusion_oof_summary.csv", index=False, encoding="utf-8-sig")
    all_preds.to_csv(OUT / "dev_meta_fusion_all_oof_predictions.csv", index=False, encoding="utf-8-sig")
    selected["dev_threshold"] = float(selected["dev_threshold"])
    selected_pred = all_preds[
        all_preds["model"].eq(selected["model"])
        & all_preds["weight_mode"].eq(selected["weight_mode"])
        & all_preds["objective"].eq(selected["objective"])
    ].copy()
    selected_pred.to_csv(OUT / "selected_dev_oof_predictions.csv", index=False, encoding="utf-8-sig")

    ext_frame = pd.read_csv(EXT_NO64, dtype={"case_id": str})
    ext_frame = add_external_dino_prob(ext_frame, DINO_B_EXT, "dino_vitb")
    ext_frame = add_external_dino_prob(ext_frame, DINO_L_EXT, "dino_vitl")
    x_ext, external_feature_names = build_features(ext_frame, external=True)
    if external_feature_names != feature_names:
        raise RuntimeError("Feature names differ between development and external.")
    external_pred, external_summary, fitted = apply_selected_model(selected, frame, x, ext_frame, x_ext)
    external_pred.to_csv(OUT / "external_locked_fusion_predictions.csv", index=False, encoding="utf-8-sig")
    external_summary.to_csv(OUT / "external_locked_fusion_summary.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"model": fitted, "selected": selected, "feature_names": feature_names}, OUT / "selected_dev_locked_fusion_model.joblib")

    report = {
        "boundary": {
            "development_data": "old batch1+batch2 plus third batch",
            "strict_external_data": str(EXT_NO64),
            "external_used_for_training_or_threshold_selection": False,
            "selection_rule": "select by development OOF selection_score_guard92 only, then apply once to strict external predictions",
        },
        "feature_names": feature_names,
        "selected": selected,
        "top10_dev": summary.head(10).to_dict(orient="records"),
        "external_summary": external_summary.to_dict(orient="records"),
    }
    write_json(OUT / "dev_locked_external_fusion_report.json", report)
    print("[selected]", json.dumps(selected, ensure_ascii=False, indent=2), flush=True)
    print("[external]", external_summary.to_string(index=False), flush=True)
    print(f"[done] outputs saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
