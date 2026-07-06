from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_task7_no64_guarded_adapt_overlay_20260522 import add_route_scores  # noqa: E402
from run_task7_no64_plus_allthird_headonly_rescue_20260523 import load_base_frames  # noqa: E402


BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
ADAPT = BASE / "task7_adaptation_runs"
EXT = BASE / "task7_external_runs"
OUT = EXT / "70_locked_536567_fullprob_external_eval_20260523"

OLD_FEATURE_DIR = BASE / "task7_gross_feature_runs" / "68_roi_whole_plus_crop_embedding_probe_20260521"
THIRD_FEATURE_DIR = EXT / "04_third_batch_whole_plus_crop_64style_20260521"
EXTERNAL_FEATURE_DIR = EXT / "20_external_thymoma_carcinoma_64style_wpc_20260522"
EXTERNAL_BASE_CSV = EXTERNAL_FEATURE_DIR / "external_folder_case_predictions.csv"
SELECTED_OVERLAY_REPORT = ADAPT / "25_no64_adapt_tuned_overlay_20260522" / "adapt_tuned_overlay_report.json"

LOCKED_MODELS = [
    ADAPT / "53_old_third_no64_meta_stack_full_prob_20260523" / "selected_guard92_final_model.joblib",
    ADAPT / "65_old_third_no64_meta_stack_plus_dinov3whole_ft_20260523" / "selected_guard92_final_model.joblib",
    ADAPT / "67_old_third_no64_meta_stack_plus_dinov3vitl_ft_20260523" / "selected_guard92_final_model.joblib",
]
DINO_VITB_EXT = EXT / "68_dinov3_locked_external_eval_20260523" / "64_dinov3_vitb16_task7_whole352_lastblock_lowlr_5fold_20260523_external_locked_predictions.csv"
DINO_VITL_EXT = EXT / "68_dinov3_locked_external_eval_20260523" / "66_dinov3_vitl16_task7_whole352_lastblock_lowlr_5fold_20260523_external_locked_predictions.csv"


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


def logit(prob: pd.Series | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def read_feature_dir(directory: Path, third_style: bool) -> tuple[pd.DataFrame, np.ndarray]:
    if third_style:
        table_name = "third_batch_dino_concat_feature_table.csv"
        npy_name = "third_batch_dino_concat_features.npy"
    else:
        table_name = "case_dino_concat_feature_table.csv"
        npy_name = "case_dino_concat_features.npy"
    table = pd.read_csv(directory / table_name, dtype={"case_id": str})
    feat = np.load(directory / npy_name).astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    return table, feat


def align_features(case_ids: pd.Series, table: pd.DataFrame, feat: np.ndarray) -> np.ndarray:
    order = case_ids.astype(str).to_frame("case_id").merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if order["feature_idx"].isna().any():
        missing = order.loc[order["feature_idx"].isna(), "case_id"].head(20).tolist()
        raise KeyError(f"Missing feature rows: {missing}")
    return feat[order["feature_idx"].astype(int).to_numpy()].astype(np.float32)


def fit_logreg(c: float, x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=20260523),
    )
    if weights is None:
        model.fit(x, y)
    else:
        model.fit(x, y, logisticregression__sample_weight=weights)
    return model


def build_external_candidate_probs(external: pd.DataFrame) -> pd.DataFrame:
    old, adapt, _, _ = load_base_frames()
    old_table, old_feat = read_feature_dir(OLD_FEATURE_DIR, third_style=False)
    third_table, third_feat = read_feature_dir(THIRD_FEATURE_DIR, third_style=True)
    ext_table, ext_feat = read_feature_dir(EXTERNAL_FEATURE_DIR, third_style=True)

    x_old = align_features(old["case_id"], old_table, old_feat)
    y_old = old["label_idx"].astype(int).to_numpy()
    x_adapt = align_features(adapt["case_id"], third_table, third_feat)
    y_adapt = adapt["label_idx"].astype(int).to_numpy()
    x_ext = align_features(external["case_id"], ext_table, ext_feat)

    out = pd.DataFrame({"case_id": external["case_id"].astype(str).to_numpy()})
    c_values = [0.0003, 0.001, 0.003, 0.01]
    for c in c_values:
        model = fit_logreg(c, x_old, y_old)
        out[f"oldonly_c{c}"] = model.predict_proba(x_ext)[:, 1].astype(float)

    x_train = np.vstack([x_old, x_adapt])
    y_train = np.concatenate([y_old, y_adapt])
    for r in [1, 2, 4, 8]:
        weights = np.concatenate([np.ones(len(y_old), dtype=float), np.ones(len(y_adapt), dtype=float) * float(r)])
        for c in c_values:
            model = fit_logreg(c, x_train, y_train, weights)
            out[f"adapt_r{r}_c{c}"] = model.predict_proba(x_ext)[:, 1].astype(float)
    return out


def apply_selected_overlay(external: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    report = json.loads(SELECTED_OVERLAY_REPORT.read_text(encoding="utf-8"))
    selected = report["selected_by_old_plus_adapt"]
    adapt_name = str(selected["adapt_candidate"])
    adapt_t = float(selected["adapt_threshold"])
    route_name = str(selected["route_name"])
    route_threshold = float(selected["route_threshold"])

    frame = external.copy()
    frame["base_pred_for_overlay"] = frame["final_pred_idx"].astype(int)
    route = add_route_scores(frame, candidate_cols, "final_prob_high", adapt_name, adapt_t)[route_name]
    routed = route >= route_threshold
    final_prob = frame["final_prob_high"].astype(float).to_numpy().copy()
    final_pred = frame["final_pred_idx"].astype(int).to_numpy().copy()
    adapt_prob = frame[adapt_name].astype(float).to_numpy()
    final_prob[routed] = adapt_prob[routed]
    final_pred[routed] = (adapt_prob[routed] >= adapt_t).astype(int)
    frame["selected_base_prob"] = final_prob
    frame["selected_base_pred"] = final_pred
    frame["selected_base_routed"] = routed.astype(int)
    frame["selected_overlay_route_score"] = route
    frame["selected_overlay_adapt_candidate"] = adapt_name
    frame["selected_overlay_adapt_threshold"] = adapt_t
    frame["selected_overlay_route_name"] = route_name
    frame["selected_overlay_route_threshold"] = route_threshold
    return frame


def add_dinov3_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for tag, path in [("dinov3whole_ft", DINO_VITB_EXT), ("dinov3vitl_ft", DINO_VITL_EXT)]:
        d = pd.read_csv(path, dtype={"case_id": str})
        out = out.merge(
            d[["case_id", "prob_high_risk_group"]].rename(columns={"prob_high_risk_group": f"{tag}_prob"}),
            on="case_id",
            how="left",
        )
        if out[f"{tag}_prob"].isna().any():
            missing = out.loc[out[f"{tag}_prob"].isna(), "case_id"].head(20).tolist()
            raise KeyError(f"Missing {tag} external probs: {missing}")
        p = out[f"{tag}_prob"].astype(float)
        out[f"{tag}_margin"] = np.abs(p - 0.5)
        out[f"{tag}_logit"] = logit(p)
        out[f"{tag}_agree_base"] = ((p >= 0.5).astype(int) == out["selected_base_pred"].astype(int)).astype(float)
        out[f"{tag}_minus_base"] = p - out["selected_base_prob"].astype(float)
    return out


def build_prob_feature_frame(frame: pd.DataFrame, prob_columns: list[str]) -> pd.DataFrame:
    candidate_cols = [c for c in frame.columns if c.startswith("oldonly_") or c.startswith("adapt_r")]
    out = frame[["selected_base_prob", "selected_base_pred", "selected_base_routed"] + candidate_cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["base_margin"] = np.abs(out["selected_base_prob"].astype(float) - 0.5)
    out["base_logit"] = logit(out["selected_base_prob"])
    adapt_cols = [c for c in candidate_cols if c.startswith("adapt_")]
    old_cols = [c for c in candidate_cols if c.startswith("oldonly_")]
    out["adapt_mean"] = out[adapt_cols].mean(axis=1)
    out["adapt_std"] = out[adapt_cols].std(axis=1).fillna(0.0)
    out["oldonly_mean"] = out[old_cols].mean(axis=1)
    out["candidate_max"] = out[candidate_cols].max(axis=1)
    out["candidate_min"] = out[candidate_cols].min(axis=1)
    out["candidate_range"] = out["candidate_max"] - out["candidate_min"]
    out["candidate_mean_minus_base"] = out[candidate_cols].mean(axis=1) - out["selected_base_prob"]

    for tag in ["dinov3whole_ft", "dinov3vitl_ft"]:
        tag_cols = [f"{tag}_prob", f"{tag}_margin", f"{tag}_logit", f"{tag}_agree_base", f"{tag}_minus_base"]
        if any(c in prob_columns for c in tag_cols):
            for col in tag_cols:
                out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    # Locked deployment convention: unseen external material is represented as third-holdout-like, not as old.
    out["domain_old"] = 0.0
    out["domain_third"] = 1.0
    out["third_split_adapt72"] = 0.0
    out["third_split_holdout234"] = 1.0
    out["third_split_old"] = 0.0
    return out.reindex(columns=prob_columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def apply_locked_model(model_path: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obj = joblib.load(model_path)
    prob_columns = list(obj["prob_columns"])
    feature_frame = build_prob_feature_frame(frame, prob_columns)
    prob = obj["model"].predict_proba(feature_frame.to_numpy(dtype=np.float32))[:, 1].astype(float)
    threshold = float(obj["threshold"])
    pred = (prob >= threshold).astype(int)
    tag = model_path.parent.name
    pred_frame = frame.copy()
    pred_frame["locked_model_tag"] = tag
    pred_frame["locked_prob_high"] = prob
    pred_frame["locked_threshold"] = threshold
    pred_frame["locked_pred_idx"] = pred
    pred_frame["locked_correct"] = (pred == pred_frame["label_idx"].astype(int).to_numpy()).astype(int)
    pred_frame["locked_feature_columns"] = len(prob_columns)

    rows: list[dict[str, Any]] = []
    for subset, mask in [
        ("all", np.ones(len(pred_frame), dtype=bool)),
        ("strict", pred_frame["strict_task7_eval"].astype(int).to_numpy() == 1),
    ]:
        g = pred_frame.loc[mask]
        metrics = metric_dict(g["label_idx"].astype(int).to_numpy(), g["locked_prob_high"].astype(float).to_numpy(), threshold)
        metrics.update(
            {
                "subset": subset,
                "locked_model_tag": tag,
                "threshold": threshold,
                "feature_set": str(obj.get("feature_set", "")),
                "selected_dev_old_accuracy": float(obj["selected"].get("old_accuracy", np.nan)),
                "selected_dev_third_accuracy": float(obj["selected"].get("third_accuracy", np.nan)),
                "selected_dev_third_balanced_accuracy": float(obj["selected"].get("third_balanced_accuracy", np.nan)),
            }
        )
        rows.append(metrics)
    return pred_frame, pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    external = pd.read_csv(EXTERNAL_BASE_CSV, dtype={"case_id": str, "original_case_id": str})
    candidate_probs = build_external_candidate_probs(external)
    candidate_cols = [c for c in candidate_probs.columns if c != "case_id"]
    external = external.merge(candidate_probs, on="case_id", how="left")
    external = apply_selected_overlay(external, candidate_cols)
    external = add_dinov3_features(external)
    external.to_csv(OUT / "external_fullprob_feature_frame.csv", index=False, encoding="utf-8-sig")

    all_summaries: list[pd.DataFrame] = []
    for model_path in LOCKED_MODELS:
        pred_frame, summary = apply_locked_model(model_path, external)
        pred_frame.to_csv(OUT / f"{model_path.parent.name}_external_predictions.csv", index=False, encoding="utf-8-sig")
        all_summaries.append(summary)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_df.to_csv(OUT / "locked_536567_external_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "boundary": {
            "external_training_or_threshold_selection": False,
            "candidate_probabilities": "reconstructed from old + third-adapt development feature data, then applied to external images",
            "locked_models": [str(p) for p in LOCKED_MODELS],
            "external_domain_encoding": "domain_third=1, third_split_holdout234=1",
        },
        "summary": summary_df.to_dict(orient="records"),
    }
    write_json(OUT / "locked_536567_external_report.json", report)
    print(summary_df.to_string(index=False), flush=True)
    print(f"[done] outputs saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
