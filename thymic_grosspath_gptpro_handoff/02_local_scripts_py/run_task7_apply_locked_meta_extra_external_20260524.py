from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a locked Task7 meta model with one extra external probability source.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-external-csv", required=True)
    parser.add_argument("--extra-external-csv", required=True)
    parser.add_argument("--extra-tag", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def logit(prob: pd.Series | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


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


def add_extra_prob(frame: pd.DataFrame, extra_csv: Path, tag: str) -> pd.DataFrame:
    extra = pd.read_csv(extra_csv, dtype={"case_id": str})
    prob_col = "prob_high_risk_group" if "prob_high_risk_group" in extra.columns else "prob_high"
    if prob_col not in extra.columns:
        raise KeyError(f"Extra external CSV needs prob_high_risk_group or prob_high: {extra_csv}")
    out = frame.merge(extra[["case_id", prob_col]].rename(columns={prob_col: f"{tag}_prob"}), on="case_id", how="left")
    if out[f"{tag}_prob"].isna().any():
        missing = out.loc[out[f"{tag}_prob"].isna(), "case_id"].head(20).tolist()
        raise KeyError(f"Missing extra external probabilities: {missing}")
    prob = out[f"{tag}_prob"].astype(float)
    out[f"{tag}_margin"] = np.abs(prob - 0.5)
    out[f"{tag}_logit"] = logit(prob)
    out[f"{tag}_agree_base"] = ((prob >= 0.5).astype(int) == out["selected_base_pred"].astype(int)).astype(float)
    out[f"{tag}_minus_base"] = prob - out["selected_base_prob"].astype(float)
    return out


def build_feature_frame(frame: pd.DataFrame, prob_columns: list[str]) -> pd.DataFrame:
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

    for col in prob_columns:
        if col not in out.columns and col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    # External material is held out; for the trained meta model, represent it as third-holdout-like.
    out["domain_old"] = 0.0
    out["domain_third"] = 1.0
    out["third_split_adapt72"] = 0.0
    out["third_split_holdout234"] = 1.0
    out["third_split_old"] = 0.0
    return out.reindex(columns=prob_columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = joblib.load(args.model_path)
    prob_columns = list(obj["prob_columns"])
    threshold = float(obj["threshold"])
    frame = pd.read_csv(args.base_external_csv, dtype={"case_id": str})
    frame = add_extra_prob(frame, Path(args.extra_external_csv), args.extra_tag.replace("-", "_"))
    feature_frame = build_feature_frame(frame, prob_columns)
    prob = obj["model"].predict_proba(feature_frame.to_numpy(dtype=np.float32))[:, 1].astype(float)
    pred = (prob >= threshold).astype(int)
    frame["locked_extra_prob_high"] = prob
    frame["locked_extra_threshold"] = threshold
    frame["locked_extra_pred_idx"] = pred
    frame["locked_extra_correct"] = (pred == frame["label_idx"].astype(int).to_numpy()).astype(int)

    rows: list[dict[str, Any]] = []
    for subset, mask in [
        ("all", np.ones(len(frame), dtype=bool)),
        ("strict", frame["strict_task7_eval"].astype(int).to_numpy() == 1),
    ]:
        group = frame.loc[mask]
        row = {"subset": subset, "model_path": str(args.model_path), "extra_tag": args.extra_tag}
        row.update(metric_dict(group["label_idx"].astype(int).to_numpy(), group["locked_extra_prob_high"].astype(float).to_numpy(), threshold))
        rows.append(row)

    frame.to_csv(out_dir / "external_locked_extra_predictions.csv", index=False, encoding="utf-8-sig")
    feature_frame.to_csv(out_dir / "external_locked_extra_feature_frame.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(out_dir / "external_locked_extra_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "external_locked_extra_report.json").write_text(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "extra_external_csv": str(args.extra_external_csv),
                "extra_tag": args.extra_tag,
                "threshold": threshold,
                "prob_columns": prob_columns,
                "boundary": "Model and extra feature choice were selected on old+third development data; external set is used only for locked audit.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
