from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "batch1_batch2_task567_20260514"
ADAPT = BASE / "task7_adaptation_runs"
EXT = BASE / "task7_external_runs"
OUT = EXT / "71_locked67_dinov3_rescue_rule_external_20260523"

DEV67 = ADAPT / "67_old_third_no64_meta_stack_plus_dinov3vitl_ft_20260523" / "selected_guard92_oof_predictions.csv"
DINO_B_DEV = ADAPT / "64_dinov3_vitb16_task7_whole352_lastblock_lowlr_5fold_20260523" / "oof_case_predictions_mean.csv"
DINO_L_DEV = ADAPT / "66_dinov3_vitl16_task7_whole352_lastblock_lowlr_5fold_20260523" / "oof_case_predictions_mean.csv"

EXT67 = EXT / "70_locked_536567_fullprob_external_eval_20260523" / "67_old_third_no64_meta_stack_plus_dinov3vitl_ft_20260523_external_predictions.csv"
DINO_B_EXT = EXT / "68_dinov3_locked_external_eval_20260523" / "64_dinov3_vitb16_task7_whole352_lastblock_lowlr_5fold_20260523_external_locked_predictions.csv"
DINO_L_EXT = EXT / "68_dinov3_locked_external_eval_20260523" / "66_dinov3_vitl16_task7_whole352_lastblock_lowlr_5fold_20260523_external_locked_predictions.csv"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "n": int(len(y)),
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


def add_dino(df: pd.DataFrame, vitb_path: Path, vitl_path: Path) -> pd.DataFrame:
    out = df.copy()
    for tag, path in [("dinov3_vitb", vitb_path), ("dinov3_vitl", vitl_path)]:
        d = pd.read_csv(path, dtype={"case_id": str})
        out = out.merge(
            d[["case_id", "prob_high_risk_group"]].rename(columns={"prob_high_risk_group": f"{tag}_prob"}),
            on="case_id",
            how="left",
        )
        if out[f"{tag}_prob"].isna().any():
            missing = out.loc[out[f"{tag}_prob"].isna(), "case_id"].head(20).tolist()
            raise KeyError(f"Missing {tag}: {missing}")
    out["dinov3_mean_prob"] = out[["dinov3_vitb_prob", "dinov3_vitl_prob"]].mean(axis=1)
    out["dinov3_max_prob"] = out[["dinov3_vitb_prob", "dinov3_vitl_prob"]].max(axis=1)
    out["dinov3_min_prob"] = out[["dinov3_vitb_prob", "dinov3_vitl_prob"]].min(axis=1)
    return out


def normalize_dev() -> pd.DataFrame:
    df = pd.read_csv(DEV67, dtype={"case_id": str})
    df = add_dino(df, DINO_B_DEV, DINO_L_DEV)
    df["base_prob"] = df["oof_prob_high"].astype(float)
    df["base_pred"] = df["oof_pred_idx"].astype(int)
    return df


def normalize_external() -> pd.DataFrame:
    df = pd.read_csv(EXT67, dtype={"case_id": str})
    df = add_dino(df, DINO_B_EXT, DINO_L_EXT)
    df["domain"] = "external"
    df["base_prob"] = df["locked_prob_high"].astype(float)
    df["base_pred"] = df["locked_pred_idx"].astype(int)
    return df


def apply_rule(frame: pd.DataFrame, rule: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_prob = frame["base_prob"].astype(float).to_numpy()
    base_pred = frame["base_pred"].astype(int).to_numpy()
    dino_mean = frame["dinov3_mean_prob"].astype(float).to_numpy()
    dino_max = frame["dinov3_max_prob"].astype(float).to_numpy()
    dino_min = frame["dinov3_min_prob"].astype(float).to_numpy()
    rescue_source = str(rule["rescue_source"])
    if rescue_source == "mean":
        dino_score = dino_mean
    elif rescue_source == "max":
        dino_score = dino_max
    elif rescue_source == "min":
        dino_score = dino_min
    else:
        raise ValueError(rescue_source)
    rescue = (
        (base_pred == 0)
        & (dino_score >= float(rule["rescue_threshold"]))
        & (base_prob >= float(rule["base_prob_min"]))
        & ((dino_score - base_prob) >= float(rule["gap_min"]))
    )
    veto = np.zeros(len(frame), dtype=bool)
    if bool(rule["use_veto"]):
        veto = (
            (base_pred == 1)
            & (dino_mean <= float(rule["veto_threshold"]))
            & (base_prob <= float(rule["veto_base_prob_max"]))
        )
    pred = base_pred.copy()
    prob = base_prob.copy()
    pred[rescue] = 1
    prob[rescue] = np.maximum(base_prob[rescue], dino_score[rescue])
    pred[veto] = 0
    prob[veto] = np.minimum(base_prob[veto], dino_mean[veto])
    routed = rescue | veto
    return pred, prob, routed


def by_domain_metrics(frame: pd.DataFrame, pred: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    y = frame["label_idx"].astype(int).to_numpy()
    row: dict[str, Any] = {f"dev_{k}": v for k, v in metric_dict(y, pred, prob).items()}
    for domain in ["old", "third"]:
        mask = frame["domain"].astype(str).eq(domain).to_numpy()
        metrics = metric_dict(y[mask], pred[mask], prob[mask])
        row.update({f"{domain}_{k}": v for k, v in metrics.items()})
    return row


def scan_rules(dev: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pred_tables: list[pd.DataFrame] = []
    for source in ["mean", "max", "min"]:
        for rescue_t in np.round(np.arange(0.50, 0.901, 0.025), 3):
            for base_min in [0.00, 0.15, 0.25, 0.35, 0.45, 0.52]:
                for gap_min in [0.00, 0.05, 0.10, 0.20, 0.30]:
                    for use_veto in [False, True]:
                        veto_thresholds = [0.30, 0.35, 0.40, 0.45] if use_veto else [0.0]
                        for veto_t in veto_thresholds:
                            rule = {
                                "rescue_source": source,
                                "rescue_threshold": float(rescue_t),
                                "base_prob_min": float(base_min),
                                "gap_min": float(gap_min),
                                "use_veto": bool(use_veto),
                                "veto_threshold": float(veto_t),
                                "veto_base_prob_max": 0.72,
                            }
                            pred, prob, routed = apply_rule(dev, rule)
                            row = by_domain_metrics(dev, pred, prob)
                            row.update(rule)
                            row["routed_n"] = int(routed.sum())
                            row["rescue_n"] = int(((dev["base_pred"].to_numpy(int) == 0) & (pred == 1)).sum())
                            row["veto_n"] = int(((dev["base_pred"].to_numpy(int) == 1) & (pred == 0)).sum())
                            old_guard = min(float(row["old_accuracy"]), float(row["old_balanced_accuracy"]))
                            third_score = (
                                0.45 * float(row["third_balanced_accuracy"])
                                + 0.25 * float(row["third_accuracy"])
                                + 0.30 * float(row["third_sensitivity_high"])
                            )
                            row["selection_score_guard92"] = third_score if old_guard >= 0.92 else third_score - (0.92 - old_guard) * 4.0
                            row["selection_score_guard90"] = third_score if old_guard >= 0.90 else third_score - (0.90 - old_guard) * 4.0
                            rows.append(row)
                            if old_guard >= 0.90 and len(pred_tables) < 300:
                                table = dev[["case_id", "original_case_id", "domain", "third_split", "fold_id", "task_l6_label", "task_l7_label", "label_idx"]].copy()
                                table["rule_idx"] = len(rows) - 1
                                table["rule_prob"] = prob
                                table["rule_pred"] = pred
                                table["rule_routed"] = routed.astype(int)
                                pred_tables.append(table)
    summary = pd.DataFrame(rows).sort_values(["selection_score_guard92", "third_balanced_accuracy", "old_accuracy"], ascending=False)
    selected = summary.iloc[0].to_dict()
    return summary, selected, pd.concat(pred_tables, ignore_index=True) if pred_tables else pd.DataFrame()


def apply_external(selected: dict[str, Any], external: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred, prob, routed = apply_rule(external, selected)
    out = external.copy()
    out["rule_prob_high"] = prob
    out["rule_pred_idx"] = pred
    out["rule_routed"] = routed.astype(int)
    out["rule_correct"] = (pred == out["label_idx"].astype(int).to_numpy()).astype(int)
    rows: list[dict[str, Any]] = []
    for subset, mask in [
        ("all", np.ones(len(out), dtype=bool)),
        ("strict", out["strict_task7_eval"].astype(int).to_numpy() == 1),
    ]:
        g = out.loc[mask]
        metrics = metric_dict(g["label_idx"].astype(int).to_numpy(), g["rule_pred_idx"].astype(int).to_numpy(), g["rule_prob_high"].astype(float).to_numpy())
        metrics.update({k: selected[k] for k in ["rescue_source", "rescue_threshold", "base_prob_min", "gap_min", "use_veto", "veto_threshold", "veto_base_prob_max"]})
        metrics["subset"] = subset
        metrics["routed_n"] = int(g["rule_routed"].sum())
        rows.append(metrics)
    return out, pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = normalize_dev()
    external = normalize_external()
    summary, selected, pred_tables = scan_rules(dev)
    summary.to_csv(OUT / "dev_rule_scan_summary.csv", index=False, encoding="utf-8-sig")
    if not pred_tables.empty:
        pred_tables.to_csv(OUT / "dev_rule_candidate_predictions_sample.csv", index=False, encoding="utf-8-sig")
    selected_pred, external_summary = apply_external(selected, external)
    selected_pred.to_csv(OUT / "external_locked67_dinov3_rescue_predictions.csv", index=False, encoding="utf-8-sig")
    external_summary.to_csv(OUT / "external_locked67_dinov3_rescue_summary.csv", index=False, encoding="utf-8-sig")
    report = {
        "boundary": {
            "selection_data": "development OOF only: old + third batch",
            "external_training_or_threshold_selection": False,
            "base_model": "67_old_third_no64_meta_stack_plus_dinov3vitl_ft selected_guard92",
        },
        "selected_rule": selected,
        "external_summary": external_summary.to_dict(orient="records"),
    }
    write_json(OUT / "locked67_dinov3_rescue_report.json", report)
    print("[selected]", json.dumps(selected, ensure_ascii=False, indent=2), flush=True)
    print("[external]", external_summary.to_string(index=False), flush=True)
    print(f"[done] outputs saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
