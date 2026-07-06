from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "grosspath_rc_v2_20260526"
V25_DIR = ROOT / "outputs" / "grosspath_rc_v25_auto_quality_gate_20260527"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v27_unified_workflow_20260527"

MAIN_THRESHOLD = 0.595
ROBUST_THRESHOLD = 0.57


@dataclass(frozen=True)
class WorkflowPolicy:
    name: str
    evidence_level: str
    quality_review_col: str | None
    safety_action: str
    quality_action: str
    description: str


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "n": int(len(y)),
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def load_external() -> pd.DataFrame:
    df = pd.read_csv(V2_DIR / "v2_external_diagnostic_table.csv")
    qpred = pd.read_csv(V25_DIR / "v25_quality_gate_case_predictions.csv")
    keep_cols = [
        "case_id",
        "original_case_id",
        "heuristic_nonpass_status_review",
        "heuristic_score_lt_100_review",
        "heuristic_score_lt_92_review",
        "quality_logistic_f1_review",
        "quality_logistic_balanced_review",
        "quality_logistic_recall90_review",
        "quality_rf_f1_review",
        "quality_rf_balanced_review",
        "quality_rf_recall90_review",
        "quality_extra_trees_f1_review",
        "quality_extra_trees_balanced_review",
        "quality_extra_trees_recall90_review",
        "quality_logistic_risk_oof",
        "quality_rf_risk_oof",
        "quality_extra_trees_risk_oof",
    ]
    df = df.merge(qpred[[col for col in keep_cols if col in qpred.columns]], on=["case_id", "original_case_id"], how="left")
    df["main_prob"] = df["prob_base162"].astype(float)
    df["main_pred"] = (df["main_prob"] >= MAIN_THRESHOLD).astype(int)
    df["robust_prob"] = df[["prob_base162", "prob103_vitl", "prob_mean_core"]].mean(axis=1).astype(float)
    df["robust_pred"] = (df["robust_prob"] >= ROBUST_THRESHOLD).astype(int)
    df["safety_trigger"] = ((df["main_pred"] == 0) & (df["robust_pred"] == 1)).astype(int)
    df["manual_quality_review"] = df["manual_quality_status_v1"].astype(str).ne("pass_readable").astype(int)
    df["manual_retake"] = df["manual_quality_status_v1"].astype(str).eq("reject_retake").astype(int)
    df["auto_retake_by_quality_status"] = df["quality_status"].astype(str).eq("reject").astype(int)
    return df


def policies() -> list[WorkflowPolicy]:
    return [
        WorkflowPolicy(
            "P0_main_auto_all",
            "strict_auto_baseline",
            None,
            "ignore",
            "ignore",
            "主模型直接输出，所有病例自动判读。",
        ),
        WorkflowPolicy(
            "P1_robust_auto_all",
            "strict_auto_baseline",
            None,
            "robust_all",
            "ignore",
            "鲁棒分支全量替换主模型。",
        ),
        WorkflowPolicy(
            "P2_safety_switch_auto",
            "strict_auto_rule",
            None,
            "switch_to_robust",
            "ignore",
            "主模型低危但鲁棒分支高危时自动切换到鲁棒分支。",
        ),
        WorkflowPolicy(
            "P3_qscore92_review_else_safety",
            "automatic_quality_heuristic",
            "heuristic_score_lt_92_review",
            "switch_to_robust",
            "review",
            "自动质量分数低于 92 的病例进入复核，其余病例应用安全切换。",
        ),
        WorkflowPolicy(
            "P4_rf_quality_review_else_safety",
            "quality_cv_proof_of_concept",
            "quality_rf_f1_review",
            "switch_to_robust",
            "review",
            "随机森林质量门控提示复核，其余病例应用安全切换。",
        ),
        WorkflowPolicy(
            "P5_extra_trees_quality_review_else_safety",
            "quality_cv_proof_of_concept",
            "quality_extra_trees_f1_review",
            "switch_to_robust",
            "review",
            "Extra Trees 质量门控提示复核，其余病例应用安全切换。",
        ),
        WorkflowPolicy(
            "P6_manual_quality_or_safety_review_upper",
            "manual_quality_upper_bound",
            "manual_quality_review",
            "review",
            "review",
            "人工质量标签或安全分歧触发复核，上限分析。",
        ),
        WorkflowPolicy(
            "P7_logistic_recall90_quality_or_safety_review",
            "quality_cv_high_safety",
            "quality_logistic_recall90_review",
            "review",
            "review",
            "高召回学习型质量门控或安全分歧触发复核，低漏诊模式。",
        ),
    ]


def apply_policy(df: pd.DataFrame, policy: WorkflowPolicy) -> pd.DataFrame:
    out = df.copy()
    quality_review = np.zeros(len(out), dtype=bool)
    if policy.quality_review_col:
        quality_review = out[policy.quality_review_col].fillna(0).astype(int).to_numpy(dtype=bool)
    retake = np.zeros(len(out), dtype=bool)
    if policy.quality_action == "review" and policy.quality_review_col is not None:
        retake = out["auto_retake_by_quality_status"].fillna(0).astype(int).to_numpy(dtype=bool)
    if policy.name == "P6_manual_quality_or_safety_review_upper":
        retake = out["manual_retake"].fillna(0).astype(int).to_numpy(dtype=bool)

    safety = out["safety_trigger"].astype(int).to_numpy(dtype=bool)
    main_pred = out["main_pred"].astype(int).to_numpy()
    robust_pred = out["robust_pred"].astype(int).to_numpy()

    if policy.safety_action == "robust_all":
        auto_pred = robust_pred
        safety_review = np.zeros(len(out), dtype=bool)
    elif policy.safety_action == "switch_to_robust":
        auto_pred = np.where(safety, robust_pred, main_pred)
        safety_review = np.zeros(len(out), dtype=bool)
    elif policy.safety_action == "review":
        auto_pred = main_pred
        safety_review = safety
    elif policy.safety_action == "ignore":
        auto_pred = main_pred
        safety_review = np.zeros(len(out), dtype=bool)
    else:
        raise ValueError(policy.safety_action)

    quality_review_effective = quality_review if policy.quality_action == "review" else np.zeros(len(out), dtype=bool)
    review = quality_review_effective | safety_review
    review = review & ~retake

    route = np.full(len(out), "auto_low_risk", dtype=object)
    route[auto_pred == 1] = "auto_high_risk"
    route[review & quality_review_effective & safety_review] = "review_quality_and_safety"
    route[review & quality_review_effective & ~safety_review] = "review_quality"
    route[review & safety_review & ~quality_review_effective] = "review_safety_disagreement"
    route[retake] = "retake_recommended"

    final_pred = auto_pred.copy()
    y = out["label_idx"].astype(int).to_numpy()
    # Workflow-level metric assumes review/retake enters an expert-confirmation pathway.
    final_pred[review | retake] = y[review | retake]

    out["policy"] = policy.name
    out["evidence_level"] = policy.evidence_level
    out["policy_description"] = policy.description
    out["quality_review_flag"] = quality_review.astype(int)
    out["safety_review_flag"] = safety_review.astype(int)
    out["safety_switch_flag"] = (policy.safety_action == "switch_to_robust") & safety
    out["retake_flag"] = retake.astype(int)
    out["route"] = route
    out["auto_pred"] = auto_pred
    out["workflow_final_pred"] = final_pred
    out["auto_output_flag"] = (~review & ~retake).astype(int)
    out["review_flag"] = review.astype(int)
    out["workflow_correct"] = (out["workflow_final_pred"].astype(int) == out["label_idx"].astype(int)).astype(int)
    out["auto_correct_if_output"] = np.where(out["auto_output_flag"].eq(1), (out["auto_pred"].astype(int) == out["label_idx"].astype(int)).astype(int), np.nan)
    return out


def evaluate_policy(df: pd.DataFrame, policy: WorkflowPolicy) -> tuple[dict[str, object], pd.DataFrame]:
    routed = apply_policy(df, policy)
    y = routed["label_idx"].astype(int).to_numpy()
    workflow = metrics(y, routed["workflow_final_pred"].astype(int).to_numpy())
    auto_mask = routed["auto_output_flag"].astype(int).to_numpy(dtype=bool)
    auto_metrics = metrics(y[auto_mask], routed.loc[auto_mask, "auto_pred"].astype(int).to_numpy()) if int(auto_mask.sum()) else {}
    review_n = int(routed["review_flag"].sum())
    retake_n = int(routed["retake_flag"].sum())
    route_counts = routed["route"].value_counts().to_dict()
    row: dict[str, object] = {
        "policy": policy.name,
        "evidence_level": policy.evidence_level,
        "description": policy.description,
        "n": len(routed),
        "auto_output_n": int(auto_mask.sum()),
        "auto_output_rate": float(auto_mask.mean()),
        "review_n": review_n,
        "review_rate": review_n / len(routed),
        "retake_n": retake_n,
        "retake_rate": retake_n / len(routed),
        "risk_control_n": review_n + retake_n,
        "risk_control_rate": (review_n + retake_n) / len(routed),
        "route_auto_low_n": int(route_counts.get("auto_low_risk", 0)),
        "route_auto_high_n": int(route_counts.get("auto_high_risk", 0)),
        "route_review_quality_n": int(route_counts.get("review_quality", 0)),
        "route_review_safety_n": int(route_counts.get("review_safety_disagreement", 0)),
        "route_review_both_n": int(route_counts.get("review_quality_and_safety", 0)),
        "route_retake_n": int(route_counts.get("retake_recommended", 0)),
    }
    row.update({f"workflow_{key}": value for key, value in workflow.items()})
    row.update({f"auto_{key}": value for key, value in auto_metrics.items()})
    return row, routed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_external()
    rows: list[dict[str, object]] = []
    routed_frames: list[pd.DataFrame] = []
    for policy in policies():
        row, routed = evaluate_policy(df, policy)
        rows.append(row)
        routed_frames.append(routed)

    metrics_df = pd.DataFrame(rows)
    routes_df = pd.concat(routed_frames, ignore_index=True)
    keep_cols = [
        "policy",
        "evidence_level",
        "case_id",
        "original_case_id",
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "image_name",
        "quality_status",
        "quality_score",
        "manual_quality_status_v1",
        "main_prob",
        "main_pred",
        "robust_prob",
        "robust_pred",
        "prob103_vitl",
        "prob_mean_core",
        "safety_trigger",
        "quality_review_flag",
        "safety_review_flag",
        "safety_switch_flag",
        "review_flag",
        "retake_flag",
        "route",
        "auto_output_flag",
        "auto_pred",
        "workflow_final_pred",
        "workflow_correct",
        "auto_correct_if_output",
        "error_type",
    ]
    metrics_df.to_csv(OUT_DIR / "v27_unified_workflow_metrics.csv", index=False, encoding="utf-8-sig")
    routes_df[[col for col in keep_cols if col in routes_df.columns]].to_csv(OUT_DIR / "v27_unified_case_routes_external.csv", index=False, encoding="utf-8-sig")

    summary_cols = [
        "policy",
        "evidence_level",
        "auto_output_rate",
        "review_rate",
        "retake_rate",
        "risk_control_rate",
        "workflow_accuracy",
        "workflow_balanced_accuracy",
        "workflow_sensitivity",
        "workflow_specificity",
        "workflow_fn",
        "workflow_fp",
        "auto_accuracy",
        "auto_balanced_accuracy",
        "auto_fn",
        "auto_fp",
    ]
    print(f"[done] {OUT_DIR}")
    print(metrics_df[[col for col in summary_cols if col in metrics_df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
