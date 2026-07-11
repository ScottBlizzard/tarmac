from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


PRIMARY_BASELINE_CANDIDATE = "main_prob"
RAW_SCORE_CANDIDATES = {
    "main_prob",
    "robust_prob",
    "prob_mean_core",
    "stack_gb_d2_multiview",
    "stack_gb_d2_core",
    "stack_logreg_c03_multiview",
    "stack_logreg_c03_core",
    "stack_extra_d3_core",
    "stack_extra_d3_multiview",
    "avg_main_core_whole",
    "median_all6",
    "max_all6",
}


def candidate_category(candidate: str) -> tuple[str, bool, str]:
    name = str(candidate)
    lowered = name.lower()
    if name == PRIMARY_BASELINE_CANDIDATE:
        return "primary_raw_forced_baseline", True, "main image-model probability; locked internal threshold evaluated on strict external"
    if any(token in lowered for token in ["subtype", "label", "oracle", "correct", "wrong"]):
        return "excluded_leaky_or_label_proxy", False, "candidate name suggests subtype/label/error information rather than deployable image-only score"
    if name in RAW_SCORE_CANDIDATES or lowered.startswith("stack_") or lowered in {"max_all6", "median_all6"}:
        return "raw_score_or_score_stack_probe", True, "image-score-derived probe; useful for discovery but not the primary clean baseline"
    if "pure_auto" in lowered:
        return "workflow_proxy", False, "derived from previous auto decision rather than a clean forced classifier"
    return "uncategorized_probe", False, "needs manual provenance check before being used as a forced baseline"


def compute_metrics(df: pd.DataFrame, pred_col: str, prob_col: str | None) -> dict:
    y = df["label_idx"].astype(int).to_numpy()
    pred = df[pred_col].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "high_risk_recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "low_risk_specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
    }
    if prob_col and prob_col in df.columns:
        try:
            out["auc"] = float(roc_auc_score(y, df[prob_col].astype(float).to_numpy()))
        except Exception:
            out["auc"] = np.nan
    else:
        out["auc"] = np.nan
    return out


def summarize_v135_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows: list[dict] = []
    for (scope, candidate, objective, domain), g in df.groupby(["scope", "candidate", "objective", "domain"], sort=True):
        category, usable, reason = candidate_category(candidate)
        metrics = compute_metrics(g, "pred_idx", "prob_high")
        rows.append(
            {
                "source_kind": "raw_forced_candidate_predictions",
                "source_csv": str(path),
                "scope": scope,
                "domain": domain,
                "candidate": candidate,
                "objective": objective,
                "threshold": float(pd.to_numeric(g["threshold"]).iloc[0]),
                "candidate_category": category,
                "usable_as_forced_baseline": bool(usable),
                "provenance_note": reason,
                **metrics,
            }
        )
    for (scope, candidate, objective), g in df.groupby(["scope", "candidate", "objective"], sort=True):
        category, usable, reason = candidate_category(candidate)
        metrics = compute_metrics(g, "pred_idx", "prob_high")
        rows.append(
            {
                "source_kind": "raw_forced_candidate_predictions",
                "source_csv": str(path),
                "scope": scope,
                "domain": "all_rows_in_scope",
                "candidate": candidate,
                "objective": objective,
                "threshold": float(pd.to_numeric(g["threshold"]).iloc[0]),
                "candidate_category": category,
                "usable_as_forced_baseline": bool(usable),
                "provenance_note": reason,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_workflow_context(path: Path, name: str) -> list[dict]:
    df = pd.read_csv(path, dtype=str)
    if "domain" not in df.columns or "label_idx" not in df.columns:
        return []
    pred_col = "system_pred" if "system_pred" in df.columns else "final_pred" if "final_pred" in df.columns else None
    if pred_col is None:
        return []
    rows = []
    for domain, g in df.groupby("domain", sort=True):
        metrics = compute_metrics(g, pred_col, "prob_mean_core" if "prob_mean_core" in g.columns else None)
        rows.append(
            {
                "source_kind": "workflow_or_reconstructed_context_not_forced_baseline",
                "source_name": name,
                "source_csv": str(path),
                "domain": domain,
                "pred_col": pred_col,
                "usable_as_forced_baseline": False,
                "provenance_note": "post-processed workflow/system output; may include review gates, auto-correction, or reconstruction",
                **metrics,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/workspace/thymic_project")
    parser.add_argument("--out-dir", default="experiments/base_model_expansion_20260706/outputs/baseline_metrics")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v135_path = project_root / "outputs/grosspath_rc_v135_stage1_base_candidate_scan_20260527/v135_stage1_candidate_predictions.csv"
    raw = summarize_v135_predictions(v135_path)
    raw.to_csv(out_dir / "raw_forced_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    raw.to_json(out_dir / "raw_forced_candidate_metrics.json", orient="records", force_ascii=False, indent=2)

    workflow_sources = [
        (
            "v195_reconstructed_system",
            project_root / "experiments/risk_control_rejection_20260621/outputs/v195_reconstructed_case_outputs.csv",
        ),
        (
            "candidate_policy_system",
            project_root / "experiments/risk_control_rejection_20260621/outputs/candidate_policy_case_outputs.csv",
        ),
        (
            "v185_unlabeled_shift_system",
            project_root / "outputs/grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527/v185_unlabeled_shift_adaptive_cases.csv",
        ),
    ]
    workflow_rows = []
    for name, path in workflow_sources:
        if path.exists():
            workflow_rows.extend(summarize_workflow_context(path, name))
    workflow = pd.DataFrame(workflow_rows)
    workflow.to_csv(out_dir / "workflow_postprocessed_context_metrics.csv", index=False, encoding="utf-8-sig")
    workflow.to_json(out_dir / "workflow_postprocessed_context_metrics.json", orient="records", force_ascii=False, indent=2)

    primary = raw[
        (raw["candidate"] == PRIMARY_BASELINE_CANDIDATE)
        & (raw["objective"] == "balanced_accuracy")
        & (raw["scope"].isin(["internal_oof_old_third", "strict_external_locked"]))
    ].copy()
    primary = primary.sort_values(["scope", "domain"])
    primary.to_csv(out_dir / "primary_main_prob_forced_baseline.csv", index=False, encoding="utf-8-sig")

    readme = [
        "# Existing Task7 Baseline Metrics",
        "",
        "This folder separates raw forced classification scores from post-processed workflow/system outputs.",
        "",
        "## Primary Clean Forced Baseline",
        "",
        "`primary_main_prob_forced_baseline.csv` uses `grosspath_rc_v135` `main_prob` with threshold 0.595 selected on internal old+third OOF and evaluated on locked strict external.",
        "",
        "## Important Guardrail",
        "",
        "`workflow_postprocessed_context_metrics.csv` is not a forced-classifier baseline. It may include review gates, auto-correction, or reconstructed final decisions, so it must not be used to claim base-model generalization.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print("Primary main_prob forced baseline:")
    print(primary.to_string(index=False))
    print(f"[ok] wrote {out_dir}")


if __name__ == "__main__":
    main()
