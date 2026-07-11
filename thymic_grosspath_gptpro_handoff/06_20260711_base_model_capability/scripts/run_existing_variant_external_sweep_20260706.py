#!/usr/bin/env python3
"""Sweep complete existing Task7 model variants on locked external registries."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


MODEL_SPECS = {
    "113_qkvb_whole448": "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/113_dinov3_vitl16_qkvb_task7_whole448_token_lastblock_lowlr_5fold_20260524",
    "119_qkvb_whole352_last2": "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/119_dinov3_vitl16_qkvb_task7_whole352_last2blocks_verylowlr_5fold_20260524",
    "154_qkvb_whole352_full": "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/154_dinov3_vitl16_qkvb_task7_whole352_full_vlowlr_5fold_20260524",
    "171_qkvb_externalmimic": "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/171_dinov3_vitl16_qkvb_task7_whole352_externalmimic_lastblock_5fold_20260525",
    "92_dinov3l_wpc352": "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/92_dinov3_vitl16_task7_wpc352_lastblock_lowlr_5fold_20260524",
    "97_convnext_wpc352_last2": "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/97_convnext_base_dinov3_task7_wpc352_last2_lowlr_5fold_20260524",
    "75b_dinov3b_wpc_domainrobust": "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/75b_grosspath_rc_v2_dinov3b16_domain_robust_weighted_20260526",
}

DOMAIN_REGISTRIES = {
    "strict_external": "experiments/base_model_expansion_20260706/outputs/registry/strict_external_registry_for_inference.csv",
    "new_external_160": "experiments/base_model_expansion_20260706/outputs/registry/new_external_160_registry_for_inference.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/base_model_expansion_20260706/outputs/existing_variant_sweep"),
    )
    parser.add_argument("--domains", default="strict_external")
    parser.add_argument("--models", default=",".join(MODEL_SPECS))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--views", default="orig,hflip,vflip,hvflip")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def metric_row(domain: str, model_tag: str, pred_csv: Path) -> dict[str, object]:
    df = pd.read_csv(pred_csv)
    y = df["label_idx"].astype(int).to_numpy()
    pred = df["pred_idx"].astype(int).to_numpy()
    prob = df["prob_high_risk_group"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "domain": domain,
        "model_tag": model_tag,
        "n": int(len(df)),
        "n_folds": int(df["n_folds"].iloc[0]) if "n_folds" in df.columns and len(df) else None,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)) if len(set(y)) == 2 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "high_recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "low_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "prediction_csv": str(pred_csv),
    }


def subtype_rows(domain: str, model_tag: str, pred_csv: Path) -> list[dict[str, object]]:
    df = pd.read_csv(pred_csv)
    rows: list[dict[str, object]] = []
    if "task_l6_label" not in df.columns:
        return rows
    for subtype, group in df.groupby("task_l6_label", dropna=False):
        rows.append(
            {
                "domain": domain,
                "model_tag": model_tag,
                "task_l6_label": subtype,
                "n": int(len(group)),
                "accuracy": float(accuracy_score(group["label_idx"].astype(int), group["pred_idx"].astype(int))),
                "mean_prob_high": float(group["prob_high_risk_group"].mean()),
                "pred_high_count": int((group["pred_idx"].astype(int) == 1).sum()),
                "pred_low_count": int((group["pred_idx"].astype(int) == 0).sum()),
            }
        )
    return rows


def assert_complete_run(project_root: Path, model_tag: str, run_dir: Path) -> None:
    if not run_dir.exists():
        raise FileNotFoundError(f"{model_tag}: run dir not found: {run_dir}")
    checkpoints = sorted(run_dir.glob("fold_*/best_model.pt"))
    if len(checkpoints) != 5:
        raise RuntimeError(f"{model_tag}: expected 5 fold checkpoints, found {len(checkpoints)}")
    for required in ("cv_fold_summary.csv", "oof_metrics.csv", "args.json"):
        if not (run_dir / required).exists():
            raise RuntimeError(f"{model_tag}: missing {required}")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = project_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    model_tags = [item.strip() for item in args.models.split(",") if item.strip()]
    script = project_root / "scripts" / "run_task7_dinov3_external_tta_fastcrop_20260706.py"

    rows: list[dict[str, object]] = []
    subrows: list[dict[str, object]] = []
    for model_tag in model_tags:
        run_dir = project_root / MODEL_SPECS[model_tag]
        assert_complete_run(project_root, model_tag, run_dir)
        for domain in domains:
            registry = project_root / DOMAIN_REGISTRIES[domain]
            out_dir = output_root / domain / f"{model_tag}_tta4_fastcrop"
            pred_csv = out_dir / "external_tta_predictions.csv"
            if args.force or not pred_csv.exists():
                cmd = [
                    sys.executable,
                    str(script),
                    "--run-dir",
                    str(run_dir),
                    "--external-registry-csv",
                    str(registry),
                    "--output-dir",
                    str(out_dir),
                    "--views",
                    args.views,
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--device",
                    args.device,
                ]
                print(f"\n[run] domain={domain} model={model_tag}", flush=True)
                subprocess.run(cmd, cwd=project_root, check=True)
            else:
                print(f"\n[skip] existing predictions domain={domain} model={model_tag}", flush=True)
            rows.append(metric_row(domain, model_tag, pred_csv))
            subrows.extend(subtype_rows(domain, model_tag, pred_csv))
            summary = pd.DataFrame(rows).sort_values(["domain", "balanced_accuracy"], ascending=[True, False])
            summary.to_csv(output_root / "existing_variant_sweep_summary_partial.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows).sort_values(["domain", "balanced_accuracy"], ascending=[True, False])
    by_subtype = pd.DataFrame(subrows).sort_values(["domain", "model_tag", "task_l6_label"])
    summary_path = output_root / "existing_variant_sweep_summary.csv"
    subtype_path = output_root / "existing_variant_sweep_by_task_l6_label.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_subtype.to_csv(subtype_path, index=False, encoding="utf-8-sig")
    print("\n[summary]")
    print(summary[["domain", "model_tag", "n", "n_folds", "accuracy", "balanced_accuracy", "auc", "high_recall", "low_specificity", "tn", "fp", "fn", "tp"]].to_string(index=False))
    print(f"[ok] {summary_path}")
    print(f"[ok] {subtype_path}")


if __name__ == "__main__":
    main()
