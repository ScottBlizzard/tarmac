from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thymic_baseline.metrics import summarize_prediction_frame


TASK6_CLASS_NAMES = ("A", "AB", "B1", "B2", "B3", "TC")
TASK7_CLASS_NAMES = ("low_risk_group", "high_risk_group")
LOW_RISK_COLS = ("prob_A", "prob_AB", "prob_B1")
HIGH_RISK_COLS = ("prob_B2", "prob_B3", "prob_TC")
LOW_RISK_IDXS = {0, 1, 2}
HIGH_RISK_IDXS = {3, 4, 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive Task7 low-risk vs high-risk(+TC) metrics from Task6 case-level predictions."
    )
    parser.add_argument("--task6-oof-csv", required=True, help="Case-level Task6 OOF predictions CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory for Task7 derived reports.")
    parser.add_argument(
        "--source-name",
        default="task6_source",
        help="Short label describing the source model/result.",
    )
    return parser.parse_args()


def build_task7_frame(task6_df: pd.DataFrame) -> pd.DataFrame:
    required = {"case_id", "label_idx", "pred_idx", "fold_id", *LOW_RISK_COLS, *HIGH_RISK_COLS}
    missing = required.difference(task6_df.columns)
    if missing:
        raise ValueError(f"Task6 OOF CSV missing required columns: {sorted(missing)}")

    out = task6_df.copy()
    out["prob_low_risk_group"] = out.loc[:, list(LOW_RISK_COLS)].sum(axis=1)
    out["prob_high_risk_group"] = out.loc[:, list(HIGH_RISK_COLS)].sum(axis=1)
    out["label_idx"] = out["label_idx"].astype(int).map(
        lambda x: 0 if x in LOW_RISK_IDXS else 1 if x in HIGH_RISK_IDXS else np.nan
    )
    out["pred_idx"] = (out["prob_high_risk_group"] > out["prob_low_risk_group"]).astype(int)
    out = out.dropna(subset=["label_idx"]).copy()
    out["label_idx"] = out["label_idx"].astype(int)
    out["pred_label"] = out["pred_idx"].map({0: TASK7_CLASS_NAMES[0], 1: TASK7_CLASS_NAMES[1]})
    out["true_label"] = out["label_idx"].map({0: TASK7_CLASS_NAMES[0], 1: TASK7_CLASS_NAMES[1]})
    keep_cols = [
        "case_id",
        "fold_id",
        "label_idx",
        "pred_idx",
        "true_label",
        "pred_label",
        "prob_low_risk_group",
        "prob_high_risk_group",
    ]
    return out.loc[:, keep_cols].reset_index(drop=True)


def compute_fold_metrics(task7_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_id, group in task7_df.groupby("fold_id", sort=True):
        metrics = summarize_prediction_frame(
            group[["case_id", "label_idx", "pred_idx", "prob_low_risk_group", "prob_high_risk_group"]].rename(
                columns={
                    "prob_low_risk_group": "prob_low_risk_group",
                    "prob_high_risk_group": "prob_high_risk_group",
                }
            ),
            TASK7_CLASS_NAMES,
        )
        row = {"fold_id": int(fold_id)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)


def compute_confusion_tables(task7_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(
        pd.Categorical(task7_df["true_label"], categories=TASK7_CLASS_NAMES, ordered=True),
        pd.Categorical(task7_df["pred_label"], categories=TASK7_CLASS_NAMES, ordered=True),
        dropna=False,
    )
    counts.index.name = "true_label"
    counts.columns.name = "pred_label"
    row_norm = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return counts.reset_index(), row_norm.reset_index()


def build_summary_md(
    source_name: str,
    task7_df: pd.DataFrame,
    oof_metrics: dict[str, float],
    fold_df: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append(f"# Task7 派生结果摘要：{source_name}")
    lines.append("")
    lines.append("Task7 定义：")
    lines.append("- 低危组：A / AB / B1")
    lines.append("- 高危组：B2 / B3 / TC")
    lines.append("")
    lines.append(f"- 纳入病例数：{len(task7_df)}")
    lines.append(f"- OOF AUC：{oof_metrics['auc']:.4f}")
    lines.append(f"- OOF Accuracy：{oof_metrics['accuracy']:.4f}")
    lines.append(f"- OOF Balanced Accuracy：{oof_metrics['balanced_accuracy']:.4f}")
    lines.append(f"- OOF Sensitivity：{oof_metrics['sensitivity']:.4f}")
    lines.append(f"- OOF Specificity：{oof_metrics['specificity']:.4f}")
    lines.append(f"- OOF F1：{oof_metrics['f1']:.4f}")
    lines.append("")
    if not fold_df.empty:
        lines.append("## 分折结果")
        lines.append("")
        lines.append("| Fold | AUC | Accuracy | BACC | Sensitivity | Specificity | F1 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in fold_df.iterrows():
            lines.append(
                f"| {int(row['fold_id'])} | {row['auc']:.4f} | {row['accuracy']:.4f} | "
                f"{row['balanced_accuracy']:.4f} | {row['sensitivity']:.4f} | "
                f"{row['specificity']:.4f} | {row['f1']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    task6_df = pd.read_csv(args.task6_oof_csv, dtype={"case_id": str})
    task7_df = build_task7_frame(task6_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task7_csv = output_dir / "oof_case_predictions_task7.csv"
    task7_df.to_csv(task7_csv, index=False, encoding="utf-8-sig")

    metrics_df_input = task7_df[["case_id", "label_idx", "pred_idx", "prob_low_risk_group", "prob_high_risk_group"]]
    oof_metrics = summarize_prediction_frame(metrics_df_input, TASK7_CLASS_NAMES)
    pd.DataFrame([oof_metrics]).to_csv(output_dir / "oof_metrics_task7.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "oof_metrics_task7.json").open("w", encoding="utf-8") as f:
        json.dump(oof_metrics, f, ensure_ascii=False, indent=2)

    fold_df = compute_fold_metrics(task7_df)
    fold_df.to_csv(output_dir / "cv_fold_summary_task7.csv", index=False, encoding="utf-8-sig")

    counts_df, row_norm_df = compute_confusion_tables(task7_df)
    counts_df.to_csv(output_dir / "confusion_matrix_counts_task7.csv", index=False, encoding="utf-8-sig")
    row_norm_df.to_csv(output_dir / "confusion_matrix_row_norm_task7.csv", index=False, encoding="utf-8-sig")

    summary_md = build_summary_md(args.source_name, task7_df, oof_metrics, fold_df)
    (output_dir / "task7_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Wrote Task7 predictions: {task7_csv}")
    print(f"Wrote Task7 metrics under: {output_dir}")
    print(summary_md)


if __name__ == "__main__":
    main()
