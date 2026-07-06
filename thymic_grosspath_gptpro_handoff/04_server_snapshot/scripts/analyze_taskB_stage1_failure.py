from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

LABEL_NAMES = ["A", "AB", "B1", "B2", "B3"]
VIEW_FILES = {
    "fine": "oof_stage1_fine_case_mean.csv",
    "fine_crop": "oof_stage1_fine_crop_case_mean.csv",
    "fine_soft_gate": "oof_stage1_fine_soft_gate_case_mean.csv",
    "fine_whole": "oof_stage1_fine_whole_case_mean.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze strict Task-B stage1 failure modes.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--early-epoch-threshold", type=int, default=3)
    parser.add_argument("--low-val-threshold", type=float, default=0.35)
    return parser.parse_args()


def per_class_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    report = classification_report(
        frame["label_idx"],
        frame["pred_idx"],
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for name in LABEL_NAMES:
        rows.append(
            {
                "class_name": name,
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_view(frame: pd.DataFrame) -> dict[str, float]:
    report = classification_report(
        frame["label_idx"],
        frame["pred_idx"],
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []
    summary_lines.append("# Task B Stage1 Failure Analysis")
    summary_lines.append("")
    summary_lines.append(f"- Run dir: `{run_dir}`")

    cv_path = run_dir / "cv_fold_summary.csv"
    cv = None
    if cv_path.exists():
        cv = pd.read_csv(cv_path).sort_values("stage1_test_fine_macro_f1")
        cv.to_csv(output_dir / "fold_ranking.csv", index=False)
        summary_lines.append("")
        summary_lines.append("## Fold Ranking")
        summary_lines.append(cv[[
            "fold_id",
            "best_stage1_epoch",
            "best_stage1_val_fine_macro_f1",
            "stage1_test_fine_macro_f1",
            "stage1_test_fine_soft_gate_macro_f1",
            "stage1_test_coarse_auc",
        ]].to_string(index=False))

    history_rows = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        hist_path = fold_dir / "stage1_history.csv"
        if not hist_path.exists():
            continue
        hist = pd.read_csv(hist_path)
        peak_idx = hist["val_fine_macro_f1"].idxmax()
        history_rows.append(
            {
                "fold_id": int(fold_dir.name.split("_")[1]),
                "epochs_ran": int(hist["epoch"].max()),
                "best_epoch_by_f1": int(hist.loc[peak_idx, "epoch"]),
                "peak_val_fine_macro_f1": float(hist["val_fine_macro_f1"].max()),
                "peak_val_fine_soft_gate_macro_f1": float(hist["val_fine_soft_gate_macro_f1"].max()) if "val_fine_soft_gate_macro_f1" in hist.columns else float("nan"),
                "peak_val_fine_macro_auc": float(hist["val_fine_macro_auc"].max()) if "val_fine_macro_auc" in hist.columns else float("nan"),
                "peak_val_coarse_auc": float(hist["val_coarse_auc"].max()) if "val_coarse_auc" in hist.columns else float("nan"),
            }
        )
    history_df = pd.DataFrame(history_rows).sort_values("fold_id") if history_rows else pd.DataFrame()
    if not history_df.empty:
        history_df.to_csv(output_dir / "history_summary.csv", index=False)
        summary_lines.append("")
        summary_lines.append("## Validation Trajectory Summary")
        summary_lines.append(history_df.to_string(index=False))
        risky = history_df[
            (history_df["best_epoch_by_f1"] <= args.early_epoch_threshold)
            & (history_df["peak_val_fine_macro_f1"] < args.low_val_threshold)
        ]
        if not risky.empty:
            summary_lines.append("")
            summary_lines.append("## Early/Low-Ceiling Folds")
            summary_lines.append(risky.to_string(index=False))

    view_rows = []
    for view_name, filename in VIEW_FILES.items():
        path = run_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        cm = confusion_matrix(frame["label_idx"], frame["pred_idx"], labels=list(range(len(LABEL_NAMES))))
        pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(output_dir / f"confusion_{view_name}.csv")
        class_df = per_class_metrics(frame).sort_values("f1")
        class_df.to_csv(output_dir / f"per_class_{view_name}.csv", index=False)
        metrics = summarize_view(frame)
        metrics["view"] = view_name
        view_rows.append(metrics)
    views_df = pd.DataFrame(view_rows).sort_values("macro_f1", ascending=False) if view_rows else pd.DataFrame()
    if not views_df.empty:
        views_df.to_csv(output_dir / "view_summary.csv", index=False)
        summary_lines.append("")
        summary_lines.append("## View Summary")
        summary_lines.append(views_df[["view", "accuracy", "macro_precision", "macro_recall", "macro_f1"]].to_string(index=False))
        summary_lines.append("")
        summary_lines.append(f"- Strongest view by macro-F1: `{views_df.iloc[0]['view']}`")
        summary_lines.append(f"- Weakest view by macro-F1: `{views_df.iloc[-1]['view']}`")
        for view_name in views_df["view"].tolist():
            class_df = pd.read_csv(output_dir / f"per_class_{view_name}.csv")
            worst_classes = ", ".join(f"{row.class_name} ({row.f1:.3f})" for row in class_df.head(3).itertuples())
            summary_lines.append(f"- `{view_name}` weakest classes: {worst_classes}")

    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
