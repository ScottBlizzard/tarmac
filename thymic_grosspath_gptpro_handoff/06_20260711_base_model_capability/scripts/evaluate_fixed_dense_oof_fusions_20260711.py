from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score


RECIPES = {
    "avg_207_211": [
        "207_qkvb_dense_gated_cw_20260711",
        "211_qkvb_dense_viewgated_moe_20260711",
    ],
    "avg_207_209": [
        "207_qkvb_dense_gated_cw_20260711",
        "209_qkvb_dense_gated_concept_subtype_proto_20260711",
    ],
    "avg_209_211": [
        "209_qkvb_dense_gated_concept_subtype_proto_20260711",
        "211_qkvb_dense_viewgated_moe_20260711",
    ],
    "avg_207_209_211": [
        "207_qkvb_dense_gated_cw_20260711",
        "209_qkvb_dense_gated_concept_subtype_proto_20260711",
        "211_qkvb_dense_viewgated_moe_20260711",
    ],
    "max_207_211": [
        "207_qkvb_dense_gated_cw_20260711",
        "211_qkvb_dense_viewgated_moe_20260711",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def summarize(frame: pd.DataFrame, probability_column: str) -> dict[str, float | int]:
    y_true = frame["label_idx"].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, probability))
    except ValueError:
        auc = float("nan")
    return {
        "n": len(frame),
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for recipe_name, tags in RECIPES.items():
        frames = []
        for index, tag in enumerate(tags):
            path = runs_root / tag / "oof_predictions.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
            keep = ["case_id", "label_idx", "domain", "source_dataset", "task_l6_label", "prob_high"]
            frame = frame[keep].rename(columns={"prob_high": f"prob_{index}"})
            frames.append(frame)
        fused = frames[0]
        for frame in frames[1:]:
            fused = fused.merge(
                frame,
                on=["case_id", "label_idx", "domain", "source_dataset", "task_l6_label"],
                how="inner",
                validate="one_to_one",
            )
        probability_columns = [f"prob_{index}" for index in range(len(tags))]
        if recipe_name.startswith("max_"):
            fused["prob_high_fused"] = fused[probability_columns].max(axis=1)
        else:
            fused["prob_high_fused"] = fused[probability_columns].mean(axis=1)
        fused["pred_idx_fused"] = (fused["prob_high_fused"] >= 0.5).astype(int)
        fused.to_csv(output_dir / f"{recipe_name}_oof_predictions.csv", index=False, encoding="utf-8-sig")

        overall = summarize(fused, "prob_high_fused")
        source_bacc = []
        for source, group in fused.groupby("source_dataset"):
            metrics = summarize(group, "prob_high_fused")
            source_bacc.append(float(metrics["balanced_accuracy"]))
        subtype_accuracy = {
            str(name): float((group["pred_idx_fused"] == group["label_idx"]).mean())
            for name, group in fused.groupby("task_l6_label")
        }
        summary_rows.append(
            {
                "recipe": recipe_name,
                "members": ",".join(tags),
                **overall,
                "min_source_bacc": float(np.min(source_bacc)),
                "mean_source_bacc": float(np.mean(source_bacc)),
                "B1_risk_accuracy": subtype_accuracy.get("B1", np.nan),
                "B2_risk_accuracy": subtype_accuracy.get("B2", np.nan),
                "AB_risk_accuracy": subtype_accuracy.get("AB", np.nan),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["min_source_bacc", "balanced_accuracy", "auc"], ascending=False
    )
    summary.to_csv(output_dir / "dense_oof_fixed_fusion_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
