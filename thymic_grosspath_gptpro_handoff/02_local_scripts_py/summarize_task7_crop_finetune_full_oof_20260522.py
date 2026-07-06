from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Task7 crop fine-tune full OOF and third holdout results.")
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/06_adapt72_highfocus_finetune_inputs_20260521/registry.csv",
    )
    parser.add_argument(
        "--fold1-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/10_adapt72_highfocus_vitb14_crop_finetune_20260521/fold_1",
    )
    parser.add_argument(
        "--folds2-5-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/30_adapt72_highfocus_vitb14_crop_finetune_full_oof_20260522",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/31_crop_finetune_full_oof_summary_20260522",
    )
    parser.add_argument("--method", default="mean", choices=["mean", "max_prob", "majority_vote"])
    parser.add_argument("--old-guard", type=float, default=0.92)
    return parser.parse_args()


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if prob is not None and len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, prob))
    return out


def read_fold_predictions(fold1_dir: Path, folds2_5_dir: Path, method: str) -> pd.DataFrame:
    paths: list[Path] = [fold1_dir / f"test_case_predictions_{method}.csv"]
    paths += [folds2_5_dir / f"fold_{fold}" / f"test_case_predictions_{method}.csv" for fold in [2, 3, 4, 5]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing fold prediction files:\n" + "\n".join(missing))

    frames = []
    for fold, path in zip([1, 2, 3, 4, 5], paths, strict=True):
        df = pd.read_csv(path, dtype={"case_id": str})
        df["fold"] = fold
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out["case_id"].duplicated().any():
        dup = out.loc[out["case_id"].duplicated(), "case_id"].head(10).tolist()
        raise ValueError(f"Duplicated case_id in test predictions: {dup}")
    return out


def add_registry(pred: pd.DataFrame, registry_csv: Path) -> pd.DataFrame:
    registry = pd.read_csv(registry_csv, dtype={"case_id": str, "original_case_id": str})
    keep = [
        "case_id",
        "original_case_id",
        "source_dataset",
        "source_case_folder",
        "task_l6_label",
        "task_l7_label",
        "selected_original_image_name",
        "training_image_path",
    ]
    merged = pred.merge(registry[keep], on="case_id", how="left", validate="one_to_one")
    if merged["source_dataset"].isna().any():
        missing = merged.loc[merged["source_dataset"].isna(), "case_id"].head(10).tolist()
        raise KeyError(f"Predictions missing registry rows: {missing}")
    merged["eval_group"] = np.select(
        [
            merged["source_dataset"].isin(["batch1", "batch2"]),
            merged["source_dataset"].eq("third_batch_adapt72_highfocus"),
            merged["source_dataset"].eq("third_batch_holdout"),
        ],
        ["old", "adapt72", "third_holdout"],
        default="other",
    )
    return merged


def collapse_replicates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eval_group, original_case_id), g in df.groupby(["eval_group", "original_case_id"], sort=False):
        label_values = sorted(g["label_idx"].astype(int).unique().tolist())
        if len(label_values) != 1:
            raise ValueError(f"Conflicting labels for {eval_group}/{original_case_id}: {label_values}")
        prob = float(g["prob_high_risk_group"].astype(float).mean())
        row = {
            "eval_group": eval_group,
            "original_case_id": original_case_id,
            "case_id_examples": ";".join(g["case_id"].astype(str).tolist()[:4]),
            "n_replicates": int(len(g)),
            "label_idx": int(label_values[0]),
            "prob_high": prob,
            "pred_idx_050": int(prob >= 0.5),
            "task_l6_label": g["task_l6_label"].iloc[0],
            "task_l7_label": g["task_l7_label"].iloc[0],
            "source_dataset": g["source_dataset"].iloc[0],
            "selected_original_image_name": g["selected_original_image_name"].iloc[0],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def metric_by_group(df: pd.DataFrame, pred_col: str, prob_col: str = "prob_high") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, g in df.groupby("eval_group", sort=False):
        out[name] = metrics(g["label_idx"].to_numpy(int), g[pred_col].to_numpy(int), g[prob_col].to_numpy(float))
    return out


def subtype_table(df: pd.DataFrame, pred_col: str, out_path: Path) -> pd.DataFrame:
    rows = []
    for (eval_group, subtype), g in df.groupby(["eval_group", "task_l6_label"], sort=False):
        row = {"eval_group": eval_group, "task_l6_label": subtype}
        row.update(metrics(g["label_idx"].to_numpy(int), g[pred_col].to_numpy(int), g["prob_high"].to_numpy(float)))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    return table


def scan_thresholds(df: pd.DataFrame, old_guard: float) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = []
    thresholds = np.round(np.arange(0.20, 0.801, 0.01), 2)
    for threshold in thresholds:
        pred = (df["prob_high"].to_numpy(float) >= threshold).astype(int)
        tmp = df.copy()
        tmp["pred_scan"] = pred
        row: dict[str, Any] = {"threshold": float(threshold)}
        group_metrics = metric_by_group(tmp, "pred_scan")
        for group, metric in group_metrics.items():
            row.update({f"{group}_{key}": value for key, value in metric.items()})
        row["old_guard"] = bool(
            row.get("old_accuracy", 0) >= old_guard and row.get("old_balanced_accuracy", 0) >= old_guard
        )
        row["selection_score_old_adapt"] = (
            float(row.get("adapt72_accuracy", 0))
            + 0.8 * float(row.get("adapt72_balanced_accuracy", 0))
            + 0.015 * float(row.get("adapt72_tp", 0))
            - 0.015 * float(row.get("old_fn", 0))
            - 0.015 * float(row.get("old_fp", 0))
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    guard = table[table["old_guard"]].copy()
    selected = (
        guard.sort_values(["selection_score_old_adapt", "old_accuracy", "old_balanced_accuracy"], ascending=False)
        .head(1)
        .to_dict("records")
    )
    hold_ref = (
        guard.sort_values(["third_holdout_accuracy", "third_holdout_balanced_accuracy"], ascending=False)
        .head(1)
        .to_dict("records")
    )
    hold_tp = guard.copy()
    base_tp = table.loc[table["threshold"].eq(0.5), "third_holdout_tp"]
    if not base_tp.empty:
        hold_tp = hold_tp[hold_tp["third_holdout_tp"] >= int(base_tp.iloc[0])]
    hold_tp_ref = (
        hold_tp.sort_values(["third_holdout_accuracy", "third_holdout_balanced_accuracy"], ascending=False)
        .head(1)
        .to_dict("records")
    )
    return table, (selected[0] if selected else {}), (hold_ref[0] if hold_ref else {}), (hold_tp_ref[0] if hold_tp_ref else {})


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred = read_fold_predictions(Path(args.fold1_dir), Path(args.folds2_5_dir), args.method)
    pred = add_registry(pred, Path(args.registry_csv))
    case = collapse_replicates(pred)
    case.to_csv(out / "crop_finetune_full_oof_case_predictions.csv", index=False, encoding="utf-8-sig")

    base_metrics = metric_by_group(case, "pred_idx_050")
    subtype = subtype_table(case, "pred_idx_050", out / "crop_finetune_full_oof_subtype_metrics.csv")
    threshold_scan, selected, hold_ref, hold_tp_ref = scan_thresholds(case, args.old_guard)
    threshold_scan.to_csv(out / "crop_finetune_threshold_scan.csv", index=False, encoding="utf-8-sig")

    error = case[case["label_idx"].astype(int) != case["pred_idx_050"].astype(int)].copy()
    error["error_direction"] = np.where(error["label_idx"].eq(1), "high_to_low", "low_to_high")
    error.sort_values(["eval_group", "error_direction", "task_l6_label", "original_case_id"]).to_csv(
        out / "crop_finetune_full_oof_errors.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "protocol": {
            "fold1_source": args.fold1_dir,
            "folds2_5_source": args.folds2_5_dir,
            "method": args.method,
            "replicate_handling": "third adapt72 replicated cases collapsed by original_case_id with mean high-risk probability",
            "threshold_selection": "selected threshold uses old + adapt72 only; third_holdout references are for analysis",
        },
        "base_threshold_0.50": base_metrics,
        "selected_by_old_plus_adapt_threshold": selected,
        "best_third_holdout_reference_under_old_guard": hold_ref,
        "best_third_holdout_tp_preserved_reference_under_old_guard": hold_tp_ref,
        "subtype_rows": subtype.to_dict("records"),
        "n_case_rows": int(len(case)),
    }
    (out / "crop_finetune_full_oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
