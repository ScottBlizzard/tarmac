from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)


KEY_COLUMNS = [
    "case_id",
    "label_idx",
    "domain",
    "source_dataset",
    "task_l6_label",
    "fold_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select an equal-weight dense fusion using internal OOF only.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-single-bacc", type=float, default=0.67)
    parser.add_argument("--min-single-source-bacc", type=float, default=0.60)
    parser.add_argument("--max-pool-size", type=int, default=12)
    parser.add_argument("--max-members", type=int, default=3)
    parser.add_argument("--min-class-recall", type=float, default=0.60)
    parser.add_argument("--candidate-manifest", default="")
    parser.add_argument("--lodo-runs-root", default="")
    parser.add_argument("--lodo-weight", type=float, default=0.45)
    parser.add_argument("--min-lodo-class-recall", type=float, default=0.50)
    return parser.parse_args()


def summarize(frame: pd.DataFrame, probability_column: str) -> dict[str, float | int]:
    y_true = frame["label_idx"].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": len(frame),
        "auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "sensitivity": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_prediction(runs_root: Path, run_tag: str, probability_name: str) -> pd.DataFrame:
    path = runs_root / run_tag / "oof_predictions.csv"
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    missing = [column for column in KEY_COLUMNS + ["prob_high"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if frame["case_id"].duplicated().any():
        raise ValueError(f"Duplicate case_id values in {path}")
    return frame[KEY_COLUMNS + ["prob_high"]].rename(columns={"prob_high": probability_name})


def evaluate_recipe(frame: pd.DataFrame, members: tuple[str, ...]) -> tuple[dict[str, object], pd.DataFrame]:
    probability_columns = [column for column in frame.columns if column.startswith("prob_")]
    frame = frame.copy()
    frame["prob_high_fused"] = frame[probability_columns].mean(axis=1)
    frame["pred_idx_fused"] = (frame["prob_high_fused"] >= 0.5).astype(int)
    overall = summarize(frame, "prob_high_fused")
    source_bacc = [
        float(summarize(group, "prob_high_fused")["balanced_accuracy"])
        for _, group in frame.groupby("source_dataset")
    ]
    fold_bacc = [
        float(summarize(group, "prob_high_fused")["balanced_accuracy"])
        for _, group in frame.groupby("fold_id")
    ]
    domain_bacc = {
        str(name): float(summarize(group, "prob_high_fused")["balanced_accuracy"])
        for name, group in frame.groupby("domain")
    }
    subtype_accuracy = {
        str(name): float((group["pred_idx_fused"] == group["label_idx"]).mean())
        for name, group in frame.groupby("task_l6_label")
    }
    min_class_recall = min(float(overall["sensitivity"]), float(overall["specificity"]))
    selection_score = (
        0.30 * float(overall["balanced_accuracy"])
        + 0.25 * float(np.min(source_bacc))
        + 0.20 * float(overall["auc"])
        + 0.15 * min_class_recall
        + 0.10 * float(np.min(fold_bacc))
    )
    row: dict[str, object] = {
        "recipe": "avg__" + "__".join(members),
        "member_count": len(members),
        "members": ",".join(members),
        **overall,
        "min_class_recall": min_class_recall,
        "min_source_bacc": float(np.min(source_bacc)),
        "mean_source_bacc": float(np.mean(source_bacc)),
        "min_fold_bacc": float(np.min(fold_bacc)),
        "mean_fold_bacc": float(np.mean(fold_bacc)),
        "old_bacc": domain_bacc.get("old_data", np.nan),
        "third_bacc": domain_bacc.get("third_batch", np.nan),
        "B1_risk_accuracy": subtype_accuracy.get("B1", np.nan),
        "B2_risk_accuracy": subtype_accuracy.get("B2", np.nan),
        "selection_score": selection_score,
    }
    return row, frame


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv, encoding="utf-8-sig")
    summary = summary[
        (pd.to_numeric(summary["overall_bacc"], errors="coerce") >= args.min_single_bacc)
        & (pd.to_numeric(summary["min_source_bacc"], errors="coerce") >= args.min_single_source_bacc)
        & ~summary["views"].astype(str).str.contains("background_only", na=False)
        & ~summary["run_tag"].astype(str).str.startswith("210_")
    ].copy()
    summary = summary[summary["run_tag"].map(lambda tag: (runs_root / str(tag) / "oof_predictions.csv").exists())]
    if summary.empty:
        raise RuntimeError("No eligible internal OOF runs for fusion selection.")
    if args.candidate_manifest:
        candidate_manifest = pd.read_csv(args.candidate_manifest, encoding="utf-8-sig")
        allowed_tags = set(candidate_manifest["run_tag"].astype(str))
        summary = summary[summary["run_tag"].astype(str).isin(allowed_tags)].copy()
        if summary.empty:
            raise RuntimeError("The candidate manifest has no eligible runs in the internal summary.")
        effective_pool_size = max(args.max_pool_size, len(summary))
    else:
        effective_pool_size = args.max_pool_size

    summary["single_rank_score"] = (
        0.45 * pd.to_numeric(summary["overall_bacc"], errors="coerce")
        + 0.35 * pd.to_numeric(summary["min_source_bacc"], errors="coerce")
        + 0.20 * pd.to_numeric(summary["overall_auc"], errors="coerce")
    )
    ranked = summary.sort_values(
        ["single_rank_score", "min_source_bacc", "overall_bacc", "overall_auc"],
        ascending=False,
    )
    diverse_tags: list[str] = []
    for _, model_rows in ranked.groupby("model_name", sort=False):
        tag = str(model_rows.iloc[0]["run_tag"])
        if tag not in diverse_tags:
            diverse_tags.append(tag)
    pool_tags = diverse_tags.copy()
    for tag in ranked["run_tag"].astype(str):
        if tag not in pool_tags:
            pool_tags.append(tag)
    pool_tags = pool_tags[: max(effective_pool_size, len(diverse_tags))]
    (output_dir / "fusion_candidate_pool.txt").write_text("\n".join(pool_tags) + "\n", encoding="utf-8")

    prediction_cache = {
        tag: load_prediction(runs_root, tag, f"prob_{index}") for index, tag in enumerate(pool_tags)
    }
    lodo_cache: dict[str, pd.DataFrame] = {}
    if args.lodo_runs_root:
        lodo_root = Path(args.lodo_runs_root)
        for index, tag in enumerate(pool_tags):
            path = lodo_root / tag / "oof_predictions.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing source-LODO prediction for locked candidate: {path}")
            lodo_cache[tag] = load_prediction(lodo_root, tag, f"prob_{index}")
    rows: list[dict[str, object]] = []
    for member_count in range(1, min(args.max_members, len(pool_tags)) + 1):
        for members in itertools.combinations(pool_tags, member_count):
            fused = prediction_cache[members[0]]
            for tag in members[1:]:
                fused = fused.merge(prediction_cache[tag], on=KEY_COLUMNS, how="inner", validate="one_to_one")
            row, predictions = evaluate_recipe(fused, members)
            if lodo_cache:
                lodo_fused = lodo_cache[members[0]]
                for tag in members[1:]:
                    lodo_fused = lodo_fused.merge(
                        lodo_cache[tag], on=KEY_COLUMNS, how="inner", validate="one_to_one"
                    )
                lodo_row, _ = evaluate_recipe(lodo_fused, members)
                row["oof_selection_score"] = row["selection_score"]
                for key, value in lodo_row.items():
                    if key not in {"recipe", "members", "member_count"}:
                        row[f"lodo_{key}"] = value
                lodo_weight = min(max(float(args.lodo_weight), 0.0), 1.0)
                row["selection_score"] = (
                    (1.0 - lodo_weight) * float(row["oof_selection_score"])
                    + lodo_weight * float(lodo_row["selection_score"])
                )
            rows.append(row)

    grid = pd.DataFrame(rows)
    eligible = grid[
        pd.to_numeric(grid["min_class_recall"], errors="coerce") >= args.min_class_recall
    ].copy()
    if lodo_cache:
        eligible = eligible[
            pd.to_numeric(eligible["lodo_min_class_recall"], errors="coerce")
            >= args.min_lodo_class_recall
        ].copy()
    if eligible.empty:
        raise RuntimeError("No fusion recipe met the minimum sensitivity/specificity guardrail.")
    eligible = eligible.sort_values(
        ["selection_score", "min_source_bacc", "balanced_accuracy", "auc", "member_count"],
        ascending=[False, False, False, False, True],
    )
    grid.sort_values(
        ["selection_score", "min_source_bacc", "balanced_accuracy"], ascending=False
    ).to_csv(output_dir / "dense_oof_fusion_grid.csv", index=False, encoding="utf-8-sig")

    best = eligible.iloc[0]
    members = str(best["members"]).split(",")
    manifest = pd.DataFrame(
        [
            {
                "member_rank": index + 1,
                "run_tag": tag,
                "recipe": best["recipe"],
                "selection_score": best["selection_score"],
                "internal_overall_bacc": best["balanced_accuracy"],
                "internal_overall_auc": best["auc"],
                "internal_min_source_bacc": best["min_source_bacc"],
                "internal_min_fold_bacc": best["min_fold_bacc"],
                "internal_min_class_recall": best["min_class_recall"],
                "source_lodo_bacc": best.get("lodo_balanced_accuracy", np.nan),
                "source_lodo_auc": best.get("lodo_auc", np.nan),
                "source_lodo_min_source_bacc": best.get("lodo_min_source_bacc", np.nan),
                "source_lodo_min_class_recall": best.get("lodo_min_class_recall", np.nan),
            }
            for index, tag in enumerate(members)
        ]
    )
    manifest_path = output_dir / "LOCKED_INTERNAL_DENSE_FUSION_MEMBERS.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, encoding="utf-8-sig")
        if existing.to_dict(orient="records") != manifest.to_dict(orient="records"):
            raise RuntimeError("A different locked internal fusion already exists; refusing to overwrite it.")
    else:
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    locked_predictions = prediction_cache[members[0]]
    for tag in members[1:]:
        locked_predictions = locked_predictions.merge(
            prediction_cache[tag], on=KEY_COLUMNS, how="inner", validate="one_to_one"
        )
    _, locked_predictions = evaluate_recipe(locked_predictions, tuple(members))
    locked_predictions.to_csv(
        output_dir / "locked_internal_dense_fusion_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if lodo_cache:
        locked_lodo_predictions = lodo_cache[members[0]]
        for tag in members[1:]:
            locked_lodo_predictions = locked_lodo_predictions.merge(
                lodo_cache[tag], on=KEY_COLUMNS, how="inner", validate="one_to_one"
            )
        _, locked_lodo_predictions = evaluate_recipe(locked_lodo_predictions, tuple(members))
        locked_lodo_predictions.to_csv(
            output_dir / "locked_internal_dense_fusion_source_lodo_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print("[candidate-pool]", pool_tags, flush=True)
    print("[top-fusions]\n" + eligible.head(15).to_string(index=False), flush=True)
    print("[locked-members]\n" + manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
