from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score


VALUE_ARGUMENTS = [
    "split_csv",
    "concept_csv",
    "concept_columns",
    "pooling",
    "expert_mode",
    "risk_objective",
    "hidden_dim",
    "attention_dim",
    "dropout",
    "epochs",
    "patience",
    "batch_size",
    "num_workers",
    "lr",
    "weight_decay",
    "grad_clip",
    "subtype_loss_weight",
    "ordinal_loss_weight",
    "concept_loss_weight",
    "prototype_loss_weight",
    "boundary_loss_weight",
    "boundary_relevance_loss_weight",
    "boundary_triplet_weight",
    "boundary_triplet_margin",
    "boundary_fusion_alpha",
    "prototype_temperature",
    "risk_from_subtype_alpha",
    "rex_weight",
    "group_dro_eta",
    "moe_specialist_weight",
    "moe_balance_weight",
    "moe_gate_supervision_weight",
    "soft_balanced_loss_weight",
    "focal_gamma",
    "visual_conflict_softening",
    "domain_adversarial_weight",
    "domain_adversarial_lambda",
    "class_conditional_align_weight",
    "sentinel_fusion_alpha",
    "sentinel_loss_weight",
    "sentinel_positive_weight",
    "sentinel_positive_gamma",
    "sentinel_negative_gamma",
    "mixstyle_probability",
    "mixstyle_alpha",
    "view_consistency_weight",
    "view_supervision_weight",
    "device",
    "seed",
]
BOOLEAN_ARGUMENTS = ["class_weighting", "subtype_balanced_sampler", "load_features_to_ram"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock and run source-held-out tests for internal dense candidates.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-candidates", type=int, default=15)
    parser.add_argument("--min-overall-bacc", type=float, default=0.67)
    parser.add_argument("--min-source-bacc", type=float, default=0.60)
    parser.add_argument("--complementary-pool-size", type=int, default=20)
    parser.add_argument("--max-complementary-recipes", type=int, default=8)
    parser.add_argument("--max-complementary-members", type=int, default=10)
    parser.add_argument("--min-fusion-class-recall", type=float, default=0.60)
    return parser.parse_args()


def rank_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["min_class_recall"] = ranked[["overall_sensitivity", "overall_specificity"]].min(axis=1)
    ranked["rank_score"] = (
        0.35 * ranked["overall_bacc"]
        + 0.30 * ranked["min_source_bacc"]
        + 0.20 * ranked["overall_auc"]
        + 0.15 * ranked["min_class_recall"]
    )
    return ranked.sort_values(
        ["rank_score", "min_source_bacc", "overall_bacc", "overall_auc"], ascending=False
    )


def evaluate_fusion(frame: pd.DataFrame, probability_columns: list[str]) -> dict[str, float]:
    probability = frame[probability_columns].mean(axis=1).to_numpy()
    label = frame["label_idx"].to_numpy(dtype=int)
    predicted = (probability >= 0.5).astype(int)
    sensitivity = float(recall_score(label, predicted, pos_label=1, zero_division=0))
    specificity = float(recall_score(label, predicted, pos_label=0, zero_division=0))
    source_bacc = [
        float(balanced_accuracy_score(group["label_idx"], (group[probability_columns].mean(axis=1) >= 0.5)))
        for _, group in frame.groupby("source_dataset")
    ]
    fold_bacc = [
        float(balanced_accuracy_score(group["label_idx"], (group[probability_columns].mean(axis=1) >= 0.5)))
        for _, group in frame.groupby("fold_id")
    ]
    overall_bacc = float(balanced_accuracy_score(label, predicted))
    overall_auc = float(roc_auc_score(label, probability))
    min_class_recall = min(sensitivity, specificity)
    selection_score = (
        0.30 * overall_bacc
        + 0.25 * min(source_bacc)
        + 0.20 * min_class_recall
        + 0.15 * overall_auc
        + 0.10 * min(fold_bacc)
    )
    return {
        "balanced_accuracy": overall_bacc,
        "auc": overall_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "min_class_recall": min_class_recall,
        "min_source_bacc": min(source_bacc),
        "min_fold_bacc": min(fold_bacc),
        "selection_score": selection_score,
    }


def find_complementary_members(
    ranked: pd.DataFrame,
    runs_root: Path,
    pool_size: int,
    maximum_recipes: int,
    maximum_members: int,
    min_class_recall: float,
) -> tuple[list[str], pd.DataFrame]:
    pool = ranked.head(max(2, pool_size)).copy()
    tags = pool["run_tag"].astype(str).tolist()
    merged: pd.DataFrame | None = None
    probability_by_tag: dict[str, str] = {}
    key_columns = ["case_id", "label_idx", "fold_id", "source_dataset"]
    for index, tag in enumerate(tags):
        path = runs_root / tag / "oof_predictions.csv"
        prediction = pd.read_csv(path, encoding="utf-8-sig", dtype={"case_id": str})
        missing = set(key_columns + ["prob_high"]) - set(prediction.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        probability_column = f"probability_{index}"
        probability_by_tag[tag] = probability_column
        prediction = prediction[key_columns + ["prob_high"]].rename(columns={"prob_high": probability_column})
        merged = prediction if merged is None else merged.merge(
            prediction, on=key_columns, how="inner", validate="one_to_one"
        )
    if merged is None or len(merged) == 0:
        raise RuntimeError("No aligned OOF predictions were available for complementarity screening.")

    rows: list[dict[str, object]] = []
    for member_count in (2, 3):
        for members in combinations(tags, member_count):
            metrics = evaluate_fusion(merged, [probability_by_tag[tag] for tag in members])
            rows.append({"members": "+".join(members), "member_count": member_count, **metrics})
    grid = pd.DataFrame(rows).sort_values(
        ["selection_score", "min_source_bacc", "balanced_accuracy", "auc"], ascending=False
    )
    eligible = grid[grid["min_class_recall"] >= min_class_recall].head(maximum_recipes)
    members: list[str] = []
    for recipe in eligible["members"].astype(str):
        for tag in recipe.split("+"):
            if tag not in members and len(members) < maximum_members:
                members.append(tag)
    return members, grid


def choose_candidates(
    summary: pd.DataFrame,
    maximum: int,
    complementary_tags: list[str],
) -> list[tuple[str, pd.Series]]:
    selected: list[tuple[str, pd.Series]] = []
    used: set[str] = set()

    def add(reason: str, row: pd.Series) -> None:
        tag = str(row["run_tag"])
        if tag not in used and len(selected) < maximum:
            selected.append((reason, row))
            used.add(tag)

    ranked = rank_candidates(summary)
    add("best_combined_internal", ranked.iloc[0])
    add("best_worst_source", summary.sort_values("min_source_bacc", ascending=False).iloc[0])
    add("best_class_balance", ranked.sort_values("min_class_recall", ascending=False).iloc[0])
    add("best_internal_auc", summary.sort_values("overall_auc", ascending=False).iloc[0])
    by_tag = ranked.set_index("run_tag", drop=False)
    for tag in complementary_tags:
        add("best_oof_complementary_fusion_member", by_tag.loc[tag])
    for _, model_rows in ranked.groupby("model_name", sort=False):
        add("best_distinct_backbone", model_rows.iloc[0])
    for _, row in ranked.iterrows():
        add("next_internal_candidate", row)
    return selected


def command_for(config: dict, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "/root/thymic_queue_scripts_20260711/run_task7_dense_feature_cv_20260711.py",
        "--feature-bank-dir",
        str(config["feature_bank_dir"]),
        "--output-dir",
        str(output_dir),
        "--split-mode",
        "source_lodo",
        "--fold",
        "all",
    ]
    for name in VALUE_ARGUMENTS:
        if name not in config or config[name] is None:
            continue
        value = config[name]
        if name == "concept_columns" and isinstance(value, list):
            value = ",".join(str(item) for item in value)
        command.extend(["--" + name.replace("_", "-"), str(value)])
    for name in BOOLEAN_ARGUMENTS:
        enabled = bool(config.get(name, False))
        command.append(("--" if enabled else "--no-") + name.replace("_", "-"))
    return command


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary_csv, encoding="utf-8-sig")
    numeric_columns = [
        "overall_bacc",
        "overall_auc",
        "overall_sensitivity",
        "overall_specificity",
        "min_source_bacc",
    ]
    for column in numeric_columns:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    summary = summary[
        (summary["overall_bacc"] >= args.min_overall_bacc)
        & (summary["min_source_bacc"] >= args.min_source_bacc)
        & ~summary["views"].astype(str).str.contains("background_only", na=False)
        & ~summary["run_tag"].astype(str).str.startswith("210_")
    ].copy()
    if summary.empty:
        raise RuntimeError("No eligible dense run for source-held-out evaluation.")
    ranked = rank_candidates(summary)
    complementary_tags, complementary_grid = find_complementary_members(
        ranked=ranked,
        runs_root=runs_root,
        pool_size=args.complementary_pool_size,
        maximum_recipes=args.max_complementary_recipes,
        maximum_members=args.max_complementary_members,
        min_class_recall=args.min_fusion_class_recall,
    )
    complementary_grid.to_csv(
        output_root / "INTERNAL_OOF_COMPLEMENTARITY_GRID.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {"member_rank": range(1, len(complementary_tags) + 1), "run_tag": complementary_tags}
    ).to_csv(
        output_root / "INTERNAL_OOF_COMPLEMENTARY_MEMBERS.csv", index=False, encoding="utf-8-sig"
    )
    print("[oof-complementary-members] " + ", ".join(complementary_tags), flush=True)
    # This dated lock protocol always reserves enough slots for complementary OOF
    # members even when an older queue wrapper still passes the former cap of six.
    effective_maximum = min(len(summary), max(args.max_candidates, 15))
    if effective_maximum != args.max_candidates:
        print(
            f"[expand-shortlist] requested={args.max_candidates} effective={effective_maximum} "
            "to preserve complementarity coverage",
            flush=True,
        )
    selected = choose_candidates(summary, effective_maximum, complementary_tags)
    manifest = pd.DataFrame(
        [
            {
                "candidate_rank": index + 1,
                "selection_reason": reason,
                "run_tag": str(row["run_tag"]),
                "model_name": str(row["model_name"]),
                "internal_overall_bacc": float(row["overall_bacc"]),
                "internal_min_source_bacc": float(row["min_source_bacc"]),
            }
            for index, (reason, row) in enumerate(selected)
        ]
    )
    manifest_path = output_root / "LOCKED_INTERNAL_SOURCE_LODO_SHORTLIST.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, encoding="utf-8-sig")
        if existing.to_dict(orient="records") != manifest.to_dict(orient="records"):
            raise RuntimeError("A different source-LODO shortlist is already locked.")
    else:
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print("[locked-source-lodo]\n" + manifest.to_string(index=False), flush=True)

    for row in manifest.to_dict(orient="records"):
        tag = str(row["run_tag"])
        output_dir = output_root / tag
        if (output_dir / "oof_metrics.csv").exists():
            print(f"[skip-source-lodo] {tag}", flush=True)
            continue
        config = json.loads((runs_root / tag / "run_config.json").read_text(encoding="utf-8"))
        command = command_for(config, output_dir)
        print("[command] " + " ".join(command), flush=True)
        subprocess.run(command, check=True)

    subprocess.run(
        [
            sys.executable,
            "/workspace/thymic_project/scripts/summarize_dense_capability_screen_20260711.py",
            "--runs-root",
            str(output_root),
            "--output-csv",
            str(output_root / "dense_source_lodo_summary.csv"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
