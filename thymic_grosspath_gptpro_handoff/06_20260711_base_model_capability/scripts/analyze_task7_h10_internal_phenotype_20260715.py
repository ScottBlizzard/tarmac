from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


METHODS = {
    "INTERNAL_NATURAL": "natural_predictions",
    "INTERNAL_RISK_BALANCED": "risk_predictions",
    "INTERNAL_SUBTYPE_TEMPERED": "tempered_predictions",
}

HIGH_RISK_CONCEPTS = [
    "boundary_unclear",
    "capsule_absent",
    "capsule_involved",
    "invasion",
    "fat_involved_or_attached",
    "lung_attached",
    "pericardium_attached",
    "pleura_attached",
    "necrosis",
    "hemorrhage",
    "cystic_change",
    "nodular_lobulated",
    "texture_tough",
]
LOW_RISK_CONCEPTS = [
    "boundary_clear",
    "capsule_complete",
    "texture_soft",
    "homogeneous",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze H10 internal H3 baselines.")
    parser.add_argument("--natural-predictions", required=True)
    parser.add_argument("--risk-predictions", required=True)
    parser.add_argument("--tempered-predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--concept-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_predictions(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    required = {
        "case_id",
        "source_dataset",
        "task_l6_label",
        "label_idx",
        "fold_id",
        "prob_high",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing {method} columns: {sorted(required - set(frame.columns))}")
    if len(frame) != 591 or frame["case_id"].duplicated().any():
        raise ValueError(f"{method} does not contain 591 unique OOF cases")
    frame = frame.copy()
    frame["label_idx"] = pd.to_numeric(frame["label_idx"], errors="raise").astype(int)
    frame["prob_high"] = pd.to_numeric(frame["prob_high"], errors="raise").astype(float)
    frame["pred_idx"] = (frame["prob_high"] >= 0.5).astype(int)
    frame["correct"] = frame["pred_idx"].eq(frame["label_idx"])
    return frame.sort_values("case_id").reset_index(drop=True)


def metric_record(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "n": int(len(labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "auc": float(roc_auc_score(labels, probability)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def balanced_accuracy(labels: np.ndarray, probability: np.ndarray) -> float:
    prediction = np.asarray(probability) >= 0.5
    low = np.asarray(labels) == 0
    high = ~low
    return float((np.mean(~prediction[low]) + np.mean(prediction[high])) / 2)


def bootstrap_difference(
    reference: pd.DataFrame,
    left_probability: np.ndarray,
    right_probability: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = reference["label_idx"].to_numpy(int)
    strata = [
        group.index.to_numpy(int)
        for _, group in reference.reset_index(drop=True).groupby("task_l6_label", sort=True)
    ]
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = np.concatenate(
            [rng.choice(rows, size=len(rows), replace=True) for rows in strata]
        )
        differences[replicate] = balanced_accuracy(
            labels[sampled], left_probability[sampled]
        ) - balanced_accuracy(labels[sampled], right_probability[sampled])
    observed = balanced_accuracy(labels, left_probability) - balanced_accuracy(
        labels, right_probability
    )
    return {
        "delta_balanced_accuracy": float(observed),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "probability_delta_gt_zero": float(np.mean(differences > 0)),
        "replicates": int(replicates),
    }


def physician_pattern(row: pd.Series) -> str:
    if pd.isna(row["gross_highrisk_score"]):
        return "missing"
    has_high = row["physician_high_concept_count"] > 0
    has_low = row["physician_low_concept_count"] > 0
    if has_high and has_low:
        return "mixed"
    if not has_high and not has_low:
        return "uninformative"
    concepts_point_high = has_high and not has_low
    return "canonical" if concepts_point_high == bool(row["label_idx"]) else "discordant"


def diagnostic_role(row: pd.Series) -> str:
    correct_count = int(row["model_correct_count"])
    pattern = str(row["physician_pattern"])
    if correct_count == 3:
        return "canonical_anchor" if pattern == "canonical" else "stable_noncanonical"
    if correct_count in {1, 2}:
        return "learnable_boundary"
    if pattern == "canonical":
        return "persistent_canonical_failure"
    if pattern in {"mixed", "discordant"}:
        return "persistent_mimic_failure"
    return "persistent_sparse_or_missing"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for method, argument_name in METHODS.items():
        frames[method] = load_predictions(Path(getattr(args, argument_name)), method)

    reference = frames["INTERNAL_NATURAL"]
    identity_columns = ["case_id", "label_idx", "task_l6_label", "source_dataset", "fold_id"]
    for method, frame in frames.items():
        if not frame[identity_columns].equals(reference[identity_columns]):
            raise ValueError(f"OOF identity mismatch for {method}")

    split = pd.read_csv(Path(args.split_csv), dtype={"case_id": str}, encoding="utf-8-sig")
    split = split.sort_values("case_id").reset_index(drop=True)
    if not split["case_id"].equals(reference["case_id"]):
        raise ValueError("H10 split and predictions have different case order")
    if not np.array_equal(split["master_fold_id"].to_numpy(int), reference["fold_id"].to_numpy(int)):
        raise ValueError("H10 OOF folds differ from the subtype-only split")

    overall_rows = []
    subtype_rows = []
    source_rows = []
    fold_rows = []
    probabilities: dict[str, np.ndarray] = {}
    for method, frame in frames.items():
        probability = frame["prob_high"].to_numpy(float)
        probabilities[method] = probability
        overall_rows.append({"method": method, **metric_record(frame["label_idx"], probability)})
        for subtype, group in frame.groupby("task_l6_label", sort=True):
            subtype_rows.append(
                {
                    "method": method,
                    "task_l6_label": subtype,
                    "n": int(len(group)),
                    "correct_n": int(group["correct"].sum()),
                    "accuracy": float(group["correct"].mean()),
                }
            )
        for source, group in frame.groupby("source_dataset", sort=True):
            source_rows.append(
                {"method": method, "source_dataset": source, **metric_record(group["label_idx"], group["prob_high"])}
            )
        for fold_id, group in frame.groupby("fold_id", sort=True):
            fold_rows.append(
                {"method": method, "fold_id": int(fold_id), **metric_record(group["label_idx"], group["prob_high"])}
            )

    overall = pd.DataFrame(overall_rows).sort_values("balanced_accuracy", ascending=False)
    subtype = pd.DataFrame(subtype_rows)
    source = pd.DataFrame(source_rows)
    folds = pd.DataFrame(fold_rows)
    comparisons = []
    names = list(METHODS)
    comparison_seed = args.bootstrap_seed
    for left_index in range(len(names)):
        for right_index in range(left_index + 1, len(names)):
            left = names[left_index]
            right = names[right_index]
            comparisons.append(
                {
                    "comparison": f"{left}_MINUS_{right}",
                    **bootstrap_difference(
                        reference,
                        probabilities[left],
                        probabilities[right],
                        args.bootstrap_replicates,
                        comparison_seed,
                    ),
                }
            )
            comparison_seed += 1
    bootstrap = pd.DataFrame(comparisons)

    metadata = pd.read_csv(
        Path(args.metadata),
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    metadata = metadata.sort_values("case_id").reset_index(drop=True)
    if not metadata["case_id"].equals(reference["case_id"]):
        raise ValueError("H10 metadata and predictions have different case order")
    concept_columns = [
        "original_case_id",
        "gross_highrisk_score",
        "gross_conflict_score",
        *HIGH_RISK_CONCEPTS,
        *LOW_RISK_CONCEPTS,
    ]
    concept = pd.read_csv(
        Path(args.concept_csv),
        usecols=concept_columns,
        dtype={"original_case_id": str},
        encoding="utf-8-sig",
    ).drop_duplicates("original_case_id")
    diagnostic = reference[identity_columns].merge(
        metadata[["case_id", "original_case_id"]], on="case_id", how="left", validate="one_to_one"
    ).merge(concept, on="original_case_id", how="left", validate="one_to_one")
    diagnostic["physician_high_concept_count"] = diagnostic[HIGH_RISK_CONCEPTS].fillna(0).sum(axis=1)
    diagnostic["physician_low_concept_count"] = diagnostic[LOW_RISK_CONCEPTS].fillna(0).sum(axis=1)
    diagnostic["physician_pattern"] = diagnostic.apply(physician_pattern, axis=1)
    for method in METHODS:
        diagnostic[f"prob_{method}"] = probabilities[method]
        diagnostic[f"correct_{method}"] = (
            (probabilities[method] >= 0.5).astype(int) == diagnostic["label_idx"].to_numpy(int)
        ).astype(int)
    correctness_columns = [f"correct_{method}" for method in METHODS]
    diagnostic["model_correct_count"] = diagnostic[correctness_columns].sum(axis=1)
    diagnostic["diagnostic_role"] = diagnostic.apply(diagnostic_role, axis=1)

    physician_rows = []
    for method in METHODS:
        for pattern, group in diagnostic.groupby("physician_pattern", sort=True):
            physician_rows.append(
                {
                    "method": method,
                    "physician_pattern": pattern,
                    "n": int(len(group)),
                    "correct_n": int(group[f"correct_{method}"].sum()),
                    "accuracy": float(group[f"correct_{method}"].mean()),
                }
            )
    physician_metrics = pd.DataFrame(physician_rows)
    role_counts = (
        diagnostic.groupby(["diagnostic_role", "task_l6_label"], sort=True)
        .size()
        .rename("n")
        .reset_index()
    )
    role_totals = diagnostic["diagnostic_role"].value_counts().rename_axis("diagnostic_role").reset_index(name="n")
    concept_valid = diagnostic["gross_highrisk_score"].notna()
    concept_auc = float(
        roc_auc_score(
            diagnostic.loc[concept_valid, "label_idx"],
            diagnostic.loc[concept_valid, "gross_highrisk_score"],
        )
    )

    top_method = str(overall.iloc[0]["method"])
    second_method = str(overall.iloc[1]["method"])
    top_comparison = bootstrap[
        bootstrap["comparison"].isin(
            [f"{top_method}_MINUS_{second_method}", f"{second_method}_MINUS_{top_method}"]
        )
    ].iloc[0].to_dict()
    if top_comparison["comparison"].startswith(second_method):
        top_delta = -float(top_comparison["delta_balanced_accuracy"])
        top_ci_low = -float(top_comparison["ci_high"])
        top_ci_high = -float(top_comparison["ci_low"])
    else:
        top_delta = float(top_comparison["delta_balanced_accuracy"])
        top_ci_low = float(top_comparison["ci_low"])
        top_ci_high = float(top_comparison["ci_high"])
    clear_winner = top_delta >= 0.0100 and top_ci_low > 0
    decision = "CLEAR_INTERNAL_SAMPLER_WINNER" if clear_winner else "NO_CLEAR_INTERNAL_SAMPLER_WINNER"

    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    subtype.to_csv(output_dir / "subtype_metrics.csv", index=False)
    source.to_csv(output_dir / "source_audit_metrics.csv", index=False)
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    physician_metrics.to_csv(output_dir / "physician_pattern_metrics.csv", index=False)
    role_counts.to_csv(output_dir / "diagnostic_role_by_subtype.csv", index=False)
    role_totals.to_csv(output_dir / "diagnostic_role_totals.csv", index=False)
    diagnostic.to_csv(
        output_dir / "server_only_internal_oof_physician_model_roles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "experiment": "H10_INTERNAL_PHENOTYPE_DIFFICULTY_REDESIGN_20260715",
        "decision": decision,
        "top_method": top_method,
        "second_method": second_method,
        "top_minus_second": {
            "delta_balanced_accuracy": top_delta,
            "ci_low": top_ci_low,
            "ci_high": top_ci_high,
        },
        "physician_concept_coverage": int(concept_valid.sum()),
        "physician_concept_auc": concept_auc,
        "diagnostic_role_totals": dict(zip(role_totals["diagnostic_role"], role_totals["n"])),
        "next_step": "recreate model roles with nested OOF inside each outer training fold",
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "RUN.status").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("\nOVERALL\n" + overall.to_string(index=False), flush=True)
    print("\nSUBTYPE\n" + subtype.to_string(index=False), flush=True)
    print("\nDIAGNOSTIC ROLES\n" + role_totals.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
