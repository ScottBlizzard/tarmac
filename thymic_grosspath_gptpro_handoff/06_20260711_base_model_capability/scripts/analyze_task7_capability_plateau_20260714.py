#!/usr/bin/env python3
"""Diagnose the Task7 full-coverage capability plateau without training a new model.

The script aligns representative source-LODO predictions at case level and writes
aggregate diagnostics for dataset confounding, model complementarity, persistent
errors, threshold transport, and retrospective gross-description concepts.

The case-level error matrix contains identifiers and is intentionally meant to stay
on the project server. Only aggregate findings should be copied into the repository.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import roc_auc_score


DEFAULT_REGISTRY = "/root/thymic_task7_internal_master_registry_cached_max2048_20260711.csv"
DEFAULT_CONCEPTS = (
    "/workspace/thymic_project/outputs/grosspath_rc_v0_20260526/gross_concepts_v1.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/workspace/thymic_project/experiments/task7_capability_plateau_audit_20260714"
)
DEFAULT_PREDICTIONS = {
    "C1": (
        "/workspace/thymic_project/experiments/base_model_capability_20260711/"
        "phase2_siglipl512_local_pyramid_screen/"
        "348_siglipl512_localpyramid6_gated_source_lodo_cw_20260711/"
        "oof_predictions.csv"
    ),
    "C2": (
        "/workspace/thymic_project/experiments/base_model_capability_20260711/"
        "phase2_siglipl512_localpyramid_plus_aimmixstyle_internal_fusion/"
        "lodo_predictions.csv"
    ),
    "H3": (
        "/workspace/thymic_project/experiments/h3_representation_renewal_20260713/"
        "h3b_runs/pe_spatial_l14_448/source_lodo/oof_predictions.csv"
    ),
    "H5": (
        "/workspace/thymic_project/experiments/h5_second_order_texture_20260713/"
        "primary_seed20260713/source_lodo/oof_predictions.csv"
    ),
    "H6": (
        "/workspace/thymic_project/experiments/h6_nuisance_anchored_csd_20260714/"
        "source_lodo/nuisance_csd/oof_predictions.csv"
    ),
    "H7": (
        "/workspace/thymic_project/experiments/h7_pe_embedding_lisa_20260714/"
        "source_lodo/oof_predictions.csv"
    ),
}

CONCEPTS = [
    "boundary_clear",
    "boundary_unclear",
    "capsule_any",
    "capsule_complete",
    "capsule_absent",
    "invasion",
    "fat_involved_or_attached",
    "lung_attached",
    "pericardium_attached",
    "pleura_attached",
    "hemorrhage",
    "necrosis",
    "cystic_change",
    "calcification",
    "nodular_lobulated",
    "septum",
    "homogeneous",
    "gray_white",
    "gray_yellow",
    "gray_red",
    "gray_brown",
    "gray_black",
    "texture_soft",
    "texture_medium",
    "texture_tough",
    "texture_fragile",
    "cut_surface_mentioned",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--concept-csv", default=DEFAULT_CONCEPTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        help="Optional TAG=/absolute/path.csv. Defaults to C1,C2,H3,H5,H6,H7.",
    )
    return parser.parse_args()


def strip_bom_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    return frame


def parse_prediction_specs(specifications: list[str]) -> dict[str, str]:
    if not specifications:
        return DEFAULT_PREDICTIONS.copy()
    parsed: dict[str, str] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Prediction must be TAG=PATH: {specification}")
        tag, path = specification.split("=", 1)
        tag = tag.strip()
        if not tag or tag in parsed:
            raise ValueError(f"Invalid or duplicate prediction tag: {tag!r}")
        parsed[tag] = path.strip()
    if len(parsed) < 2:
        raise ValueError("At least two prediction files are required for complementarity analysis")
    return parsed


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    prediction = (probabilities >= threshold).astype(int)
    negative = y_true == 0
    positive = y_true == 1
    tn = int(np.sum(negative & (prediction == 0)))
    fp = int(np.sum(negative & (prediction == 1)))
    fn = int(np.sum(positive & (prediction == 0)))
    tp = int(np.sum(positive & (prediction == 1)))
    specificity = tn / (tn + fp) if tn + fp else np.nan
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    balanced_accuracy = (
        (specificity + sensitivity) / 2
        if np.isfinite(specificity) and np.isfinite(sensitivity)
        else np.nan
    )
    auc = (
        float(roc_auc_score(y_true, probabilities))
        if np.unique(y_true).size == 2
        else np.nan
    )
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(np.mean(prediction == y_true)),
        "balanced_accuracy": float(balanced_accuracy),
        "auc": auc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(prediction)),
        "mean_probability": float(np.mean(probabilities)),
        "brier": float(np.mean((probabilities - y_true) ** 2)),
    }


def best_balanced_accuracy_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict]:
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scored = []
    for threshold in candidates:
        metrics = binary_metrics(y_true, probabilities, float(threshold))
        score = metrics["balanced_accuracy"]
        if np.isfinite(score):
            scored.append((float(score), -abs(float(threshold) - 0.5), float(threshold), metrics))
    if not scored:
        return 0.5, binary_metrics(y_true, probabilities, 0.5)
    _, _, threshold, metrics = max(scored, key=lambda row: (row[0], row[1]))
    return threshold, metrics


def cramers_v(table: pd.DataFrame) -> dict:
    observed = table.to_numpy(dtype=float)
    chi2, p_value, degrees_freedom, _ = chi2_contingency(observed)
    denominator = observed.sum() * min(observed.shape[0] - 1, observed.shape[1] - 1)
    value = math.sqrt(chi2 / denominator) if denominator > 0 else np.nan
    return {
        "chi2": float(chi2),
        "degrees_freedom": int(degrees_freedom),
        "p_value": float(p_value),
        "cramers_v_uncorrected": float(value),
        "n": int(observed.sum()),
    }


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    count = len(ordered)
    adjusted = ordered.to_numpy() * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.clip(adjusted, 0.0, 1.0)
    return result


def load_registry(path: str) -> pd.DataFrame:
    frame = strip_bom_columns(
        pd.read_csv(path, dtype={"case_id": str, "original_case_id": str}, encoding="utf-8-sig")
    )
    required = {"case_id", "original_case_id", "source_dataset", "task_l6_label", "label_idx"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Registry lacks columns: {sorted(missing)}")
    if frame["case_id"].duplicated().any():
        raise ValueError("Registry contains duplicate case_id values")
    frame["label_idx"] = pd.to_numeric(frame["label_idx"], errors="raise").astype(int)
    frame["original_case_id"] = frame["original_case_id"].astype(str).str.strip()
    return frame


def align_predictions(registry: pd.DataFrame, specifications: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    keep = ["case_id", "original_case_id", "source_dataset", "task_l6_label", "label_idx"]
    aligned = registry[keep].copy()
    tags = list(specifications)
    for tag, path in specifications.items():
        prediction = strip_bom_columns(
            pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
        )
        required = {"case_id", "prob_high"}
        missing = required - set(prediction.columns)
        if missing:
            raise ValueError(f"{tag} lacks columns: {sorted(missing)}")
        if prediction["case_id"].duplicated().any():
            raise ValueError(f"{tag} contains duplicate case_id values")
        probability = pd.to_numeric(prediction["prob_high"], errors="raise")
        if probability.lt(0).any() or probability.gt(1).any():
            raise ValueError(f"{tag} probabilities are outside [0, 1]")
        candidate = prediction[["case_id"]].copy()
        candidate[f"prob_{tag}"] = probability
        aligned = aligned.merge(candidate, on="case_id", how="left", validate="one_to_one")
        if aligned[f"prob_{tag}"].isna().any():
            missing_count = int(aligned[f"prob_{tag}"].isna().sum())
            raise ValueError(f"{tag} is missing {missing_count} registry cases")
        extra = set(prediction["case_id"]) - set(registry["case_id"])
        if extra:
            raise ValueError(f"{tag} contains {len(extra)} cases absent from the registry")
        aligned[f"pred_{tag}"] = (aligned[f"prob_{tag}"] >= 0.5).astype(int)
        aligned[f"wrong_{tag}"] = (aligned[f"pred_{tag}"] != aligned["label_idx"]).astype(int)
    return aligned, tags


def dataset_structure(registry: pd.DataFrame, output_dir: Path) -> dict:
    source_subtype = pd.crosstab(registry["source_dataset"], registry["task_l6_label"])
    source_risk = pd.crosstab(registry["source_dataset"], registry["label_idx"])
    source_subtype.to_csv(output_dir / "dataset_source_by_subtype.csv", encoding="utf-8-sig")
    source_risk.to_csv(output_dir / "dataset_source_by_risk.csv", encoding="utf-8-sig")

    source_summary = (
        registry.groupby("source_dataset", observed=True)
        .agg(n=("case_id", "size"), high_risk_n=("label_idx", "sum"))
        .reset_index()
    )
    source_summary["low_risk_n"] = source_summary["n"] - source_summary["high_risk_n"]
    source_summary["high_risk_rate"] = source_summary["high_risk_n"] / source_summary["n"]
    source_summary.to_csv(output_dir / "dataset_source_risk_summary.csv", index=False, encoding="utf-8-sig")

    subtype_summary = (
        registry.groupby("task_l6_label", observed=True)
        .agg(n=("case_id", "size"), high_risk_rate=("label_idx", "mean"))
        .reset_index()
    )
    subtype_summary.to_csv(output_dir / "dataset_subtype_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "source_vs_risk": cramers_v(source_risk),
        "source_vs_subtype": cramers_v(source_subtype),
        "source_summary": source_summary.to_dict(orient="records"),
        "subtype_summary": subtype_summary.to_dict(orient="records"),
        "empty_source_subtype_cells": int((source_subtype == 0).sum().sum()),
        "total_source_subtype_cells": int(source_subtype.size),
    }


def model_stratum_metrics(aligned: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    strata: list[tuple[str, str, pd.Series]] = [
        ("overall", "all", pd.Series(True, index=aligned.index))
    ]
    for source in sorted(aligned["source_dataset"].unique()):
        strata.append(("source", str(source), aligned["source_dataset"].eq(source)))
    for subtype in sorted(aligned["task_l6_label"].unique()):
        strata.append(("subtype", str(subtype), aligned["task_l6_label"].eq(subtype)))
    for tag in tags:
        for stratum_type, stratum, mask in strata:
            subset = aligned.loc[mask]
            metrics = binary_metrics(
                subset["label_idx"].to_numpy(), subset[f"prob_{tag}"].to_numpy(), 0.5
            )
            rows.append(
                {"model": tag, "stratum_type": stratum_type, "stratum": stratum, **metrics}
            )
    return pd.DataFrame(rows)


def pairwise_complementarity(aligned: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    rows = []
    for first, second in combinations(tags, 2):
        wrong_first = aligned[f"wrong_{first}"].astype(bool)
        wrong_second = aligned[f"wrong_{second}"].astype(bool)
        union = wrong_first | wrong_second
        intersection = wrong_first & wrong_second
        rows.append(
            {
                "model_a": first,
                "model_b": second,
                "n": len(aligned),
                "prediction_agreement": float(
                    np.mean(aligned[f"pred_{first}"] == aligned[f"pred_{second}"])
                ),
                "probability_pearson": float(
                    aligned[f"prob_{first}"].corr(aligned[f"prob_{second}"], method="pearson")
                ),
                "probability_spearman": float(
                    aligned[f"prob_{first}"].corr(aligned[f"prob_{second}"], method="spearman")
                ),
                "both_correct": int((~wrong_first & ~wrong_second).sum()),
                "both_wrong": int(intersection.sum()),
                "a_correct_b_wrong": int((~wrong_first & wrong_second).sum()),
                "a_wrong_b_correct": int((wrong_first & ~wrong_second).sum()),
                "error_jaccard": float(intersection.sum() / union.sum()) if union.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def source_subtype_error_table(aligned: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    rows = []
    for (source, subtype), subset in aligned.groupby(
        ["source_dataset", "task_l6_label"], observed=True
    ):
        row = {
            "source_dataset": source,
            "task_l6_label": subtype,
            "n": len(subset),
            "label_idx": int(subset["label_idx"].iloc[0]),
            "mean_wrong_models": float(subset["wrong_count"].mean()),
            "persistent_ge5_n": int(subset["persistent_ge5"].sum()),
            "persistent_ge5_rate": float(subset["persistent_ge5"].mean()),
            "all_models_wrong_n": int(subset["all_models_wrong"].sum()),
            "all_models_wrong_rate": float(subset["all_models_wrong"].mean()),
            "six_model_mean_accuracy": float(
                np.mean(subset["mean_probability_pred"] == subset["label_idx"])
            ),
        }
        for tag in tags:
            row[f"{tag}_accuracy"] = float(
                np.mean(subset[f"pred_{tag}"] == subset["label_idx"])
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_dataset", "task_l6_label"])


def ensemble_and_persistence(aligned: pd.DataFrame, tags: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    probabilities = aligned[[f"prob_{tag}" for tag in tags]].to_numpy(dtype=float)
    predictions = aligned[[f"pred_{tag}" for tag in tags]].to_numpy(dtype=int)
    wrong = aligned[[f"wrong_{tag}" for tag in tags]].to_numpy(dtype=int)
    y_true = aligned["label_idx"].to_numpy(dtype=int)

    aligned["wrong_count"] = wrong.sum(axis=1)
    aligned["persistent_ge5"] = (aligned["wrong_count"] >= len(tags) - 1).astype(int)
    aligned["all_models_wrong"] = (aligned["wrong_count"] == len(tags)).astype(int)
    aligned["mean_probability"] = probabilities.mean(axis=1)
    aligned["mean_probability_pred"] = (aligned["mean_probability"] >= 0.5).astype(int)
    vote_count = predictions.sum(axis=1)
    majority_prediction = (vote_count > len(tags) / 2).astype(int)
    ties = vote_count == len(tags) / 2
    majority_prediction[ties] = aligned.loc[ties, "mean_probability_pred"].to_numpy(dtype=int)
    aligned["majority_vote_pred"] = majority_prediction

    any_correct = aligned["wrong_count"].to_numpy() < len(tags)
    oracle_prediction = np.where(any_correct, y_true, 1 - y_true)

    ensemble_rows = []
    mean_metrics = binary_metrics(y_true, aligned["mean_probability"].to_numpy(), 0.5)
    ensemble_rows.append({"method": "mean_probability_at_0.5", **mean_metrics})
    majority_metrics = binary_metrics(y_true, majority_prediction.astype(float), 0.5)
    ensemble_rows.append({"method": "majority_vote_tie_mean_probability", **majority_metrics})
    oracle_metrics = binary_metrics(y_true, oracle_prediction.astype(float), 0.5)
    ensemble_rows.append(
        {
            "method": "oracle_any_model_correct_non_deployable",
            **oracle_metrics,
        }
    )

    persistence_rows: list[dict] = []
    grouping = [("overall", "all", pd.Series(True, index=aligned.index))]
    grouping.extend(
        ("source", str(value), aligned["source_dataset"].eq(value))
        for value in sorted(aligned["source_dataset"].unique())
    )
    grouping.extend(
        ("subtype", str(value), aligned["task_l6_label"].eq(value))
        for value in sorted(aligned["task_l6_label"].unique())
    )
    for stratum_type, stratum, mask in grouping:
        subset = aligned.loc[mask]
        persistence_rows.append(
            {
                "stratum_type": stratum_type,
                "stratum": stratum,
                "n": len(subset),
                "mean_wrong_models": float(subset["wrong_count"].mean()),
                "median_wrong_models": float(subset["wrong_count"].median()),
                "all_models_correct_n": int((subset["wrong_count"] == 0).sum()),
                "persistent_ge5_n": int(subset["persistent_ge5"].sum()),
                "persistent_ge5_rate": float(subset["persistent_ge5"].mean()),
                "all_models_wrong_n": int(subset["all_models_wrong"].sum()),
                "all_models_wrong_rate": float(subset["all_models_wrong"].mean()),
            }
        )
    return pd.DataFrame(ensemble_rows), pd.DataFrame(persistence_rows)


def candidate_model_sets(tags: list[str]) -> list[tuple[str, list[str]]]:
    requested_sets = [
        ("all_representatives", tags),
        ("without_locked_c1", [tag for tag in tags if tag != "C1"]),
        ("pe_family_h3_h5_h6_h7", [tag for tag in ["H3", "H5", "H6", "H7"] if tag in tags]),
    ]
    model_sets = []
    seen = set()
    for name, members in requested_sets:
        key = tuple(members)
        if len(members) >= 2 and key not in seen:
            model_sets.append((name, members))
            seen.add(key)
    return model_sets


def model_set_diagnostics(
    aligned: pd.DataFrame, tags: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_sets = candidate_model_sets(tags)

    y_true = aligned["label_idx"].to_numpy(dtype=int)
    summary_rows = []
    stratum_rows = []
    for set_name, members in model_sets:
        probabilities = aligned[[f"prob_{tag}" for tag in members]].to_numpy(dtype=float)
        predictions = (probabilities >= 0.5).astype(int)
        wrong = predictions != y_true[:, None]
        mean_probability = probabilities.mean(axis=1)
        vote_count = predictions.sum(axis=1)
        majority_prediction = (vote_count > len(members) / 2).astype(int)
        ties = vote_count == len(members) / 2
        majority_prediction[ties] = (mean_probability[ties] >= 0.5).astype(int)
        confidence_choice = np.abs(probabilities - 0.5).argmax(axis=1)
        selected_probability = probabilities[np.arange(len(aligned)), confidence_choice]
        any_correct = ~wrong.all(axis=1)
        oracle_prediction = np.where(any_correct, y_true, 1 - y_true)

        methods = [
            ("mean_probability_at_0.5", mean_probability, 0.5),
            ("majority_vote_tie_mean_probability", majority_prediction.astype(float), 0.5),
            ("highest_confidence_model", selected_probability, 0.5),
            ("oracle_any_model_correct_non_deployable", oracle_prediction.astype(float), 0.5),
        ]
        mean_oracle_threshold, mean_oracle_metrics = best_balanced_accuracy_threshold(
            y_true, mean_probability
        )
        methods.insert(
            1,
            (
                "mean_probability_pooled_oracle_threshold_non_deployable",
                mean_probability,
                mean_oracle_threshold,
            ),
        )
        for method, score, threshold in methods:
            metrics = (
                mean_oracle_metrics
                if method == "mean_probability_pooled_oracle_threshold_non_deployable"
                else binary_metrics(y_true, score, threshold)
            )
            summary_rows.append(
                {
                    "model_set": set_name,
                    "models": "+".join(members),
                    "model_count": len(members),
                    "method": method,
                    **metrics,
                }
            )

        grouping = [("overall", "all", pd.Series(True, index=aligned.index))]
        grouping.extend(
            ("source", str(value), aligned["source_dataset"].eq(value))
            for value in sorted(aligned["source_dataset"].unique())
        )
        grouping.extend(
            ("subtype", str(value), aligned["task_l6_label"].eq(value))
            for value in sorted(aligned["task_l6_label"].unique())
        )
        all_wrong = wrong.all(axis=1)
        stratum_methods = [
            ("mean_probability_at_0.5", mean_probability),
            ("highest_confidence_model", selected_probability),
            ("oracle_any_model_correct_non_deployable", oracle_prediction.astype(float)),
        ]
        for stratum_type, stratum, mask in grouping:
            selected = mask.to_numpy(dtype=bool)
            y_subset = y_true[selected]
            for method, score in stratum_methods:
                metrics = binary_metrics(y_subset, score[selected], 0.5)
                stratum_rows.append(
                    {
                        "model_set": set_name,
                        "models": "+".join(members),
                        "method": method,
                        "stratum_type": stratum_type,
                        "stratum": stratum,
                        "all_models_wrong_n": int(all_wrong[selected].sum()),
                        "all_models_wrong_rate": float(all_wrong[selected].mean()),
                        **metrics,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(stratum_rows)


def balanced_accuracy_from_prediction(y_true: np.ndarray, prediction: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    sensitivity = np.mean(prediction[y_true == 1] == 1)
    specificity = np.mean(prediction[y_true == 0] == 0)
    return float((sensitivity + specificity) / 2)


def bootstrap_mean_ensembles_against_h3(
    aligned: pd.DataFrame,
    tags: list[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    if "H3" not in tags:
        return pd.DataFrame()
    y_true = aligned["label_idx"].to_numpy(dtype=int)
    reference_prediction = (aligned["prob_H3"].to_numpy(dtype=float) >= 0.5).astype(int)
    low_indices = np.flatnonzero(y_true == 0)
    high_indices = np.flatnonzero(y_true == 1)
    rng = np.random.default_rng(seed)
    rows = []
    for set_offset, (set_name, members) in enumerate(candidate_model_sets(tags)):
        probabilities = aligned[[f"prob_{tag}" for tag in members]].to_numpy(dtype=float)
        ensemble_prediction = (probabilities.mean(axis=1) >= 0.5).astype(int)
        point_delta = balanced_accuracy_from_prediction(
            y_true, ensemble_prediction
        ) - balanced_accuracy_from_prediction(y_true, reference_prediction)
        deltas = np.empty(iterations, dtype=float)
        set_rng = np.random.default_rng(rng.integers(0, 2**32 - 1) + set_offset)
        for iteration in range(iterations):
            sampled = np.concatenate(
                [
                    set_rng.choice(low_indices, size=len(low_indices), replace=True),
                    set_rng.choice(high_indices, size=len(high_indices), replace=True),
                ]
            )
            sampled_y = y_true[sampled]
            deltas[iteration] = balanced_accuracy_from_prediction(
                sampled_y, ensemble_prediction[sampled]
            ) - balanced_accuracy_from_prediction(
                sampled_y, reference_prediction[sampled]
            )
        rows.append(
            {
                "model_set": set_name,
                "models": "+".join(members),
                "reference": "H3_at_0.5",
                "method": "mean_probability_at_0.5",
                "iterations": iterations,
                "point_delta_balanced_accuracy": point_delta,
                "bootstrap_ci_low": float(np.quantile(deltas, 0.025)),
                "bootstrap_ci_high": float(np.quantile(deltas, 0.975)),
                "bootstrap_probability_delta_gt_zero": float(np.mean(deltas > 0)),
            }
        )
    return pd.DataFrame(rows)


def threshold_transport(aligned: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    y_all = aligned["label_idx"].to_numpy(dtype=int)
    sources = sorted(aligned["source_dataset"].unique())
    for tag in tags:
        probability_column = f"prob_{tag}"
        probabilities = aligned[probability_column].to_numpy(dtype=float)
        rows.append(
            {
                "model": tag,
                "selection": "fixed_prespecified",
                "fit_source": "none",
                "eval_source": "all",
                **binary_metrics(y_all, probabilities, 0.5),
            }
        )
        pooled_threshold, pooled_metrics = best_balanced_accuracy_threshold(y_all, probabilities)
        rows.append(
            {
                "model": tag,
                "selection": "pooled_oracle_same_cases_non_deployable",
                "fit_source": "all",
                "eval_source": "all",
                **pooled_metrics,
                "threshold": pooled_threshold,
            }
        )
        source_oracle_decision = np.zeros(len(aligned), dtype=int)
        transferred_decision = np.zeros(len(aligned), dtype=int)
        source_oracle_thresholds = {}
        transferred_thresholds = {}
        for source in sources:
            target = aligned["source_dataset"].eq(source)
            other = ~target
            y_target = aligned.loc[target, "label_idx"].to_numpy(dtype=int)
            p_target = aligned.loc[target, probability_column].to_numpy(dtype=float)
            y_other = aligned.loc[other, "label_idx"].to_numpy(dtype=int)
            p_other = aligned.loc[other, probability_column].to_numpy(dtype=float)

            source_threshold, source_metrics = best_balanced_accuracy_threshold(y_target, p_target)
            target_positions = target.to_numpy(dtype=bool)
            source_oracle_decision[target_positions] = (p_target >= source_threshold).astype(int)
            source_oracle_thresholds[str(source)] = source_threshold
            rows.append(
                {
                    "model": tag,
                    "selection": "source_oracle_same_cases_non_deployable",
                    "fit_source": str(source),
                    "eval_source": str(source),
                    **source_metrics,
                    "threshold": source_threshold,
                }
            )
            transferred_threshold, _ = best_balanced_accuracy_threshold(y_other, p_other)
            transferred_decision[target_positions] = (p_target >= transferred_threshold).astype(int)
            transferred_thresholds[str(source)] = transferred_threshold
            transferred_metrics = binary_metrics(y_target, p_target, transferred_threshold)
            rows.append(
                {
                    "model": tag,
                    "selection": "other_sources_threshold_transfer",
                    "fit_source": "+".join(str(item) for item in sources if item != source),
                    "eval_source": str(source),
                    **transferred_metrics,
                    "threshold": transferred_threshold,
                }
            )
        source_oracle_combined = binary_metrics(y_all, source_oracle_decision.astype(float), 0.5)
        source_oracle_combined["threshold"] = np.nan
        rows.append(
            {
                "model": tag,
                "selection": "per_source_oracle_combined_non_deployable",
                "fit_source": "each_eval_source",
                "eval_source": "all",
                "component_thresholds": json.dumps(source_oracle_thresholds, sort_keys=True),
                **source_oracle_combined,
            }
        )
        transferred_combined = binary_metrics(y_all, transferred_decision.astype(float), 0.5)
        transferred_combined["threshold"] = np.nan
        rows.append(
            {
                "model": tag,
                "selection": "other_sources_threshold_transfer_combined",
                "fit_source": "other_sources_per_eval_source",
                "eval_source": "all",
                "component_thresholds": json.dumps(transferred_thresholds, sort_keys=True),
                **transferred_combined,
            }
        )
    return pd.DataFrame(rows)


def concept_associations(
    aligned: pd.DataFrame, concept_path: str, output_dir: Path
) -> tuple[pd.DataFrame, dict]:
    concept = strip_bom_columns(
        pd.read_csv(
            concept_path,
            usecols=["original_case_id", "concept_has_gross_text", *CONCEPTS],
            dtype={"original_case_id": str},
            encoding="utf-8-sig",
        )
    )
    concept["original_case_id"] = concept["original_case_id"].astype(str).str.strip()
    concept = concept.drop_duplicates("original_case_id")
    merged = aligned.merge(concept, on="original_case_id", how="left", validate="many_to_one")
    has_text = pd.to_numeric(merged["concept_has_gross_text"], errors="coerce").fillna(0).gt(0)
    rows = []
    for label_idx, comparison in [(0, "low_risk_persistent_ge5_vs_other"), (1, "high_risk_persistent_ge5_vs_other")]:
        subset = merged[merged["label_idx"].eq(label_idx) & has_text].copy()
        persistent = subset[subset["persistent_ge5"].eq(1)]
        other = subset[subset["persistent_ge5"].eq(0)]
        for concept_name in CONCEPTS:
            persistent_positive = int(
                pd.to_numeric(persistent[concept_name], errors="coerce").fillna(0).gt(0).sum()
            )
            other_positive = int(
                pd.to_numeric(other[concept_name], errors="coerce").fillna(0).gt(0).sum()
            )
            persistent_negative = len(persistent) - persistent_positive
            other_negative = len(other) - other_positive
            odds_ratio, p_value = fisher_exact(
                [[persistent_positive, persistent_negative], [other_positive, other_negative]],
                alternative="two-sided",
            )
            rows.append(
                {
                    "comparison": comparison,
                    "concept": concept_name,
                    "persistent_n": len(persistent),
                    "other_n": len(other),
                    "persistent_positive": persistent_positive,
                    "persistent_negative": persistent_negative,
                    "other_positive": other_positive,
                    "other_negative": other_negative,
                    "persistent_prevalence": persistent_positive / len(persistent)
                    if len(persistent)
                    else np.nan,
                    "other_prevalence": other_positive / len(other) if len(other) else np.nan,
                    "prevalence_difference": (
                        persistent_positive / len(persistent) - other_positive / len(other)
                        if len(persistent) and len(other)
                        else np.nan
                    ),
                    "odds_ratio": float(odds_ratio),
                    "fisher_p": float(p_value),
                }
            )
    association = pd.DataFrame(rows)
    association["fisher_q_bh"] = benjamini_hochberg(association["fisher_p"])
    association = association.sort_values(["fisher_q_bh", "fisher_p", "comparison", "concept"])
    coverage = {
        "registry_n": int(len(merged)),
        "matched_concept_id_n": int(merged["concept_has_gross_text"].notna().sum()),
        "has_gross_text_n": int(has_text.sum()),
        "missing_or_no_text_n": int((~has_text).sum()),
    }
    coverage_by_source = (
        merged.assign(has_gross_text=has_text.astype(int))
        .groupby("source_dataset", observed=True)
        .agg(n=("case_id", "size"), has_gross_text_n=("has_gross_text", "sum"))
        .reset_index()
    )
    coverage_by_source["coverage"] = coverage_by_source["has_gross_text_n"] / coverage_by_source["n"]
    coverage_by_source.to_csv(
        output_dir / "concept_coverage_by_source.csv", index=False, encoding="utf-8-sig"
    )
    coverage["by_source"] = coverage_by_source.to_dict(orient="records")
    return association, coverage


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(args.registry)
    specifications = parse_prediction_specs(args.prediction)
    aligned, tags = align_predictions(registry, specifications)

    structure_summary = dataset_structure(registry, output_dir)

    model_metrics = model_stratum_metrics(aligned, tags)
    model_metrics.to_csv(output_dir / "model_metrics_by_stratum.csv", index=False, encoding="utf-8-sig")

    pairwise = pairwise_complementarity(aligned, tags)
    pairwise.to_csv(output_dir / "pairwise_model_complementarity.csv", index=False, encoding="utf-8-sig")

    ensemble, persistence = ensemble_and_persistence(aligned, tags)
    ensemble.to_csv(output_dir / "ensemble_oracle_summary.csv", index=False, encoding="utf-8-sig")
    persistence.to_csv(output_dir / "error_persistence_by_stratum.csv", index=False, encoding="utf-8-sig")
    wrong_distribution = (
        aligned["wrong_count"].value_counts().sort_index().rename_axis("wrong_model_count").reset_index(name="n")
    )
    wrong_distribution["rate"] = wrong_distribution["n"] / len(aligned)
    wrong_distribution.to_csv(
        output_dir / "error_persistence_distribution.csv", index=False, encoding="utf-8-sig"
    )
    source_subtype_errors = source_subtype_error_table(aligned, tags)
    source_subtype_errors.to_csv(
        output_dir / "error_by_source_and_subtype.csv", index=False, encoding="utf-8-sig"
    )

    model_set_summary, model_set_oracle = model_set_diagnostics(aligned, tags)
    model_set_summary.to_csv(
        output_dir / "model_set_ensemble_and_oracle.csv", index=False, encoding="utf-8-sig"
    )
    model_set_oracle.to_csv(
        output_dir / "model_set_metrics_by_stratum.csv", index=False, encoding="utf-8-sig"
    )
    ensemble_bootstrap = bootstrap_mean_ensembles_against_h3(
        aligned,
        tags,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    ensemble_bootstrap.to_csv(
        output_dir / "mean_ensemble_vs_h3_paired_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    thresholds = threshold_transport(aligned, tags)
    thresholds.to_csv(output_dir / "threshold_transport_diagnostics.csv", index=False, encoding="utf-8-sig")

    associations, concept_coverage = concept_associations(aligned, args.concept_csv, output_dir)
    associations.to_csv(
        output_dir / "persistent_error_concept_associations.csv", index=False, encoding="utf-8-sig"
    )

    case_columns = [
        "case_id",
        "original_case_id",
        "source_dataset",
        "task_l6_label",
        "label_idx",
        *[f"prob_{tag}" for tag in tags],
        *[f"pred_{tag}" for tag in tags],
        *[f"wrong_{tag}" for tag in tags],
        "wrong_count",
        "persistent_ge5",
        "all_models_wrong",
        "mean_probability",
        "mean_probability_pred",
        "majority_vote_pred",
    ]
    aligned[case_columns].to_csv(
        output_dir / "case_level_error_matrix_SERVER_ONLY.csv", index=False, encoding="utf-8-sig"
    )

    overall_models = model_metrics[
        model_metrics["stratum_type"].eq("overall")
    ].set_index("model")
    persistent_overall = persistence[persistence["stratum_type"].eq("overall")].iloc[0].to_dict()
    summary = {
        "analysis_scope": {
            "n_cases": int(len(aligned)),
            "models": tags,
            "protocol": "source-LODO full-coverage Task7 predictions",
            "fixed_threshold": 0.5,
        },
        "dataset_structure": structure_summary,
        "overall_model_metrics": {
            tag: {
                column: overall_models.loc[tag, column]
                for column in [
                    "balanced_accuracy",
                    "auc",
                    "sensitivity",
                    "specificity",
                    "accuracy",
                    "brier",
                ]
            }
            for tag in tags
        },
        "ensemble_and_oracle": ensemble.to_dict(orient="records"),
        "model_set_ensemble_and_oracle": model_set_summary.to_dict(orient="records"),
        "mean_ensemble_vs_h3_paired_bootstrap": ensemble_bootstrap.to_dict(orient="records"),
        "persistent_errors": persistent_overall,
        "concept_coverage": concept_coverage,
        "significant_persistent_error_concepts_bh_0_05": associations[
            associations["fisher_q_bh"].lt(0.05)
        ].to_dict(orient="records"),
        "caveats": [
            "Pooled and same-source oracle thresholds are retrospective diagnostics, not deployable estimates.",
            "The oracle-any-correct row is a non-deployable ceiling and does not specify a routing rule.",
            "Gross-description concepts are retrospective explanatory variables, not image evidence available at deployment.",
            "Source-LODO uses acquisition batches from one institutional project and is not a true external-hospital validation.",
        ],
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, ensure_ascii=False, indent=2)

    print(f"Wrote Task7 capability plateau audit to {output_dir}")


if __name__ == "__main__":
    main()
