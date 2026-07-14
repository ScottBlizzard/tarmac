from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from run_task7_spatial_relational_20260713 import metric_record


SUBTYPES = ("A", "AB", "B1", "B2", "B3", "TC")
SOURCES = ("batch1", "batch2", "third_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze preregistered H6 CSD runs.")
    parser.add_argument("--h6-root", required=True)
    parser.add_argument("--h3-oof", required=True)
    parser.add_argument("--h3-lodo", required=True)
    parser.add_argument("--c2-oof", required=True)
    parser.add_argument("--c2-lodo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--confirmation-root", default=None)
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def canonical_source(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"^third_batch.*", "third_batch", regex=True)


def load_prediction(path: Path, model: str, protocol: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    required = {"case_id", "label_idx", "source_dataset", "task_l6_label", "prob_high"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prediction columns are incomplete: {path}")
    if len(frame) != 591 or frame["case_id"].duplicated().any():
        raise ValueError(f"Prediction coverage is not 591 unique cases: {path}")
    if not np.isfinite(frame["prob_high"].to_numpy(dtype=float)).all():
        raise ValueError(f"Prediction probabilities are non-finite: {path}")
    frame = frame.copy()
    frame["source_dataset"] = canonical_source(frame["source_dataset"])
    frame["label_idx"] = frame["label_idx"].astype(int)
    frame["task_l6_label"] = frame["task_l6_label"].astype(str)
    frame["pred_idx"] = (frame["prob_high"].to_numpy(float) >= 0.5).astype(int)
    frame["correct"] = frame["pred_idx"] == frame["label_idx"]
    frame["model"] = model
    frame["protocol"] = protocol
    return frame.sort_values("case_id").reset_index(drop=True)


def validate_alignment(frames: dict[tuple[str, str], pd.DataFrame]) -> None:
    reference = frames[("H3", "fivefold")]
    columns = ["case_id", "label_idx", "source_dataset", "task_l6_label"]
    for key, frame in frames.items():
        if not reference[columns].equals(frame[columns]):
            raise ValueError(f"Case metadata differs for {key}")
    if tuple(sorted(reference["source_dataset"].unique())) != tuple(sorted(SOURCES)):
        raise ValueError("Unexpected source set")
    if tuple(sorted(reference["task_l6_label"].unique())) != tuple(sorted(SUBTYPES)):
        raise ValueError("Unexpected subtype set")


def point_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return metric_record(frame["label_idx"], frame["prob_high"])


def subtype_accuracy(frame: pd.DataFrame, subtype: str) -> float:
    selected = frame["task_l6_label"] == subtype
    if not selected.any():
        return float("nan")
    return float(frame.loc[selected, "correct"].mean())


def subtype_correct_n(frame: pd.DataFrame, subtype: str) -> int:
    selected = frame["task_l6_label"] == subtype
    return int(frame.loc[selected, "correct"].sum())


def model_table(frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (model, protocol), frame in frames.items():
        rows.append(
            {
                "protocol": protocol,
                "model": model,
                **point_metrics(frame),
                "coverage_n": int(len(frame)),
                "B1_accuracy": subtype_accuracy(frame, "B1"),
                "B2_accuracy": subtype_accuracy(frame, "B2"),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol", "model"]).reset_index(drop=True)


def source_table(frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (model, protocol), frame in frames.items():
        for source in SOURCES:
            group = frame[frame["source_dataset"] == source]
            rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "source_dataset": source,
                    **point_metrics(group),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["protocol", "model", "source_dataset"]
    ).reset_index(drop=True)


def subtype_table(frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (model, protocol), frame in frames.items():
        for subtype in SUBTYPES:
            group = frame[frame["task_l6_label"] == subtype]
            rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "subtype": subtype,
                    "n": int(len(group)),
                    "risk_accuracy": float(group["correct"].mean()),
                    "correct_n": int(group["correct"].sum()),
                    "predicted_high_n": int(group["pred_idx"].sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["protocol", "model", "subtype"]
    ).reset_index(drop=True)


def sampled_metric(
    frame: pd.DataFrame,
    indices: np.ndarray,
    metric: str,
) -> float:
    labels = frame["label_idx"].to_numpy(dtype=int)[indices]
    predicted = frame["pred_idx"].to_numpy(dtype=int)[indices]
    if metric == "sensitivity":
        selected = labels == 1
        return float(np.mean(predicted[selected] == 1))
    if metric == "specificity":
        selected = labels == 0
        return float(np.mean(predicted[selected] == 0))
    if metric == "balanced_accuracy":
        sensitivity = np.mean(predicted[labels == 1] == 1)
        specificity = np.mean(predicted[labels == 0] == 0)
        return float(0.5 * (sensitivity + specificity))
    if metric in {"B1_accuracy", "B2_accuracy"}:
        subtype = metric.split("_")[0]
        selected = frame["task_l6_label"].to_numpy(str)[indices] == subtype
        if not np.any(selected):
            return float("nan")
        return float(np.mean(predicted[selected] == labels[selected]))
    raise ValueError(metric)


def paired_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    repetitions: int,
    seed: int,
    comparison: str,
    protocol: str,
) -> list[dict[str, Any]]:
    if not candidate["case_id"].equals(reference["case_id"]):
        raise ValueError("Bootstrap predictions are not aligned")
    strata = []
    for source in SOURCES:
        for risk in (0, 1):
            selected = np.flatnonzero(
                (candidate["source_dataset"].to_numpy(str) == source)
                & (candidate["label_idx"].to_numpy(int) == risk)
            )
            if len(selected) == 0:
                raise ValueError(f"Empty source/risk bootstrap stratum: {source}/{risk}")
            strata.append(selected)
    rng = np.random.default_rng(seed)
    metrics = (
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "B1_accuracy",
        "B2_accuracy",
    )
    distributions = {metric: np.empty(repetitions, dtype=np.float64) for metric in metrics}
    for repetition in range(repetitions):
        sampled = np.concatenate(
            [rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata]
        )
        for metric in metrics:
            distributions[metric][repetition] = sampled_metric(
                candidate, sampled, metric
            ) - sampled_metric(reference, sampled, metric)
    rows = []
    for metric, values in distributions.items():
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "comparison": comparison,
                "protocol": protocol,
                "metric": metric,
                "repetitions": int(len(finite)),
                "mean_delta": float(np.mean(finite)),
                "ci_lower_95": float(np.quantile(finite, 0.025)),
                "ci_upper_95": float(np.quantile(finite, 0.975)),
                "probability_delta_gt_0": float(np.mean(finite > 0.0)),
            }
        )
    return rows


def load_fold_summaries(run_dir: Path, protocol: str, variant: str) -> list[dict[str, Any]]:
    expected = 3 if protocol == "source_lodo" else 5
    status = run_dir / "RUN.status"
    if not status.exists() or status.read_text(encoding="utf-8").strip() != "complete":
        raise ValueError(f"H6 run is not complete: {run_dir}")
    summaries = []
    for fold_id in range(1, expected + 1):
        path = run_dir / f"fold_{fold_id}" / "fold_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("fold_id") != fold_id or summary.get("variant") != variant:
            raise ValueError(f"Fold summary identity mismatch: {path}")
        summaries.append(summary)
    return summaries


def mechanism_table(
    h6_root: Path,
    summaries: dict[tuple[str, str], list[dict[str, Any]]],
) -> pd.DataFrame:
    del h6_root
    rows = []
    for (protocol, variant), fold_summaries in summaries.items():
        for summary in fold_summaries:
            mechanism = summary["mechanism"]
            rows.append(
                {
                    "protocol": protocol,
                    "variant": variant,
                    "fold_id": int(summary["fold_id"]),
                    "held_out_source": summary.get("held_out_source", ""),
                    "best_epoch": int(summary["best_epoch"]),
                    "common_validation_bacc": float(
                        summary["common_validation_metrics"]["balanced_accuracy"]
                    ),
                    "specific_validation_bacc": float(
                        summary["specific_validation_metrics_diagnostic"][
                            "balanced_accuracy"
                        ]
                    ),
                    "specific_minus_common_validation_bacc": float(
                        mechanism["specific_minus_common_validation_bacc"]
                    ),
                    "median_specific_to_common_ratio": float(
                        mechanism["median_specific_to_common_ratio"]
                    ),
                    "common_weight_norm": float(
                        mechanism["common_weight_frobenius_norm"]
                    ),
                    "specific_weight_norm": float(
                        mechanism["specific_weight_frobenius_norm"]
                    ),
                    "parameter_count": int(summary["parameter_count"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["protocol", "variant", "fold_id"]
    ).reset_index(drop=True)


def row_for_bootstrap(
    table: pd.DataFrame, comparison: str, protocol: str, metric: str
) -> pd.Series:
    selected = table[
        (table["comparison"] == comparison)
        & (table["protocol"] == protocol)
        & (table["metric"] == metric)
    ]
    if len(selected) != 1:
        raise ValueError(f"Missing bootstrap row: {comparison}/{protocol}/{metric}")
    return selected.iloc[0]


def fold_minimum_bacc(frame: pd.DataFrame) -> float:
    if "fold_id" not in frame.columns:
        raise ValueError("H6 predictions are missing fold_id")
    return float(
        min(
            point_metrics(group)["balanced_accuracy"]
            for _, group in frame.groupby("fold_id")
        )
    )


def gate_record(name: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "gate": name,
        "passed": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def point_gate_values(
    nuisance: pd.DataFrame,
    h3: pd.DataFrame,
    source_csd: pd.DataFrame,
    protocol: str,
) -> dict[str, Any]:
    nuisance_metrics = point_metrics(nuisance)
    h3_by_source = {
        source: point_metrics(h3[h3["source_dataset"] == source])
        for source in SOURCES
    }
    nuisance_by_source = {
        source: point_metrics(nuisance[nuisance["source_dataset"] == source])
        for source in SOURCES
    }
    source_deltas = {
        source: float(
            nuisance_by_source[source]["balanced_accuracy"]
            - h3_by_source[source]["balanced_accuracy"]
        )
        for source in SOURCES
    }
    values = {
        "metrics": nuisance_metrics,
        "B1_accuracy": subtype_accuracy(nuisance, "B1"),
        "B2_accuracy": subtype_accuracy(nuisance, "B2"),
        "B1_correct_n": subtype_correct_n(nuisance, "B1"),
        "B2_correct_n": subtype_correct_n(nuisance, "B2"),
        "minimum_source_bacc": float(
            min(value["balanced_accuracy"] for value in nuisance_by_source.values())
        ),
        "source_bacc": {
            source: nuisance_by_source[source]["balanced_accuracy"] for source in SOURCES
        },
        "source_delta_vs_h3": source_deltas,
        "sources_improved_vs_h3": int(sum(delta > 0.0 for delta in source_deltas.values())),
        "maximum_source_harm_vs_h3": float(min(source_deltas.values())),
        "source_csd_bacc": point_metrics(source_csd)["balanced_accuracy"],
        "source_csd_sensitivity": point_metrics(source_csd)["sensitivity"],
        "source_csd_B1_accuracy": subtype_accuracy(source_csd, "B1"),
        "source_csd_B2_accuracy": subtype_accuracy(source_csd, "B2"),
        "source_csd_B1_correct_n": subtype_correct_n(source_csd, "B1"),
        "protocol": protocol,
    }
    return values


def primary_gates(
    frames: dict[tuple[str, str], pd.DataFrame],
    bootstrap: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nuisance = frames[("NA-CSD", "source_lodo")]
    h3 = frames[("H3", "source_lodo")]
    source_csd = frames[("SOURCE_ONLY_CSD", "source_lodo")]
    values = point_gate_values(nuisance, h3, source_csd, "source_lodo")
    metrics = values["metrics"]
    vs_h3 = row_for_bootstrap(
        bootstrap, "NA-CSD_minus_H3", "source_lodo", "balanced_accuracy"
    )
    vs_c2_sensitivity = row_for_bootstrap(
        bootstrap, "NA-CSD_minus_C2", "source_lodo", "sensitivity"
    )
    mechanism_pass = (
        metrics["balanced_accuracy"] >= values["source_csd_bacc"] + 0.005
        and (
            metrics["sensitivity"] > values["source_csd_sensitivity"]
            or values["B2_accuracy"] > values["source_csd_B2_accuracy"]
        )
        and values["B1_correct_n"] >= values["source_csd_B1_correct_n"] - 1
    )
    gates = [
        gate_record("LODO BAcc", metrics["balanced_accuracy"] >= 0.7641, metrics["balanced_accuracy"], ">= 0.7641"),
        gate_record("LODO sensitivity", metrics["sensitivity"] >= 0.7354, metrics["sensitivity"], ">= 0.7354"),
        gate_record("LODO specificity", metrics["specificity"] >= 0.7800, metrics["specificity"], ">= 0.7800"),
        gate_record("LODO B1", values["B1_accuracy"] >= 0.6129, values["B1_accuracy"], ">= 0.6129 (38/62)"),
        gate_record("LODO B2", values["B2_accuracy"] >= 0.6629, values["B2_accuracy"], ">= 0.6629 (59/89)"),
        gate_record("Minimum held-source BAcc", values["minimum_source_bacc"] >= 0.7381, values["minimum_source_bacc"], ">= 0.7381"),
        gate_record("Source direction", values["sources_improved_vs_h3"] >= 2, values["sources_improved_vs_h3"], ">= 2/3 improve vs H3"),
        gate_record("Maximum source harm", values["maximum_source_harm_vs_h3"] >= -0.015, values["maximum_source_harm_vs_h3"], "no source delta vs H3 < -0.015"),
        gate_record(
            "Bootstrap versus H3",
            float(vs_h3["mean_delta"]) > 0.0
            and float(vs_h3["probability_delta_gt_0"]) >= 0.80
            and float(vs_h3["ci_lower_95"]) > -0.010,
            {
                "mean_delta": float(vs_h3["mean_delta"]),
                "probability_delta_gt_0": float(vs_h3["probability_delta_gt_0"]),
                "ci_lower_95": float(vs_h3["ci_lower_95"]),
            },
            "mean > 0; P(delta>0) >= 0.80; 95% CI lower > -0.010",
        ),
        gate_record(
            "Sensitivity versus C2",
            float(vs_c2_sensitivity["ci_lower_95"]) > -0.020,
            float(vs_c2_sensitivity["ci_lower_95"]),
            "95% CI lower > -0.020",
        ),
        gate_record(
            "Mechanism control",
            mechanism_pass,
            {
                "bacc_delta_vs_source_only": float(
                    metrics["balanced_accuracy"] - values["source_csd_bacc"]
                ),
                "sensitivity_delta_vs_source_only": float(
                    metrics["sensitivity"] - values["source_csd_sensitivity"]
                ),
                "B2_delta_vs_source_only": float(
                    values["B2_accuracy"] - values["source_csd_B2_accuracy"]
                ),
                "B1_case_delta_vs_source_only": int(
                    values["B1_correct_n"] - values["source_csd_B1_correct_n"]
                ),
            },
            "BAcc >= source-only +0.005; sensitivity or B2 higher; lose <=1 B1 case",
        ),
        gate_record("Coverage/threshold", len(nuisance) == 591, int(len(nuisance)), "591/591 at threshold 0.5"),
    ]
    return gates, values


def secondary_gates(
    frames: dict[tuple[str, str], pd.DataFrame]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nuisance = frames[("NA-CSD", "fivefold")]
    metrics = point_metrics(nuisance)
    values = {
        "metrics": metrics,
        "B1_accuracy": subtype_accuracy(nuisance, "B1"),
        "B2_accuracy": subtype_accuracy(nuisance, "B2"),
        "minimum_fold_bacc": fold_minimum_bacc(nuisance),
    }
    gates = [
        gate_record("Fivefold BAcc", metrics["balanced_accuracy"] >= 0.7903, metrics["balanced_accuracy"], ">= 0.7903"),
        gate_record("Fivefold sensitivity", metrics["sensitivity"] >= 0.7800, metrics["sensitivity"], ">= 0.7800"),
        gate_record("Fivefold specificity", metrics["specificity"] >= 0.7800, metrics["specificity"], ">= 0.7800"),
        gate_record("Fivefold B1", values["B1_accuracy"] >= 0.6452, values["B1_accuracy"], ">= 0.6452 (40/62)"),
        gate_record("Fivefold B2", values["B2_accuracy"] >= 0.6742, values["B2_accuracy"], ">= 0.6742 (60/89)"),
        gate_record("Minimum fold BAcc", values["minimum_fold_bacc"] >= 0.70, values["minimum_fold_bacc"], ">= 0.70"),
    ]
    return gates, values


def fishr_trigger(
    mechanism: pd.DataFrame,
    primary_gates_list: list[dict[str, Any]],
    primary_values: dict[str, Any],
) -> dict[str, Any]:
    nuisance = mechanism[
        (mechanism["protocol"] == "source_lodo")
        & (mechanism["variant"] == "nuisance_csd")
    ]
    ratio_condition = bool(
        len(nuisance) == 3
        and (nuisance["median_specific_to_common_ratio"] >= 0.10).all()
    )
    validation_condition_count = int(
        (nuisance["specific_minus_common_validation_bacc"] >= 0.03).sum()
    )
    capability_floor_failure = bool(
        primary_values["metrics"]["sensitivity"] < 0.7354
        or primary_values["B2_accuracy"] < 0.6629
    )
    primary_failed = not all(gate["passed"] for gate in primary_gates_list)
    triggered = (
        primary_failed
        and ratio_condition
        and validation_condition_count >= 2
        and capability_floor_failure
    )
    return {
        "triggered": bool(triggered),
        "all_lodo_fold_median_ratios_at_least_0_10": ratio_condition,
        "lodo_folds_specific_validation_advantage_at_least_0_03": validation_condition_count,
        "common_sensitivity_or_B2_floor_failed": capability_floor_failure,
    }


def single_seed_point_gates(
    lodo: pd.DataFrame,
    fivefold: pd.DataFrame,
    h3_lodo: pd.DataFrame,
) -> dict[str, bool]:
    lm = point_metrics(lodo)
    fm = point_metrics(fivefold)
    source_deltas = []
    source_bacc = []
    for source in SOURCES:
        candidate_metric = point_metrics(lodo[lodo["source_dataset"] == source])
        reference_metric = point_metrics(
            h3_lodo[h3_lodo["source_dataset"] == source]
        )
        source_bacc.append(float(candidate_metric["balanced_accuracy"]))
        source_deltas.append(
            float(
                candidate_metric["balanced_accuracy"]
                - reference_metric["balanced_accuracy"]
            )
        )
    gates = {
        "lodo_bacc": lm["balanced_accuracy"] >= 0.7641,
        "lodo_sensitivity": lm["sensitivity"] >= 0.7354,
        "lodo_specificity": lm["specificity"] >= 0.7800,
        "lodo_B1": subtype_accuracy(lodo, "B1") >= 0.6129,
        "lodo_B2": subtype_accuracy(lodo, "B2") >= 0.6629,
        "minimum_source_bacc": min(source_bacc) >= 0.7381,
        "at_least_two_sources_improve_vs_h3": sum(
            delta > 0.0 for delta in source_deltas
        )
        >= 2,
        "no_source_harm_beyond_0_015_vs_h3": min(source_deltas) >= -0.015,
        "fivefold_bacc": fm["balanced_accuracy"] >= 0.7903,
        "fivefold_sensitivity": fm["sensitivity"] >= 0.7800,
        "fivefold_specificity": fm["specificity"] >= 0.7800,
        "fivefold_B1": subtype_accuracy(fivefold, "B1") >= 0.6452,
        "fivefold_B2": subtype_accuracy(fivefold, "B2") >= 0.6742,
        "minimum_fold_bacc": fold_minimum_bacc(fivefold) >= 0.70,
        "coverage": len(lodo) == 591 and len(fivefold) == 591,
    }
    return {name: bool(value) for name, value in gates.items()}


def average_predictions(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    if not first["case_id"].equals(second["case_id"]):
        raise ValueError("Two-seed predictions are not aligned")
    result = first.copy()
    result["prob_high"] = 0.5 * (
        first["prob_high"].to_numpy(float) + second["prob_high"].to_numpy(float)
    )
    result["pred_idx"] = (result["prob_high"] >= 0.5).astype(int)
    result["correct"] = result["pred_idx"] == result["label_idx"]
    return result


def confirmation_point_gates(
    primary_frames: dict[tuple[str, str], pd.DataFrame],
    confirmation_frames: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, Any]:
    confirmation_lodo = confirmation_frames[
        ("NA-CSD-confirmation", "source_lodo")
    ]
    confirmation_fivefold = confirmation_frames[
        ("NA-CSD-confirmation", "fivefold")
    ]
    primary_lodo = primary_frames[("NA-CSD", "source_lodo")]
    primary_fivefold = primary_frames[("NA-CSD", "fivefold")]
    h3_lodo = primary_frames[("H3", "source_lodo")]
    confirmation_gates = single_seed_point_gates(
        confirmation_lodo, confirmation_fivefold, h3_lodo
    )
    mean_gates = single_seed_point_gates(
        average_predictions(primary_lodo, confirmation_lodo),
        average_predictions(primary_fivefold, confirmation_fivefold),
        h3_lodo,
    )
    return {
        "confirmation_seed_gates": confirmation_gates,
        "two_seed_mean_gates": mean_gates,
        "confirmation_seed_all_pass": bool(all(confirmation_gates.values())),
        "two_seed_mean_all_pass": bool(all(mean_gates.values())),
        "all_pass": bool(
            all(confirmation_gates.values()) and all(mean_gates.values())
        ),
    }


def markdown_report(
    model_metrics: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    def metric_row(protocol: str, model: str) -> pd.Series:
        return model_metrics[
            (model_metrics["protocol"] == protocol)
            & (model_metrics["model"] == model)
        ].iloc[0]

    lines = [
        "# H6 Nuisance-Anchored CSD Results",
        "",
        "## Decision",
        "",
        f"- Primary source-LODO gates: **{'PASS' if decision['primary_source_lodo_pass'] else 'FAIL'}**.",
        f"- Secondary five-fold gates: **{'PASS' if decision['secondary_fivefold_pass'] else 'FAIL'}**.",
        f"- Next action: **{decision['next_action']}**.",
        "",
        "## Locked Metrics",
        "",
        "| Protocol | Model | BAcc | AUC | Sensitivity | Specificity | B1 | B2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in ("source_lodo", "fivefold"):
        for model in ("H3", "C2", "SOURCE_ONLY_CSD", "NA-CSD"):
            row = metric_row(protocol, model)
            lines.append(
                f"| {protocol} | {model} | {row['balanced_accuracy']:.4f} | "
                f"{row['auc']:.4f} | {row['sensitivity']:.4f} | "
                f"{row['specificity']:.4f} | {row['B1_accuracy']:.4f} | "
                f"{row['B2_accuracy']:.4f} |"
            )
    lines.extend(["", "## Gate Detail", ""])
    for section in ("primary_gates", "secondary_gates"):
        lines.append(f"### {section.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Gate | Requirement | Pass | Observed |")
        lines.append("|---|---|---:|---|")
        for gate in decision[section]:
            observed = json.dumps(gate["observed"], ensure_ascii=False)
            lines.append(
                f"| {gate['gate']} | {gate['requirement']} | "
                f"{'YES' if gate['passed'] else 'NO'} | {observed} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Evidence Ceiling",
            "",
            "This is fixed-data exploratory engineering evidence and internal acquisition-batch robustness testing. It is not independent external or multicenter validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions != 20000:
        raise ValueError("H6 preregistration requires exactly 20,000 bootstrap repetitions")
    h6_root = Path(args.h6_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        ("H3", "fivefold"): Path(args.h3_oof),
        ("H3", "source_lodo"): Path(args.h3_lodo),
        ("C2", "fivefold"): Path(args.c2_oof),
        ("C2", "source_lodo"): Path(args.c2_lodo),
        ("NA-CSD", "fivefold"): h6_root / "fivefold" / "nuisance_csd" / "oof_predictions.csv",
        ("NA-CSD", "source_lodo"): h6_root / "source_lodo" / "nuisance_csd" / "oof_predictions.csv",
        ("SOURCE_ONLY_CSD", "fivefold"): h6_root / "fivefold" / "source_csd" / "oof_predictions.csv",
        ("SOURCE_ONLY_CSD", "source_lodo"): h6_root / "source_lodo" / "source_csd" / "oof_predictions.csv",
    }
    frames = {
        key: load_prediction(path, key[0], key[1]) for key, path in paths.items()
    }
    validate_alignment(frames)
    summaries = {
        (protocol, variant): load_fold_summaries(
            h6_root / protocol / variant, protocol, variant
        )
        for protocol in ("source_lodo", "fivefold")
        for variant in ("nuisance_csd", "source_csd")
    }
    model_metrics = model_table(frames)
    source_metrics = source_table(frames)
    subtype_metrics = subtype_table(frames)
    mechanism = mechanism_table(h6_root, summaries)

    bootstrap_rows: list[dict[str, Any]] = []
    comparison_index = 0
    for protocol in ("source_lodo", "fivefold"):
        for reference_name in ("H3", "C2", "SOURCE_ONLY_CSD"):
            comparison = f"NA-CSD_minus_{reference_name}"
            bootstrap_rows.extend(
                paired_bootstrap(
                    frames[("NA-CSD", protocol)],
                    frames[(reference_name, protocol)],
                    args.bootstrap_repetitions,
                    args.seed + comparison_index,
                    comparison,
                    protocol,
                )
            )
            comparison_index += 1
    bootstrap = pd.DataFrame(bootstrap_rows)
    primary_gate_rows, primary_values = primary_gates(frames, bootstrap)
    secondary_gate_rows, secondary_values = secondary_gates(frames)
    primary_pass = bool(all(row["passed"] for row in primary_gate_rows))
    secondary_pass = bool(all(row["passed"] for row in secondary_gate_rows))
    trigger = fishr_trigger(mechanism, primary_gate_rows, primary_values)
    if primary_pass and secondary_pass:
        next_action = "RUN_CONFIRMATION_SEED_20260715"
    elif trigger["triggered"]:
        next_action = "RUN_SINGLE_CONDITIONAL_FISHR_BACKUP"
    else:
        next_action = "STOP_CURRENT_DATA_VISUAL_DEVELOPMENT"
    decision: dict[str, Any] = {
        "experiment": "H6_NUISANCE_ANCHORED_CSD_20260714",
        "primary_source_lodo_pass": primary_pass,
        "secondary_fivefold_pass": secondary_pass,
        "all_primary_and_secondary_gates_pass": primary_pass and secondary_pass,
        "primary_gates": primary_gate_rows,
        "secondary_gates": secondary_gate_rows,
        "primary_values": primary_values,
        "secondary_values": secondary_values,
        "fishr_trigger": trigger,
        "next_action": next_action,
        "evidence_ceiling": "exploratory fixed-data engineering and internal acquisition-batch robustness",
    }

    if args.confirmation_root is not None:
        confirmation_root = Path(args.confirmation_root)
        confirmation_frames = {
            ("NA-CSD-confirmation", protocol): load_prediction(
                confirmation_root / protocol / "nuisance_csd" / "oof_predictions.csv",
                "NA-CSD-confirmation",
                protocol,
            )
            for protocol in ("source_lodo", "fivefold")
        }
        confirmation_result = confirmation_point_gates(frames, confirmation_frames)
        decision["confirmation"] = confirmation_result
        decision["next_action"] = (
            "ENGINEERING_GO_FREEZE_H6"
            if confirmation_result["all_pass"]
            else "NO_GO_CONFIRMATION_FAILED"
        )

    atomic_write_csv(output_dir / "model_comparison.csv", model_metrics)
    atomic_write_csv(output_dir / "source_comparison.csv", source_metrics)
    atomic_write_csv(output_dir / "subtype_comparison.csv", subtype_metrics)
    atomic_write_csv(
        output_dir / "bootstrap_vs_h3.csv",
        bootstrap[bootstrap["comparison"] == "NA-CSD_minus_H3"].reset_index(drop=True),
    )
    atomic_write_csv(
        output_dir / "bootstrap_vs_c2.csv",
        bootstrap[bootstrap["comparison"] == "NA-CSD_minus_C2"].reset_index(drop=True),
    )
    atomic_write_csv(
        output_dir / "bootstrap_vs_source_only_csd.csv",
        bootstrap[
            bootstrap["comparison"] == "NA-CSD_minus_SOURCE_ONLY_CSD"
        ].reset_index(drop=True),
    )
    atomic_write_csv(output_dir / "mechanism_diagnostics.csv", mechanism)
    atomic_write_json(output_dir / "decision.json", decision)
    (output_dir / "RESULTS.md").write_text(
        markdown_report(model_metrics, decision), encoding="utf-8"
    )
    print(json.dumps({"next_action": decision["next_action"]}, indent=2))


if __name__ == "__main__":
    main()
