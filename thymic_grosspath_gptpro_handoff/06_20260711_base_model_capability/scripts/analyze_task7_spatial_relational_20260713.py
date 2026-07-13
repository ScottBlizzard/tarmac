from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


EXPECTED_SOURCES = ("batch1", "batch2", "third_batch")
EXPECTED_SUBTYPES = {"A": 44, "AB": 262, "B1": 62, "B2": 89, "B3": 24, "TC": 110}
LOW_RISK = {"A", "AB", "B1"}
METRIC_NAMES = ("accuracy", "balanced_accuracy", "auc", "sensitivity", "specificity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the locked Task7 canonical spatial-relational experiment."
    )
    parser.add_argument("--feature-bank-dir", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--integrity-manifest", required=True)
    parser.add_argument("--expected-integrity-sha256", required=True)
    parser.add_argument("--c1-oof", required=True)
    parser.add_argument("--c1-lodo", required=True)
    parser.add_argument("--c2-oof", required=True)
    parser.add_argument("--c2-lodo", required=True)
    parser.add_argument("--matched-oof", required=True)
    parser.add_argument("--matched-lodo", required=True)
    parser.add_argument("--permuted-oof", required=True)
    parser.add_argument("--permuted-lodo", required=True)
    parser.add_argument("--relational-oof", required=True)
    parser.add_argument("--relational-lodo", required=True)
    parser.add_argument("--confirmation-relational-oof")
    parser.add_argument("--confirmation-relational-lodo")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_source(value: object) -> str:
    text = str(value)
    if text.startswith("third_batch"):
        return "third_batch"
    return text


def normalize_label(frame: pd.DataFrame) -> pd.Series:
    column = "label_idx" if "label_idx" in frame.columns else "risk_label"
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any():
        mapped = frame[column].astype(str).str.lower().map(
            {
                "low": 0,
                "low_risk_group": 0,
                "high": 1,
                "high_risk_group": 1,
            }
        )
        numeric = numeric.fillna(mapped)
    if numeric.isna().any():
        raise ValueError(f"Could not normalize labels from {column}")
    return numeric.astype(int)


def load_metadata(feature_bank_dir: Path, split_csv: Path) -> pd.DataFrame:
    metadata = pd.read_csv(
        feature_bank_dir / "metadata.csv",
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    metadata.columns = [str(column).lstrip("\ufeff") for column in metadata.columns]
    split = pd.read_csv(split_csv, dtype={"case_id": str}, encoding="utf-8-sig")
    split.columns = [str(column).lstrip("\ufeff") for column in split.columns]
    split = split[["case_id", "master_fold_id"]].drop_duplicates("case_id")
    authoritative = metadata[["case_id"]].merge(split, on="case_id", how="left")[
        "master_fold_id"
    ]
    fallback = pd.to_numeric(metadata.get("master_fold_id"), errors="coerce")
    metadata["master_fold_id"] = pd.to_numeric(authoritative, errors="coerce").fillna(
        fallback
    )
    metadata["source_dataset"] = metadata["source_dataset"].map(canonical_source)
    metadata["task_l6_label"] = metadata["task_l6_label"].astype(str)
    metadata["label_idx"] = (~metadata["task_l6_label"].isin(LOW_RISK)).astype(int)
    metadata["feature_row"] = np.arange(len(metadata), dtype=int)
    if len(metadata) != 591 or metadata["case_id"].duplicated().any():
        raise ValueError("Expected exactly 591 unique cases")
    if metadata["master_fold_id"].isna().any():
        raise ValueError("Locked fold assignment is incomplete")
    metadata["master_fold_id"] = metadata["master_fold_id"].astype(int)
    if metadata["task_l6_label"].value_counts().to_dict() != EXPECTED_SUBTYPES:
        raise ValueError("Subtype counts differ from the preregistration")
    if tuple(sorted(metadata["source_dataset"].unique())) != tuple(
        sorted(EXPECTED_SOURCES)
    ):
        raise ValueError("Source set differs from the preregistration")
    return metadata


def validate_code_lock(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.integrity_manifest).resolve(strict=True)
    if sha256_file(manifest_path) != args.expected_integrity_sha256:
        raise ValueError("Integrity manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("case_count") != 591:
        raise ValueError("Integrity manifest is incomplete")
    current_code = Path(__file__).resolve()
    matching = [
        record
        for key, record in manifest["assets"].items()
        if key.startswith("code_") and Path(record["path"]).resolve() == current_code
    ]
    if not matching or matching[0]["sha256"] != sha256_file(current_code):
        raise ValueError("Analyzer code is not locked in the integrity manifest")
    return manifest


def load_prediction(path: Path, metadata: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    probability_column = "prob_high" if "prob_high" in frame.columns else "probability"
    if not {"case_id", probability_column}.issubset(frame.columns):
        raise ValueError(f"{name} is missing case_id or high-risk probability")
    if len(frame) != len(metadata) or frame["case_id"].duplicated().any():
        raise ValueError(f"{name} must contain 591 unique predictions")
    values = frame[["case_id", probability_column]].rename(
        columns={probability_column: "prob_high"}
    )
    values["prob_high"] = pd.to_numeric(values["prob_high"], errors="raise")
    if not np.isfinite(values["prob_high"]).all() or not values["prob_high"].between(
        0.0, 1.0
    ).all():
        raise ValueError(f"{name} has invalid probabilities")
    if "label_idx" in frame.columns or "risk_label" in frame.columns:
        values["prediction_label"] = normalize_label(frame)
    result = metadata[
        [
            "feature_row",
            "case_id",
            "source_dataset",
            "task_l6_label",
            "label_idx",
            "master_fold_id",
        ]
    ].merge(values, on="case_id", how="left", validate="one_to_one")
    if result["prob_high"].isna().any():
        raise ValueError(f"{name} does not align with the immutable metadata")
    if "prediction_label" in result.columns and not np.array_equal(
        result["label_idx"].to_numpy(), result["prediction_label"].to_numpy()
    ):
        raise ValueError(f"{name} labels disagree with immutable metadata")
    result["model"] = name
    return result


def metric_record(y_true: Iterable[int], probability: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    predicted = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def subtype_accuracy(frame: pd.DataFrame, subtype: str) -> float:
    subset = frame.loc[frame["task_l6_label"].eq(subtype)]
    predicted = subset["prob_high"].to_numpy(dtype=float) >= 0.5
    return float(np.mean(predicted == subset["label_idx"].to_numpy(dtype=int)))


def aggregate_tables(
    predictions: dict[str, dict[str, pd.DataFrame]], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    subtype_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for protocol, models in predictions.items():
        for model, frame in models.items():
            overall_rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    **metric_record(frame["label_idx"], frame["prob_high"]),
                }
            )
            for source, group in frame.groupby("source_dataset", sort=False):
                source_rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "source_dataset": source,
                        **metric_record(group["label_idx"], group["prob_high"]),
                    }
                )
            for subtype, group in frame.groupby("task_l6_label", sort=False):
                predicted = group["prob_high"].to_numpy(dtype=float) >= 0.5
                subtype_rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "subtype": subtype,
                        "n": int(len(group)),
                        "risk_accuracy": float(
                            np.mean(predicted == group["label_idx"].to_numpy(dtype=int))
                        ),
                        "predicted_high_n": int(predicted.sum()),
                    }
                )
            group_column = "master_fold_id" if protocol == "fivefold_oof" else "source_dataset"
            for fold_value, group in frame.groupby(group_column, sort=False):
                fold_rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "held_out_partition": str(fold_value),
                        **metric_record(group["label_idx"], group["prob_high"]),
                    }
                )
    overall = pd.DataFrame(overall_rows)
    source = pd.DataFrame(source_rows)
    subtype = pd.DataFrame(subtype_rows)
    folds = pd.DataFrame(fold_rows)
    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    source.to_csv(output_dir / "source_metrics.csv", index=False)
    subtype.to_csv(output_dir / "subtype_metrics.csv", index=False)
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    return overall, source, subtype, folds


def paired_stratified_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    comparison: str,
    protocol: str,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not np.array_equal(candidate["case_id"].to_numpy(), baseline["case_id"].to_numpy()):
        raise ValueError(f"Case alignment failed for {comparison} {protocol}")
    if not np.array_equal(candidate["label_idx"].to_numpy(), baseline["label_idx"].to_numpy()):
        raise ValueError(f"Label alignment failed for {comparison} {protocol}")
    y = candidate["label_idx"].to_numpy(dtype=int)
    candidate_probability = candidate["prob_high"].to_numpy(dtype=float)
    baseline_probability = baseline["prob_high"].to_numpy(dtype=float)
    source = candidate["source_dataset"].astype(str).to_numpy()
    strata = [
        np.flatnonzero((source == source_name) & (y == risk_label))
        for source_name in EXPECTED_SOURCES
        for risk_label in (0, 1)
    ]
    if any(len(indices) == 0 for indices in strata):
        raise ValueError("Every source-by-risk bootstrap stratum must be non-empty")
    rng = np.random.default_rng(seed)
    differences = {metric: np.empty(repetitions, dtype=float) for metric in METRIC_NAMES}
    for repetition in range(repetitions):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        )
        candidate_metrics = metric_record(y[sampled], candidate_probability[sampled])
        baseline_metrics = metric_record(y[sampled], baseline_probability[sampled])
        for metric in METRIC_NAMES:
            differences[metric][repetition] = (
                candidate_metrics[metric] - baseline_metrics[metric]
            )
    candidate_point = metric_record(y, candidate_probability)
    baseline_point = metric_record(y, baseline_probability)
    rows = []
    for metric in METRIC_NAMES:
        rows.append(
            {
                "protocol": protocol,
                "comparison": comparison,
                "metric": metric,
                "candidate": float(candidate_point[metric]),
                "baseline": float(baseline_point[metric]),
                "delta": float(candidate_point[metric] - baseline_point[metric]),
                "ci_lower": float(np.quantile(differences[metric], 0.025)),
                "ci_upper": float(np.quantile(differences[metric], 0.975)),
                "bootstrap_repetitions": int(repetitions),
            }
        )
    return rows


def source_bacc(frame: pd.DataFrame) -> dict[str, float]:
    return {
        source: float(metric_record(group["label_idx"], group["prob_high"])["balanced_accuracy"])
        for source, group in frame.groupby("source_dataset", sort=False)
    }


def point_gate_records(
    candidate_oof: pd.DataFrame,
    candidate_lodo: pd.DataFrame,
    c2_lodo: pd.DataFrame,
    permuted_lodo: pd.DataFrame,
) -> list[dict[str, Any]]:
    oof = metric_record(candidate_oof["label_idx"], candidate_oof["prob_high"])
    lodo = metric_record(candidate_lodo["label_idx"], candidate_lodo["prob_high"])
    c2_sources = source_bacc(c2_lodo)
    candidate_sources = source_bacc(candidate_lodo)
    source_deltas = {
        source: candidate_sources[source] - c2_sources[source] for source in EXPECTED_SOURCES
    }
    minimum_source = min(candidate_sources.values())
    improved_sources = sum(delta > 0.0 for delta in source_deltas.values())
    worst_source_delta = min(source_deltas.values())
    b1 = subtype_accuracy(candidate_lodo, "B1")
    b2 = subtype_accuracy(candidate_lodo, "B2")
    c2_b1 = subtype_accuracy(c2_lodo, "B1")
    c2_b2 = subtype_accuracy(c2_lodo, "B2")
    permuted = metric_record(permuted_lodo["label_idx"], permuted_lodo["prob_high"])
    mechanism_delta = lodo["balanced_accuracy"] - permuted["balanced_accuracy"]
    return [
        {
            "gate_id": 1,
            "gate": "OOF BAcc >= 0.7664",
            "observed": oof["balanced_accuracy"],
            "passed": oof["balanced_accuracy"] >= 0.7664,
        },
        {
            "gate_id": 2,
            "gate": "LODO BAcc >= 0.7641",
            "observed": lodo["balanced_accuracy"],
            "passed": lodo["balanced_accuracy"] >= 0.7641,
        },
        {
            "gate_id": 3,
            "gate": "LODO sensitivity >= 0.7354 and specificity >= 0.7527",
            "observed": f"sensitivity={lodo['sensitivity']:.6f}; specificity={lodo['specificity']:.6f}",
            "passed": lodo["sensitivity"] >= 0.7354 and lodo["specificity"] >= 0.7527,
        },
        {
            "gate_id": 6,
            "gate": "At least 2/3 sources improve, worst delta >= -0.02, min BAcc >= 0.7083",
            "observed": (
                f"improved={improved_sources}; worst_delta={worst_source_delta:.6f}; "
                f"min_bacc={minimum_source:.6f}; deltas={json.dumps(source_deltas, sort_keys=True)}"
            ),
            "passed": improved_sources >= 2
            and worst_source_delta >= -0.02
            and minimum_source >= 0.7083,
        },
        {
            "gate_id": 7,
            "gate": "LODO B1 >= 0.5000 and B2 >= 0.6629",
            "observed": f"B1={b1:.6f}; B2={b2:.6f}",
            "passed": b1 >= 0.5000 and b2 >= 0.6629,
        },
        {
            "gate_id": 8,
            "gate": "Mean B1/B2 >= 0.6115 and neither below C2",
            "observed": (
                f"mean={(b1 + b2) / 2:.6f}; B1={b1:.6f} vs C2={c2_b1:.6f}; "
                f"B2={b2:.6f} vs C2={c2_b2:.6f}"
            ),
            "passed": (b1 + b2) / 2 >= 0.6115 and b1 >= c2_b1 and b2 >= c2_b2,
        },
        {
            "gate_id": 9,
            "gate": "Relational LODO BAcc - permuted LODO BAcc >= 0.01",
            "observed": mechanism_delta,
            "passed": mechanism_delta >= 0.01,
        },
    ]


def bootstrap_gate_records(bootstrap: pd.DataFrame) -> list[dict[str, Any]]:
    c2_lodo = bootstrap[
        bootstrap["protocol"].eq("source_lodo")
        & bootstrap["comparison"].eq("relational_vs_C2")
    ].set_index("metric")
    bacc_lower = float(c2_lodo.loc["balanced_accuracy", "ci_lower"])
    sensitivity_lower = float(c2_lodo.loc["sensitivity", "ci_lower"])
    return [
        {
            "gate_id": 4,
            "gate": "LODO delta BAcc vs C2 bootstrap lower bound > 0",
            "observed": bacc_lower,
            "passed": bacc_lower > 0.0,
        },
        {
            "gate_id": 5,
            "gate": "LODO sensitivity delta vs C2 bootstrap lower bound > -0.02",
            "observed": sensitivity_lower,
            "passed": sensitivity_lower > -0.02,
        },
    ]


def mean_prediction(primary: pd.DataFrame, confirmation: pd.DataFrame, name: str) -> pd.DataFrame:
    if not np.array_equal(primary["case_id"].to_numpy(), confirmation["case_id"].to_numpy()):
        raise ValueError(f"Primary and confirmation predictions do not align for {name}")
    result = primary.copy()
    result["prob_high"] = 0.5 * (
        primary["prob_high"].to_numpy(dtype=float)
        + confirmation["prob_high"].to_numpy(dtype=float)
    )
    result["model"] = name
    return result


def report_markdown(
    decision: str,
    overall: pd.DataFrame,
    source: pd.DataFrame,
    subtype: pd.DataFrame,
    bootstrap: pd.DataFrame,
    gates: pd.DataFrame,
    manifest_hash: str,
) -> str:
    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No rows."
        def render(value: object) -> str:
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.4f}"
            return str(value).replace("|", "\\|").replace("\n", " ")

        header = "| " + " | ".join(map(str, frame.columns)) + " |"
        divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
        rows = [
            "| " + " | ".join(render(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, divider, *rows])

    selected_models = [
        "C1",
        "C2",
        "matched_gated",
        "relational_permuted",
        "relational",
        "relational_confirmation",
        "relational_two_seed_mean",
    ]
    overall_view = overall[overall["model"].isin(selected_models)][
        [
            "protocol",
            "model",
            "n",
            "accuracy",
            "balanced_accuracy",
            "auc",
            "sensitivity",
            "specificity",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ]
    source_view = source[
        source["protocol"].eq("source_lodo") & source["model"].isin(selected_models)
    ][["model", "source_dataset", "n", "balanced_accuracy", "sensitivity", "specificity"]]
    subtype_view = subtype[
        subtype["protocol"].eq("source_lodo") & subtype["model"].isin(selected_models)
    ][["model", "subtype", "n", "risk_accuracy"]]
    bootstrap_view = bootstrap[
        ["protocol", "comparison", "metric", "delta", "ci_lower", "ci_upper"]
    ]
    gate_view = gates[["gate_id", "gate", "observed", "passed"]].sort_values("gate_id")
    return "\n".join(
        [
            "# H2 Canonical Spatial-Relational Final Report",
            "",
            f"**Decision: {decision}**",
            "",
            f"Integrity manifest SHA-256: `{manifest_hash}`",
            "",
            "All metrics use the preregistered threshold of 0.5. The three sources are internal acquisition batches, not external hospitals.",
            "",
            "## Primary Gates",
            "",
            table(gate_view),
            "",
            "## Overall Metrics",
            "",
            table(overall_view),
            "",
            "## LODO Source Metrics",
            "",
            table(source_view),
            "",
            "## LODO Subtype Risk Accuracy",
            "",
            table(subtype_view),
            "",
            "## Paired Source-by-Risk Bootstrap",
            "",
            table(bootstrap_view),
            "",
            "## Interpretation Boundary",
            "",
            "A NO-GO closes further optimization of the repeatedly reused 591-case single-photograph cohort. It does not show that standardized multi-view gross pathology images or independent multicenter data are uninformative.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions != 5000 or args.seed != 20260713:
        raise ValueError("The locked analysis requires 5,000 bootstrap repetitions and seed 20260713")
    confirmation_arguments = (
        args.confirmation_relational_oof,
        args.confirmation_relational_lodo,
    )
    if any(confirmation_arguments) and not all(confirmation_arguments):
        raise ValueError("Both confirmation OOF and LODO predictions are required together")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = validate_code_lock(args)
    metadata = load_metadata(Path(args.feature_bank_dir), Path(args.split_csv))
    path_map = {
        "fivefold_oof": {
            "C1": args.c1_oof,
            "C2": args.c2_oof,
            "matched_gated": args.matched_oof,
            "relational_permuted": args.permuted_oof,
            "relational": args.relational_oof,
        },
        "source_lodo": {
            "C1": args.c1_lodo,
            "C2": args.c2_lodo,
            "matched_gated": args.matched_lodo,
            "relational_permuted": args.permuted_lodo,
            "relational": args.relational_lodo,
        },
    }
    if all(confirmation_arguments):
        path_map["fivefold_oof"]["relational_confirmation"] = args.confirmation_relational_oof
        path_map["source_lodo"]["relational_confirmation"] = args.confirmation_relational_lodo
    predictions = {
        protocol: {
            model: load_prediction(Path(path), metadata, model)
            for model, path in paths.items()
        }
        for protocol, paths in path_map.items()
    }
    for protocol, models in predictions.items():
        reference = models["C2"]
        for model, frame in models.items():
            if not np.array_equal(reference["case_id"].to_numpy(), frame["case_id"].to_numpy()):
                raise ValueError(f"Explicit alignment failed for {protocol} {model}")

    bootstrap_rows: list[dict[str, Any]] = []
    comparison_index = 0
    for protocol in ("fivefold_oof", "source_lodo"):
        models = predictions[protocol]
        for comparison, candidate_name, baseline_name in (
            ("relational_vs_C2", "relational", "C2"),
            ("relational_vs_permuted", "relational", "relational_permuted"),
        ):
            bootstrap_rows.extend(
                paired_stratified_bootstrap(
                    models[candidate_name],
                    models[baseline_name],
                    comparison,
                    protocol,
                    args.bootstrap_repetitions,
                    args.seed + comparison_index,
                )
            )
            comparison_index += 1

    confirmation_present = all(confirmation_arguments)
    if confirmation_present:
        for protocol in ("fivefold_oof", "source_lodo"):
            models = predictions[protocol]
            models["relational_two_seed_mean"] = mean_prediction(
                models["relational"],
                models["relational_confirmation"],
                "relational_two_seed_mean",
            )
            for comparison, candidate_name in (
                ("confirmation_relational_vs_C2", "relational_confirmation"),
                ("two_seed_mean_vs_C2", "relational_two_seed_mean"),
            ):
                bootstrap_rows.extend(
                    paired_stratified_bootstrap(
                        models[candidate_name],
                        models["C2"],
                        comparison,
                        protocol,
                        args.bootstrap_repetitions,
                        args.seed + comparison_index,
                    )
                )
                comparison_index += 1

    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    primary_gate_rows = point_gate_records(
        predictions["fivefold_oof"]["relational"],
        predictions["source_lodo"]["relational"],
        predictions["source_lodo"]["C2"],
        predictions["source_lodo"]["relational_permuted"],
    ) + bootstrap_gate_records(bootstrap)
    gates = pd.DataFrame(primary_gate_rows).sort_values("gate_id")
    primary_passed = bool(gates["passed"].all())
    decision = "PROVISIONAL-GO: CONFIRMATION REQUIRED" if primary_passed else "NO-GO"

    confirmation_summary: dict[str, Any] | None = None
    if confirmation_present and primary_passed:
        confirmation_lodo = metric_record(
            predictions["source_lodo"]["relational_confirmation"]["label_idx"],
            predictions["source_lodo"]["relational_confirmation"]["prob_high"],
        )
        c2_lodo = metric_record(
            predictions["source_lodo"]["C2"]["label_idx"],
            predictions["source_lodo"]["C2"]["prob_high"],
        )
        confirmation_positive = (
            confirmation_lodo["balanced_accuracy"] - c2_lodo["balanced_accuracy"] > 0.0
        )
        mean_point_gates = point_gate_records(
            predictions["fivefold_oof"]["relational_two_seed_mean"],
            predictions["source_lodo"]["relational_two_seed_mean"],
            predictions["source_lodo"]["C2"],
            predictions["source_lodo"]["relational_permuted"],
        )
        mean_point_passed = all(bool(row["passed"]) for row in mean_point_gates)
        decision = "GO" if confirmation_positive and mean_point_passed else "NO-GO"
        confirmation_summary = {
            "confirmation_lodo_bacc_delta_vs_c2": float(
                confirmation_lodo["balanced_accuracy"] - c2_lodo["balanced_accuracy"]
            ),
            "confirmation_positive": confirmation_positive,
            "two_seed_mean_point_gates": mean_point_gates,
            "two_seed_mean_point_gates_passed": mean_point_passed,
        }

    overall, source, subtype, _ = aggregate_tables(predictions, output_dir)
    gates.to_csv(output_dir / "primary_gate_results.csv", index=False)
    manifest_hash = sha256_file(Path(args.integrity_manifest))
    decision_record = {
        "protocol": "H2_CANONICAL_SPATIAL_RELATIONAL_20260713",
        "decision": decision,
        "primary_gates_passed": primary_passed,
        "confirmation_present": confirmation_present,
        "confirmation": confirmation_summary,
        "integrity_manifest_sha256": manifest_hash,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "case_count": manifest["case_count"],
    }
    write_json(output_dir / "decision.json", decision_record)
    write_json(output_dir / "analysis_config.json", vars(args))
    (output_dir / "SPATIAL_RELATIONAL_FINAL_REPORT.md").write_text(
        report_markdown(
            decision,
            overall,
            source,
            subtype,
            bootstrap,
            gates,
            manifest_hash,
        ),
        encoding="utf-8",
    )
    (output_dir / "ANALYSIS.status").write_text("complete\n", encoding="utf-8")
    print(json.dumps(decision_record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
