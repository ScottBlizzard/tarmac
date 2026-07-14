from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_task7_h6_nuisance_csd_20260714 import (
    SOURCES,
    atomic_write_csv,
    atomic_write_json,
    fold_minimum_bacc,
    gate_record,
    load_prediction,
    model_table,
    paired_bootstrap,
    point_metrics,
    source_table,
    subtype_accuracy,
    subtype_correct_n,
    subtype_table,
    validate_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze bounded H7 LISA exploration.")
    parser.add_argument("--h7-root", required=True)
    parser.add_argument("--h3-oof", required=True)
    parser.add_argument("--h3-lodo", required=True)
    parser.add_argument("--c2-oof", required=True)
    parser.add_argument("--c2-lodo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--confirmation-root", default=None)
    return parser.parse_args()


def bootstrap_row(
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


def point_values(candidate: pd.DataFrame, h3: pd.DataFrame) -> dict[str, Any]:
    candidate_metrics = point_metrics(candidate)
    candidate_sources = {
        source: point_metrics(candidate[candidate["source_dataset"] == source])
        for source in SOURCES
    }
    h3_sources = {
        source: point_metrics(h3[h3["source_dataset"] == source])
        for source in SOURCES
    }
    source_deltas = {
        source: float(
            candidate_sources[source]["balanced_accuracy"]
            - h3_sources[source]["balanced_accuracy"]
        )
        for source in SOURCES
    }
    return {
        "metrics": candidate_metrics,
        "B1_accuracy": subtype_accuracy(candidate, "B1"),
        "B2_accuracy": subtype_accuracy(candidate, "B2"),
        "B1_correct_n": subtype_correct_n(candidate, "B1"),
        "B2_correct_n": subtype_correct_n(candidate, "B2"),
        "source_bacc": {
            source: candidate_sources[source]["balanced_accuracy"]
            for source in SOURCES
        },
        "source_delta_vs_h3": source_deltas,
        "minimum_source_bacc": float(
            min(value["balanced_accuracy"] for value in candidate_sources.values())
        ),
        "sources_improved_vs_h3": int(sum(value > 0.0 for value in source_deltas.values())),
        "maximum_source_harm_vs_h3": float(min(source_deltas.values())),
    }


def pairing_integrity(run_dir: Path) -> tuple[bool, dict[str, Any]]:
    summaries_path = run_dir / "fold_summaries.json"
    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    fractions = [
        float(item["pairing_diagnostics"]["actual_cross_source_fraction"])
        for item in summaries
    ]
    counts = [
        item["pairing_diagnostics"]["actual_mode_counts"] for item in summaries
    ]
    passed = bool(
        fractions
        and all(0.48 <= value <= 0.52 for value in fractions)
        and all(all(int(value) > 0 for value in record.values()) for record in counts)
    )
    return passed, {"fold_fractions": fractions, "fold_mode_counts": counts}


def primary_gates(
    candidate: pd.DataFrame,
    h3: pd.DataFrame,
    bootstrap: pd.DataFrame,
    pairing: tuple[bool, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = point_values(candidate, h3)
    metrics = values["metrics"]
    versus_h3 = bootstrap_row(
        bootstrap, "H7_LISA_minus_H3", "source_lodo", "balanced_accuracy"
    )
    versus_c2_sensitivity = bootstrap_row(
        bootstrap, "H7_LISA_minus_C2", "source_lodo", "sensitivity"
    )
    gates = [
        gate_record("LODO BAcc", metrics["balanced_accuracy"] >= 0.7641, metrics["balanced_accuracy"], ">= 0.7641"),
        gate_record("LODO sensitivity", metrics["sensitivity"] >= 0.7354, metrics["sensitivity"], ">= 0.7354"),
        gate_record("LODO specificity", metrics["specificity"] >= 0.7800, metrics["specificity"], ">= 0.7800"),
        gate_record("LODO B1", values["B1_correct_n"] >= 38, {"accuracy": values["B1_accuracy"], "correct_n": values["B1_correct_n"]}, ">= 38/62"),
        gate_record("LODO B2", values["B2_correct_n"] >= 59, {"accuracy": values["B2_accuracy"], "correct_n": values["B2_correct_n"]}, ">= 59/89"),
        gate_record("Minimum held-source BAcc", values["minimum_source_bacc"] >= 0.7381, values["minimum_source_bacc"], ">= 0.7381"),
        gate_record("Source direction", values["sources_improved_vs_h3"] >= 2, values["sources_improved_vs_h3"], ">= 2/3 sources improve vs H3"),
        gate_record("Maximum source harm", values["maximum_source_harm_vs_h3"] >= -0.015, values["maximum_source_harm_vs_h3"], "no source delta vs H3 < -0.015"),
        gate_record(
            "Bootstrap versus H3",
            float(versus_h3["mean_delta"]) > 0.0
            and float(versus_h3["probability_delta_gt_0"]) >= 0.80
            and float(versus_h3["ci_lower_95"]) > -0.010,
            {
                "mean_delta": float(versus_h3["mean_delta"]),
                "probability_delta_gt_0": float(versus_h3["probability_delta_gt_0"]),
                "ci_lower_95": float(versus_h3["ci_lower_95"]),
            },
            "mean > 0; P(delta>0) >= 0.80; 95% CI lower > -0.010",
        ),
        gate_record(
            "Sensitivity versus C2",
            float(versus_c2_sensitivity["ci_lower_95"]) > -0.020,
            float(versus_c2_sensitivity["ci_lower_95"]),
            "95% CI lower > -0.020",
        ),
        gate_record("Pairing integrity", pairing[0], pairing[1], "both modes present and cross-source fraction in [0.48, 0.52]"),
        gate_record("Coverage/threshold", len(candidate) == 591, int(len(candidate)), "591/591 at threshold 0.5"),
    ]
    return gates, values


def secondary_gates(
    candidate: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = point_metrics(candidate)
    values = {
        "metrics": metrics,
        "B1_accuracy": subtype_accuracy(candidate, "B1"),
        "B2_accuracy": subtype_accuracy(candidate, "B2"),
        "B1_correct_n": subtype_correct_n(candidate, "B1"),
        "B2_correct_n": subtype_correct_n(candidate, "B2"),
        "minimum_fold_bacc": fold_minimum_bacc(candidate),
    }
    gates = [
        gate_record("Fivefold BAcc", metrics["balanced_accuracy"] >= 0.7903, metrics["balanced_accuracy"], ">= 0.7903"),
        gate_record("Fivefold sensitivity", metrics["sensitivity"] >= 0.7800, metrics["sensitivity"], ">= 0.7800"),
        gate_record("Fivefold specificity", metrics["specificity"] >= 0.7800, metrics["specificity"], ">= 0.7800"),
        gate_record("Fivefold B1", values["B1_correct_n"] >= 40, {"accuracy": values["B1_accuracy"], "correct_n": values["B1_correct_n"]}, ">= 40/62"),
        gate_record("Fivefold B2", values["B2_correct_n"] >= 60, {"accuracy": values["B2_accuracy"], "correct_n": values["B2_correct_n"]}, ">= 60/89"),
        gate_record("Minimum fold BAcc", values["minimum_fold_bacc"] >= 0.70, values["minimum_fold_bacc"], ">= 0.70"),
        gate_record("Coverage/threshold", len(candidate) == 591, int(len(candidate)), "591/591 at threshold 0.5"),
    ]
    return gates, values


def point_only_primary_gates(
    candidate: pd.DataFrame, h3: pd.DataFrame
) -> list[dict[str, Any]]:
    values = point_values(candidate, h3)
    metrics = values["metrics"]
    return [
        gate_record("Confirmation LODO BAcc", metrics["balanced_accuracy"] >= 0.7641, metrics["balanced_accuracy"], ">= 0.7641"),
        gate_record("Confirmation LODO sensitivity", metrics["sensitivity"] >= 0.7354, metrics["sensitivity"], ">= 0.7354"),
        gate_record("Confirmation LODO specificity", metrics["specificity"] >= 0.7800, metrics["specificity"], ">= 0.7800"),
        gate_record("Confirmation LODO B1", values["B1_correct_n"] >= 38, values["B1_correct_n"], ">= 38/62"),
        gate_record("Confirmation LODO B2", values["B2_correct_n"] >= 59, values["B2_correct_n"], ">= 59/89"),
        gate_record("Confirmation minimum source", values["minimum_source_bacc"] >= 0.7381, values["minimum_source_bacc"], ">= 0.7381"),
        gate_record("Confirmation source direction", values["sources_improved_vs_h3"] >= 2, values["sources_improved_vs_h3"], ">= 2/3 improve"),
        gate_record("Confirmation maximum source harm", values["maximum_source_harm_vs_h3"] >= -0.015, values["maximum_source_harm_vs_h3"], ">= -0.015"),
    ]


def render_report(
    models: pd.DataFrame,
    sources: pd.DataFrame,
    subtypes: pd.DataFrame,
    gates: list[dict[str, Any]],
    next_action: str,
) -> str:
    lines = [
        "# H7 PE-Embedding LISA Interim Result",
        "",
        "This is a post-H6-stop exploratory experiment. It is not independent external validation.",
        "",
        f"Locked next action: `{next_action}`.",
        "",
        "## Overall metrics",
        "",
        "| Protocol | Model | BAcc | AUC | Sensitivity | Specificity | B1 | B2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in models.itertuples(index=False):
        lines.append(
            f"| {row.protocol} | {row.model} | {row.balanced_accuracy:.4f} | "
            f"{row.auc:.4f} | {row.sensitivity:.4f} | {row.specificity:.4f} | "
            f"{row.B1_accuracy:.4f} | {row.B2_accuracy:.4f} |"
        )
    lines.extend(["", "## Gates", "", "| Gate | Pass | Observed | Requirement |", "|---|---|---|---|"])
    for gate in gates:
        observed = json.dumps(gate["observed"], ensure_ascii=False)
        lines.append(
            f"| {gate['gate']} | {'PASS' if gate['passed'] else 'FAIL'} | "
            f"`{observed}` | {gate['requirement']} |"
        )
    lines.extend(["", "## Held-source metrics", "", "```csv", sources.to_csv(index=False).strip(), "```", "", "## Subtype metrics", "", "```csv", subtypes.to_csv(index=False).strip(), "```", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    h7_root = Path(args.h7_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[tuple[str, str], pd.DataFrame] = {
        ("H3", "fivefold"): load_prediction(Path(args.h3_oof), "H3", "fivefold"),
        ("H3", "source_lodo"): load_prediction(Path(args.h3_lodo), "H3", "source_lodo"),
        ("C2", "fivefold"): load_prediction(Path(args.c2_oof), "C2", "fivefold"),
        ("C2", "source_lodo"): load_prediction(Path(args.c2_lodo), "C2", "source_lodo"),
        ("H7-LISA", "source_lodo"): load_prediction(
            h7_root / "source_lodo" / "oof_predictions.csv", "H7-LISA", "source_lodo"
        ),
    }
    fivefold_path = h7_root / "fivefold" / "oof_predictions.csv"
    if fivefold_path.exists():
        frames[("H7-LISA", "fivefold")] = load_prediction(
            fivefold_path, "H7-LISA", "fivefold"
        )
    validate_alignment(frames)
    bootstrap_rows = []
    bootstrap_rows.extend(
        paired_bootstrap(
            frames[("H7-LISA", "source_lodo")],
            frames[("H3", "source_lodo")],
            args.bootstrap_repetitions,
            args.seed,
            "H7_LISA_minus_H3",
            "source_lodo",
        )
    )
    bootstrap_rows.extend(
        paired_bootstrap(
            frames[("H7-LISA", "source_lodo")],
            frames[("C2", "source_lodo")],
            args.bootstrap_repetitions,
            args.seed + 1,
            "H7_LISA_minus_C2",
            "source_lodo",
        )
    )
    bootstrap = pd.DataFrame(bootstrap_rows)
    pairing = pairing_integrity(h7_root / "source_lodo")
    primary, primary_values = primary_gates(
        frames[("H7-LISA", "source_lodo")],
        frames[("H3", "source_lodo")],
        bootstrap,
        pairing,
    )
    primary_pass = all(item["passed"] for item in primary)
    secondary: list[dict[str, Any]] = []
    secondary_values: dict[str, Any] | None = None
    if ("H7-LISA", "fivefold") in frames:
        secondary, secondary_values = secondary_gates(
            frames[("H7-LISA", "fivefold")]
        )
    if not primary_pass:
        next_action = "STOP_H7_SOURCE_LODO_NO_GO"
    elif not secondary:
        next_action = "RUN_FIVEFOLD"
    elif not all(item["passed"] for item in secondary):
        next_action = "STOP_H7_FIVEFOLD_NO_GO"
    else:
        next_action = "RUN_CONFIRMATION_SEED_20260717"

    confirmation_gates: list[dict[str, Any]] = []
    if args.confirmation_root:
        confirmation_root = Path(args.confirmation_root)
        confirmation_lodo = load_prediction(
            confirmation_root / "source_lodo" / "oof_predictions.csv",
            "H7-LISA-confirmation",
            "source_lodo",
        )
        confirmation_fivefold = load_prediction(
            confirmation_root / "fivefold" / "oof_predictions.csv",
            "H7-LISA-confirmation",
            "fivefold",
        )
        confirmation_gates.extend(
            point_only_primary_gates(
                confirmation_lodo, frames[("H3", "source_lodo")]
            )
        )
        confirmation_secondary, _ = secondary_gates(confirmation_fivefold)
        confirmation_gates.extend(
            [
                {**gate, "gate": f"Confirmation {gate['gate']}"}
                for gate in confirmation_secondary
            ]
        )
        next_action = (
            "FREEZE_H7_ENGINEERING_GO"
            if all(item["passed"] for item in confirmation_gates)
            else "STOP_H7_CONFIRMATION_NO_GO"
        )

    models = model_table(frames)
    sources = source_table(frames)
    subtypes = subtype_table(frames)
    all_gates = primary + secondary + confirmation_gates
    atomic_write_csv(output_dir / "model_metrics.csv", models)
    atomic_write_csv(output_dir / "source_metrics.csv", sources)
    atomic_write_csv(output_dir / "subtype_metrics.csv", subtypes)
    atomic_write_csv(output_dir / "paired_bootstrap.csv", bootstrap)
    atomic_write_csv(output_dir / "gate_summary.csv", pd.DataFrame(all_gates))
    decision = {
        "experiment": "H7_PE_EMBEDDING_LISA_20260714",
        "post_h6_stop_rule_exploration": True,
        "primary_source_lodo_pass": primary_pass,
        "secondary_fivefold_available": bool(secondary),
        "secondary_fivefold_pass": bool(secondary) and all(
            item["passed"] for item in secondary
        ),
        "confirmation_available": bool(confirmation_gates),
        "next_action": next_action,
        "primary_values": primary_values,
        "secondary_values": secondary_values,
    }
    atomic_write_json(output_dir / "decision.json", decision)
    report = render_report(models, sources, subtypes, all_gates, next_action)
    report_path = output_dir / "H7_INTERIM_RESULTS.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
