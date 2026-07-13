from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from summarize_task7_h4_gate_20260713 import (
    EXPECTED_SOURCES,
    EXPECTED_SUBTYPES,
    align,
    gate_row,
    load_predictions,
    metrics,
    paired_source_risk_bootstrap,
    paired_transitions,
    source_metrics,
    subtype_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the preregistered H5 second-order texture gates."
    )
    parser.add_argument("--c2-oof", required=True)
    parser.add_argument("--c2-lodo", required=True)
    parser.add_argument("--h3-oof", required=True)
    parser.add_argument("--h3-lodo", required=True)
    parser.add_argument("--h5-oof", required=True)
    parser.add_argument("--h5-lodo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# H5 low-rank second-order texture locked gate summary",
        "",
        "All metrics use threshold 0.5 and 100% coverage. Source-LODO is an internal acquisition-batch stress test, not independent external validation.",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Model | OOF BAcc | OOF AUC | OOF Sens | OOF Spec | LODO BAcc | LODO AUC | LODO Sens | LODO Spec | B1 | B2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ("c2", "h3_pe", "h5_texture"):
        oof = result["metrics"][model]["fivefold"]
        lodo = result["metrics"][model]["source_lodo"]
        subtype = result["by_subtype"][model]
        lines.append(
            f"| {model} | {oof['balanced_accuracy']:.4f} | {oof['auc']:.4f} | "
            f"{oof['sensitivity']:.4f} | {oof['specificity']:.4f} | "
            f"{lodo['balanced_accuracy']:.4f} | {lodo['auc']:.4f} | "
            f"{lodo['sensitivity']:.4f} | {lodo['specificity']:.4f} | "
            f"{subtype['B1']:.4f} | {subtype['B2']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Preregistered gates",
            "",
            "| # | Gate | Status |",
            "| ---: | --- | --- |",
        ]
    )
    for row in result["gates"]:
        lines.append(f"| {row['gate_number']} | {row['gate']} | {row['status']} |")

    lines.extend(
        [
            "",
            "## Held-source results",
            "",
            "| Source | C2 BAcc | H3 PE BAcc | H5 BAcc | H5 - C2 | H5 - H3 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source in EXPECTED_SOURCES:
        row = result["by_source"][source]
        lines.append(
            f"| {source} | {row['c2']['balanced_accuracy']:.4f} | "
            f"{row['h3_pe']['balanced_accuracy']:.4f} | "
            f"{row['h5_texture']['balanced_accuracy']:.4f} | "
            f"{row['h5_minus_c2_bacc']:+.4f} | {row['h5_minus_h3_bacc']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "| Subtype | C2 accuracy | H3 PE accuracy | H5 accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for subtype in EXPECTED_SUBTYPES:
        lines.append(
            f"| {subtype} | {result['by_subtype']['c2'][subtype]:.4f} | "
            f"{result['by_subtype']['h3_pe'][subtype]:.4f} | "
            f"{result['by_subtype']['h5_texture'][subtype]:.4f} |"
        )

    for reference in ("c2", "h3_pe"):
        bootstrap = result["paired_bootstrap"][reference]["balanced_accuracy"]
        lines.append(
            f"\nH5 minus {reference} LODO BAcc: {bootstrap['point_delta']:+.4f}, "
            f"95% CI [{bootstrap['ci95_lower']:+.4f}, {bootstrap['ci95_upper']:+.4f}]."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("Use at least 1000 bootstrap replicates")

    c2_oof, h3_oof, h5_oof = align(
        load_predictions(Path(args.c2_oof)),
        load_predictions(Path(args.h3_oof)),
        load_predictions(Path(args.h5_oof)),
    )
    c2_lodo, h3_lodo, h5_lodo = align(
        load_predictions(Path(args.c2_lodo)),
        load_predictions(Path(args.h3_lodo)),
        load_predictions(Path(args.h5_lodo)),
    )
    frames = {
        "c2": (c2_oof, c2_lodo),
        "h3_pe": (h3_oof, h3_lodo),
        "h5_texture": (h5_oof, h5_lodo),
    }
    result: dict[str, Any] = {
        "candidate": "pe_spatial_lowrank_covariance_v1",
        "threshold": 0.5,
        "coverage": 1.0,
        "metrics": {
            name: {"fivefold": metrics(pair[0]), "source_lodo": metrics(pair[1])}
            for name, pair in frames.items()
        },
        "by_subtype": {
            name: subtype_accuracy(pair[1]) for name, pair in frames.items()
        },
    }

    by_source = {name: source_metrics(pair[1]) for name, pair in frames.items()}
    result["by_source"] = {
        source: {
            "c2": by_source["c2"][source],
            "h3_pe": by_source["h3_pe"][source],
            "h5_texture": by_source["h5_texture"][source],
            "h5_minus_c2_bacc": float(by_source["h5_texture"][source]["balanced_accuracy"])
            - float(by_source["c2"][source]["balanced_accuracy"]),
            "h5_minus_h3_bacc": float(by_source["h5_texture"][source]["balanced_accuracy"])
            - float(by_source["h3_pe"][source]["balanced_accuracy"]),
        }
        for source in EXPECTED_SOURCES
    }
    result["paired_bootstrap"] = {
        "c2": paired_source_risk_bootstrap(
            c2_lodo, h5_lodo, args.bootstrap_replicates, args.seed
        ),
        "h3_pe": paired_source_risk_bootstrap(
            h3_lodo, h5_lodo, args.bootstrap_replicates, args.seed + 1
        ),
    }
    result["error_transitions"] = {
        "versus_c2": paired_transitions(c2_lodo, h5_lodo),
        "versus_h3_pe": paired_transitions(h3_lodo, h5_lodo),
    }

    h5_oof_metrics = result["metrics"]["h5_texture"]["fivefold"]
    h5_lodo_metrics = result["metrics"]["h5_texture"]["source_lodo"]
    h3_lodo_metrics = result["metrics"]["h3_pe"]["source_lodo"]
    h5_subtypes = result["by_subtype"]["h5_texture"]
    h3_subtypes = result["by_subtype"]["h3_pe"]
    source_deltas = [
        result["by_source"][source]["h5_minus_c2_bacc"]
        for source in EXPECTED_SOURCES
    ]
    primary_gates = [
        gate_row(1, "Five-fold OOF BAcc >= 0.7903", h5_oof_metrics["balanced_accuracy"] >= 0.7903),
        gate_row(
            2,
            "Five-fold sensitivity >= 0.7772 and specificity >= 0.7635",
            h5_oof_metrics["sensitivity"] >= 0.7772
            and h5_oof_metrics["specificity"] >= 0.7635,
        ),
        gate_row(3, "Source-LODO BAcc >= 0.7641", h5_lodo_metrics["balanced_accuracy"] >= 0.7641),
        gate_row(4, "Source-LODO sensitivity >= 0.7354", h5_lodo_metrics["sensitivity"] >= 0.7354),
        gate_row(5, "Source-LODO specificity >= 0.7527", h5_lodo_metrics["specificity"] >= 0.7527),
        gate_row(6, "Source-LODO B1 accuracy >= 0.6000", h5_subtypes["B1"] >= 0.6000),
        gate_row(7, "Source-LODO B2 accuracy >= 0.6629", h5_subtypes["B2"] >= 0.6629),
        gate_row(8, "At least two held-out sources improve versus C2", sum(delta > 0 for delta in source_deltas) >= 2),
        gate_row(9, "No held-out source declines by more than 0.02 versus C2", min(source_deltas) >= -0.02),
        gate_row(
            10,
            "Source-LODO BAcc and sensitivity both exceed H3 PE",
            h5_lodo_metrics["balanced_accuracy"] > h3_lodo_metrics["balanced_accuracy"]
            and h5_lodo_metrics["sensitivity"] > h3_lodo_metrics["sensitivity"],
        ),
        gate_row(
            11,
            "B2 exceeds H3 PE and B1 does not decline",
            h5_subtypes["B2"] > h3_subtypes["B2"]
            and h5_subtypes["B1"] >= h3_subtypes["B1"],
        ),
        gate_row(
            12,
            "Paired LODO BAcc delta versus C2 has CI95 lower bound > 0",
            result["paired_bootstrap"]["c2"]["balanced_accuracy"]["ci95_lower"] > 0,
        ),
    ]
    primary_passed = all(row["passed"] for row in primary_gates)
    confirmation_gate = gate_row(
        13,
        "Confirmation seed remains directionally positive",
        False,
        "required_not_run" if primary_passed else "not_evaluated_primary_failed",
    )
    coverage_gate = gate_row(
        14,
        "Threshold 0.5 and coverage 100%",
        len(h5_oof) == 591 and len(h5_lodo) == 591,
    )
    result["gates"] = primary_gates + [confirmation_gate, coverage_gate]
    result["decision"] = (
        "PRIMARY_PASS_CONFIRMATION_REQUIRED" if primary_passed else "NO_GO"
    )
    result["bootstrap_replicates"] = args.bootstrap_replicates
    result["seed"] = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "h5_gate_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(result["gates"]).to_csv(output_dir / "h5_gate_table.csv", index=False)
    (output_dir / "h5_gate_summary.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
