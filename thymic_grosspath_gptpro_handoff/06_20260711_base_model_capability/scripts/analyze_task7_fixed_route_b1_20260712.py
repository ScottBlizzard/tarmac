from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final predeclared decision audit for the Task7 B1 cascade.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def comparison_row(frame: pd.DataFrame, scope: str, baseline: str, candidate: str) -> pd.Series:
    selected = frame[
        (frame["scope"] == scope)
        & (frame["baseline"] == baseline)
        & (frame["candidate"] == candidate)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one comparison row for {scope} {baseline} {candidate}")
    return selected.iloc[0]


def overall_metric(frame: pd.DataFrame, model: str) -> pd.Series:
    selected = frame[
        (frame["scope"] == "full")
        & (frame["group_type"] == "overall")
        & (frame["group"] == "all")
        & (frame["model"] == model)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one overall metric row for {model}")
    return selected.iloc[0]


def subgroup_accuracy(frame: pd.DataFrame, model: str, subtype: str) -> float:
    selected = frame[
        (frame["scope"] == "full")
        & (frame["group_type"] == "task_l6_label")
        & (frame["group"] == subtype)
        & (frame["model"] == model)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one subtype row for {model} {subtype}")
    return float(selected.iloc[0]["accuracy"])


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "final_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split_mode in ["fivefold", "source_lodo"]:
        split_dir = run_dir / split_mode
        comparisons = pd.read_csv(split_dir / "paired_comparisons.csv", encoding="utf-8-sig")
        metrics = pd.read_csv(split_dir / "summary_metrics.csv", encoding="utf-8-sig")
        predictions = pd.read_csv(
            split_dir / "oof_predictions.csv", dtype={"case_id": str, "outer_key": str}, encoding="utf-8-sig"
        )
        random_route = json.loads((split_dir / "matched_random_route_analysis.json").read_text(encoding="utf-8"))
        routed_m2 = comparison_row(comparisons, "routed", "m0_c1", "m2_image")
        routed_m3_m1 = comparison_row(comparisons, "routed", "m1_behavior", "m3_fusion")
        full_m2 = comparison_row(comparisons, "full", "m0_c1", "m2_image")
        full_m3 = comparison_row(comparisons, "full", "m0_c1", "m3_fusion")
        route_counts = (
            predictions.groupby("outer_key", dropna=False)
            .agg(n=("case_id", "size"), routed=("routed", "sum"))
            .reset_index()
        )
        route_counts["route_rate"] = route_counts["routed"] / route_counts["n"]
        results[split_mode] = {
            "comparisons": comparisons,
            "metrics": metrics,
            "random_route": random_route,
            "route_counts": route_counts,
            "routed_m2": routed_m2,
            "routed_m3_m1": routed_m3_m1,
            "full_m2": full_m2,
            "full_m3": full_m3,
        }

    lodo_metrics = results["source_lodo"]["metrics"]
    source_rows = lodo_metrics[
        (lodo_metrics["scope"] == "full") & (lodo_metrics["group_type"] == "source_dataset")
    ]
    source_pivot = source_rows.pivot(index="group", columns="model", values="balanced_accuracy")
    positive_lodo_sources = int((source_pivot["m2_image"] > source_pivot["m0_c1"]).sum())

    boundary = {}
    for split_mode in ["fivefold", "source_lodo"]:
        metrics = results[split_mode]["metrics"]
        boundary[split_mode] = {}
        for subtype in ["B1", "B2"]:
            baseline = subgroup_accuracy(metrics, "m0_c1", subtype)
            image = subgroup_accuracy(metrics, "m2_image", subtype)
            boundary[split_mode][subtype] = {
                "m0_accuracy": baseline,
                "m2_accuracy": image,
                "delta": image - baseline,
            }

    oof = results["fivefold"]
    lodo = results["source_lodo"]
    checks = {
        "oof_routed_m2_minus_m0_at_least_0_06": float(oof["routed_m2"]["delta_bacc"]) >= 0.06,
        "oof_routed_m2_ci_lower_above_zero": float(oof["routed_m2"]["ci_low"]) > 0.0,
        "oof_routed_m3_minus_m1_at_least_0_03": float(oof["routed_m3_m1"]["delta_bacc"]) >= 0.03,
        "oof_full_m2_minus_m0_at_least_0_02": float(oof["full_m2"]["delta_bacc"]) >= 0.02,
        "lodo_full_m2_minus_m0_at_least_0_015": float(lodo["full_m2"]["delta_bacc"]) >= 0.015,
        "positive_lodo_sources_at_least_2": positive_lodo_sources >= 2,
        "oof_actual_route_exceeds_random_95th": bool(oof["random_route"]["actual_exceeds_random_95th"]),
        "lodo_b1_nonnegative": boundary["source_lodo"]["B1"]["delta"] >= 0.0,
        "lodo_b2_nonnegative": boundary["source_lodo"]["B2"]["delta"] >= 0.0,
    }
    decision = {
        "checks": checks,
        "passes_predeclared_b1_gate": bool(all(checks.values())),
        "positive_lodo_sources": positive_lodo_sources,
        "boundary": boundary,
        "fivefold": {
            "routed_m2_minus_m0": float(oof["routed_m2"]["delta_bacc"]),
            "routed_m2_ci": [float(oof["routed_m2"]["ci_low"]), float(oof["routed_m2"]["ci_high"])],
            "routed_m3_minus_m1": float(oof["routed_m3_m1"]["delta_bacc"]),
            "full_m2_minus_m0": float(oof["full_m2"]["delta_bacc"]),
            "full_m3_minus_m0": float(oof["full_m3"]["delta_bacc"]),
            "random_route": oof["random_route"],
        },
        "source_lodo": {
            "routed_m2_minus_m0": float(lodo["routed_m2"]["delta_bacc"]),
            "routed_m2_ci": [float(lodo["routed_m2"]["ci_low"]), float(lodo["routed_m2"]["ci_high"])],
            "routed_m3_minus_m1": float(lodo["routed_m3_m1"]["delta_bacc"]),
            "full_m2_minus_m0": float(lodo["full_m2"]["delta_bacc"]),
            "full_m3_minus_m0": float(lodo["full_m3"]["delta_bacc"]),
            "random_route": lodo["random_route"],
        },
        "interpretation": "NO-GO: fixed confidence routing does not identify a transferable visual-specialist population.",
    }
    write_json(output_dir / "b1_final_decision.json", decision)

    lines = [
        "# Task7 B1 Fixed-Route M0-M4 Results",
        "",
        "Stage 1 is locked C1. The router uses the outer-training 40th percentile of absolute C1 logit margin. M2 receives image tokens and tile metadata only; M3 is fit from inner-crossfit M2 predictions.",
        "",
        "## Overall full-coverage metrics",
        "",
        "| Split | Model | BAcc | AUC | Sensitivity | Specificity |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split_mode in ["fivefold", "source_lodo"]:
        metrics = results[split_mode]["metrics"]
        for model in ["m0_c1", "m1_behavior", "m2_image", "m3_fusion"]:
            row = overall_metric(metrics, model)
            lines.append(
                f"| {split_mode} | {model} | {row['balanced_accuracy']:.4f} | {row['auc']:.4f} | "
                f"{row['sensitivity']:.4f} | {row['specificity']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Same-routed-case tests",
            "",
            "| Split | Comparison | Delta BAcc [95% CI] | Rescue/Harm |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for split_mode in ["fivefold", "source_lodo"]:
        for label, row in [
            ("M2-M0", results[split_mode]["routed_m2"]),
            ("M3-M1", results[split_mode]["routed_m3_m1"]),
        ]:
            lines.append(
                f"| {split_mode} | {label} | {row['delta_bacc']:.4f} "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {int(row['rescued'])}/{int(row['harmed'])} |"
            )
    lines.extend(["", "## Router stability", ""])
    for split_mode in ["fivefold", "source_lodo"]:
        rates = results[split_mode]["route_counts"]
        rendered = ", ".join(
            f"{row.outer_key}: {int(row.routed)}/{int(row.n)} ({row.route_rate:.1%})"
            for row in rates.itertuples()
        )
        lines.append(f"- `{split_mode}`: {rendered}.")
    lines.extend(
        [
            "",
            "## Boundary effects",
            "",
            "| Split | Subtype | M0 accuracy | M2 accuracy | Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for split_mode in ["fivefold", "source_lodo"]:
        for subtype in ["B1", "B2"]:
            row = boundary[split_mode][subtype]
            lines.append(
                f"| {split_mode} | {subtype} | {row['m0_accuracy']:.4f} | "
                f"{row['m2_accuracy']:.4f} | {row['delta']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**{'PASS' if decision['passes_predeclared_b1_gate'] else 'NO-GO'}**. Positive LODO sources: {positive_lodo_sources}/3.",
            "",
            "The routed image-only gain is small and uncertain in five-fold OOF, reverses significantly under source LODO, and the actual confidence router underperforms matched random routing. This fixed-grid confidence-routed cascade is closed.",
        ]
    )
    (output_dir / "B1_FIXED_ROUTE_M0_M4_RESULTS_20260712.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
