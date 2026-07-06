from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


ROOT = Path(r"D:\影响分析")
OUT = ROOT / "outputs" / "grosspath_rc_v107_scorecard_evidence_ladder_20260527"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def first(rows: list[dict[str, str]], **conds: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(k) == v for k, v in conds.items()):
            return row
    raise KeyError(f"No row matching {conds}")


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "" or value is None:
        return None
    return float(value)


def metric_row(
    version: str,
    module: str,
    scope: str,
    evidence_type: str,
    source_file: str,
    row: dict[str, str] | None,
    note: str,
) -> dict[str, str]:
    if row is None:
        return {
            "version": version,
            "module": module,
            "scope": scope,
            "evidence_type": evidence_type,
            "source_file": source_file,
            "control_rate": "",
            "balanced_accuracy": "",
            "fn": "",
            "fp": "",
            "remaining_error_n": "",
            "note": note,
        }

    return {
        "version": version,
        "module": module,
        "scope": scope,
        "evidence_type": evidence_type,
        "source_file": source_file,
        "control_rate": f"{as_float(row, 'control_rate'):.6f}",
        "balanced_accuracy": f"{as_float(row, 'balanced_accuracy'):.6f}",
        "fn": row.get("fn", ""),
        "fp": row.get("fp", ""),
        "remaining_error_n": row.get("remaining_error_n", ""),
        "note": note,
    }


def pct(value: str) -> str:
    if value == "":
        return ""
    return f"{float(value) * 100:.2f}%"


def make_markdown(rows: list[dict[str, str]], path: Path) -> None:
    headers = [
        "version",
        "module",
        "scope",
        "evidence_type",
        "BAcc",
        "control",
        "FN",
        "FP",
        "remaining",
        "note",
    ]
    lines = [
        "# v107 Scorecard Evidence Ladder",
        "",
        "This table separates internal OOF evidence, approximate external stress tests, and external-compatible refit evidence.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["version"],
            row["module"],
            row["scope"],
            row["evidence_type"],
            pct(row["balanced_accuracy"]),
            pct(row["control_rate"]),
            row["fn"],
            row["fp"],
            row["remaining_error_n"],
            row["note"],
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    v91 = read_csv(
        ROOT
        / "outputs"
        / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527"
        / "v91_integrated_summary.csv"
    )
    v101 = read_csv(
        ROOT
        / "outputs"
        / "grosspath_rc_v101_multimodel_oof_fn_sentinel_20260527"
        / "v101_nested_internal_comparison.csv"
    )
    v102 = read_csv(
        ROOT
        / "outputs"
        / "grosspath_rc_v102_external_approx_multimodel_sentinel_20260527"
        / "v102_external_approx_summary.csv"
    )
    v104 = read_csv(
        ROOT
        / "outputs"
        / "grosspath_rc_v104_multimodel_ab_filtered_sentinel_20260527"
        / "v104_nested_internal_comparison.csv"
    )
    v105_internal = read_csv(
        ROOT
        / "outputs"
        / "grosspath_rc_v105_stability_distilled_scorecard_20260527"
        / "v105_internal_scorecard_summary.csv"
    )
    v105_external = read_csv(
        ROOT
        / "outputs"
        / "grosspath_rc_v105_stability_distilled_scorecard_20260527"
        / "v105_strict_external_approx_scorecard_summary.csv"
    )
    v106 = read_csv(
        ROOT
        / "outputs"
        / "grosspath_rc_v106_external_compatible_wholecrop_scorecard_20260527"
        / "v106_scorecard_summary.csv"
    )

    rows: list[dict[str, str]] = []

    # Baseline workflow: use v101 internal copy for old+third OOF and v91 for strict external.
    for scope in ["internal_all", "third_batch"]:
        rows.append(
            metric_row(
                "v91",
                "Batch-adaptive main",
                scope,
                "internal_oof",
                "v101_nested_internal_comparison.csv",
                first(v101, workflow="Batch-adaptive main", scope=scope),
                "Primary workflow before extra FN sentinel.",
            )
        )
    rows.append(
        metric_row(
            "v91",
            "Batch-adaptive main",
            "strict_external",
            "locked_external",
            "v91_integrated_summary.csv",
            first(v91, policy_label="Batch-adaptive main", scope="strict_external"),
            "Locked severe-shift branch uses v79-light.",
        )
    )

    for scope in ["internal_all", "third_batch"]:
        rows.append(
            metric_row(
                "v101",
                "Nested multimodel FN sentinel",
                scope,
                "internal_nested_oof",
                "v101_nested_internal_comparison.csv",
                first(v101, workflow="Nested multimodel FN sentinel", scope=scope),
                "Strong internal FN rescue; exact external features pending.",
            )
        )

    fold_rows = [r for r in v102 if r.get("workflow") == "v102 approx fold-selected sentinel"]
    approx_row = dict(fold_rows[0])
    approx_row["control_rate"] = f"{mean(float(r['control_rate']) for r in fold_rows):.12f}"
    rows.append(
        metric_row(
            "v102",
            "Approx external multimodel sentinel",
            "strict_external",
            "external_approx",
            "v102_external_approx_summary.csv",
            approx_row,
            "Compatibility stress test only; not exact v101 validation.",
        )
    )

    for scope in ["internal_all", "third_batch"]:
        rows.append(
            metric_row(
                "v104",
                "AB-filtered multimodel sentinel",
                scope,
                "internal_nested_oof",
                "v104_nested_internal_comparison.csv",
                first(v104, workflow="v104 AB-filtered multimodel sentinel", scope=scope),
                "Keeps v101 rescue while reducing clean AB review.",
            )
        )
    rows.append(
        metric_row(
            "v104",
            "AB-filtered multimodel sentinel",
            "strict_external",
            "not_available",
            "",
            None,
            "Exact external-compatible validation not completed.",
        )
    )

    for scope in ["all", "third_batch"]:
        rows.append(
            metric_row(
                "v105",
                "Stability-distilled scorecard",
                "internal_all" if scope == "all" else scope,
                "internal_oof",
                "v105_internal_scorecard_summary.csv",
                first(
                    v105_internal,
                    scorecard="v105_majority_unified025_core045",
                    scope=scope,
                ),
                "Best internal scorecard candidate.",
            )
        )
    rows.append(
        metric_row(
            "v105",
            "Stability-distilled scorecard",
            "strict_external",
            "external_approx",
            "v105_strict_external_approx_scorecard_summary.csv",
            first(
                v105_external,
                scorecard="v105_majority_unified025_core045",
                scope="all",
            ),
            "Approximate external mapping only.",
        )
    )

    for scope in ["all", "third_batch"]:
        rows.append(
            metric_row(
                "v106",
                "External-compatible whole-crop scorecard",
                "internal_all" if scope == "all" else scope,
                "internal_oof",
                "v106_scorecard_summary.csv",
                first(v106, setting="internal_oof", scope=scope),
                "Same high internal ceiling using whole-crop family.",
            )
        )
    rows.append(
        metric_row(
            "v106",
            "External-compatible whole-crop scorecard",
            "strict_external",
            "external_compatible_refit",
            "v106_scorecard_summary.csv",
            first(v106, setting="strict_external_refit", scope="strict_external"),
            "External-compatible surrogate; not exact v105 crop validation.",
        )
    )

    out_csv = OUT / "v107_scorecard_evidence_ladder.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_markdown(rows, OUT / "v107_scorecard_evidence_ladder.md")

    key = OUT / "v107_key_messages.md"
    key.write_text(
        "\n".join(
            [
                "# v107 Key Messages",
                "",
                "- v91 remains the clean primary locked workflow: strict external BAcc 99.18%, FN=0, FP=1, control 82.41%.",
                "- v101/v104 show internal nested FN-rescue signals, but exact strict external feature parity is still missing.",
                "- v105 is the strongest internal scorecard: old+third BAcc 99.37%, third batch BAcc 98.94%, third FN=1.",
                "- v106 is the strongest external-compatible surrogate: internal all BAcc 99.37%, third BAcc 98.94%, strict external BAcc 99.18%, FN=0, FP=1, control 84.26%.",
                "- The clean writing hierarchy is: v91/v79-light as primary locked workflow; v105 as internal upper-bound scorecard; v106 as external-compatible scorecard evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {out_csv}")
    print(f"Wrote {OUT / 'v107_scorecard_evidence_ladder.md'}")
    print(f"Wrote {key}")


if __name__ == "__main__":
    main()
