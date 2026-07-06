from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for ancestor in current.parents:
        if (
            (ancestor / "outputs" / "grosspath_rc_v185_unlabeled_shift_adaptive_policy_20260527").exists()
            and (ancestor / "experiments" / "risk_control_rejection_20260621").exists()
        ):
            return ancestor
    return current.parents[3]


ROOT = _resolve_project_root()
EXP_ROOT = ROOT / "experiments" / "risk_control_rejection_20260621"
SIDECAR_ROOT = EXP_ROOT / "v195_plus_sidecar"
OUT_DIR = SIDECAR_ROOT / "outputs"
REPORT_DIR = EXP_ROOT / "reports"
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from v195_plus_sidecar_runner import (  # noqa: E402
    AUDIT_COLUMNS,
    DEFAULT_MANIFEST,
    RUNTIME_COLUMNS,
    build_v195_plus_sidecar_decisions,
)


REPORT_JSON = OUT_DIR / "v195_plus_sidecar_audit_report.json"
REPORT_MD = REPORT_DIR / "v195_plus_sidecar_audit.md"

EXPECTED_FILES = [
    OUT_DIR / "v195_plus_runtime_decisions.csv",
    OUT_DIR / "v195_plus_audit_decisions.csv",
    OUT_DIR / "v195_plus_summary.csv",
    OUT_DIR / "v195_plus_sidecar_report.json",
]

FORBIDDEN_RUNTIME_PATTERNS = [
    "label",
    "task_l6",
    "error",
    "high_risk",
    "correct",
    "fold",
]

EXPECTED_STRICT = {
    "v195": {"auto_n": 52, "review_n": 56, "auto_error_n": 0, "auto_high_risk_fn_n": 0},
    "v195_plus": {"auto_n": 57, "review_n": 51, "auto_error_n": 0, "auto_high_risk_fn_n": 0},
}


def check(condition: bool, check_id: str, detail: str) -> dict[str, object]:
    return {"check_id": check_id, "passed": bool(condition), "detail": detail}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"case_id": str, "original_case_id": str})


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> bool:
    if list(left.columns) != list(right.columns):
        return False
    left_cmp = left[columns].fillna("").astype(str).reset_index(drop=True)
    right_cmp = right[columns].fillna("").astype(str).reset_index(drop=True)
    return left_cmp.equals(right_cmp)


def file_inventory() -> list[dict[str, object]]:
    rows = []
    for path in EXPECTED_FILES:
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


def strict_metric_checks(summary: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for policy, expected in EXPECTED_STRICT.items():
        sub = summary.loc[summary["policy"].astype(str).eq(policy) & summary["domain"].astype(str).eq("strict_external")]
        if sub.empty:
            rows.append(check(False, f"{policy}_strict_row_exists", "missing strict_external summary row"))
            continue
        row = sub.iloc[0]
        for metric, value in expected.items():
            observed = int(row[metric])
            rows.append(
                check(
                    observed == value,
                    f"{policy}_strict_{metric}",
                    f"observed={observed}, expected={value}",
                )
            )
    return rows


def runtime_schema_checks(runtime: pd.DataFrame, audit: pd.DataFrame) -> list[dict[str, object]]:
    forbidden_present = [
        col
        for col in runtime.columns
        if col in AUDIT_COLUMNS or any(pattern in str(col).lower() for pattern in FORBIDDEN_RUNTIME_PATTERNS)
    ]
    return [
        check(list(runtime.columns) == RUNTIME_COLUMNS, "runtime_exact_column_contract", str(runtime.columns.tolist())),
        check(not forbidden_present, "runtime_no_audit_or_label_columns", str(forbidden_present)),
        check(len(runtime) == 699, "runtime_row_count_699", f"runtime_rows={len(runtime)}"),
        check(len(audit) == 699, "audit_row_count_699", f"audit_rows={len(audit)}"),
        check(all(col in audit.columns for col in AUDIT_COLUMNS), "audit_contains_audit_columns", str(AUDIT_COLUMNS)),
        check(not runtime["case_id"].duplicated().any(), "runtime_case_id_unique", "case_id duplicates absent"),
    ]


def manifest_checks() -> list[dict[str, object]]:
    if not DEFAULT_MANIFEST.exists():
        return [check(False, "manifest_exists", str(DEFAULT_MANIFEST))]
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    candidate = manifest.get("candidates", {}).get("phase2_min10_both_domain_union", {})
    records = candidate.get("candidate_records", [])
    return [
        check(True, "manifest_exists", str(DEFAULT_MANIFEST.relative_to(ROOT))),
        check(len(records) == 7, "manifest_phase2_candidate_n_7", f"candidate_n={len(records)}"),
        check(
            manifest.get("candidate_reconstruction", {}).get("strict_external_labels_used_for_selection") is False,
            "manifest_strict_external_not_selection",
            str(manifest.get("candidate_reconstruction", {}).get("strict_external_labels_used_for_selection")),
        ),
    ]


def idempotence_checks(runtime_disk: pd.DataFrame, audit_disk: pd.DataFrame, summary_disk: pd.DataFrame) -> list[dict[str, object]]:
    runtime_a, audit_a, summary_a, report_a = build_v195_plus_sidecar_decisions()
    runtime_b, audit_b, summary_b, report_b = build_v195_plus_sidecar_decisions()
    return [
        check(report_a["passed"] and report_b["passed"], "builder_reports_pass", f"a={report_a['passed']}, b={report_b['passed']}"),
        check(compare_frames(runtime_a, runtime_b, RUNTIME_COLUMNS), "builder_runtime_idempotent", "two in-memory builds match"),
        check(compare_frames(audit_a, audit_b, RUNTIME_COLUMNS + AUDIT_COLUMNS), "builder_audit_idempotent", "two in-memory builds match"),
        check(compare_frames(runtime_disk, runtime_a, RUNTIME_COLUMNS), "runtime_disk_matches_builder", "disk output matches builder"),
        check(compare_frames(audit_disk, audit_a, RUNTIME_COLUMNS + AUDIT_COLUMNS), "audit_disk_matches_builder", "disk output matches builder"),
        check(
            summary_disk.fillna("").astype(str).reset_index(drop=True).equals(summary_a.fillna("").astype(str).reset_index(drop=True)),
            "summary_disk_matches_builder",
            "summary output matches builder",
        ),
    ]


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_empty_"
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    runtime = load_csv(OUT_DIR / "v195_plus_runtime_decisions.csv")
    audit = load_csv(OUT_DIR / "v195_plus_audit_decisions.csv")
    summary = load_csv(OUT_DIR / "v195_plus_summary.csv")
    sidecar_report_path = OUT_DIR / "v195_plus_sidecar_report.json"
    sidecar_report = json.loads(sidecar_report_path.read_text(encoding="utf-8")) if sidecar_report_path.exists() else {}

    checks: list[dict[str, object]] = []
    inventory = file_inventory()
    checks.extend(
        check(bool(row["exists"]) and int(row["size"]) > 0, f"file_exists::{Path(str(row['path'])).name}", str(row))
        for row in inventory
    )
    checks.extend(manifest_checks())
    checks.extend(runtime_schema_checks(runtime, audit))
    checks.extend(strict_metric_checks(summary))
    checks.extend(idempotence_checks(runtime, audit, summary))
    checks.append(
        check(
            bool(sidecar_report.get("sidecar_only")) and sidecar_report.get("original_project_code_modified") is False,
            "sidecar_boundary_flags",
            f"sidecar_only={sidecar_report.get('sidecar_only')}, original_project_code_modified={sidecar_report.get('original_project_code_modified')}",
        )
    )

    passed = all(bool(row["passed"]) for row in checks)
    failed = [row for row in checks if not row["passed"]]
    report = {
        "passed": bool(passed),
        "check_n": int(len(checks)),
        "failed_n": int(len(failed)),
        "failed_checks": failed,
        "file_inventory": inventory,
        "checks": checks,
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# v195+ Sidecar Stability and Interface Audit",
        "",
        "This audit checks that the isolated v195+ sidecar has a stable, label-free runtime interface and reproduces the frozen Phase2 metrics.",
        "",
        "## Result",
        "",
        f"- Passed: `{passed}`.",
        f"- Checks: `{len(checks)}`.",
        f"- Failed: `{len(failed)}`.",
        "",
        "## Failed Checks",
        "",
        markdown_table(failed),
        "",
        "## File Inventory",
        "",
        markdown_table(inventory),
        "",
        "## All Checks",
        "",
        markdown_table(checks),
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": passed, "failed_n": len(failed), "report": str(REPORT_JSON.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
