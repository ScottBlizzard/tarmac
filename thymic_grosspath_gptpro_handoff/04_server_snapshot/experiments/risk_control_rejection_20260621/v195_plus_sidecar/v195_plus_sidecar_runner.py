from __future__ import annotations

import argparse
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
SCRIPT_DIR = EXP_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v195_phase1_union_agreement_release_audit import (  # noqa: E402
    SELECTED_V195_CANDIDATE_ID,
    V173_CORRECTORS,
    V185_CASES,
    load_inputs,
    reconstruct_agreement_candidates,
)


SIDECAR_ROOT = EXP_ROOT / "v195_plus_sidecar"
DEFAULT_MANIFEST = EXP_ROOT / "configs" / "v195_plus_candidate_manifest.json"
DEFAULT_OUT_DIR = SIDECAR_ROOT / "outputs"

RUNTIME_COLUMNS = [
    "domain",
    "case_id",
    "original_case_id",
    "final_pred",
    "prob_mean_core",
    "v185_auto_decision",
    "v185_review_or_reject",
    "v195_release_from_review",
    "v195_auto_decision",
    "v195_review_or_reject",
    "v195_plus_release_from_review",
    "v195_plus_new_release_vs_v195",
    "v195_plus_auto_decision",
    "v195_plus_review_or_reject",
    "v195_plus_release_reason",
    "v195_plus_releasing_candidate_n",
    "v195_plus_releasing_candidate_ids",
    "v195_plus_policy_id",
]

AUDIT_COLUMNS = [
    "task_l6_label",
    "label_idx",
    "v195_auto_error",
    "v195_auto_high_risk_fn",
    "v195_plus_auto_error",
    "v195_plus_auto_high_risk_fn",
    "v195_plus_new_release_error",
    "v195_plus_new_release_high_risk_fn",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing v195+ candidate manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_ids(manifest: dict[str, Any], candidate_key: str) -> set[str]:
    records = manifest["candidates"][candidate_key]["candidate_records"]
    return {str(row["candidate_id"]) for row in records}


def _case_candidate_map(actions: pd.DataFrame, candidate_ids: set[str]) -> dict[str, list[str]]:
    selected = actions.loc[actions["candidate_id"].astype(str).isin(candidate_ids)].copy()
    if selected.empty:
        return {}
    grouped = selected.groupby("case_id")["candidate_id"].apply(lambda values: sorted(set(map(str, values))))
    return {str(case_id): ids for case_id, ids in grouped.items()}


def _summarize(frame: pd.DataFrame, prefix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain in ["old_data", "third_batch", "strict_external", "all"]:
        mask = frame["domain"].isin(["old_data", "third_batch", "strict_external"]) if domain == "all" else frame["domain"].eq(domain)
        sub = frame.loc[mask]
        if sub.empty:
            continue
        auto = sub[f"{prefix}_auto_decision"].astype(bool)
        review = sub[f"{prefix}_review_or_reject"].astype(bool)
        row = {
            "policy": prefix,
            "domain": domain,
            "n": int(len(sub)),
            "auto_n": int(auto.sum()),
            "review_n": int(review.sum()),
            "release_from_review_n": int(sub.get(f"{prefix}_release_from_review", pd.Series(False, index=sub.index)).astype(bool).sum()),
        }
        if f"{prefix}_auto_error" in sub.columns:
            row["auto_error_n"] = int(sub[f"{prefix}_auto_error"].astype(bool).sum())
            row["auto_high_risk_fn_n"] = int(sub[f"{prefix}_auto_high_risk_fn"].astype(bool).sum())
        rows.append(row)
    return rows


def build_v195_plus_sidecar_decisions(
    manifest_path: Path = DEFAULT_MANIFEST,
    candidate_key: str = "phase2_min10_both_domain_union",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    phase2_ids = _candidate_ids(manifest, candidate_key)
    v195_ids = {SELECTED_V195_CANDIDATE_ID}

    base, corr = load_inputs()
    _, actions = reconstruct_agreement_candidates(base, corr)
    base = base.copy()
    base["case_id"] = base["case_id"].astype(str)
    base["adaptive_review"] = _as_bool(base["adaptive_review"])

    v195_map = _case_candidate_map(actions, v195_ids)
    phase2_map = _case_candidate_map(actions, phase2_ids)
    v195_release_cases = set(v195_map)
    phase2_release_cases = set(phase2_map)

    case_ids = base["case_id"].astype(str)
    base["v185_auto_decision"] = ~base["adaptive_review"]
    base["v185_review_or_reject"] = base["adaptive_review"]
    base["v195_release_from_review"] = base["adaptive_review"] & case_ids.isin(v195_release_cases)
    base["v195_auto_decision"] = base["v185_auto_decision"] | base["v195_release_from_review"]
    base["v195_review_or_reject"] = ~base["v195_auto_decision"]
    base["v195_plus_release_from_review"] = base["adaptive_review"] & case_ids.isin(phase2_release_cases)
    base["v195_plus_auto_decision"] = base["v185_auto_decision"] | base["v195_plus_release_from_review"]
    base["v195_plus_review_or_reject"] = ~base["v195_plus_auto_decision"]
    base["v195_plus_new_release_vs_v195"] = base["v195_plus_auto_decision"] & ~base["v195_auto_decision"]
    base["v195_plus_releasing_candidate_ids"] = case_ids.map(lambda cid: ";".join(phase2_map.get(str(cid), [])))
    base["v195_plus_releasing_candidate_n"] = case_ids.map(lambda cid: len(phase2_map.get(str(cid), []))).astype(int)
    base["v195_plus_release_reason"] = "not_released"
    base.loc[base["v185_auto_decision"], "v195_plus_release_reason"] = "v185_already_auto"
    base.loc[base["v195_release_from_review"], "v195_plus_release_reason"] = "v195_selected_release"
    base.loc[base["v195_plus_new_release_vs_v195"], "v195_plus_release_reason"] = candidate_key
    base["v195_plus_policy_id"] = candidate_key

    base["v195_auto_error"] = base["v195_auto_decision"] & base["final_pred"].ne(base["label_idx"])
    base["v195_auto_high_risk_fn"] = base["v195_auto_decision"] & base["label_idx"].eq(1) & base["final_pred"].eq(0)
    base["v195_plus_auto_error"] = base["v195_plus_auto_decision"] & base["final_pred"].ne(base["label_idx"])
    base["v195_plus_auto_high_risk_fn"] = base["v195_plus_auto_decision"] & base["label_idx"].eq(1) & base["final_pred"].eq(0)
    base["v195_plus_new_release_error"] = base["v195_plus_new_release_vs_v195"] & base["final_pred"].ne(base["label_idx"])
    base["v195_plus_new_release_high_risk_fn"] = (
        base["v195_plus_new_release_vs_v195"] & base["label_idx"].eq(1) & base["final_pred"].eq(0)
    )

    runtime = base[RUNTIME_COLUMNS].copy()
    audit = base[RUNTIME_COLUMNS + AUDIT_COLUMNS].copy()
    summary = pd.DataFrame(_summarize(audit, "v195") + _summarize(audit, "v195_plus"))

    strict_phase2 = summary.loc[
        summary["policy"].eq("v195_plus") & summary["domain"].eq("strict_external")
    ].iloc[0]
    report = {
        "passed": bool(int(strict_phase2["auto_n"]) == 57 and int(strict_phase2["auto_error_n"]) == 0),
        "sidecar_only": True,
        "original_project_code_modified": False,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "candidate_key": candidate_key,
        "candidate_n": int(len(phase2_ids)),
        "source_files": {
            "base_cases": str(V185_CASES.relative_to(ROOT)),
            "corrector_outputs": str(V173_CORRECTORS.relative_to(ROOT)),
        },
        "runtime_columns": RUNTIME_COLUMNS,
        "audit_only_columns": AUDIT_COLUMNS,
        "summary": summary.to_dict(orient="records"),
        "strict_external_expected": {
            "v195_auto_n": 52,
            "v195_plus_auto_n": 57,
            "v195_plus_auto_error_n": 0,
        },
    }
    return runtime, audit, summary, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v195+ Phase2 sidecar flow without modifying original project code.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Frozen v195+ candidate manifest JSON.")
    parser.add_argument("--candidate-key", default="phase2_min10_both_domain_union")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Sidecar output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime, audit, summary, report = build_v195_plus_sidecar_decisions(
        manifest_path=Path(args.manifest),
        candidate_key=str(args.candidate_key),
    )
    runtime_path = out_dir / "v195_plus_runtime_decisions.csv"
    audit_path = out_dir / "v195_plus_audit_decisions.csv"
    summary_path = out_dir / "v195_plus_summary.csv"
    report_path = out_dir / "v195_plus_sidecar_report.json"
    runtime.to_csv(runtime_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report.update(
        {
            "runtime_decisions": str(runtime_path.relative_to(ROOT)),
            "audit_decisions": str(audit_path.relative_to(ROOT)),
            "summary_csv": str(summary_path.relative_to(ROOT)),
            "report_json": str(report_path.relative_to(ROOT)),
            "runtime_rows": int(len(runtime)),
            "audit_rows": int(len(audit)),
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
