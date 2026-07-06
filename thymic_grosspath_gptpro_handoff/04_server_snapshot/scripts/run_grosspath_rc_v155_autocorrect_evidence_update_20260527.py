from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v155_autocorrect_evidence_update_20260527"
V153 = ROOT / "outputs" / "grosspath_rc_v153_high_precision_autoflip_trigger_20260527" / "v153_autoflip_trigger_passed_internal_rules.csv"
V154 = ROOT / "outputs" / "grosspath_rc_v154_pseudo_severe_shift_autoflip_selection_20260527" / "v154_pseudo_severe_selected_rules.csv"
V154_ALL = ROOT / "outputs" / "grosspath_rc_v154_pseudo_severe_shift_autoflip_selection_20260527" / "v154_pseudo_severe_all_rule_scope_summary.csv"
V152_MODULE = ROOT / "outputs" / "grosspath_rc_v152_framework_evidence_pack_20260527" / "v152_module_evidence_table.csv"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def pp(x: float, digits: int = 2) -> str:
    return f"{100 * float(x):+.{digits}f} pp"


def best_v154_rule(v154: pd.DataFrame, signal_filter: str | None = None) -> pd.Series:
    eligible = v154.loc[
        v154["pseudo_selection_status"].eq("selected_by_pseudo_subset")
        & (pd.to_numeric(v154["subset_net_errors_reduced"], errors="coerce") > 0)
        & (pd.to_numeric(v154["strict_delta_bacc"], errors="coerce") > 0)
    ].copy()
    if signal_filter:
        eligible = eligible.loc[eligible["signal"].astype(str).str.contains(signal_filter, regex=True, na=False)].copy()
    if eligible.empty:
        raise RuntimeError(f"No eligible v154 rule for signal_filter={signal_filter!r}")
    return eligible.sort_values(
        ["strict_delta_bacc", "strict_net_errors_reduced", "subset_delta_bacc"],
        ascending=False,
    ).iloc[0]


def scope_line(v154_all: pd.DataFrame, rule_name: str, scope: str) -> str:
    rows = v154_all.loc[v154_all["rule_name"].eq(rule_name) & v154_all["eval_scope"].eq(scope)]
    if rows.empty:
        return f"{scope}: not available"
    r = rows.iloc[0]
    return (
        f"{scope}: flips {int(r['flip_n'])}, net {int(r['net_errors_reduced'])}, "
        f"rescued {int(r['rescued_n'])}, hurt {int(r['hurt_n'])}, Delta BAcc {pp(r['delta_bacc'])}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v153 = pd.read_csv(V153)
    v154 = pd.read_csv(V154)
    v154_all = pd.read_csv(V154_ALL)
    module = pd.read_csv(V152_MODULE)

    best_v153 = v153.sort_values(["internal_flip_n", "old_net_errors_reduced", "third_net_errors_reduced"], ascending=False).iloc[0]
    best_v154 = best_v154_rule(v154)
    best_concept_v154 = best_v154_rule(v154, r"v144_concept|mean_v143_v144")

    rows = [
        {
            "line": "all-domain high-precision auto-corrector",
            "selection_data": "old+third full internal no-harm",
            "selected_rule": best_v153["rule_name"],
            "internal_behavior": f"flips {int(best_v153['internal_flip_n'])} internal cases; old net {int(best_v153['old_net_errors_reduced'])}, third net {int(best_v153['third_net_errors_reduced'])}",
            "strict_external_behavior": f"flips {int(best_v153['strict_flip_n'])} external cases; Delta BAcc {pp(best_v153['strict_delta_bacc'])}",
            "status": "safe but too narrow",
            "paper_use": "Supports the safety boundary: strict no-harm internal locking yields almost no severe-shift coverage.",
        },
        {
            "line": "pseudo-severe concept-direction auto-corrector",
            "selection_data": str(best_concept_v154["pseudo_subset"]),
            "selected_rule": best_concept_v154["rule_name"],
            "internal_behavior": f"pseudo subset n={int(best_concept_v154['subset_n'])}, flips {int(best_concept_v154['subset_flip_n'])}, rescued {int(best_concept_v154['subset_rescued'])}, hurt {int(best_concept_v154['subset_hurt'])}",
            "strict_external_behavior": f"flips {int(best_concept_v154['strict_flip_n'])}, rescued {int(best_concept_v154['strict_rescued'])}, hurt {int(best_concept_v154['strict_hurt'])}, BAcc {pct(best_concept_v154['strict_final_bacc'])}, Delta BAcc {pp(best_concept_v154['strict_delta_bacc'])}",
            "status": "promising severe-shift candidate",
            "paper_use": "Supports the next method claim: quality-proxy pseudo-shift can select concept-direction correction without external labels, but needs stability validation.",
            "ordinary_domain_boundary": "; ".join(
                [
                    scope_line(v154_all, str(best_concept_v154["rule_name"]), "old_data"),
                    scope_line(v154_all, str(best_concept_v154["rule_name"]), "third_batch"),
                ]
            ),
        },
    ]
    if str(best_v154["rule_name"]) != str(best_concept_v154["rule_name"]):
        rows.append(
            {
                "line": "pseudo-severe best-overall auto-corrector",
                "selection_data": str(best_v154["pseudo_subset"]),
                "selected_rule": best_v154["rule_name"],
                "internal_behavior": f"pseudo subset n={int(best_v154['subset_n'])}, flips {int(best_v154['subset_flip_n'])}, rescued {int(best_v154['subset_rescued'])}, hurt {int(best_v154['subset_hurt'])}",
                "strict_external_behavior": f"flips {int(best_v154['strict_flip_n'])}, rescued {int(best_v154['strict_rescued'])}, hurt {int(best_v154['strict_hurt'])}, BAcc {pct(best_v154['strict_final_bacc'])}, Delta BAcc {pp(best_v154['strict_delta_bacc'])}",
                "status": "exploratory benchmark",
                "paper_use": "Kept as a numerical benchmark, not the preferred mechanistic claim if it is not concept-directional.",
                "ordinary_domain_boundary": "; ".join(
                    [
                        scope_line(v154_all, str(best_v154["rule_name"]), "old_data"),
                        scope_line(v154_all, str(best_v154["rule_name"]), "third_batch"),
                    ]
                ),
            }
        )
    update = pd.DataFrame(rows)
    update.to_csv(OUT_DIR / "v155_autocorrect_evidence_update.csv", index=False, encoding="utf-8-sig")

    updated_module = pd.concat(
        [
            module,
            pd.DataFrame(
                [
                    {
                        "module": "Quality-proxy pseudo-severe auto-correction",
                        "main_result": f"Strict external BAcc {pct(best_concept_v154['strict_final_bacc'])}, Delta BAcc {pp(best_concept_v154['strict_delta_bacc'])}",
                        "evidence": f"v154 selected by {best_concept_v154['pseudo_subset']}; strict rescued {int(best_concept_v154['strict_rescued'])}, hurt {int(best_concept_v154['strict_hurt'])}",
                        "status": "promising candidate; not locked",
                        "interpretation": "Pseudo severe-shift development can identify external-effective concept correction, but ordinary-domain harm and small pseudo subset require validation.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    updated_module.to_csv(OUT_DIR / "v155_updated_module_evidence_table.csv", index=False, encoding="utf-8-sig")

    md = [
        "# v155 Auto-correction Evidence Update",
        "",
        "## Key Change After v152",
        "",
        "v151 showed that all-domain no-harm selection forces auto-flip to 0%. v154 adds a more targeted result: when internal quality-proxy pseudo-severe subsets are used for selection, concept-direction correction can improve strict external without using external labels for rule selection.",
        "",
        "## Evidence Rows",
        "",
    ]
    for r in update.itertuples():
        md.append(f"- {r.line}: {r.strict_external_behavior}. Status: {r.status}.")
        boundary = getattr(r, "ordinary_domain_boundary", "")
        if isinstance(boundary, str) and boundary:
            md.append(f"  Ordinary-domain boundary: {boundary}.")
    md += [
        "",
        "## Boundary",
        "",
        "This does not upgrade auto-correction to a locked main workflow. The selected pseudo-severe subset is small, and the same rule can hurt ordinary old/third domains if the severe-shift gate is wrong. It should be written as a promising candidate that motivates a dedicated severe-shift validation batch.",
    ]
    (OUT_DIR / "v155_autocorrect_evidence_update.md").write_text("\n".join(md), encoding="utf-8")
    report = {
        "best_v153_rule": str(best_v153["rule_name"]),
        "best_v154_rule": str(best_v154["rule_name"]),
        "best_concept_v154_rule": str(best_concept_v154["rule_name"]),
        "main_boundary": "auto-correction remains candidate; pseudo-severe selection is promising but not final locked policy.",
    }
    (OUT_DIR / "v155_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v155] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
