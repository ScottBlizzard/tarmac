from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT
from run_grosspath_rc_v153_high_precision_autoflip_trigger_20260527 import load_data
from run_grosspath_rc_v156_severe_shift_gate_safety_boundary_20260527 import (
    V154,
    apply_gated_rule,
    choose_concept_rule,
    evaluate_predictions,
)


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v158_unlabeled_gate_threshold_stability_20260527"
V77 = ROOT / "outputs" / "grosspath_rc_v77_batch_shift_audit_policy_switch_20260527" / "v77_unlabeled_batch_shift_audit.csv"


def pp(x: float, digits: int = 2) -> str:
    return f"{100 * float(x):+.{digits}f} pp"


def domain_table(audit: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "pseudo_third_to_old": "old_data",
        "pseudo_old_to_third": "third_batch",
        "strict_external_vs_dev": "strict_external",
    }
    rows = []
    for audit_name, eval_domain in mapping.items():
        r = audit.loc[audit["audit_name"].eq(audit_name)].iloc[0]
        rows.append(
            {
                "eval_domain": eval_domain,
                "batch_shift_index": float(r["batch_shift_index"]),
                "domain_auc_cv": float(r["domain_auc_cv"]),
                "quality_proxy_mean": float(r["quality_proxy_mean"]),
                "mean_outside_ref_05_95_rate": float(r["mean_outside_ref_05_95_rate"]),
                "shift_category_v77": str(r["shift_category"]),
            }
        )
    return pd.DataFrame(rows)


def gate_domains(tab: pd.DataFrame, gate_name: str, internal_max: pd.Series) -> set[str]:
    shift = tab["batch_shift_index"] > internal_max["batch_shift_index"] * 1.35
    auc = tab["domain_auc_cv"] > max(0.90, internal_max["domain_auc_cv"] + 0.05)
    quality = tab["quality_proxy_mean"] > internal_max["quality_proxy_mean"] * 1.50
    outside = tab["mean_outside_ref_05_95_rate"] > internal_max["mean_outside_ref_05_95_rate"] * 1.50
    if gate_name == "combined_any_v77_style":
        mask = shift | auc | quality
    elif gate_name == "combined_any_plus_outside":
        mask = shift | auc | quality | outside
    elif gate_name == "strict_all_core_components":
        mask = shift & auc & quality
    elif gate_name == "shift_index_only_1p35x":
        mask = shift
    elif gate_name == "domain_auc_only_0p90":
        mask = auc
    elif gate_name == "quality_mean_only_1p50x":
        mask = quality
    elif gate_name == "outside_rate_only_1p50x":
        mask = outside
    elif gate_name == "oversensitive_shift_index_0p75x":
        mask = tab["batch_shift_index"] > internal_max["batch_shift_index"] * 0.75
    elif gate_name == "too_strict_shift_index_2p50x":
        mask = tab["batch_shift_index"] > internal_max["batch_shift_index"] * 2.50
    else:
        raise ValueError(gate_name)
    return set(tab.loc[mask, "eval_domain"].tolist())


def evaluate_gate_outcome(gate_name: str, domains: set[str], rule: pd.Series) -> pd.DataFrame:
    df_all = load_data()
    df = df_all.loc[df_all["base_model"].eq(str(rule["base_model"]))].copy().reset_index(drop=True)
    final, flip = apply_gated_rule(df, rule, domains)
    out = pd.DataFrame(evaluate_predictions(df, gate_name, final, flip))
    out["severe_domains"] = ",".join(sorted(domains)) if domains else "none"
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(V77)
    tab = domain_table(audit)
    internal = tab.loc[tab["eval_domain"].isin(["old_data", "third_batch"])]
    strict = tab.loc[tab["eval_domain"].eq("strict_external")].iloc[0]
    internal_max = internal[
        [
            "batch_shift_index",
            "domain_auc_cv",
            "quality_proxy_mean",
            "mean_outside_ref_05_95_rate",
        ]
    ].max()

    margin_rows = []
    for metric in [
        "batch_shift_index",
        "domain_auc_cv",
        "quality_proxy_mean",
        "mean_outside_ref_05_95_rate",
    ]:
        imax = float(internal_max[metric])
        sval = float(strict[metric])
        midpoint = (imax + sval) / 2.0
        margin_rows.append(
            {
                "metric": metric,
                "internal_max": imax,
                "strict_external": sval,
                "absolute_margin": sval - imax,
                "strict_to_internal_ratio": sval / max(imax, 1e-12),
                "midpoint_threshold_current_batches": midpoint,
                "separates_current_batches": bool(sval > imax),
            }
        )
    margin = pd.DataFrame(margin_rows)
    margin.to_csv(OUT_DIR / "v158_gate_metric_margin.csv", index=False, encoding="utf-8-sig")

    gate_names = [
        "combined_any_v77_style",
        "combined_any_plus_outside",
        "strict_all_core_components",
        "shift_index_only_1p35x",
        "domain_auc_only_0p90",
        "quality_mean_only_1p50x",
        "outside_rate_only_1p50x",
        "oversensitive_shift_index_0p75x",
        "too_strict_shift_index_2p50x",
    ]
    component_rows = []
    outcome_rows = []
    rule = choose_concept_rule(pd.read_csv(V154))
    for gate_name in gate_names:
        domains = gate_domains(tab, gate_name, internal_max)
        component_rows.append(
            {
                "gate_name": gate_name,
                "severe_domains": ",".join(sorted(domains)) if domains else "none",
                "false_internal_trigger": bool({"old_data", "third_batch"} & domains),
                "missed_strict_external": "strict_external" not in domains,
                "current_safety_status": (
                    "clean_current_split"
                    if domains == {"strict_external"}
                    else "false_internal_trigger"
                    if {"old_data", "third_batch"} & domains
                    else "missed_external"
                ),
            }
        )
        outcome_rows.extend(evaluate_gate_outcome(gate_name, domains, rule).to_dict("records"))
    components = pd.DataFrame(component_rows)
    outcomes = pd.DataFrame(outcome_rows)
    components.to_csv(OUT_DIR / "v158_gate_component_stability.csv", index=False, encoding="utf-8-sig")
    outcomes.to_csv(OUT_DIR / "v158_gate_policy_outcome.csv", index=False, encoding="utf-8-sig")

    focus = outcomes.loc[outcomes["eval_domain"].eq("all_three_domains")].copy()
    focus = focus[
        [
            "scenario",
            "severe_domains",
            "flip_n",
            "net_errors_reduced",
            "rescued_n",
            "hurt_n",
            "base_balanced_accuracy",
            "final_balanced_accuracy",
            "delta_bacc",
        ]
    ]
    focus.to_csv(OUT_DIR / "v158_all_domain_gate_outcome_focus.csv", index=False, encoding="utf-8-sig")

    clean_count = int(components["current_safety_status"].eq("clean_current_split").sum())
    false_count = int(components["false_internal_trigger"].sum())
    missed_count = int(components["missed_strict_external"].sum())
    combined = focus.loc[focus["scenario"].eq("combined_any_v77_style")].iloc[0]
    oversensitive = focus.loc[focus["scenario"].eq("oversensitive_shift_index_0p75x")].iloc[0]
    toostrict = focus.loc[focus["scenario"].eq("too_strict_shift_index_2p50x")].iloc[0]

    md = [
        "# v158 Unlabeled Severe-shift Gate Threshold Stability",
        "",
        "## Main Findings",
        "",
        f"- Among {len(gate_names)} no-label gate variants, {clean_count} cleanly flag only the strict external batch in the current data.",
        f"- False internal triggers appear in {false_count} variants; missed strict-external triggers appear in {missed_count} variants.",
        (
            f"- The v77-style combined gate changes all-domain BAcc by {pp(combined['delta_bacc'])} "
            f"with severe domains `{combined['severe_domains']}`."
        ),
        (
            f"- An over-sensitive shift-index gate changes all-domain BAcc by {pp(oversensitive['delta_bacc'])}; "
            f"a too-strict shift-index gate changes all-domain BAcc by {pp(toostrict['delta_bacc'])}."
        ),
        "",
        "## Interpretation",
        "",
        "The current strict external batch is not separated by a single fragile statistic. Batch separability AUC, quality proxy mean, outside-reference rate, and the composite shift index all point in the same direction. However, the stress gates show why the severe-shift trigger must be validated prospectively: an over-sensitive gate can falsely activate automatic correction on internal-like data, while an overly strict gate misses the external shift and loses the correction opportunity.",
    ]
    (OUT_DIR / "v158_unlabeled_gate_threshold_stability.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "gate_variant_count": int(len(gate_names)),
        "clean_current_split_count": clean_count,
        "false_internal_trigger_count": false_count,
        "missed_strict_external_count": missed_count,
        "combined_any_v77_style_delta_bacc": float(combined["delta_bacc"]),
        "oversensitive_shift_index_delta_bacc": float(oversensitive["delta_bacc"]),
        "too_strict_shift_index_delta_bacc": float(toostrict["delta_bacc"]),
    }
    (OUT_DIR / "v158_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v158] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
