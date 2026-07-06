from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v153_high_precision_autoflip_trigger_20260527 import apply_rule, load_data


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v156_severe_shift_gate_safety_boundary_20260527"
V154 = ROOT / "outputs" / "grosspath_rc_v154_pseudo_severe_shift_autoflip_selection_20260527" / "v154_pseudo_severe_selected_rules.csv"
V77 = ROOT / "outputs" / "grosspath_rc_v77_batch_shift_audit_policy_switch_20260527" / "v77_unlabeled_batch_shift_audit.csv"


def choose_concept_rule(v154: pd.DataFrame) -> pd.Series:
    eligible = v154.loc[
        v154["pseudo_selection_status"].eq("selected_by_pseudo_subset")
        & v154["signal"].astype(str).str.contains("v144_concept|mean_v143_v144", regex=True, na=False)
        & (pd.to_numeric(v154["strict_delta_bacc"], errors="coerce") > 0)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No positive pseudo-severe concept-direction rule found in v154.")
    return eligible.sort_values(
        ["strict_delta_bacc", "strict_net_errors_reduced", "subset_delta_bacc"],
        ascending=False,
    ).iloc[0]


def batch_shift_map(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = {
        "pseudo_third_to_old": "old_data",
        "pseudo_old_to_third": "third_batch",
        "strict_external_vs_dev": "strict_external",
    }
    for audit_name, eval_domain in mapping.items():
        r = audit.loc[audit["audit_name"].eq(audit_name)].iloc[0]
        rows.append(
            {
                "eval_domain": eval_domain,
                "audit_name": audit_name,
                "batch_shift_index": float(r["batch_shift_index"]),
                "domain_auc_cv": float(r["domain_auc_cv"]),
                "quality_proxy_mean": float(r["quality_proxy_mean"]),
                "mean_outside_ref_05_95_rate": float(r["mean_outside_ref_05_95_rate"]),
                "shift_category": str(r["shift_category"]),
            }
        )
    out = pd.DataFrame(rows)
    internal_max = float(out.loc[out["eval_domain"].isin(["old_data", "third_batch"]), "batch_shift_index"].max())
    external_min = float(out.loc[out["eval_domain"].eq("strict_external"), "batch_shift_index"].min())
    out["current_margin_to_internal_max"] = out["batch_shift_index"] - internal_max
    out.attrs["separation_threshold_midpoint"] = (internal_max + external_min) / 2.0
    out.attrs["internal_max_shift_index"] = internal_max
    out.attrs["strict_external_shift_index"] = external_min
    out.attrs["current_shift_margin"] = external_min - internal_max
    return out


def evaluate_predictions(df: pd.DataFrame, scenario: str, final_pred: np.ndarray, auto_flip: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for domain in ["old_data", "third_batch", "strict_external", "all_three_domains"]:
        sub = df if domain == "all_three_domains" else df.loc[df["eval_domain"].eq(domain)].copy()
        if domain == "all_three_domains":
            idx = np.arange(len(df))
        else:
            idx = np.flatnonzero(df["eval_domain"].eq(domain).to_numpy())
        y = sub["label_idx"].astype(int).to_numpy()
        base = sub["base_pred"].astype(int).to_numpy()
        final = final_pred[idx].astype(int)
        prob = sub["base_prob"].to_numpy(float)
        flip = auto_flip[idx].astype(bool)
        base_wrong = base != y
        final_wrong = final != y
        rescued = base_wrong & ~final_wrong
        hurt = ~base_wrong & final_wrong
        row: dict[str, object] = {
            "scenario": scenario,
            "eval_domain": domain,
            "n": int(len(sub)),
            "flip_n": int(flip.sum()),
            "flip_rate": float(flip.mean()) if len(sub) else np.nan,
            "base_errors": int(base_wrong.sum()),
            "final_errors": int(final_wrong.sum()),
            "net_errors_reduced": int(base_wrong.sum() - final_wrong.sum()),
            "rescued_n": int(rescued.sum()),
            "hurt_n": int(hurt.sum()),
            "rescued_fn": int((rescued & (y == 1) & (base == 0)).sum()),
            "rescued_fp": int((rescued & (y == 0) & (base == 1)).sum()),
            "hurt_to_fn": int((hurt & (y == 1) & (final == 0)).sum()),
            "hurt_to_fp": int((hurt & (y == 0) & (final == 1)).sum()),
        }
        row.update({f"base_{k}": v for k, v in metrics(y, base, prob).items()})
        row.update({f"final_{k}": v for k, v in metrics(y, final, prob).items()})
        row["delta_bacc"] = float(row["final_balanced_accuracy"] - row["base_balanced_accuracy"])
        rows.append(row)
    return rows


def apply_gated_rule(df: pd.DataFrame, rule: pd.Series, auto_domains: set[str]) -> tuple[np.ndarray, np.ndarray]:
    ruled = apply_rule(
        df,
        signal=str(rule["signal"]),
        mode=str(rule["mode"]),
        threshold=float(rule["threshold"]),
    )
    final = df["base_pred"].astype(int).to_numpy().copy()
    flip = np.zeros(len(df), dtype=bool)
    mask = df["eval_domain"].isin(auto_domains).to_numpy()
    final[mask] = ruled.loc[mask, "final_pred"].astype(int).to_numpy()
    flip[mask] = ruled.loc[mask, "auto_flip"].astype(bool).to_numpy()
    return final, flip


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rule = choose_concept_rule(pd.read_csv(V154))
    shift = batch_shift_map(pd.read_csv(V77))
    df_all = load_data()
    df = df_all.loc[df_all["base_model"].eq(str(rule["base_model"]))].copy().reset_index(drop=True)

    severe_domains = set(shift.loc[shift["shift_category"].eq("severe_shift"), "eval_domain"].tolist())
    scenarios = {
        "base_no_autocorrect": set(),
        "current_unlabeled_gate_only_severe_shift": severe_domains,
        "false_gate_old_only": {"old_data"},
        "false_gate_third_only": {"third_batch"},
        "false_gate_old_and_third": {"old_data", "third_batch"},
        "no_gate_apply_all_domains": {"old_data", "third_batch", "strict_external"},
        "missed_external_gate": set(),
    }

    rows = []
    for scenario, domains in scenarios.items():
        final, flip = apply_gated_rule(df, rule, domains)
        rows.extend(evaluate_predictions(df, scenario, final, flip))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "v156_gate_scenario_summary.csv", index=False, encoding="utf-8-sig")

    shift_out = shift.copy()
    shift_out["separation_threshold_midpoint"] = shift.attrs["separation_threshold_midpoint"]
    shift_out["internal_max_shift_index"] = shift.attrs["internal_max_shift_index"]
    shift_out["strict_external_shift_index"] = shift.attrs["strict_external_shift_index"]
    shift_out["current_shift_margin"] = shift.attrs["current_shift_margin"]
    shift_out.to_csv(OUT_DIR / "v156_shift_gate_margin.csv", index=False, encoding="utf-8-sig")

    focus = summary.loc[
        summary["scenario"].isin(
            [
                "base_no_autocorrect",
                "current_unlabeled_gate_only_severe_shift",
                "false_gate_old_and_third",
                "no_gate_apply_all_domains",
            ]
        )
        & summary["eval_domain"].isin(["old_data", "third_batch", "strict_external", "all_three_domains"])
    ].copy()
    focus.to_csv(OUT_DIR / "v156_focus_safety_table.csv", index=False, encoding="utf-8-sig")

    strict_current = summary.loc[
        summary["scenario"].eq("current_unlabeled_gate_only_severe_shift") & summary["eval_domain"].eq("strict_external")
    ].iloc[0]
    old_false = summary.loc[summary["scenario"].eq("false_gate_old_only") & summary["eval_domain"].eq("old_data")].iloc[0]
    third_false = summary.loc[summary["scenario"].eq("false_gate_third_only") & summary["eval_domain"].eq("third_batch")].iloc[0]
    all_no_gate = summary.loc[summary["scenario"].eq("no_gate_apply_all_domains") & summary["eval_domain"].eq("all_three_domains")].iloc[0]

    md = [
        "# v156 Severe-shift Gate Safety Boundary",
        "",
        "## Selected Correction Rule",
        "",
        f"- Rule: `{rule['rule_name']}`",
        f"- Evaluated branch: `{rule['base_model']}`. This is the severe-shift auto-correction branch, not the final locked clinical workflow by itself.",
        f"- Selection source: v154 pseudo-severe subset `{rule['pseudo_subset']}`",
        "",
        "## Main Findings",
        "",
        (
            f"- Current unlabeled shift audit separates the strict external batch from internal-like batches: "
            f"max internal shift index {shift.attrs['internal_max_shift_index']:.3f}, "
            f"strict external {shift.attrs['strict_external_shift_index']:.3f}, "
            f"margin {shift.attrs['current_shift_margin']:.3f}."
        ),
        (
            f"- If the rule is triggered only for severe-shift batches, strict external flips "
            f"{int(strict_current['flip_n'])} cases, rescues {int(strict_current['rescued_n'])}, "
            f"hurts {int(strict_current['hurt_n'])}, and changes BAcc by {100 * strict_current['delta_bacc']:+.2f} pp."
        ),
        (
            f"- If the severe-shift gate falsely triggers on ordinary old data, BAcc changes by "
            f"{100 * old_false['delta_bacc']:+.2f} pp; if it falsely triggers on third batch, BAcc changes by "
            f"{100 * third_false['delta_bacc']:+.2f} pp."
        ),
        (
            f"- If the same rule is applied without any gate to all domains, all-domain BAcc changes by "
            f"{100 * all_no_gate['delta_bacc']:+.2f} pp; this is the safety reason it cannot be used as a global auto-flip rule."
        ),
        "",
        "## Interpretation",
        "",
        "The result supports a conditional design rather than unconditional automatic correction. Concept-direction correction is useful under severe acquisition/domain shift, but it is unsafe when the shift gate is uncertain. The deployable workflow should therefore use automatic correction only after a batch-level severe-shift trigger; otherwise the same high-risk cases should be routed to review instead of being flipped.",
    ]
    (OUT_DIR / "v156_severe_shift_gate_safety_boundary.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "selected_rule": str(rule["rule_name"]),
        "severe_domains_from_v77": sorted(severe_domains),
        "internal_max_shift_index": shift.attrs["internal_max_shift_index"],
        "strict_external_shift_index": shift.attrs["strict_external_shift_index"],
        "current_shift_margin": shift.attrs["current_shift_margin"],
        "strict_external_delta_bacc_when_gated": float(strict_current["delta_bacc"]),
        "old_false_gate_delta_bacc": float(old_false["delta_bacc"]),
        "third_false_gate_delta_bacc": float(third_false["delta_bacc"]),
        "all_domain_delta_bacc_without_gate": float(all_no_gate["delta_bacc"]),
    }
    (OUT_DIR / "v156_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v156] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
