from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v151_no_leak_autoflip_budget_selection_20260527"
V149_SUMMARY = ROOT / "outputs" / "grosspath_rc_v149_directional_auto_flip_corrector_20260527" / "v149_auto_flip_corrector_summary.csv"

FOCUS_POLICIES = [
    "baseline_low_conf_dev_selected",
    "dev_stable_router_all_domains",
    "shift_aware_image_directional_candidate",
    "shift_aware_concept_directional_candidate",
]


def select_budget(summary: pd.DataFrame, policy: str) -> tuple[float, pd.DataFrame]:
    dev = summary.loc[
        summary["policy"].eq(policy)
        & summary["eval_domain"].isin(["old_data", "third_batch"])
    ].copy()
    piv = dev.pivot_table(
        index="flip_budget",
        columns="eval_domain",
        values=["net_errors_reduced", "auto_flip_balanced_accuracy", "base_balanced_accuracy"],
        aggfunc="first",
    )
    rows = []
    for budget in sorted(dev["flip_budget"].unique()):
        row = {"policy": policy, "flip_budget": float(budget)}
        ok = True
        min_net = np.inf
        min_delta_bacc = np.inf
        mean_delta_bacc = 0.0
        for domain in ["old_data", "third_batch"]:
            d = dev.loc[np.isclose(dev["flip_budget"].astype(float), budget) & dev["eval_domain"].eq(domain)].iloc[0]
            delta = float(d["auto_flip_balanced_accuracy"] - d["base_balanced_accuracy"])
            net = int(d["net_errors_reduced"])
            row[f"{domain}_net_errors_reduced"] = net
            row[f"{domain}_delta_bacc"] = delta
            min_net = min(min_net, net)
            min_delta_bacc = min(min_delta_bacc, delta)
            mean_delta_bacc += delta / 2.0
            if net < 0 or delta < -1e-12:
                ok = False
        row["passes_internal_no_harm"] = bool(ok)
        row["min_internal_net_errors_reduced"] = int(min_net)
        row["min_internal_delta_bacc"] = float(min_delta_bacc)
        row["mean_internal_delta_bacc"] = float(mean_delta_bacc)
        rows.append(row)
    ranked = pd.DataFrame(rows)
    passed = ranked.loc[ranked["passes_internal_no_harm"]].copy()
    if passed.empty:
        selected = 0.0
    else:
        passed = passed.sort_values(["flip_budget", "mean_internal_delta_bacc"], ascending=[False, False])
        selected = float(passed.iloc[0]["flip_budget"])
    return selected, ranked


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(V149_SUMMARY)
    rows = []
    ranks = []
    for policy in FOCUS_POLICIES:
        selected, rank = select_budget(summary, policy)
        ranks.append(rank)
        for domain in ["old_data", "third_batch", "strict_external", "all_three_domains"]:
            row = summary.loc[
                summary["policy"].eq(policy)
                & summary["eval_domain"].eq(domain)
                & np.isclose(summary["flip_budget"].astype(float), selected)
            ].iloc[0].to_dict()
            row["selected_by_internal_no_harm"] = selected
            rows.append(row)
    selected_df = pd.DataFrame(rows)
    rank_df = pd.concat(ranks, ignore_index=True)
    rank_df.to_csv(OUT_DIR / "v151_internal_autoflip_budget_rank.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(OUT_DIR / "v151_no_leak_selected_autoflip_external_check.csv", index=False, encoding="utf-8-sig")
    report = {
        "selection_rule": "Select the largest auto-flip budget whose old and third batch net_errors_reduced are both nonnegative and BAcc does not drop. Strict external is only checked after selection.",
        "interpretation": "If this selects 0, the current auto-corrector is not lockable without a severe-shift validation batch.",
    }
    (OUT_DIR / "v151_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v151] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
