from __future__ import annotations

import json

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v160_safety_efficiency_operating_frontier_20260527"
V147_SUMMARY = ROOT / "outputs" / "grosspath_rc_v147_unlabeled_shift_aware_router_card_20260527" / "v147_shift_aware_policy_budget_summary.csv"
V159 = ROOT / "outputs" / "grosspath_rc_v159_unified_workflow_operating_table_20260527" / "v159_unified_workflow_operating_table.csv"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def row_from_v147(g: pd.DataFrame) -> dict[str, object]:
    all_row = g.loc[g["eval_domain"].eq("all_three_domains")].iloc[0]
    domains = g.loc[g["eval_domain"].isin(["old_data", "third_batch", "strict_external"])].copy()
    strict = domains.loc[domains["eval_domain"].eq("strict_external")].iloc[0]
    return {
        "operating_point": f"{all_row['policy']}@review{float(all_row['review_budget']):.1f}",
        "source": "v147_direction_router_curve",
        "policy": str(all_row["policy"]),
        "review_rate": float(all_row["review_rate"]),
        "auto_pass_rate": float(all_row["auto_rate"]),
        "auto_correct_rate": 0.0,
        "all_bacc": float(all_row["system_balanced_accuracy"]),
        "all_acc": float(all_row["system_accuracy"]),
        "all_f1": float(all_row["system_f1"]),
        "all_fn": int(all_row["system_fn"]),
        "all_fp": int(all_row["system_fp"]),
        "strict_bacc": float(strict["system_balanced_accuracy"]),
        "strict_fn": int(strict["system_fn"]),
        "strict_fp": int(strict["system_fp"]),
        "min_domain_bacc": float(domains["system_balanced_accuracy"].min()),
        "max_domain_fn": int(domains["system_fn"].max()),
        "max_domain_fp": int(domains["system_fp"].max()),
        "status": "selective review curve",
    }


def scorecard_rows(v159: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for evidence in [
        "locked_high_safety_two_signal_scorecard",
        "severe_shift_gated_concept_direction_autocorrect",
    ]:
        sub = v159.loc[v159["evidence_line"].eq(evidence)].copy()
        all_row = sub.loc[sub["eval_domain"].eq("all_domains")].iloc[0]
        domains = sub.loc[sub["eval_domain"].isin(["old_data", "third_batch", "strict_external"])].copy()
        strict = domains.loc[domains["eval_domain"].eq("strict_external")].iloc[0]
        rows.append(
            {
                "operating_point": evidence,
                "source": "v159_unified_table",
                "policy": evidence,
                "review_rate": float(all_row["review_or_reject_rate"]),
                "auto_pass_rate": float(all_row["auto_pass_rate"]),
                "auto_correct_rate": float(all_row["auto_correct_rate"]),
                "all_bacc": float(all_row["balanced_accuracy"]),
                "all_acc": float(all_row["accuracy"]),
                "all_f1": float(all_row["f1"]),
                "all_fn": int(all_row["fn"]),
                "all_fp": int(all_row["fp"]),
                "strict_bacc": float(strict["balanced_accuracy"]),
                "strict_fn": int(strict["fn"]),
                "strict_fp": int(strict["fp"]),
                "min_domain_bacc": float(domains["balanced_accuracy"].min()),
                "max_domain_fn": int(domains["fn"].max()),
                "max_domain_fp": int(domains["fp"].max()),
                "status": "main high-safety workflow" if evidence.startswith("locked") else "auto-correction module candidate",
            }
        )
    return rows


def build_pool() -> pd.DataFrame:
    v147 = pd.read_csv(V147_SUMMARY)
    rows = []
    for _, g in v147.groupby(["policy", "review_budget"], sort=False):
        rows.append(row_from_v147(g))
    v159 = pd.read_csv(V159)
    rows.extend(scorecard_rows(v159))
    pool = pd.DataFrame(rows)
    pool["review_rate_round"] = pool["review_rate"].round(3)
    return pool.sort_values(["review_rate", "all_bacc"], ascending=[True, False])


def select_point(pool: pd.DataFrame, name: str, mask: pd.Series, sort_cols: list[str], ascending: list[bool]) -> dict[str, object]:
    cand = pool.loc[mask].copy()
    if cand.empty:
        return {"constraint": name, "selected": False, "reason": "no candidate"}
    top = cand.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
    top["constraint"] = name
    top["selected"] = True
    return top


def constraint_table(pool: pd.DataFrame) -> pd.DataFrame:
    rows = [
        select_point(
            pool,
            "max_30pct_review_best_all_bacc",
            pool["review_rate"].le(0.305),
            ["all_bacc", "strict_bacc", "review_rate"],
            [False, False, True],
        ),
        select_point(
            pool,
            "max_40pct_review_best_all_bacc",
            pool["review_rate"].le(0.405),
            ["all_bacc", "strict_bacc", "review_rate"],
            [False, False, True],
        ),
        select_point(
            pool,
            "max_60pct_review_best_all_bacc",
            pool["review_rate"].le(0.605),
            ["all_bacc", "strict_bacc", "review_rate"],
            [False, False, True],
        ),
        select_point(
            pool,
            "max_60pct_review_best_min_domain_bacc",
            pool["review_rate"].le(0.605),
            ["min_domain_bacc", "all_bacc", "review_rate"],
            [False, False, True],
        ),
        select_point(
            pool,
            "all_bacc_at_least_95_min_review",
            pool["all_bacc"].ge(0.95),
            ["review_rate", "all_bacc"],
            [True, False],
        ),
        select_point(
            pool,
            "min_domain_bacc_at_least_95_min_review",
            pool["min_domain_bacc"].ge(0.95),
            ["review_rate", "min_domain_bacc", "all_bacc"],
            [True, False, False],
        ),
        select_point(
            pool,
            "near_zero_error_high_safety",
            (pool["all_fn"].le(1)) & (pool["all_fp"].eq(0)),
            ["review_rate", "all_bacc"],
            [True, False],
        ),
    ]
    return pd.DataFrame(rows)


def frontier(pool: pd.DataFrame) -> pd.DataFrame:
    candidates = pool.copy().sort_values(["review_rate", "all_bacc"], ascending=[True, False])
    rows = []
    best = -np.inf
    for _, r in candidates.iterrows():
        if float(r["all_bacc"]) > best + 1e-12:
            rows.append(r.to_dict())
            best = float(r["all_bacc"])
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = build_pool()
    selected = constraint_table(pool)
    front = frontier(pool)

    pool.to_csv(OUT_DIR / "v160_operating_point_pool.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v160_constraint_selected_points.csv", index=False, encoding="utf-8-sig")
    front.to_csv(OUT_DIR / "v160_all_bacc_review_frontier.csv", index=False, encoding="utf-8-sig")

    p30 = selected.loc[selected["constraint"].eq("max_30pct_review_best_all_bacc")].iloc[0]
    p60 = selected.loc[selected["constraint"].eq("max_60pct_review_best_all_bacc")].iloc[0]
    p60_min = selected.loc[selected["constraint"].eq("max_60pct_review_best_min_domain_bacc")].iloc[0]
    p95 = selected.loc[selected["constraint"].eq("min_domain_bacc_at_least_95_min_review")].iloc[0]
    pzero = selected.loc[selected["constraint"].eq("near_zero_error_high_safety")].iloc[0]

    md = [
        "# v160 Safety-efficiency Operating Frontier",
        "",
        "## Main Findings",
        "",
        (
            f"- With about 30% review, the best all-domain point is `{p30['operating_point']}`: "
            f"all-domain BAcc {pct(p30['all_bacc'])}, strict external BAcc {pct(p30['strict_bacc'])}, "
            f"auto-pass {pct(p30['auto_pass_rate'])}."
        ),
        (
            f"- With about 60% review, the best all-domain point is `{p60['operating_point']}`: "
            f"all-domain BAcc {pct(p60['all_bacc'])}, but strict external BAcc {pct(p60['strict_bacc'])}."
        ),
        (
            f"- If the objective is best worst-domain safety under 60% review, `{p60_min['operating_point']}` is selected: "
            f"minimum domain BAcc {pct(p60_min['min_domain_bacc'])}, all-domain BAcc {pct(p60_min['all_bacc'])}."
        ),
        (
            f"- The first point with minimum-domain BAcc >=95% is `{p95['operating_point']}` with review/reject "
            f"{pct(p95['review_rate'])}."
        ),
        (
            f"- The near-zero-error point is `{pzero['operating_point']}` with all-domain FN={int(pzero['all_fn'])}, "
            f"FP={int(pzero['all_fp'])}, review/reject {pct(pzero['review_rate'])}."
        ),
        "",
        "## Interpretation",
        "",
        "The current system has a clear safety-efficiency gap. A 30%-60% review workflow is more deployable but does not yet reach the high-safety regime across domains, especially on strict external. The two-signal scorecard reaches near-zero error but at high review burden. Therefore the next model-development target should be efficiency: raise auto-pass rate while preserving the scorecard's FN/FP safety, rather than only increasing the already high reviewed-system accuracy.",
    ]
    (OUT_DIR / "v160_safety_efficiency_operating_frontier.md").write_text("\n".join(md), encoding="utf-8")

    report = {
        "pool_rows": int(len(pool)),
        "frontier_rows": int(len(front)),
        "best_30_review": str(p30["operating_point"]),
        "best_30_review_all_bacc": float(p30["all_bacc"]),
        "best_60_review": str(p60["operating_point"]),
        "best_60_review_all_bacc": float(p60["all_bacc"]),
        "min_domain_ge95_min_review": str(p95["operating_point"]),
        "near_zero_error_point": str(pzero["operating_point"]),
    }
    (OUT_DIR / "v160_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v160] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
