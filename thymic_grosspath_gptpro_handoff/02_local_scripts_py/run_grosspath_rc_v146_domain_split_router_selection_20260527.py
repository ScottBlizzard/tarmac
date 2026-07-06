from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import BUDGETS, ROOT, review_budget_rows


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v146_domain_split_router_selection_20260527"
V145_LONG = ROOT / "outputs" / "grosspath_rc_v145_fusion_router_20260527" / "v145_fusion_router_scores_long.csv"
V143_CASES = ROOT / "outputs" / "grosspath_rc_v143_image_feature_error_router_20260527" / "v143_image_feature_case_risks.csv"


def prepare() -> pd.DataFrame:
    scores = pd.read_csv(V145_LONG, dtype={"case_id": str})
    meta = pd.read_csv(V143_CASES, dtype={"case_id": str})
    meta = meta[["scope", "base_model", "case_id", "domain", "task_l6_label"]].drop_duplicates()
    df = scores.merge(meta, on=["scope", "base_model", "case_id"], how="left", validate="many_to_one")
    df["eval_domain"] = df["domain"].fillna(df["scope"])
    df.loc[df["scope"].eq("strict_external_locked"), "eval_domain"] = "strict_external"
    return df


def evaluate_group(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for base in sorted(df["base_model"].unique()):
        for router in sorted(df["router"].unique()):
            for domain in ["old_data", "third_batch", "strict_external"]:
                sub = df.loc[df["base_model"].eq(base) & df["router"].eq(router) & df["eval_domain"].eq(domain)].copy()
                if sub.empty:
                    continue
                y = sub["label_idx"].astype(int).to_numpy()
                pred = sub["base_pred"].astype(int).to_numpy()
                prob = sub["base_prob"].to_numpy(float)
                risk = sub["risk_score"].to_numpy(float)
                group_rows = review_budget_rows(domain, base, router, y, pred, prob, risk)
                for row in group_rows:
                    row["eval_domain"] = domain
                    rows.append(row)
    return pd.DataFrame(rows)


def select_by_third(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    b03 = curve.loc[np.isclose(curve["review_budget"].astype(float), 0.3)].copy()
    for base in sorted(b03["base_model"].unique()):
        third = b03.loc[b03["base_model"].eq(base) & b03["eval_domain"].eq("third_batch")].copy()
        third = third.sort_values(
            ["system_if_review_corrected_balanced_accuracy", "error_capture_rate", "auto_balanced_accuracy"],
            ascending=False,
        )
        if third.empty:
            continue
        selected_router = third.iloc[0]["router"]
        for domain in ["old_data", "third_batch", "strict_external"]:
            sub = b03.loc[b03["base_model"].eq(base) & b03["router"].eq(selected_router) & b03["eval_domain"].eq(domain)].copy()
            if sub.empty:
                continue
            row = sub.iloc[0].to_dict()
            row["selection_basis"] = "best_on_third_batch_budget03"
            row["selected_router"] = selected_router
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare()
    curve = evaluate_group(df)
    selected = select_by_third(curve)
    curve = curve.sort_values(
        ["eval_domain", "base_model", "review_budget", "system_if_review_corrected_balanced_accuracy"],
        ascending=[True, True, True, False],
    )
    curve.to_csv(OUT_DIR / "v146_domain_split_router_budget_curve.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v146_third_selected_router_external_check.csv", index=False, encoding="utf-8-sig")
    report = {
        "selection": "Router selected on third_batch only at 30% review budget; strict_external reported without selection.",
        "source": str(V145_LONG),
    }
    (OUT_DIR / "v146_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v146] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
