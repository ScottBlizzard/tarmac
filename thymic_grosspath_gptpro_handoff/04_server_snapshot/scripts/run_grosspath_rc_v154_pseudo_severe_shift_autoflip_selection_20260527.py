from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics
from run_grosspath_rc_v153_high_precision_autoflip_trigger_20260527 import (
    apply_rule,
    load_data,
)


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v154_pseudo_severe_shift_autoflip_selection_20260527"
V74_CASES = ROOT / "outputs" / "grosspath_rc_v74_dev_prespecified_quality_gate_20260527" / "v74_dev_prespecified_quality_gate_case_routes.csv"
MODES = ["all_preds", "pred_high_only"]
SIGNALS = [
    "single:v144_concept_directional",
    "single:v144_concept_plus_base_directional",
    "signal:mean_v143_v144_directional",
]


def attach_quality(df: pd.DataFrame) -> pd.DataFrame:
    q = pd.read_csv(V74_CASES, dtype={"case_id": str})
    q = q.loc[q["domain"].eq("development_all") & q["gate_name"].eq("v50_base")].copy()
    q = q[["case_id", "quality_proxy_risk"]].drop_duplicates("case_id")
    out = df.merge(q, on="case_id", how="left", validate="many_to_one")
    out["quality_proxy_risk"] = pd.to_numeric(out["quality_proxy_risk"], errors="coerce").fillna(0.0)
    return out


def pseudo_masks(dev: pd.DataFrame) -> dict[str, np.ndarray]:
    q = dev["quality_proxy_risk"].to_numpy(float)
    out: dict[str, np.ndarray] = {}
    for rate in [0.20]:
        threshold = float(np.quantile(q, 1.0 - rate))
        out[f"quality_top{int(rate * 100)}_internal"] = q >= threshold
    out["quality_ge_0p10_internal"] = q >= 0.10
    third = dev["eval_domain"].eq("third_batch").to_numpy()
    if third.any():
        tq = q[third]
        threshold = float(np.quantile(tq, 0.80))
        out["third_quality_top20"] = third & (q >= threshold)
    old = dev["eval_domain"].eq("old_data").to_numpy()
    if old.any():
        oq = q[old]
        threshold = float(np.quantile(oq, 0.80))
        out["old_quality_top20"] = old & (q >= threshold)
    return {k: v for k, v in out.items() if int(v.sum()) >= 10}


def build_fast_thresholds(dev: pd.DataFrame, signal: str, mode: str) -> list[float]:
    pred = dev["base_pred"].astype(int).to_numpy()
    if mode == "all_preds":
        mask = np.ones(len(dev), dtype=bool)
    elif mode == "pred_high_only":
        mask = pred == 1
    else:
        mask = pred == 0
    scores = dev.loc[mask, signal].dropna().to_numpy(float)
    if len(scores) == 0:
        return []
    thresholds = []
    for q in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]:
        thresholds.append(float(np.quantile(scores, q)))
    for rate in [0.02, 0.05, 0.10]:
        k = max(1, int(round(rate * len(dev))))
        if k <= len(scores):
            thresholds.append(float(np.partition(scores, max(0, len(scores) - k))[max(0, len(scores) - k)]))
    return sorted(set(x for x in thresholds if np.isfinite(x)))


def eval_subset(df: pd.DataFrame, mask: np.ndarray) -> dict[str, object]:
    sub = df.loc[mask].copy()
    y = sub["label_idx"].astype(int).to_numpy()
    base = sub["base_pred"].astype(int).to_numpy()
    final = sub["final_pred"].astype(int).to_numpy()
    prob = sub["base_prob"].to_numpy(float)
    flip = sub["auto_flip"].astype(bool).to_numpy()
    base_wrong = base != y
    final_wrong = final != y
    rescued = base_wrong & ~final_wrong
    hurt = ~base_wrong & final_wrong
    row: dict[str, object] = {
        "n": int(len(sub)),
        "flip_n": int(flip.sum()),
        "flip_rate": float(flip.mean()) if len(sub) else np.nan,
        "base_errors": int(base_wrong.sum()),
        "final_errors": int(final_wrong.sum()),
        "net_errors_reduced": int(base_wrong.sum() - final_wrong.sum()),
        "rescued_n": int(rescued.sum()),
        "hurt_n": int(hurt.sum()),
    }
    row.update({f"base_{k}": v for k, v in metrics(y, base, prob).items()})
    row.update({f"final_{k}": v for k, v in metrics(y, final, prob).items()})
    row["delta_bacc"] = float(row["final_balanced_accuracy"] - row["base_balanced_accuracy"])
    return row


def evaluate_rule_all(df: pd.DataFrame, ruled: pd.DataFrame, subset_name: str, subset_mask: np.ndarray) -> list[dict[str, object]]:
    rows = []
    domains = {
        subset_name: subset_mask,
        "old_data": ruled["eval_domain"].eq("old_data").to_numpy(),
        "third_batch": ruled["eval_domain"].eq("third_batch").to_numpy(),
        "strict_external": ruled["eval_domain"].eq("strict_external").to_numpy(),
        "all_three_domains": np.ones(len(ruled), dtype=bool),
    }
    for domain, mask in domains.items():
        if int(mask.sum()) == 0:
            continue
        row = eval_subset(ruled, mask)
        row["eval_scope"] = domain
        rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = attach_quality(load_data())
    all_rows = []
    selected_rows = []

    for base_model in ["robust_prob", "prob_mean_core"]:
        work = df.loc[df["base_model"].eq(base_model)].copy()
        dev = work.loc[work["eval_domain"].isin(["old_data", "third_batch"])].copy()
        masks = pseudo_masks(dev)
        for subset_name, dev_mask in masks.items():
            full_subset_mask = np.zeros(len(work), dtype=bool)
            full_subset_mask[np.flatnonzero(work["eval_domain"].isin(["old_data", "third_batch"]).to_numpy())[dev_mask]] = True
            candidates = []
            for signal in [s for s in SIGNALS if s in work.columns]:
                for mode in MODES:
                    for threshold in build_fast_thresholds(dev, signal, mode):
                        rule_name = f"{base_model}|{subset_name}|{signal}|{mode}|t={threshold:.6f}"
                        ruled = apply_rule(work, signal, mode, threshold)
                        rows = evaluate_rule_all(work, ruled, subset_name, full_subset_mask)
                        for row in rows:
                            row.update(
                                {
                                    "rule_name": rule_name,
                                    "base_model": base_model,
                                    "pseudo_subset": subset_name,
                                    "signal": signal,
                                    "mode": mode,
                                    "threshold": float(threshold),
                                }
                            )
                        all_rows.extend(rows)
                        subset_row = next(r for r in rows if r["eval_scope"] == subset_name)
                        strict_row = next(r for r in rows if r["eval_scope"] == "strict_external")
                        candidates.append(
                            {
                                "rule_name": rule_name,
                                "base_model": base_model,
                                "pseudo_subset": subset_name,
                                "signal": signal,
                                "mode": mode,
                                "threshold": float(threshold),
                                "subset_n": subset_row["n"],
                                "subset_flip_n": subset_row["flip_n"],
                                "subset_net_errors_reduced": subset_row["net_errors_reduced"],
                                "subset_delta_bacc": subset_row["delta_bacc"],
                                "subset_rescued": subset_row["rescued_n"],
                                "subset_hurt": subset_row["hurt_n"],
                                "strict_flip_n": strict_row["flip_n"],
                                "strict_net_errors_reduced": strict_row["net_errors_reduced"],
                                "strict_delta_bacc": strict_row["delta_bacc"],
                                "strict_final_bacc": strict_row["final_balanced_accuracy"],
                                "strict_rescued": strict_row["rescued_n"],
                                "strict_hurt": strict_row["hurt_n"],
                            }
                        )
            cand = pd.DataFrame(candidates)
            eligible = cand.loc[
                (cand["subset_flip_n"] >= 1)
                & (cand["subset_net_errors_reduced"] > 0)
                & (cand["subset_delta_bacc"] > 0)
                & (cand["subset_rescued"] >= cand["subset_hurt"])
            ].copy()
            if eligible.empty:
                chosen = cand.sort_values(["subset_delta_bacc", "subset_net_errors_reduced"], ascending=False).head(1)
                chosen["pseudo_selection_status"] = "no_positive_safe_rule"
            else:
                chosen = eligible.sort_values(
                    ["subset_delta_bacc", "subset_net_errors_reduced", "subset_flip_n"],
                    ascending=False,
                ).head(1)
                chosen["pseudo_selection_status"] = "selected_by_pseudo_subset"
            selected_rows.append(chosen)

    all_df = pd.DataFrame(all_rows)
    selected = pd.concat(selected_rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "v154_pseudo_severe_all_rule_scope_summary.csv", index=False, encoding="utf-8-sig")
    selected.sort_values(["strict_delta_bacc", "subset_delta_bacc"], ascending=False).to_csv(
        OUT_DIR / "v154_pseudo_severe_selected_rules.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "pseudo_subsets": sorted(selected["pseudo_subset"].unique().tolist()),
        "selection": "Rules are selected by positive net correction inside internal pseudo-severe quality subsets only; strict external is reported after selection.",
        "selected_count": int(len(selected)),
    }
    (OUT_DIR / "v154_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v154] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
