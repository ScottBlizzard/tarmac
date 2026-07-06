from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v71_pseudodomain_selected_external_ood_20260527"
V70 = ROOT / "outputs" / "grosspath_rc_v70_pseudo_domain_ood_validation_20260527" / "v70_pseudo_domain_ood_summary.csv"
V68 = ROOT / "outputs" / "grosspath_rc_v68_rank_ood_overlay_20260527" / "v68_rank_ood_overlay_summary.csv"


def build_pseudo_candidate_table(v70: pd.DataFrame) -> pd.DataFrame:
    base = v70.loc[v70["split"].eq("target") & v70["policy"].eq("v50_base")].copy()
    base_bacc = base.set_index("direction")["balanced_accuracy"].to_dict()
    base_control = base.set_index("direction")["control_rate"].to_dict()
    cand = v70.loc[v70["split"].eq("target") & v70["policy"].eq("v50_plus_rank_ood")].copy()
    cand["target_bacc_gain"] = cand.apply(lambda r: r["balanced_accuracy"] - base_bacc[r["direction"]], axis=1)
    cand["target_control_delta"] = cand.apply(lambda r: r["control_rate"] - base_control[r["direction"]], axis=1)

    metric_cols = [
        "balanced_accuracy",
        "control_rate",
        "target_bacc_gain",
        "target_control_delta",
        "sensitivity",
        "specificity",
        "fn",
        "fp",
        "extra_captured_wrong_n",
        "remaining_error_n",
    ]
    piv = cand.pivot_table(index=["score_name", "extra_rate"], columns="direction", values=metric_cols, aggfunc="first")
    piv.columns = [f"{metric}__{direction}" for metric, direction in piv.columns]
    piv = piv.reset_index()
    gain_cols = [c for c in piv.columns if c.startswith("target_bacc_gain__")]
    control_cols = [c for c in piv.columns if c.startswith("control_rate__")]
    extra_wrong_cols = [c for c in piv.columns if c.startswith("extra_captured_wrong_n__")]
    piv["min_bacc_gain"] = piv[gain_cols].min(axis=1)
    piv["mean_bacc_gain"] = piv[gain_cols].mean(axis=1)
    piv["max_control_rate"] = piv[control_cols].max(axis=1)
    piv["mean_control_rate"] = piv[control_cols].mean(axis=1)
    piv["total_extra_captured_wrong"] = piv[extra_wrong_cols].sum(axis=1)
    piv["is_image_only"] = piv["score_name"].str.startswith("image_common5")
    piv["is_model_disagreement"] = piv["score_name"].eq("main_robust_disagreement")
    return piv


def choose_candidates(pseudo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = [
        {
            "scenario": "pseudo_robust_best_gain",
            "filter": lambda d: d.loc[d["min_bacc_gain"].ge(-1e-12)],
            "sort": ["mean_bacc_gain", "min_bacc_gain", "max_control_rate"],
            "ascending": [False, False, True],
        },
        {
            "scenario": "pseudo_robust_low_control",
            "filter": lambda d: d.loc[d["min_bacc_gain"].ge(-1e-12) & d["max_control_rate"].le(0.76)],
            "sort": ["mean_bacc_gain", "min_bacc_gain", "max_control_rate"],
            "ascending": [False, False, True],
        },
        {
            "scenario": "pseudo_image_only_best_gain",
            "filter": lambda d: d.loc[d["min_bacc_gain"].ge(-1e-12) & d["is_image_only"]],
            "sort": ["mean_bacc_gain", "min_bacc_gain", "max_control_rate"],
            "ascending": [False, False, True],
        },
        {
            "scenario": "pseudo_image_only_low_control",
            "filter": lambda d: d.loc[d["min_bacc_gain"].ge(-1e-12) & d["is_image_only"] & d["max_control_rate"].le(0.80)],
            "sort": ["mean_bacc_gain", "min_bacc_gain", "max_control_rate"],
            "ascending": [False, False, True],
        },
        {
            "scenario": "pseudo_disagreement_best",
            "filter": lambda d: d.loc[d["min_bacc_gain"].ge(-1e-12) & d["is_model_disagreement"]],
            "sort": ["mean_bacc_gain", "min_bacc_gain", "max_control_rate"],
            "ascending": [False, False, True],
        },
    ]
    for sc in scenarios:
        ok = sc["filter"](pseudo).copy()
        if ok.empty:
            continue
        chosen = ok.sort_values(sc["sort"], ascending=sc["ascending"]).iloc[0].to_dict()
        chosen["scenario"] = sc["scenario"]
        rows.append(chosen)
    out = pd.DataFrame(rows)
    return out.drop_duplicates(["score_name", "extra_rate", "scenario"]).reset_index(drop=True)


def attach_external(selected: pd.DataFrame, v68: pd.DataFrame) -> pd.DataFrame:
    ext = v68.loc[v68["split"].eq("external")].copy()
    base = ext.loc[ext["policy"].eq("v50_base")].iloc[0]
    rows = []
    for _, row in selected.iterrows():
        match = ext.loc[
            ext["policy"].eq("v50_plus_rank_ood")
            & ext["score_name"].eq(row["score_name"])
            & np.isclose(ext["extra_rate"].astype(float), float(row["extra_rate"]))
        ]
        if match.empty:
            continue
        e = match.iloc[0]
        out = row.to_dict()
        for col in [
            "control_rate",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "fn",
            "fp",
            "extra_captured_wrong_n",
            "remaining_error_n",
        ]:
            out[f"external_{col}"] = e[col]
            out[f"external_delta_{col}"] = e[col] - base[col] if col in base.index and pd.api.types.is_number(e[col]) else np.nan
        out["external_base_bacc"] = base["balanced_accuracy"]
        out["external_base_control_rate"] = base["control_rate"]
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v70 = pd.read_csv(V70)
    v68 = pd.read_csv(V68)
    pseudo = build_pseudo_candidate_table(v70)
    selected = choose_candidates(pseudo)
    external_eval = attach_external(selected, v68)

    pseudo.to_csv(OUT_DIR / "v71_pseudodomain_candidate_table.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v71_pseudodomain_selected_candidates.csv", index=False, encoding="utf-8-sig")
    external_eval.to_csv(OUT_DIR / "v71_pseudodomain_selected_external_eval.csv", index=False, encoding="utf-8-sig")

    show_cols = [
        "scenario",
        "score_name",
        "extra_rate",
        "min_bacc_gain",
        "mean_bacc_gain",
        "max_control_rate",
        "total_extra_captured_wrong",
        "external_control_rate",
        "external_balanced_accuracy",
        "external_sensitivity",
        "external_specificity",
        "external_fn",
        "external_fp",
        "external_extra_captured_wrong_n",
        "external_delta_balanced_accuracy",
    ]
    print(external_eval[show_cols].to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
