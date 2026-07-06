from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import BUDGETS, ROOT, metrics, review_budget_rows


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v145_fusion_router_20260527"
V143_CASES = ROOT / "outputs" / "grosspath_rc_v143_image_feature_error_router_20260527" / "v143_image_feature_case_risks.csv"
V144_SCORES = ROOT / "outputs" / "grosspath_rc_v144_distilled_concept_router_20260527" / "v144_distilled_concept_router_risk_scores_long.csv"

BASE_MODELS = ["robust_prob", "prob_mean_core"]
FUSION_SIGNALS = [
    "low_conf",
    "v143_pca_directional",
    "v144_concept_directional",
]


def load_v144() -> pd.DataFrame:
    df = pd.read_csv(V144_SCORES, dtype={"case_id": str})
    want = {
        "heuristic_low_conf": "low_conf",
        "pred_concept:logreg_c03:directional": "v144_concept_directional",
        "pred_concept_plus_base_probs:logreg_c03:directional": "v144_concept_plus_base_directional",
        "pred_concept:logreg_c03:any": "v144_concept_any",
    }
    df = df.loc[df["router"].isin(want)].copy()
    df["router"] = df["router"].map(want)
    piv = df.pivot_table(
        index=["scope", "base_model", "case_id", "label_idx", "base_pred", "base_prob", "base_correct"],
        columns="router",
        values="risk_score",
        aggfunc="first",
    ).reset_index()
    return piv


def load_v143_long() -> pd.DataFrame:
    df = pd.read_csv(V143_CASES, dtype={"case_id": str})
    rows = []
    mapping = {
        "pca64_logreg_c03_any_risk": "v143_pca_any",
        "pca64_logreg_c03_directional_risk": "v143_pca_directional",
        "extra_d3_directional_risk": "v143_extra_directional",
    }
    for col, name in mapping.items():
        if col not in df:
            continue
        part = df[["scope", "base_model", "case_id", col]].copy()
        part = part.rename(columns={col: name})
        rows.append(part)
    out = df[["scope", "base_model", "case_id"]].drop_duplicates().copy()
    for part in rows:
        out = out.merge(part, on=["scope", "base_model", "case_id"], how="left", validate="one_to_one")
    return out


def prepare() -> pd.DataFrame:
    v144 = load_v144()
    v143 = load_v143_long()
    df = v144.merge(v143, on=["scope", "base_model", "case_id"], how="left", validate="one_to_one")
    for col in [c for c in df.columns if c.startswith("v143_") or c.startswith("v144_") or c == "low_conf"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_by_internal(df: pd.DataFrame, base_model: str, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    internal = out["scope"].eq("internal_oof") & out["base_model"].eq(base_model)
    for col in cols:
        lo = float(out.loc[internal, col].min())
        hi = float(out.loc[internal, col].max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[f"norm_{col}"] = 0.5
        else:
            out[f"norm_{col}"] = ((out[col] - lo) / (hi - lo)).clip(0.0, 1.0)
    return out


def candidate_scores(df: pd.DataFrame, base_model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "low_conf",
        "v143_pca_directional",
        "v143_pca_any",
        "v143_extra_directional",
        "v144_concept_directional",
        "v144_concept_plus_base_directional",
        "v144_concept_any",
    ]
    cols = [c for c in cols if c in df and df[c].notna().any()]
    work = normalize_by_internal(df, base_model, cols)
    work = work.loc[work["base_model"].eq(base_model)].copy()

    score_cols = []
    for col in cols:
        name = f"single:{col}"
        work[name] = work[f"norm_{col}"]
        score_cols.append(name)

    trio = [c for c in FUSION_SIGNALS if c in cols]
    if len(trio) == 3:
        norm_trio = [f"norm_{c}" for c in trio]
        work["fusion:rank_mean_lowconf_v143_v144"] = work[norm_trio].rank(pct=True).mean(axis=1)
        score_cols.append("fusion:rank_mean_lowconf_v143_v144")

        rows = []
        best = None
        internal = work["scope"].eq("internal_oof")
        y = work.loc[internal, "label_idx"].astype(int).to_numpy()
        pred = work.loc[internal, "base_pred"].astype(int).to_numpy()
        prob = work.loc[internal, "base_prob"].to_numpy(float)
        for w0 in [0.0, 0.25, 0.50, 0.75, 1.0]:
            for w1 in [0.0, 0.25, 0.50, 0.75, 1.0]:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9 or w2 > 1.0:
                    continue
                weights = np.array([w0, w1, w2], dtype=float)
                score = (work.loc[internal, norm_trio].to_numpy(float) * weights).sum(axis=1)
                rows_03 = review_budget_rows("internal_oof", base_model, "tmp", y, pred, prob, score)
                b03 = [r for r in rows_03 if abs(float(r["review_budget"]) - 0.3) < 1e-9][0]
                item = {
                    "base_model": base_model,
                    "w_low_conf": w0,
                    "w_v143_pca_directional": w1,
                    "w_v144_concept_directional": w2,
                    "internal_budget03_system_bacc": b03["system_if_review_corrected_balanced_accuracy"],
                    "internal_budget03_error_capture": b03["error_capture_rate"],
                }
                rows.append(item)
                key = (float(item["internal_budget03_system_bacc"]), float(item["internal_budget03_error_capture"]), -abs(w0 - 0.34))
                if best is None or key > best[0]:
                    best = (key, item)
        selected = best[1]
        weights = np.array(
            [selected["w_low_conf"], selected["w_v143_pca_directional"], selected["w_v144_concept_directional"]],
            dtype=float,
        )
        work["fusion:selected_internal_budget03"] = (work[norm_trio].to_numpy(float) * weights).sum(axis=1)
        score_cols.append("fusion:selected_internal_budget03")
        weights_df = pd.DataFrame(rows).sort_values(
            ["internal_budget03_system_bacc", "internal_budget03_error_capture"], ascending=False
        )
    else:
        weights_df = pd.DataFrame()
    return work[["scope", "base_model", "case_id", "label_idx", "base_pred", "base_prob", "base_correct"] + score_cols], weights_df


def evaluate_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cases = []
    score_cols = [c for c in scores.columns if c.startswith("single:") or c.startswith("fusion:")]
    for base_model in sorted(scores["base_model"].unique()):
        for scope in ["internal_oof", "strict_external_locked"]:
            sub = scores.loc[scores["base_model"].eq(base_model) & scores["scope"].eq(scope)].copy()
            y = sub["label_idx"].astype(int).to_numpy()
            pred = sub["base_pred"].astype(int).to_numpy()
            prob = sub["base_prob"].to_numpy(float)
            for col in score_cols:
                if col not in sub or sub[col].isna().all():
                    continue
                risk = sub[col].fillna(sub[col].median()).to_numpy(float)
                rows.extend(review_budget_rows(scope, base_model, col, y, pred, prob, risk))
                case_part = sub[["scope", "base_model", "case_id", "label_idx", "base_pred", "base_prob", "base_correct"]].copy()
                case_part["router"] = col
                case_part["risk_score"] = risk
                cases.append(case_part)
    return pd.DataFrame(rows), pd.concat(cases, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare()
    all_scores = []
    all_weights = []
    for base in BASE_MODELS:
        scores, weights = candidate_scores(df, base)
        all_scores.append(scores)
        if not weights.empty:
            all_weights.append(weights)
    scores_df = pd.concat(all_scores, ignore_index=True)
    weights_df = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    budget_df, case_df = evaluate_scores(scores_df)
    budget_df = budget_df.sort_values(
        ["scope", "base_model", "review_budget", "system_if_review_corrected_balanced_accuracy"],
        ascending=[True, True, True, False],
    )
    scores_df.to_csv(OUT_DIR / "v145_fusion_router_scores_wide.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(OUT_DIR / "v145_fusion_router_scores_long.csv", index=False, encoding="utf-8-sig")
    budget_df.to_csv(OUT_DIR / "v145_fusion_router_budget_curve.csv", index=False, encoding="utf-8-sig")
    weights_df.to_csv(OUT_DIR / "v145_fusion_router_internal_weight_grid.csv", index=False, encoding="utf-8-sig")
    report = {
        "base_models": BASE_MODELS,
        "signals": FUSION_SIGNALS,
        "selection": "Fusion weights selected only by internal OOF system BAcc at 30% review budget; strict external is reported once.",
    }
    (OUT_DIR / "v145_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v145] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
