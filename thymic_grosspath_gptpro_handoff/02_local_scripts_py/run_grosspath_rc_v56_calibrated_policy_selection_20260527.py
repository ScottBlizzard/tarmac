from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v51_workflow_validation_20260527 as v51  # noqa: E402
import run_grosspath_rc_v54_constrained_policy_search_20260527 as v54  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v56_calibrated_policy_selection_20260527"
N_SPLITS = 1000
CAL_FRACTION = 0.70
SEED = 20260527 + 56


SCENARIOS = [
    {"scenario": "empirical_sens98_spec95", "sens_target": 0.98, "spec_min": 0.95, "bound": "empirical"},
    {"scenario": "empirical_sens985_spec95", "sens_target": 0.985, "spec_min": 0.95, "bound": "empirical"},
    {"scenario": "empirical_sens99_spec95", "sens_target": 0.99, "spec_min": 0.95, "bound": "empirical"},
    {"scenario": "wilson95_sens95_spec90", "sens_target": 0.95, "spec_min": 0.90, "bound": "wilson95"},
    {"scenario": "wilson95_sens97_spec90", "sens_target": 0.97, "spec_min": 0.90, "bound": "wilson95"},
]


def stratified_calibration_indices(y: np.ndarray, rng: np.random.Generator, frac: float) -> np.ndarray:
    parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        n = max(1, int(round(len(idx) * frac)))
        parts.append(rng.choice(idx, size=n, replace=False))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def wilson_lower(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(k, dtype=float)
    ok = n > 0
    p = np.zeros_like(k, dtype=float)
    p[ok] = k[ok] / n[ok]
    denom = 1.0 + (z * z) / np.maximum(n, 1)
    center = p + (z * z) / (2 * np.maximum(n, 1))
    adj = z * np.sqrt((p * (1 - p) + (z * z) / (4 * np.maximum(n, 1))) / np.maximum(n, 1))
    out[ok] = (center[ok] - adj[ok]) / denom[ok]
    return out


def make_candidate_predictions() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    candidates = []
    dev_preds = []
    ext_preds = []

    for cand in v54.build_score_candidates(dev, ext, dev_scores, ext_scores):
        for budget in v54.BUDGETS:
            policy = f"{cand['family']}::{cand['name']}::budget={budget:.3f}"
            dev_review = v54.top_by_score(cand["dev_score"], float(budget))
            ext_review = v54.top_by_score(cand["ext_score"], float(budget))
            dev_pred = v51.final_prediction(dev, dev_review)
            ext_pred = v51.final_prediction(ext, ext_review)
            dm = v30.metrics_binary(dev["label_idx"].to_numpy(dtype=int), dev_pred)
            em = v30.metrics_binary(ext["label_idx"].to_numpy(dtype=int), ext_pred)
            candidates.append(
                {
                    "policy": policy,
                    "family": cand["family"],
                    "name": cand["name"],
                    "budget": float(budget),
                    "base_budget": np.nan,
                    "addon_candidate": "",
                    "addon_rate": np.nan,
                    "dev_review_rate": float(dev_review.mean()),
                    "external_review_rate": float(ext_review.mean()),
                    **{f"dev_{k}": v for k, v in dm.items()},
                    **{f"external_{k}": v for k, v in em.items()},
                }
            )
            dev_preds.append(dev_pred.astype(np.int8))
            ext_preds.append(ext_pred.astype(np.int8))

    addon_defs = [
        ("pred_low_fn", "fn"),
        ("pred_high_fp", "fp"),
        ("all_direction", "direction"),
        ("all_any", "any"),
    ]
    for base_budget in v54.BASE_BUDGETS:
        dev_base = v54.top_by_score(dev_scores["any"], float(base_budget))
        ext_base = v54.top_by_score(ext_scores["any"], float(base_budget))
        for addon_candidate, score_name in addon_defs:
            for addon_rate in v54.ADDON_RATES:
                policy = f"buffer::{addon_candidate}::{base_budget:.3f}+{addon_rate:.3f}"
                if addon_candidate == "all_any":
                    dev_review = v50.add_top_candidates(dev, dev_base, dev_scores["any"], float(addon_rate), "all_direction")
                    ext_review = v50.add_top_candidates(ext, ext_base, ext_scores["any"], float(addon_rate), "all_direction")
                else:
                    dev_review = v50.add_top_candidates(dev, dev_base, dev_scores[score_name], float(addon_rate), addon_candidate)
                    ext_review = v50.add_top_candidates(ext, ext_base, ext_scores[score_name], float(addon_rate), addon_candidate)
                dev_pred = v51.final_prediction(dev, dev_review)
                ext_pred = v51.final_prediction(ext, ext_review)
                dm = v30.metrics_binary(dev["label_idx"].to_numpy(dtype=int), dev_pred)
                em = v30.metrics_binary(ext["label_idx"].to_numpy(dtype=int), ext_pred)
                candidates.append(
                    {
                        "policy": policy,
                        "family": "buffer",
                        "name": addon_candidate,
                        "budget": np.nan,
                        "base_budget": float(base_budget),
                        "addon_candidate": addon_candidate,
                        "addon_rate": float(addon_rate),
                        "dev_review_rate": float(dev_review.mean()),
                        "external_review_rate": float(ext_review.mean()),
                        **{f"dev_{k}": v for k, v in dm.items()},
                        **{f"external_{k}": v for k, v in em.items()},
                    }
                )
                dev_preds.append(dev_pred.astype(np.int8))
                ext_preds.append(ext_pred.astype(np.int8))

    cand_df = pd.DataFrame(candidates)
    return cand_df, np.vstack(dev_preds), np.vstack(ext_preds), dev["label_idx"].to_numpy(dtype=np.int8), ext["label_idx"].to_numpy(dtype=np.int8)


def calibration_metrics(y: np.ndarray, pred_mat: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    y_cal = y[idx]
    p = pred_mat[:, idx]
    y1 = y_cal == 1
    y0 = y_cal == 0
    tp = ((p == 1) & y1).sum(axis=1)
    fn = ((p == 0) & y1).sum(axis=1)
    tn = ((p == 0) & y0).sum(axis=1)
    fp = ((p == 1) & y0).sum(axis=1)
    sens = tp / np.maximum(tp + fn, 1)
    spec = tn / np.maximum(tn + fp, 1)
    acc = (tp + tn) / len(idx)
    bacc = (sens + spec) / 2
    return {
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sens,
        "specificity": spec,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "sensitivity_lcb": wilson_lower(tp, tp + fn),
        "specificity_lcb": wilson_lower(tn, tn + fp),
    }


def select_candidate(cand: pd.DataFrame, cm: dict[str, np.ndarray], scenario: dict[str, object]) -> tuple[int, bool]:
    if scenario["bound"] == "wilson95":
        sens = cm["sensitivity_lcb"]
        spec = cm["specificity_lcb"]
    else:
        sens = cm["sensitivity"]
        spec = cm["specificity"]
    ok = (sens >= float(scenario["sens_target"])) & (spec >= float(scenario["spec_min"]))
    if ok.any():
        subset = cand.loc[ok].copy()
        subset["_cal_bacc"] = cm["balanced_accuracy"][ok]
        subset["_cal_sens"] = cm["sensitivity"][ok]
        subset["_cal_spec"] = cm["specificity"][ok]
        chosen_idx = subset.sort_values(
            ["dev_review_rate", "_cal_bacc", "_cal_spec", "_cal_sens"],
            ascending=[True, False, False, False],
        ).index[0]
        return int(chosen_idx), True
    tmp = cand.copy()
    tmp["_sel_sens"] = sens
    tmp["_sel_spec"] = spec
    tmp["_cal_bacc"] = cm["balanced_accuracy"]
    chosen_idx = tmp.sort_values(
        ["_sel_sens", "_sel_spec", "_cal_bacc", "dev_review_rate"],
        ascending=[False, False, False, True],
    ).index[0]
    return int(chosen_idx), False


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, g in detail.groupby("scenario"):
        mode = g["policy"].mode()
        rows.append(
            {
                "scenario": scenario,
                "n_splits": int(len(g)),
                "constraints_met_rate": float(g["constraints_met"].mean()),
                "external_review_rate_median": float(g["external_review_rate"].median()),
                "external_review_rate_ci025": float(g["external_review_rate"].quantile(0.025)),
                "external_review_rate_ci975": float(g["external_review_rate"].quantile(0.975)),
                "external_bacc_median": float(g["external_balanced_accuracy"].median()),
                "external_bacc_ci025": float(g["external_balanced_accuracy"].quantile(0.025)),
                "external_bacc_ci975": float(g["external_balanced_accuracy"].quantile(0.975)),
                "external_sensitivity_median": float(g["external_sensitivity"].median()),
                "external_specificity_median": float(g["external_specificity"].median()),
                "external_fn_median": float(g["external_fn"].median()),
                "external_fp_median": float(g["external_fp"].median()),
                "p_external_fn_le_1": float((g["external_fn"] <= 1).mean()),
                "p_external_bacc_ge_95": float((g["external_balanced_accuracy"] >= 0.95).mean()),
                "most_common_policy": mode.iloc[0] if not mode.empty else "",
                "most_common_policy_rate": float((g["policy"] == mode.iloc[0]).mean()) if not mode.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand, dev_preds, _ext_preds, y_dev, _y_ext = make_candidate_predictions()
    cand = cand.reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    rows = []
    for split_id in range(N_SPLITS):
        idx = stratified_calibration_indices(y_dev, rng, CAL_FRACTION)
        cm = calibration_metrics(y_dev, dev_preds, idx)
        for scenario in SCENARIOS:
            chosen_idx, met = select_candidate(cand, cm, scenario)
            chosen = cand.iloc[chosen_idx]
            rows.append(
                {
                    "split_id": split_id,
                    "scenario": scenario["scenario"],
                    "bound": scenario["bound"],
                    "target_sensitivity": scenario["sens_target"],
                    "specificity_min": scenario["spec_min"],
                    "constraints_met": int(met),
                    "policy": chosen["policy"],
                    "family": chosen["family"],
                    "name": chosen["name"],
                    "cal_sensitivity": float(cm["sensitivity"][chosen_idx]),
                    "cal_specificity": float(cm["specificity"][chosen_idx]),
                    "cal_sensitivity_lcb": float(cm["sensitivity_lcb"][chosen_idx]),
                    "cal_specificity_lcb": float(cm["specificity_lcb"][chosen_idx]),
                    "cal_balanced_accuracy": float(cm["balanced_accuracy"][chosen_idx]),
                    "dev_review_rate": float(chosen["dev_review_rate"]),
                    "external_review_rate": float(chosen["external_review_rate"]),
                    "external_accuracy": float(chosen["external_accuracy"]),
                    "external_balanced_accuracy": float(chosen["external_balanced_accuracy"]),
                    "external_sensitivity": float(chosen["external_sensitivity"]),
                    "external_specificity": float(chosen["external_specificity"]),
                    "external_fn": int(chosen["external_fn"]),
                    "external_fp": int(chosen["external_fp"]),
                }
            )

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    cand.drop(columns=[]).to_csv(OUT_DIR / "v56_candidate_policy_table.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT_DIR / "v56_calibration_split_selection_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v56_calibration_split_selection_summary.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
