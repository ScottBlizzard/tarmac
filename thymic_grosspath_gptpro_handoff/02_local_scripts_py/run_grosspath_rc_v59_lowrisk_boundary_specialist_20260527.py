from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v48_directional_risk_controller_20260527 as v48  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v51_workflow_validation_20260527 as v51  # noqa: E402
import run_grosspath_rc_v54_constrained_policy_search_20260527 as v54  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v59_lowrisk_boundary_specialist_20260527"
BASE_POLICY = "fusion_rank::dir_plus_pred_low_fn::budget=0.650"
ADDON_RATES = np.round(np.arange(0.0, 0.181, 0.01), 3)
SCENARIOS = [
    {"scenario": "sens985_spec965", "sens_min": 0.985, "spec_min": 0.965},
    {"scenario": "sens985_spec970", "sens_min": 0.985, "spec_min": 0.970},
    {"scenario": "sens985_spec980", "sens_min": 0.985, "spec_min": 0.980},
    {"scenario": "sens990_spec970", "sens_min": 0.990, "spec_min": 0.970},
    {"scenario": "sens990_spec980", "sens_min": 0.990, "spec_min": 0.980},
]


def feature_columns(dev: pd.DataFrame, ext: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = [
        c
        for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES
        if c in dev.columns and c in ext.columns
    ]
    # These are model-derived or image-derived concept features available for both splits.
    extra_numeric = [
        "prob103_vitl",
        "prob107_qkvb",
        "prob162_blend",
        "pred_stack_plain",
        "pred_stack_balanced",
        "concept_highrisk_z",
        "concept_direction",
        "concept_conflicts_mean_pred",
        "gross_highrisk_score",
        "gross_conflict_score",
        "tumor_max_dim_mm",
        "tumor_max_area_mm2",
        "all_max_area_mm2",
        "pleura_attached",
        "main_prob",
        "robust_prob",
        "main_margin_abs",
        "robust_margin_abs",
    ]
    for c in extra_numeric:
        if c in dev.columns and c in ext.columns and c not in numeric:
            numeric.append(c)
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical
    return numeric, categorical, features


def specialist_scores(
    dev: pd.DataFrame,
    ext: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    dev_high = dev["p2_pred"].astype(int).eq(1).to_numpy()
    ext_high = ext["p2_pred"].astype(int).eq(1).to_numpy()
    train = dev.loc[dev_high].reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    y = train["label_idx"].astype(int).eq(0).astype(int).to_numpy()  # 1 means likely false-positive high-risk output.
    models = v30.make_models(numeric, categorical)
    if model_name == "specialist_logistic":
        model = models["hard_logistic"]
    elif model_name == "specialist_rf":
        model = models["hard_rf"]
    elif model_name == "specialist_extra_trees":
        model = models["hard_extra_trees"]
    elif model_name == "specialist_gbdt":
        model = models["hard_gbdt"]
    else:
        raise ValueError(model_name)

    oof_subset = np.zeros(len(train), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260527)
    for tr, te in cv.split(train[features], y):
        fold_model = clone(model)
        fold_model.fit(train.iloc[tr][features], y[tr])
        oof_subset[te] = fold_model.predict_proba(train.iloc[te][features])[:, 1]
    final_model = clone(model)
    final_model.fit(train[features], y)
    ext_subset = ext.loc[ext_high].reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    ext_subset_score = final_model.predict_proba(ext_subset[features])[:, 1] if len(ext_subset) else np.array([])

    dev_score = np.zeros(len(dev), dtype=float)
    ext_score = np.zeros(len(ext), dtype=float)
    dev_score[train["_orig_idx"].to_numpy(dtype=int)] = oof_subset
    ext_score[ext_subset["_orig_idx"].to_numpy(dtype=int)] = ext_subset_score

    ext_target = ext.loc[ext_high, "label_idx"].astype(int).eq(0).astype(int).to_numpy()
    try:
        dev_auc = float(roc_auc_score(y, oof_subset))
    except ValueError:
        dev_auc = float("nan")
    try:
        ext_auc = float(roc_auc_score(ext_target, ext_subset_score)) if len(np.unique(ext_target)) > 1 else float("nan")
    except ValueError:
        ext_auc = float("nan")
    meta = {
        "model_name": model_name,
        "dev_pred_high_n": int(dev_high.sum()),
        "dev_false_positive_target_n": int(y.sum()),
        "external_pred_high_n": int(ext_high.sum()),
        "external_false_positive_target_n": int(ext_target.sum()),
        "dev_auc": dev_auc,
        "external_auc": ext_auc,
    }
    return dev_score, ext_score, meta


def add_pred_high_buffer(df: pd.DataFrame, base_review: np.ndarray, score: np.ndarray, addon_rate: float) -> np.ndarray:
    review = base_review.copy()
    n_add = int(round(len(df) * addon_rate))
    if n_add <= 0:
        return review
    p2 = df["p2_pred"].to_numpy(dtype=int)
    idx = np.flatnonzero((~review) & (p2 == 1))
    if len(idx) == 0:
        return review
    order = idx[np.argsort(-score[idx], kind="mergesort")]
    review[order[: min(n_add, len(order))]] = True
    return review


def metric(df: pd.DataFrame, review: np.ndarray) -> dict[str, float | int]:
    y = df["label_idx"].to_numpy(dtype=int)
    final = v51.final_prediction(df, review)
    m = v30.metrics_binary(y, final)
    masks = v48.error_masks(df)
    m.update(
        {
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
            "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
            "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
            "remaining_error_n": int((final != y).sum()),
        }
    )
    return m


def build_grid() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    numeric, categorical, features = feature_columns(dev, ext)
    grid54, ext_reviews = v54.build_policy_grid(dev, ext, dev_scores, ext_scores)
    base_ext = ext_reviews[BASE_POLICY]
    base_dev = None
    for cand in v54.build_score_candidates(dev, ext, dev_scores, ext_scores):
        if cand["family"] == "fusion_rank" and cand["name"] == "dir_plus_pred_low_fn":
            base_dev = v54.top_by_score(cand["dev_score"], 0.650)
            break
    if base_dev is None:
        raise RuntimeError("Base dev review not found")

    rows = []
    meta_rows = []
    score_cache: dict[str, np.ndarray] = {}
    for model_name in ["specialist_logistic", "specialist_rf", "specialist_extra_trees", "specialist_gbdt"]:
        dev_score, ext_score, meta = specialist_scores(dev, ext, features, numeric, categorical, model_name)
        meta_rows.append(meta)
        score_cache[f"dev_{model_name}"] = dev_score
        score_cache[f"ext_{model_name}"] = ext_score
        for addon_rate in ADDON_RATES:
            dev_review = add_pred_high_buffer(dev, base_dev, dev_score, float(addon_rate))
            ext_review = add_pred_high_buffer(ext, base_ext, ext_score, float(addon_rate))
            rows.append(
                {
                    "model_name": model_name,
                    "addon_rate": float(addon_rate),
                    **{f"dev_{k}": v for k, v in metric(dev, dev_review).items()},
                    **{f"external_{k}": v for k, v in metric(ext, ext_review).items()},
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(meta_rows), score_cache


def select_by_dev(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        ok = grid.loc[(grid["dev_sensitivity"] >= sc["sens_min"]) & (grid["dev_specificity"] >= sc["spec_min"])].copy()
        if ok.empty:
            chosen = grid.sort_values(["dev_sensitivity", "dev_specificity", "dev_balanced_accuracy"], ascending=[False, False, False]).iloc[0]
            met = 0
        else:
            chosen = ok.sort_values(["dev_review_rate", "dev_specificity", "dev_balanced_accuracy"], ascending=[True, False, False]).iloc[0]
            met = 1
        row = chosen.to_dict()
        row.update({"scenario": sc["scenario"], "sens_min": sc["sens_min"], "spec_min": sc["spec_min"], "constraints_met": met})
        rows.append(row)
    return pd.DataFrame(rows)


def external_oracle(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        ok = grid.loc[(grid["external_sensitivity"] >= sc["sens_min"]) & (grid["external_specificity"] >= sc["spec_min"])].copy()
        if ok.empty:
            continue
        chosen = ok.sort_values(["external_review_rate", "external_balanced_accuracy"], ascending=[True, False]).iloc[0]
        row = chosen.to_dict()
        row.update({"scenario": sc["scenario"], "sens_min": sc["sens_min"], "spec_min": sc["spec_min"]})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid, meta, _score_cache = build_grid()
    selected = select_by_dev(grid)
    oracle = external_oracle(grid)
    grid.to_csv(OUT_DIR / "v59_specialist_fp_buffer_grid.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(OUT_DIR / "v59_specialist_fp_model_auc.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v59_dev_selected_specialist_fp_buffer.csv", index=False, encoding="utf-8-sig")
    oracle.to_csv(OUT_DIR / "v59_external_oracle_specialist_fp_buffer.csv", index=False, encoding="utf-8-sig")
    show_cols = [
        "scenario",
        "model_name",
        "addon_rate",
        "dev_review_rate",
        "dev_sensitivity",
        "dev_specificity",
        "external_review_rate",
        "external_balanced_accuracy",
        "external_sensitivity",
        "external_specificity",
        "external_fn",
        "external_fp",
    ]
    print("Specialist AUC:")
    print(meta.to_string(index=False))
    print("\nDev-selected specialist FP buffer:")
    print(selected[show_cols].to_string(index=False))
    if not oracle.empty:
        print("\nExternal oracle upper bound:")
        print(oracle[show_cols].to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
