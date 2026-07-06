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
import run_grosspath_rc_v31_two_stage_auto_reviewer_20260527 as v31  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v33_calibrated_auto_reviewer_20260527"
BUDGETS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75]


def oof_reviewer_scores(dev: pd.DataFrame, ext: pd.DataFrame, features: list[str], model) -> tuple[np.ndarray, np.ndarray]:
    y = dev["label_idx"].to_numpy(dtype=int)
    oof = np.zeros(len(dev), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260527)
    for train_idx, test_idx in cv.split(dev[features], y):
        fold_model = clone(model)
        fold_model.fit(dev.iloc[train_idx][features], y[train_idx])
        oof[test_idx] = fold_model.predict_proba(dev.iloc[test_idx][features])[:, 1]
    full_model = clone(model)
    full_model.fit(dev[features], y)
    ext_scores = full_model.predict_proba(ext[features])[:, 1]
    return oof, ext_scores


def policy_predict(
    p2_pred: np.ndarray,
    reviewer_prob: np.ndarray,
    review_flag: np.ndarray,
    policy: str,
    low_thr: float | None,
    high_thr: float | None,
) -> np.ndarray:
    final = p2_pred.copy()
    if policy == "two_sided":
        assert low_thr is not None and high_thr is not None
        final[review_flag & (reviewer_prob <= low_thr)] = 0
        final[review_flag & (reviewer_prob >= high_thr)] = 1
    elif policy == "high_only":
        assert high_thr is not None
        final[review_flag & (p2_pred == 0) & (reviewer_prob >= high_thr)] = 1
    elif policy == "low_only":
        assert low_thr is not None
        final[review_flag & (p2_pred == 1) & (reviewer_prob <= low_thr)] = 0
    elif policy == "all_review_threshold":
        assert high_thr is not None
        final[review_flag] = (reviewer_prob[review_flag] >= high_thr).astype(int)
    else:
        raise ValueError(policy)
    return final


def summarize_policy(y: np.ndarray, p2_pred: np.ndarray, final: np.ndarray, review_flag: np.ndarray) -> dict[str, float | int]:
    before_wrong = p2_pred != y
    after_wrong = final != y
    changed = final != p2_pred
    rescued = before_wrong & ~after_wrong & changed
    harmed = ~before_wrong & after_wrong & changed
    m = v30.metrics_binary(y, final)
    m.update(
        {
            "review_n": int(review_flag.sum()),
            "review_rate": float(review_flag.mean()),
            "changed_n": int(changed.sum()),
            "changed_rate": float(changed.mean()),
            "rescued_n": int(rescued.sum()),
            "harmed_n": int(harmed.sum()),
            "net_rescue_n": int(rescued.sum() - harmed.sum()),
            "p2_errors_in_review_n": int((before_wrong & review_flag).sum()),
            "p2_errors_missed_by_review_n": int((before_wrong & ~review_flag).sum()),
        }
    )
    return m


def candidate_configs() -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    lows = np.round(np.arange(0.05, 0.50, 0.05), 2)
    highs = np.round(np.arange(0.50, 0.96, 0.05), 2)
    for high in highs:
        configs.append({"policy": "high_only", "low_thr": None, "high_thr": float(high)})
        configs.append({"policy": "all_review_threshold", "low_thr": None, "high_thr": float(high)})
    for low in lows:
        configs.append({"policy": "low_only", "low_thr": float(low), "high_thr": None})
    for low in lows:
        for high in highs:
            if low < high:
                configs.append({"policy": "two_sided", "low_thr": float(low), "high_thr": float(high)})
    return configs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dev = v30.load_development()
    ext = v30.load_external()
    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical

    hard_router = v30.make_models(numeric, categorical)["hard_logistic"]
    router_oof, router_ext = v30.oof_and_external_scores(dev, ext, features, hard_router)

    y_dev = dev["label_idx"].to_numpy(dtype=int)
    p2_dev = dev["p2_pred"].to_numpy(dtype=int)
    y_ext = ext["label_idx"].to_numpy(dtype=int)
    p2_ext = ext["p2_pred"].to_numpy(dtype=int)

    p2_dev_metrics = v30.metrics_binary(y_dev, p2_dev)
    p2_ext_metrics = v30.metrics_binary(y_ext, p2_ext)

    all_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []

    for reviewer_name, reviewer_model in v31.make_reviewer_models(numeric, categorical).items():
        reviewer_oof, reviewer_ext = oof_reviewer_scores(dev, ext, features, reviewer_model)
        try:
            reviewer_dev_auc = roc_auc_score(y_dev, reviewer_oof)
            reviewer_ext_auc = roc_auc_score(y_ext, reviewer_ext)
        except ValueError:
            reviewer_dev_auc = float("nan")
            reviewer_ext_auc = float("nan")

        for budget in BUDGETS:
            dev_review = v30.top_budget(router_oof, budget)
            ext_review = v30.top_budget(router_ext, budget)
            budget_rows: list[dict[str, object]] = []

            for cfg in candidate_configs():
                dev_final = policy_predict(
                    p2_dev,
                    reviewer_oof,
                    dev_review,
                    str(cfg["policy"]),
                    cfg["low_thr"],
                    cfg["high_thr"],
                )
                dev_m = summarize_policy(y_dev, p2_dev, dev_final, dev_review)
                row = {
                    "reviewer": reviewer_name,
                    "budget": budget,
                    "reviewer_dev_auc": reviewer_dev_auc,
                    "reviewer_external_auc": reviewer_ext_auc,
                    **cfg,
                    **{f"dev_{k}": v for k, v in dev_m.items()},
                }
                # Dev-only selection objective: require some actual changes, then maximize BAcc;
                # tie-break by net rescue and lower changed rate to avoid broad unstable overrides.
                row["dev_objective"] = (
                    float(dev_m["balanced_accuracy"])
                    + 0.001 * float(dev_m["net_rescue_n"])
                    - 0.0005 * float(dev_m["changed_n"])
                )
                if int(dev_m["changed_n"]) < 3:
                    row["dev_objective"] -= 1.0
                budget_rows.append(row)

            budget_df = pd.DataFrame(budget_rows)
            all_rows.extend(budget_rows)
            best = budget_df.sort_values(
                ["dev_objective", "dev_balanced_accuracy", "dev_net_rescue_n", "dev_changed_n"],
                ascending=[False, False, False, True],
            ).iloc[0].to_dict()

            ext_final = policy_predict(
                p2_ext,
                reviewer_ext,
                ext_review,
                str(best["policy"]),
                best["low_thr"] if pd.notna(best["low_thr"]) else None,
                best["high_thr"] if pd.notna(best["high_thr"]) else None,
            )
            ext_m = summarize_policy(y_ext, p2_ext, ext_final, ext_review)
            selected = {
                **best,
                **{f"external_{k}": v for k, v in ext_m.items()},
                "p2_dev_balanced_accuracy": p2_dev_metrics["balanced_accuracy"],
                "p2_external_balanced_accuracy": p2_ext_metrics["balanced_accuracy"],
            }
            selected_rows.append(selected)

            tmp = ext[
                [
                    "case_id",
                    "original_case_id",
                    "source_folder",
                    "task_l6_label",
                    "task_l7_label",
                    "label_idx",
                    "image_name",
                    "quality_status",
                    "quality_score",
                    "main_prob",
                    "main_pred",
                    "robust_prob",
                    "robust_pred",
                    "prob_mean_core",
                    "p2_pred",
                    "p2_wrong",
                ]
            ].copy()
            tmp.insert(0, "reviewer", reviewer_name)
            tmp.insert(1, "budget", budget)
            tmp["policy"] = best["policy"]
            tmp["low_thr"] = best["low_thr"]
            tmp["high_thr"] = best["high_thr"]
            tmp["router_hard_score"] = router_ext
            tmp["review_flag"] = ext_review.astype(int)
            tmp["reviewer_prob"] = reviewer_ext
            tmp["final_pred"] = ext_final
            tmp["changed_flag"] = (ext_final != p2_ext).astype(int)
            tmp["final_correct"] = (ext_final == y_ext).astype(int)
            before_wrong = p2_ext != y_ext
            after_wrong = ext_final != y_ext
            tmp["bucket"] = np.select(
                [
                    (ext_final != p2_ext) & before_wrong & ~after_wrong,
                    (ext_final != p2_ext) & ~before_wrong & after_wrong,
                    ext_review & before_wrong & after_wrong,
                    ext_review & ~before_wrong,
                    ~ext_review & before_wrong,
                ],
                ["rescued", "harmed", "review_still_wrong", "review_kept_correct", "missed_p2_error"],
                default="auto_correct",
            )
            case_frames.append(tmp)

    all_df = pd.DataFrame(all_rows)
    selected_df = pd.DataFrame(selected_rows)
    cases = pd.concat(case_frames, ignore_index=True)

    all_df.to_csv(OUT_DIR / "v33_all_dev_calibration_candidates.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(OUT_DIR / "v33_selected_dev_calibrated_policies_external_eval.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v33_selected_policy_case_routes_external.csv", index=False, encoding="utf-8-sig")

    show_cols = [
        "reviewer",
        "budget",
        "policy",
        "low_thr",
        "high_thr",
        "dev_balanced_accuracy",
        "dev_net_rescue_n",
        "dev_changed_n",
        "external_balanced_accuracy",
        "external_accuracy",
        "external_fn",
        "external_fp",
        "external_net_rescue_n",
        "external_changed_n",
        "external_review_rate",
    ]
    print(selected_df[show_cols].sort_values(["external_balanced_accuracy", "external_changed_n"], ascending=[False, True]).head(30).to_string(index=False))
    print(f"\nP2 external BAcc={p2_ext_metrics['balanced_accuracy']:.4f}, Acc={p2_ext_metrics['accuracy']:.4f}")
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
