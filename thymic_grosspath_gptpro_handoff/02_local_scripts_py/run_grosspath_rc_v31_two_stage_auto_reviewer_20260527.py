from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v31_two_stage_auto_reviewer_20260527"
BUDGETS = [0.50, 0.60, 0.75]


def make_reviewer_models(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    linear_pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    tree_pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    return {
        "review_logistic_c025": Pipeline(
            [
                ("prep", linear_pre),
                ("clf", LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000, random_state=20260527)),
            ]
        ),
        "review_logistic_c1": Pipeline(
            [
                ("prep", linear_pre),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=20260527)),
            ]
        ),
        "review_rf": Pipeline(
            [
                ("prep", tree_pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=600,
                        max_depth=5,
                        min_samples_leaf=6,
                        class_weight="balanced_subsample",
                        random_state=20260527,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "review_extra_trees": Pipeline(
            [
                ("prep", tree_pre),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        max_depth=5,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=20260527,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "review_gbdt": Pipeline(
            [
                ("prep", tree_pre),
                ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.035, max_depth=2, random_state=20260527)),
            ]
        ),
    }


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return v30.metrics_binary(y, pred)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = v30.load_development()
    ext = v30.load_external()

    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical

    # Freeze the best v30 router: development-trained hard_logistic.
    hard_router = v30.make_models(numeric, categorical)["hard_logistic"]
    hard_router.fit(dev[features], dev["p2_wrong"].astype(int))
    ext_hard_score = hard_router.predict_proba(ext[features])[:, 1]

    y_ext = ext["label_idx"].to_numpy(dtype=int)
    p2_ext = ext["p2_pred"].to_numpy(dtype=int)

    rows = []
    case_frames = []
    base = evaluate(y_ext, p2_ext)
    rows.append(
        {
            "router": "none",
            "reviewer": "P2_auto_baseline",
            "budget": 0.0,
            "review_n": 0,
            "review_rate": 0.0,
            **{f"external_{k}": v for k, v in base.items()},
        }
    )

    for reviewer_name, reviewer in make_reviewer_models(numeric, categorical).items():
        reviewer.fit(dev[features], dev["label_idx"].astype(int))
        reviewer_prob = reviewer.predict_proba(ext[features])[:, 1]
        reviewer_pred = (reviewer_prob >= 0.5).astype(int)
        reviewer_all_metrics = evaluate(y_ext, reviewer_pred)
        rows.append(
            {
                "router": "none",
                "reviewer": f"{reviewer_name}_all_cases",
                "budget": 1.0,
                "review_n": len(ext),
                "review_rate": 1.0,
                **{f"external_{k}": v for k, v in reviewer_all_metrics.items()},
            }
        )

        for budget in BUDGETS:
            review = v30.top_budget(ext_hard_score, budget)
            final = p2_ext.copy()
            final[review] = reviewer_pred[review]
            m = evaluate(y_ext, final)
            before_wrong = p2_ext != y_ext
            after_wrong = final != y_ext
            rescued = before_wrong & ~after_wrong & review
            harmed = ~before_wrong & after_wrong & review
            rows.append(
                {
                    "router": "v30_hard_logistic",
                    "reviewer": reviewer_name,
                    "budget": budget,
                    "review_n": int(review.sum()),
                    "review_rate": float(review.mean()),
                    "rescued_n": int(rescued.sum()),
                    "harmed_n": int(harmed.sum()),
                    "net_rescue_n": int(rescued.sum() - harmed.sum()),
                    "p2_errors_enter_review_n": int((before_wrong & review).sum()),
                    "p2_errors_missed_n": int((before_wrong & ~review).sum()),
                    **{f"external_{k}": v for k, v in m.items()},
                }
            )

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
                    "p2_pred",
                    "p2_wrong",
                    "main_prob",
                    "robust_prob",
                    "prob_mean_core",
                ]
            ].copy()
            tmp.insert(0, "reviewer", reviewer_name)
            tmp.insert(1, "budget", budget)
            tmp["hard_risk_score"] = ext_hard_score
            tmp["review_flag"] = review.astype(int)
            tmp["reviewer_prob"] = reviewer_prob
            tmp["reviewer_pred"] = reviewer_pred
            tmp["final_pred"] = final
            tmp["final_correct"] = final == y_ext
            tmp["bucket"] = np.select(
                [rescued, harmed, review & before_wrong, review & ~before_wrong, ~review & before_wrong],
                ["rescued", "harmed", "review_still_wrong", "review_kept_correct", "missed_p2_error"],
                default="auto_correct",
            )
            case_frames.append(tmp)

    metrics = pd.DataFrame(rows)
    cases = pd.concat(case_frames, ignore_index=True)
    metrics.to_csv(OUT_DIR / "v31_two_stage_auto_reviewer_metrics.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v31_two_stage_auto_reviewer_case_routes.csv", index=False, encoding="utf-8-sig")

    show_cols = [
        "router",
        "reviewer",
        "budget",
        "review_rate",
        "rescued_n",
        "harmed_n",
        "net_rescue_n",
        "external_accuracy",
        "external_balanced_accuracy",
        "external_fn",
        "external_fp",
    ]
    print(metrics[[c for c in show_cols if c in metrics.columns]].sort_values(["external_balanced_accuracy", "review_rate"], ascending=[False, True]).head(30).to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
