from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_grosspath_rc_v143_image_feature_error_router_20260527 import (
    BASE_CANDIDATES,
    BUDGETS,
    OUT_DIR as V143_OUT_DIR,
    ROOT,
    metrics,
    prepare,
    review_budget_rows,
    risk_summary_row,
    safe_ap,
    safe_auc,
    choose_threshold,
)


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v144_distilled_concept_router_20260527"
CONCEPTS = ROOT / "outputs" / "grosspath_rc_v0_20260526" / "gross_concepts_v1.csv"

CONCEPT_COLS = [
    "boundary_clear",
    "boundary_unclear",
    "capsule_any",
    "capsule_complete",
    "capsule_absent",
    "invasion",
    "lung_attached",
    "pericardium_attached",
    "hemorrhage",
    "necrosis",
    "cystic_change",
    "nodular_lobulated",
    "texture_tough",
    "gray_white",
    "gray_yellow",
    "sex_male",
]


def concept_model(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=180,
        max_depth=4,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def risk_models(seed: int) -> dict[str, object]:
    return {
        "logreg_c03": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
        ),
        "extra_d3": ExtraTreesClassifier(
            n_estimators=220,
            max_depth=3,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def attach_concepts(internal: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    concepts = pd.read_csv(CONCEPTS, dtype={"original_case_id": str})
    keep = ["original_case_id"] + [c for c in CONCEPT_COLS if c in concepts.columns]
    concepts = concepts[keep].copy()
    for col in keep[1:]:
        concepts[col] = pd.to_numeric(concepts[col], errors="coerce")
    out = internal.merge(concepts, on="original_case_id", how="left", validate="many_to_one")
    return out, keep[1:]


def fit_predict_one_concept(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, object | None]:
    mask = np.isfinite(y_train)
    if mask.sum() < 10 or len(np.unique(y_train[mask].astype(int))) < 2:
        rate = float(np.nanmean(y_train)) if np.isfinite(y_train).any() else 0.5
        return np.full(x_pred.shape[0], rate, dtype=float), None
    clf = concept_model(seed)
    clf.fit(x_train[mask], y_train[mask].astype(int))
    return clf.predict_proba(x_pred)[:, 1], clf


def precompute_concept_predictions(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    feat_cols: list[str],
    concept_cols: list[str],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], np.ndarray, np.ndarray, pd.DataFrame]:
    x = np.nan_to_num(internal[feat_cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    x_ext = np.nan_to_num(external[feat_cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    folds = internal["fold_id"].astype(int).to_numpy()
    y_concepts = internal[concept_cols].to_numpy(float)

    fold_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    concept_oof = np.zeros((len(internal), len(concept_cols)), dtype=float)
    perf_rows: list[dict[str, object]] = []

    for fold in sorted(np.unique(folds)):
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]
        train_pred = np.zeros((len(train_idx), len(concept_cols)), dtype=float)
        test_pred = np.zeros((len(test_idx), len(concept_cols)), dtype=float)
        for j, concept in enumerate(concept_cols):
            y_train = y_concepts[train_idx, j]
            pred_train, clf = fit_predict_one_concept(x[train_idx], y_train, x[train_idx], 20260527 + j)
            train_pred[:, j] = pred_train
            if clf is None:
                rate = float(np.nanmean(y_train)) if np.isfinite(y_train).any() else 0.5
                test_pred[:, j] = rate
            else:
                test_pred[:, j] = clf.predict_proba(x[test_idx])[:, 1]
        concept_oof[test_idx] = test_pred
        fold_cache[int(fold)] = (train_idx, test_idx, train_pred, test_pred)

    full_internal = np.zeros((len(internal), len(concept_cols)), dtype=float)
    external_pred = np.zeros((len(external), len(concept_cols)), dtype=float)
    for j, concept in enumerate(concept_cols):
        yj = y_concepts[:, j]
        pred_internal, clf = fit_predict_one_concept(x, yj, x, 20260627 + j)
        full_internal[:, j] = pred_internal
        if clf is None:
            rate = float(np.nanmean(yj)) if np.isfinite(yj).any() else 0.5
            external_pred[:, j] = rate
        else:
            external_pred[:, j] = clf.predict_proba(x_ext)[:, 1]
        valid = np.isfinite(yj)
        perf_rows.append(
            {
                "concept": concept,
                "positive_n": int(np.nansum(yj)),
                "labeled_n": int(valid.sum()),
                "positive_rate": float(np.nanmean(yj)) if valid.any() else np.nan,
                "oof_auroc": safe_auc(yj[valid].astype(int), concept_oof[valid, j]) if valid.any() else np.nan,
                "oof_average_precision": safe_ap(yj[valid].astype(int), concept_oof[valid, j]) if valid.any() else np.nan,
            }
        )

    return fold_cache, full_internal, external_pred, pd.DataFrame(perf_rows)


def base_feature_frame(df: pd.DataFrame, base_name: str, pred: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    conf = np.where(pred == 1, prob, 1.0 - prob)
    out = pd.DataFrame(
        {
            "base_prob": prob,
            "base_pred": pred.astype(float),
            "base_conf": conf,
            "base_uncertainty": 1.0 - conf,
            "base_entropy": -(prob * np.log(np.clip(prob, 1e-6, 1.0)) + (1.0 - prob) * np.log(np.clip(1.0 - prob, 1e-6, 1.0))),
        },
        index=df.index,
    )
    for candidate in BASE_CANDIDATES:
        out[f"candidate_{candidate}"] = df[candidate].to_numpy(float)
    return out


def build_features(base_feats: pd.DataFrame, concept_pred: np.ndarray, concept_cols: list[str], feature_set: str) -> np.ndarray:
    if feature_set == "confidence_only":
        return base_feats[["base_prob", "base_pred", "base_conf", "base_uncertainty", "base_entropy"]].to_numpy(float)
    concept_df = pd.DataFrame(concept_pred, columns=[f"pred_{c}" for c in concept_cols], index=base_feats.index)
    if feature_set == "pred_concept":
        return pd.concat([base_feats[["base_prob", "base_pred", "base_conf", "base_uncertainty", "base_entropy"]], concept_df], axis=1).to_numpy(float)
    if feature_set == "pred_concept_plus_base_probs":
        return pd.concat([base_feats, concept_df], axis=1).to_numpy(float)
    raise ValueError(feature_set)


def oof_external_risk(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    fold_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    full_internal_concepts: np.ndarray,
    external_concepts: np.ndarray,
    concept_cols: list[str],
    base_name: str,
    target: np.ndarray,
    model: object,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_ext_dummy = np.zeros(len(external), dtype=int)
    folds = internal["fold_id"].astype(int).to_numpy()
    prob = internal[base_name].to_numpy(float)
    threshold = choose_threshold(internal["label_idx"].astype(int).to_numpy(), prob)
    pred = (prob >= threshold).astype(int)
    base_feats = base_feature_frame(internal, base_name, pred, prob)
    prob_ext = external[base_name].to_numpy(float)
    pred_ext = (prob_ext >= threshold).astype(int)
    base_feats_ext = base_feature_frame(external, base_name, pred_ext, prob_ext)

    oof = np.zeros(len(internal), dtype=float)
    for fold in sorted(np.unique(folds)):
        train_idx, test_idx, train_concepts, test_concepts = fold_cache[int(fold)]
        x_train = build_features(base_feats.iloc[train_idx].reset_index(drop=True), train_concepts, concept_cols, feature_set)
        x_test = build_features(base_feats.iloc[test_idx].reset_index(drop=True), test_concepts, concept_cols, feature_set)
        if len(np.unique(target[train_idx])) < 2:
            oof[test_idx] = float(np.mean(target[train_idx]))
            continue
        clf = clone(model)
        clf.fit(x_train, target[train_idx])
        oof[test_idx] = clf.predict_proba(x_test)[:, 1]

    x_full = build_features(base_feats.reset_index(drop=True), full_internal_concepts, concept_cols, feature_set)
    x_ext = build_features(base_feats_ext.reset_index(drop=True), external_concepts, concept_cols, feature_set)
    if len(np.unique(target)) < 2:
        ext = np.full(len(external), float(np.mean(target)), dtype=float)
    else:
        clf = clone(model)
        clf.fit(x_full, target)
        ext = clf.predict_proba(x_ext)[:, 1]
    return oof, ext


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    internal0, external, feat_cols = prepare()
    internal, concept_cols = attach_concepts(internal0)
    fold_cache, full_concepts, external_concepts, concept_perf = precompute_concept_predictions(internal, external, feat_cols, concept_cols)
    concept_perf.sort_values("oof_auroc", ascending=False).to_csv(OUT_DIR / "v144_predicted_concept_performance.csv", index=False, encoding="utf-8-sig")

    models = risk_models(20260527)
    feature_sets = ["confidence_only", "pred_concept", "pred_concept_plus_base_probs"]
    risk_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []

    y = internal["label_idx"].astype(int).to_numpy()
    y_ext = external["label_idx"].astype(int).to_numpy()

    for base_name in BASE_CANDIDATES:
        p = internal[base_name].to_numpy(float)
        t = choose_threshold(y, p)
        pred = (p >= t).astype(int)
        p_ext = external[base_name].to_numpy(float)
        pred_ext = (p_ext >= t).astype(int)
        targets = {
            "any_error": (pred != y).astype(int),
            "fn_error": ((y == 1) & (pred == 0)).astype(int),
            "fp_error": ((y == 0) & (pred == 1)).astype(int),
        }
        targets_ext = {
            "any_error": (pred_ext != y_ext).astype(int),
            "fn_error": ((y_ext == 1) & (pred_ext == 0)).astype(int),
            "fp_error": ((y_ext == 0) & (pred_ext == 1)).astype(int),
        }
        low_conf = 1.0 - np.where(pred == 1, p, 1.0 - p)
        low_conf_ext = 1.0 - np.where(pred_ext == 1, p_ext, 1.0 - p_ext)
        for scope, yy, ppred, pprob, target, score in [
            ("internal_oof", y, pred, p, targets["any_error"], low_conf),
            ("strict_external_locked", y_ext, pred_ext, p_ext, targets_ext["any_error"], low_conf_ext),
        ]:
            risk_rows.append(risk_summary_row(scope, base_name, "any_error", "heuristic_low_conf", target, score))
            budget_rows.extend(review_budget_rows(scope, base_name, "heuristic_low_conf", yy, ppred, pprob, score))
            source_df = internal if scope == "internal_oof" else external
            score_frames.append(
                pd.DataFrame(
                    {
                        "scope": scope,
                        "base_model": base_name,
                        "router": "heuristic_low_conf",
                        "case_id": source_df["case_id"].astype(str).to_numpy(),
                        "label_idx": yy,
                        "base_pred": ppred,
                        "base_prob": pprob,
                        "base_correct": ppred == yy,
                        "risk_score": score,
                    }
                )
            )

        directional_internal: dict[str, np.ndarray] = {}
        directional_external: dict[str, np.ndarray] = {}
        for feature_set in feature_sets:
            for target_name, target in targets.items():
                for model_name, model in models.items():
                    oof, ext = oof_external_risk(
                        internal,
                        external,
                        fold_cache,
                        full_concepts,
                        external_concepts,
                        concept_cols,
                        base_name,
                        target,
                        model,
                        feature_set,
                    )
                    router = f"{feature_set}:{model_name}"
                    risk_rows.append(risk_summary_row("internal_oof", base_name, target_name, router, target, oof))
                    risk_rows.append(risk_summary_row("strict_external_locked", base_name, target_name, router, targets_ext[target_name], ext))
                    if target_name == "any_error":
                        budget_rows.extend(review_budget_rows("internal_oof", base_name, f"{router}:any", y, pred, p, oof))
                        budget_rows.extend(review_budget_rows("strict_external_locked", base_name, f"{router}:any", y_ext, pred_ext, p_ext, ext))
                        score_frames.append(
                            pd.DataFrame(
                                {
                                    "scope": "internal_oof",
                                    "base_model": base_name,
                                    "router": f"{router}:any",
                                    "case_id": internal["case_id"].astype(str).to_numpy(),
                                    "label_idx": y,
                                    "base_pred": pred,
                                    "base_prob": p,
                                    "base_correct": pred == y,
                                    "risk_score": oof,
                                }
                            )
                        )
                        score_frames.append(
                            pd.DataFrame(
                                {
                                    "scope": "strict_external_locked",
                                    "base_model": base_name,
                                    "router": f"{router}:any",
                                    "case_id": external["case_id"].astype(str).to_numpy(),
                                    "label_idx": y_ext,
                                    "base_pred": pred_ext,
                                    "base_prob": p_ext,
                                    "base_correct": pred_ext == y_ext,
                                    "risk_score": ext,
                                }
                            )
                        )
                    else:
                        directional_internal[f"{router}:{target_name}"] = oof
                        directional_external[f"{router}:{target_name}"] = ext

            for model_name in models:
                router = f"{feature_set}:{model_name}"
                fn_oof = directional_internal[f"{router}:fn_error"]
                fp_oof = directional_internal[f"{router}:fp_error"]
                fn_ext = directional_external[f"{router}:fn_error"]
                fp_ext = directional_external[f"{router}:fp_error"]
                dir_oof = np.where(pred == 0, fn_oof, fp_oof)
                dir_ext = np.where(pred_ext == 0, fn_ext, fp_ext)
                risk_rows.append(risk_summary_row("internal_oof", base_name, "any_error", f"{router}:directional", targets["any_error"], dir_oof))
                risk_rows.append(risk_summary_row("strict_external_locked", base_name, "any_error", f"{router}:directional", targets_ext["any_error"], dir_ext))
                budget_rows.extend(review_budget_rows("internal_oof", base_name, f"{router}:directional", y, pred, p, dir_oof))
                budget_rows.extend(review_budget_rows("strict_external_locked", base_name, f"{router}:directional", y_ext, pred_ext, p_ext, dir_ext))
                score_frames.append(
                    pd.DataFrame(
                        {
                            "scope": "internal_oof",
                            "base_model": base_name,
                            "router": f"{router}:directional",
                            "case_id": internal["case_id"].astype(str).to_numpy(),
                            "label_idx": y,
                            "base_pred": pred,
                            "base_prob": p,
                            "base_correct": pred == y,
                            "risk_score": dir_oof,
                        }
                    )
                )
                score_frames.append(
                    pd.DataFrame(
                        {
                            "scope": "strict_external_locked",
                            "base_model": base_name,
                            "router": f"{router}:directional",
                            "case_id": external["case_id"].astype(str).to_numpy(),
                            "label_idx": y_ext,
                            "base_pred": pred_ext,
                            "base_prob": p_ext,
                            "base_correct": pred_ext == y_ext,
                            "risk_score": dir_ext,
                        }
                    )
                )

        cases = internal[["case_id", "original_case_id", "domain", "fold_id", "task_l6_label", "label_idx"]].copy()
        cases["scope"] = "internal_oof"
        cases["base_model"] = base_name
        cases["base_prob"] = p
        cases["base_pred"] = pred
        cases["base_correct"] = pred == y
        cases_ext = external[["case_id", "original_case_id", "domain", "task_l6_label", "label_idx"]].copy()
        cases_ext["fold_id"] = -1
        cases_ext["scope"] = "strict_external_locked"
        cases_ext["base_model"] = base_name
        cases_ext["base_prob"] = p_ext
        cases_ext["base_pred"] = pred_ext
        cases_ext["base_correct"] = pred_ext == y_ext
        case_frames.append(pd.concat([cases, cases_ext], ignore_index=True))

    risk_df = pd.DataFrame(risk_rows).sort_values(["scope", "base_model", "target", "auroc"], ascending=[True, True, True, False])
    budget_df = pd.DataFrame(budget_rows).sort_values(
        ["scope", "base_model", "review_budget", "system_if_review_corrected_balanced_accuracy"],
        ascending=[True, True, True, False],
    )
    risk_df.to_csv(OUT_DIR / "v144_distilled_concept_router_risk_summary.csv", index=False, encoding="utf-8-sig")
    budget_df.to_csv(OUT_DIR / "v144_distilled_concept_router_review_budget_curve.csv", index=False, encoding="utf-8-sig")
    pd.concat(case_frames, ignore_index=True).to_csv(OUT_DIR / "v144_distilled_concept_router_cases.csv", index=False, encoding="utf-8-sig")
    pd.concat(score_frames, ignore_index=True).to_csv(OUT_DIR / "v144_distilled_concept_router_risk_scores_long.csv", index=False, encoding="utf-8-sig")
    report = {
        "internal_n": int(len(internal)),
        "strict_external_n": int(len(external)),
        "concepts": concept_cols,
        "base_candidates": BASE_CANDIDATES,
        "note": "Concepts are predicted from image features inside each outer fold, not read from doctor text at test time.",
        "v143_output_reference": str(V143_OUT_DIR),
        "budgets": [float(x) for x in BUDGETS],
    }
    (OUT_DIR / "v144_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v144] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
