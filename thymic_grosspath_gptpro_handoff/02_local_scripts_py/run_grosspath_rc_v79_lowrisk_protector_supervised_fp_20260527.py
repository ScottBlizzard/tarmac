from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v73_pseudodomain_policy_search_20260527 as v73  # noqa: E402
import run_grosspath_rc_v74_dev_prespecified_quality_gate_20260527 as v74  # noqa: E402
import run_grosspath_rc_v75_joint_quality_risk_ood_search_20260527 as v75  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v79_lowrisk_protector_supervised_fp_20260527"
RATES = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]
SEED = 20260527


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["domain"] = np.where(out["source_folder"].notna(), "third_batch", "old_data")
    return out


def v75_base_review(reference: pd.DataFrame, target: pd.DataFrame, scores: dict[str, np.ndarray]) -> np.ndarray:
    base = v73.v50_review(target, scores)
    joint = v75.joint_scores(reference, target, scores)
    extra = v73.top_by_rate(joint["quality_plus_lowconf_mean"], 0.30) & (~base)
    return base | extra


def normalize_image_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "contrast" not in out.columns and "contrast_std" in out.columns:
        out["contrast"] = out["contrast_std"]
    if "contrast_std" not in out.columns and "contrast" in out.columns:
        out["contrast_std"] = out["contrast"]
    if "saturation_mean" in out.columns:
        sat = pd.to_numeric(out["saturation_mean"], errors="coerce")
        if sat.median(skipna=True) > 2:
            out["saturation_mean"] = sat / 255.0
    return out


def add_derived_features(reference: pd.DataFrame, target: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    out = normalize_image_columns(target)
    out = out.copy()
    out["risk_any_score"] = np.asarray(scores["any"], dtype=float)
    out["risk_direction_score"] = np.asarray(scores["direction"], dtype=float)
    out["quality_proxy_raw"] = v74.quality_proxy_risk(reference, target)
    out["quality_proxy_rank"] = v75.rank01(out["quality_proxy_raw"].to_numpy(float))
    if {"main_margin_abs", "robust_margin_abs"}.issubset(out.columns):
        main = pd.to_numeric(out["main_margin_abs"], errors="coerce").to_numpy(float)
        robust = pd.to_numeric(out["robust_margin_abs"], errors="coerce").to_numpy(float)
        out["lowconf_raw"] = 1.0 - np.minimum(main, robust)
        out["lowconf_rank"] = v75.rank01(out["lowconf_raw"].to_numpy(float))
    return out


def feature_sets(train: pd.DataFrame, target: pd.DataFrame) -> dict[str, list[str]]:
    common = lambda cols: [c for c in cols if c in train.columns and c in target.columns]
    prob = common(v30.PROB_FEATURES + ["risk_any_score", "risk_direction_score", "lowconf_raw", "lowconf_rank"])
    bins = common(v30.BIN_FEATURES)
    image = common(v30.IMAGE_FEATURES + ["contrast_std", "quality_proxy_raw", "quality_proxy_rank"])
    compact = common(
        [
            "prob_base162",
            "prob103_vitl",
            "prob107_qkvb",
            "prob_mean_core",
            "core_prob_std",
            "core_prob_range",
            "margin_mean_core",
            "main_prob",
            "robust_prob",
            "main_margin_abs",
            "robust_margin_abs",
            "main_robust_abs_diff",
            "score_margin_agree",
            "core_agree_count",
            "risk_any_score",
            "risk_direction_score",
            "lowconf_raw",
            "quality_proxy_raw",
        ]
    )
    return {
        "model_compact": compact + bins,
        "model_full": prob + bins,
        "image_quality": image,
        "model_quality": compact + bins + common(["quality_proxy_raw", "quality_proxy_rank"]),
        "model_image_quality": compact + bins + image,
    }


def classifiers() -> dict[str, object]:
    return {
        "logreg_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler(quantile_range=(10, 90))),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)),
            ]
        ),
        "rf_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
            ]
        ),
        "extra_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", ExtraTreesClassifier(n_estimators=700, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)),
            ]
        ),
    }


def supervised_scores(
    train_ref: pd.DataFrame,
    train_df: pd.DataFrame,
    train_scores: dict[str, np.ndarray],
    target_ref: pd.DataFrame,
    target_df: pd.DataFrame,
    target_scores: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    train_feat = add_derived_features(train_ref, train_df, train_scores)
    target_feat = add_derived_features(target_ref, target_df, target_scores)
    train_mask = train_feat["p2_pred"].to_numpy(int) == 1
    y_train = ((train_feat["label_idx"].to_numpy(int) == 0) & train_mask).astype(int)
    out: dict[str, np.ndarray] = {}
    if train_mask.sum() < 12 or y_train[train_mask].sum() < 2 or (train_mask.sum() - y_train[train_mask].sum()) < 2:
        return out
    fs = feature_sets(train_feat, target_feat)
    for set_name, cols in fs.items():
        cols = list(dict.fromkeys(cols))
        if len(cols) < 3:
            continue
        x_train = train_feat.loc[train_mask, cols]
        y = y_train[train_mask]
        x_target = target_feat[cols]
        for clf_name, clf in classifiers().items():
            try:
                clf.fit(x_train, y)
                score = clf.predict_proba(x_target)[:, 1]
            except Exception as exc:
                print(f"[skip] {set_name}/{clf_name}: {exc}")
                continue
            out[f"supfp_{set_name}_{clf_name}"] = np.asarray(score, dtype=float)
    return out


def direct_scores(reference: pd.DataFrame, target: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    feat = add_derived_features(reference, target, scores)
    out: dict[str, np.ndarray] = {}
    for col in ["main_prob", "robust_prob", "prob_mean_core", "prob_stack_balanced", "prob_base162"]:
        if col in feat.columns:
            out[f"direct_low_{col}"] = -pd.to_numeric(feat[col], errors="coerce").to_numpy(float)
    for col in ["main_margin_abs", "robust_margin_abs", "margin_mean_core", "score_margin_agree", "core_agree_count"]:
        if col in feat.columns:
            out[f"direct_low_{col}"] = -pd.to_numeric(feat[col], errors="coerce").to_numpy(float)
    out["direct_quality_proxy"] = pd.to_numeric(feat["quality_proxy_raw"], errors="coerce").to_numpy(float)
    out["direct_lowconf_raw"] = pd.to_numeric(feat.get("lowconf_raw", pd.Series(np.zeros(len(feat)))), errors="coerce").to_numpy(float)
    return out


def masked_score(score: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    s = np.asarray(score, dtype=float).copy()
    s[~eligible] = -np.inf
    return s


def run_split(
    split: str,
    train_ref: pd.DataFrame,
    train_df: pd.DataFrame,
    train_scores: dict[str, np.ndarray],
    target_ref: pd.DataFrame,
    target_df: pd.DataFrame,
    target_scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    base = v75_base_review(target_ref, target_df, target_scores)
    eligible = (~base) & (target_df["p2_pred"].to_numpy(int) == 1)
    rows = [v73.evaluate(split, "v75_base", target_df, base)]
    scores = {}
    scores.update(direct_scores(target_ref, target_df, target_scores))
    scores.update(supervised_scores(train_ref, train_df, train_scores, target_ref, target_df, target_scores))
    for name, score in scores.items():
        masked = masked_score(score, eligible)
        for rate in RATES:
            extra = v73.top_by_rate(masked, rate) & (~base)
            review = base | extra
            row = v73.evaluate(split, f"v75_plus_lowrisk_protect_{name}_r{int(rate * 1000):03d}", target_df, review, extra)
            row["candidate"] = name
            row["rate"] = rate
            row["eligible_auto_high_n"] = int(eligible.sum())
            rows.append(row)
    out = pd.DataFrame(rows)
    out["candidate"] = out["candidate"].fillna("none")
    out["rate"] = out["rate"].fillna(0.0)
    return out


def aggregate(pseudo: pd.DataFrame) -> pd.DataFrame:
    agg = v73.pseudo_aggregate(pseudo)
    third = pseudo.loc[pseudo["split"].eq("pseudo_old_to_third"), ["candidate", "rate", "balanced_accuracy", "control_rate", "fn", "fp", "specificity", "remaining_error_n"]]
    third = third.rename(
        columns={
            "balanced_accuracy": "third_pseudo_bacc",
            "control_rate": "third_pseudo_control",
            "fn": "third_pseudo_fn",
            "fp": "third_pseudo_fp",
            "specificity": "third_pseudo_specificity",
            "remaining_error_n": "third_pseudo_error_n",
        }
    )
    old = pseudo.loc[pseudo["split"].eq("pseudo_third_to_old"), ["candidate", "rate", "balanced_accuracy", "control_rate", "fn", "fp", "specificity", "remaining_error_n"]]
    old = old.rename(
        columns={
            "balanced_accuracy": "old_pseudo_bacc",
            "control_rate": "old_pseudo_control",
            "fn": "old_pseudo_fn",
            "fp": "old_pseudo_fp",
            "specificity": "old_pseudo_specificity",
            "remaining_error_n": "old_pseudo_error_n",
        }
    )
    return agg.merge(third, on=["candidate", "rate"], how="left").merge(old, on=["candidate", "rate"], how="left")


def select(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule, mask in [
        ("best_bacc_le082", agg["pseudo_max_control"].le(0.82)),
        ("best_bacc_le085", agg["pseudo_max_control"].le(0.85)),
        ("min_fp_le082", agg["pseudo_max_control"].le(0.82)),
        ("min_error_le085", agg["pseudo_max_control"].le(0.85)),
        ("best_specificity_le085", agg["pseudo_max_control"].le(0.85)),
    ]:
        pool = agg.loc[mask].copy()
        if pool.empty:
            continue
        if rule == "min_fp_le082":
            pick = pool.sort_values(["pseudo_total_fp", "pseudo_total_fn", "pseudo_min_bacc", "pseudo_mean_control"], ascending=[True, True, False, True]).head(1)
        elif rule == "min_error_le085":
            pick = pool.sort_values(["pseudo_total_remaining_error", "pseudo_total_fp", "pseudo_mean_control"], ascending=[True, True, True]).head(1)
        elif rule == "best_specificity_le085":
            pick = pool.sort_values(["pseudo_min_specificity", "pseudo_total_fn", "pseudo_mean_control"], ascending=[False, True, True]).head(1)
        else:
            pick = pool.sort_values(["pseudo_min_bacc", "pseudo_mean_control"], ascending=[False, True]).head(1)
        pick = pick.copy()
        pick.insert(0, "selection_rule", rule)
        rows.append(pick)
    return pd.concat(rows, ignore_index=True).drop_duplicates(["selection_rule", "candidate", "rate"])


def run_external(dev: pd.DataFrame, dev_scores: dict[str, np.ndarray], ext: pd.DataFrame, ext_scores: dict[str, np.ndarray], selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = v75_base_review(dev, ext, ext_scores)
    eligible = (~base) & (ext["p2_pred"].to_numpy(int) == 1)
    scores = {}
    scores.update(direct_scores(dev, ext, ext_scores))
    scores.update(supervised_scores(dev, dev, dev_scores, dev, ext, ext_scores))
    rows = [v73.evaluate("strict_external", "v75_base", ext, base)]
    all_rows = [v73.evaluate("strict_external_exploratory", "v75_base", ext, base)]
    for name, score in scores.items():
        masked = masked_score(score, eligible)
        for rate in RATES:
            extra = v73.top_by_rate(masked, rate) & (~base)
            review = base | extra
            row = v73.evaluate("strict_external_exploratory", f"v75_plus_lowrisk_protect_{name}_r{int(rate * 1000):03d}", ext, review, extra)
            row["candidate"] = name
            row["rate"] = rate
            all_rows.append(row)
    for _, pick in selected.iterrows():
        name = str(pick["candidate"])
        rate = float(pick["rate"])
        if name == "none" or name not in scores:
            extra = np.zeros(len(ext), dtype=bool)
        else:
            extra = v73.top_by_rate(masked_score(scores[name], eligible), rate) & (~base)
        review = base | extra
        policy = f"selected_{pick['selection_rule']}__{name}_r{int(rate * 1000):03d}"
        row = v73.evaluate("strict_external", policy, ext, review, extra)
        row["selection_rule"] = pick["selection_rule"]
        row["candidate"] = name
        row["rate"] = rate
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(all_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev = add_domain(dev)
    old_df, old_scores = v73.subset(dev, dev_scores, dev["domain"].eq("old_data").to_numpy())
    third_df, third_scores = v73.subset(dev, dev_scores, dev["domain"].eq("third_batch").to_numpy())
    pseudo = pd.concat(
        [
            run_split("pseudo_third_to_old", third_df, third_df, third_scores, third_df, old_df, old_scores),
            run_split("pseudo_old_to_third", old_df, old_df, old_scores, old_df, third_df, third_scores),
        ],
        ignore_index=True,
    )
    agg = aggregate(pseudo)
    selected = select(agg)
    ext_selected, ext_all = run_external(dev, dev_scores, ext, ext_scores, selected)

    pseudo.to_csv(OUT_DIR / "v79_lowrisk_protector_pseudodomain_all_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "v79_lowrisk_protector_pseudodomain_aggregate.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v79_lowrisk_protector_selected_policies.csv", index=False, encoding="utf-8-sig")
    ext_selected.to_csv(OUT_DIR / "v79_lowrisk_protector_selected_strict_external_eval.csv", index=False, encoding="utf-8-sig")
    ext_all.to_csv(OUT_DIR / "v79_lowrisk_protector_all_strict_external_exploratory.csv", index=False, encoding="utf-8-sig")

    print("Pseudo-domain low-risk protector top 20:")
    print(
        agg[
            [
                "candidate",
                "rate",
                "pseudo_mean_control",
                "pseudo_max_control",
                "pseudo_min_bacc",
                "pseudo_total_fn",
                "pseudo_total_fp",
                "pseudo_total_remaining_error",
                "third_pseudo_specificity",
                "old_pseudo_specificity",
            ]
        ]
        .sort_values(["pseudo_total_fp", "pseudo_total_fn", "pseudo_min_bacc", "pseudo_mean_control"], ascending=[True, True, False, True])
        .head(20)
        .to_string(index=False)
    )
    print("\nSelected strict external:")
    print(
        ext_selected[
            [
                "policy",
                "control_rate",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "fn",
                "fp",
                "remaining_error_n",
                "extra_captured_wrong_n",
            ]
        ].to_string(index=False)
    )
    print("\nStrict external exploratory top 15:")
    print(
        ext_all[
            [
                "candidate",
                "rate",
                "control_rate",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "fn",
                "fp",
                "remaining_error_n",
                "extra_captured_wrong_n",
            ]
        ]
        .sort_values(["balanced_accuracy", "specificity", "control_rate"], ascending=[False, False, True])
        .head(15)
        .to_string(index=False)
    )
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
