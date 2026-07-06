from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v67_devfit_ood_quality_controller_20260527 as v67  # noqa: E402
import run_grosspath_rc_v73_pseudodomain_policy_search_20260527 as v73  # noqa: E402
import run_grosspath_rc_v74_dev_prespecified_quality_gate_20260527 as v74  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v75_joint_quality_risk_ood_search_20260527"
RATES = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30]


def rank01(score: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(score, dtype=float))
    if s.notna().sum() <= 1:
        return np.zeros(len(s), dtype=float)
    return s.rank(method="average", pct=True).fillna(0.0).to_numpy(float)


def image_maha_rank(train_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
    cols = v67.common_features(train_df, target_df).get("image_common5", [])
    if len(cols) < 2:
        return np.zeros(len(target_df), dtype=float)
    _train_score, target_score = v67.mahalanobis_scores(train_df, target_df, cols)
    return rank01(target_score)


def image_iso_rank(train_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
    cols = v67.common_features(train_df, target_df).get("image_common5", [])
    if len(cols) < 2:
        return np.zeros(len(target_df), dtype=float)
    _train_score, target_score = v67.isolation_scores(train_df, target_df, cols)
    return rank01(target_score)


def low_conf_rank(target_df: pd.DataFrame) -> np.ndarray:
    if {"main_margin_abs", "robust_margin_abs"}.issubset(target_df.columns):
        main = pd.to_numeric(target_df["main_margin_abs"], errors="coerce").to_numpy(float)
        robust = pd.to_numeric(target_df["robust_margin_abs"], errors="coerce").to_numpy(float)
        return rank01(1.0 - np.minimum(main, robust))
    return np.zeros(len(target_df), dtype=float)


def joint_scores(train_df: pd.DataFrame, target_df: pd.DataFrame, target_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    q = rank01(v74.quality_proxy_risk(train_df, target_df))
    any_r = rank01(target_scores["any"])
    dir_r = rank01(target_scores["direction"])
    conf_r = low_conf_rank(target_df)
    maha_r = image_maha_rank(train_df, target_df)
    iso_r = image_iso_rank(train_df, target_df)

    return {
        "quality_proxy": q,
        "risk_any": any_r,
        "risk_direction": dir_r,
        "image_maha": maha_r,
        "image_iso": iso_r,
        "quality_plus_direction_mean": (q + dir_r) / 2,
        "quality_plus_any_mean": (q + any_r) / 2,
        "quality_plus_lowconf_mean": (q + conf_r) / 2,
        "quality_plus_image_maha_mean": (q + maha_r) / 2,
        "quality_plus_image_iso_mean": (q + iso_r) / 2,
        "direction_plus_image_maha_mean": (dir_r + maha_r) / 2,
        "quality_direction_image_maha_mean": (q + dir_r + maha_r) / 3,
        "quality_or_direction": np.maximum(q, dir_r),
        "quality_or_image_maha": np.maximum(q, maha_r),
        "quality_and_direction": np.minimum(q, dir_r),
        "quality_and_image_maha": np.minimum(q, maha_r),
        "quality_x_direction": q * dir_r,
        "quality_x_image_maha": q * maha_r,
        "quality_x_direction_x_image": q * dir_r * maha_r,
    }


def run_split(split: str, train_df: pd.DataFrame, target_df: pd.DataFrame, target_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    base = v73.v50_review(target_df, target_scores)
    rows = [v73.evaluate(split, "v50_base", target_df, base)]
    scores = joint_scores(train_df, target_df, target_scores)
    for name, score in scores.items():
        for rate in RATES:
            extra = v73.top_by_rate(score, rate) & (~base)
            review = base | extra
            row = v73.evaluate(split, f"v50_plus_joint_{name}_r{int(rate * 1000):03d}", target_df, review, extra)
            row["candidate"] = name
            row["rate"] = rate
            rows.append(row)
    out = pd.DataFrame(rows)
    out["candidate"] = out["candidate"].fillna("none")
    out["rate"] = out["rate"].fillna(0.0)
    return out


def aggregate(pseudo: pd.DataFrame) -> pd.DataFrame:
    agg = v73.pseudo_aggregate(pseudo)
    # Prefer policies that improve the weaker third-batch pseudo direction without sacrificing old-data safety.
    third = pseudo.loc[pseudo["split"].eq("pseudo_old_to_third"), ["candidate", "rate", "balanced_accuracy", "control_rate", "fn", "fp"]]
    third = third.rename(
        columns={
            "balanced_accuracy": "third_pseudo_bacc",
            "control_rate": "third_pseudo_control",
            "fn": "third_pseudo_fn",
            "fp": "third_pseudo_fp",
        }
    )
    old = pseudo.loc[pseudo["split"].eq("pseudo_third_to_old"), ["candidate", "rate", "balanced_accuracy", "control_rate", "fn", "fp"]]
    old = old.rename(
        columns={
            "balanced_accuracy": "old_pseudo_bacc",
            "control_rate": "old_pseudo_control",
            "fn": "old_pseudo_fn",
            "fp": "old_pseudo_fp",
        }
    )
    return agg.merge(third, on=["candidate", "rate"], how="left").merge(old, on=["candidate", "rate"], how="left")


def select_policies(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = [
        ("best_le078", agg["pseudo_max_control"].le(0.78)),
        ("best_le082", agg["pseudo_max_control"].le(0.82)),
        ("best_le085", agg["pseudo_max_control"].le(0.85)),
        ("best_third_le082", agg["pseudo_max_control"].le(0.82)),
        ("lowest_fn_le085", agg["pseudo_max_control"].le(0.85)),
    ]
    for rule, mask in rules:
        pool = agg.loc[mask].copy()
        if pool.empty:
            continue
        if rule == "best_third_le082":
            pick = pool.sort_values(["third_pseudo_bacc", "pseudo_min_bacc", "pseudo_mean_control"], ascending=[False, False, True]).head(1)
        elif rule == "lowest_fn_le085":
            pick = pool.sort_values(["pseudo_total_fn", "pseudo_total_remaining_error", "pseudo_min_bacc", "pseudo_mean_control"], ascending=[True, True, False, True]).head(1)
        else:
            pick = pool.sort_values(["pseudo_min_bacc", "pseudo_mean_control"], ascending=[False, True]).head(1)
        pick = pick.copy()
        pick.insert(0, "selection_rule", rule)
        rows.append(pick)
    return pd.concat(rows, ignore_index=True).drop_duplicates(["selection_rule", "candidate", "rate"])


def run_external(ext: pd.DataFrame, ext_scores: dict[str, np.ndarray], dev: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = v73.v50_review(ext, ext_scores)
    scores = joint_scores(dev, ext, ext_scores)
    rows = [v73.evaluate("strict_external", "v50_base", ext, base)]
    all_rows = [v73.evaluate("strict_external_exploratory", "v50_base", ext, base)]
    for name, score in scores.items():
        for rate in RATES:
            extra = v73.top_by_rate(score, rate) & (~base)
            review = base | extra
            row = v73.evaluate("strict_external_exploratory", f"v50_plus_joint_{name}_r{int(rate * 1000):03d}", ext, review, extra)
            row["candidate"] = name
            row["rate"] = rate
            all_rows.append(row)
    for _, row in selected.iterrows():
        name = str(row["candidate"])
        rate = float(row["rate"])
        if name == "none":
            extra = np.zeros(len(ext), dtype=bool)
        else:
            extra = v73.top_by_rate(scores[name], rate) & (~base)
        review = base | extra
        policy = f"selected_{row['selection_rule']}__{name}_r{int(rate * 1000):03d}"
        out = v73.evaluate("strict_external", policy, ext, review, extra)
        out["selection_rule"] = row["selection_rule"]
        out["candidate"] = name
        out["rate"] = rate
        rows.append(out)
    return pd.DataFrame(rows), pd.DataFrame(all_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev = v73.add_domain(dev)
    old_df, old_scores = v73.subset(dev, dev_scores, dev["domain"].eq("old_data").to_numpy())
    third_df, third_scores = v73.subset(dev, dev_scores, dev["domain"].eq("third_batch").to_numpy())

    pseudo = pd.concat(
        [
            run_split("pseudo_third_to_old", third_df, old_df, old_scores),
            run_split("pseudo_old_to_third", old_df, third_df, third_scores),
        ],
        ignore_index=True,
    )
    agg = aggregate(pseudo)
    selected = select_policies(agg)
    ext_selected, ext_all = run_external(ext, ext_scores, dev, selected)

    pseudo.to_csv(OUT_DIR / "v75_joint_pseudodomain_all_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "v75_joint_pseudodomain_aggregate.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v75_joint_selected_policies.csv", index=False, encoding="utf-8-sig")
    ext_selected.to_csv(OUT_DIR / "v75_joint_selected_strict_external_eval.csv", index=False, encoding="utf-8-sig")
    ext_all.to_csv(OUT_DIR / "v75_joint_all_strict_external_exploratory.csv", index=False, encoding="utf-8-sig")

    print("Pseudo-domain joint top 18:")
    print(
        agg[
            [
                "candidate",
                "rate",
                "pseudo_mean_control",
                "pseudo_max_control",
                "pseudo_min_bacc",
                "third_pseudo_bacc",
                "old_pseudo_bacc",
                "pseudo_total_fn",
                "pseudo_total_fp",
                "pseudo_total_remaining_error",
            ]
        ]
        .sort_values(["pseudo_min_bacc", "pseudo_mean_control"], ascending=[False, True])
        .head(18)
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
    print("\nStrict external exploratory top 12:")
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
        .sort_values(["balanced_accuracy", "control_rate"], ascending=[False, True])
        .head(12)
        .to_string(index=False)
    )
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
