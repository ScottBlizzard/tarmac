from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v73_pseudodomain_policy_search_20260527 as v73  # noqa: E402
import run_grosspath_rc_v75_joint_quality_risk_ood_search_20260527 as v75  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v76_fp_targeted_overlay_after_v75_20260527"
RATES = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]


def v75_base_review(train_df: pd.DataFrame, target_df: pd.DataFrame, target_scores: dict[str, np.ndarray]) -> np.ndarray:
    base = v73.v50_review(target_df, target_scores)
    scores = v75.joint_scores(train_df, target_df, target_scores)
    extra = v73.top_by_rate(scores["quality_plus_lowconf_mean"], 0.30) & (~base)
    return base | extra


def masked_rank(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = v75.rank01(score)
    out = out.copy()
    out[~mask] = -np.inf
    return out


def fp_scores(train_df: pd.DataFrame, target_df: pd.DataFrame, target_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pred_high = target_df["p2_pred"].to_numpy(int) == 1
    scores = v75.joint_scores(train_df, target_df, target_scores)
    return {
        f"pred_high_{name}": masked_rank(score, pred_high)
        for name, score in scores.items()
        if name
        in [
            "quality_proxy",
            "risk_any",
            "risk_direction",
            "image_maha",
            "image_iso",
            "quality_plus_lowconf_mean",
            "quality_plus_image_maha_mean",
            "quality_or_image_maha",
            "quality_and_image_maha",
            "quality_x_image_maha",
        ]
    }


def run_split(split: str, train_df: pd.DataFrame, target_df: pd.DataFrame, target_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    base = v75_base_review(train_df, target_df, target_scores)
    rows = [v73.evaluate(split, "v75_quality_lowconf_base", target_df, base)]
    scores = fp_scores(train_df, target_df, target_scores)
    for name, score in scores.items():
        for rate in RATES:
            extra = v73.top_by_rate(score, rate) & (~base)
            review = base | extra
            row = v73.evaluate(split, f"v75_plus_{name}_r{int(rate * 1000):03d}", target_df, review, extra)
            row["candidate"] = name
            row["rate"] = rate
            rows.append(row)
    out = pd.DataFrame(rows)
    out["candidate"] = out["candidate"].fillna("none")
    out["rate"] = out["rate"].fillna(0.0)
    return out


def aggregate(pseudo: pd.DataFrame) -> pd.DataFrame:
    agg = v73.pseudo_aggregate(pseudo)
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


def select(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = [
        ("best_le082", agg["pseudo_max_control"].le(0.82)),
        ("best_le085", agg["pseudo_max_control"].le(0.85)),
        ("min_fp_le085", agg["pseudo_max_control"].le(0.85)),
        ("min_total_error_le085", agg["pseudo_max_control"].le(0.85)),
    ]
    for rule, mask in rules:
        pool = agg.loc[mask].copy()
        if pool.empty:
            continue
        if rule == "min_fp_le085":
            pick = pool.sort_values(["pseudo_total_fp", "pseudo_total_fn", "pseudo_min_bacc", "pseudo_mean_control"], ascending=[True, True, False, True]).head(1)
        elif rule == "min_total_error_le085":
            pick = pool.sort_values(["pseudo_total_remaining_error", "pseudo_min_bacc", "pseudo_mean_control"], ascending=[True, False, True]).head(1)
        else:
            pick = pool.sort_values(["pseudo_min_bacc", "pseudo_mean_control"], ascending=[False, True]).head(1)
        pick = pick.copy()
        pick.insert(0, "selection_rule", rule)
        rows.append(pick)
    return pd.concat(rows, ignore_index=True).drop_duplicates(["selection_rule", "candidate", "rate"])


def run_external(dev: pd.DataFrame, ext: pd.DataFrame, ext_scores: dict[str, np.ndarray], selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = v75_base_review(dev, ext, ext_scores)
    scores = fp_scores(dev, ext, ext_scores)
    rows = [v73.evaluate("strict_external", "v75_quality_lowconf_base", ext, base)]
    all_rows = [v73.evaluate("strict_external_exploratory", "v75_quality_lowconf_base", ext, base)]
    for name, score in scores.items():
        for rate in RATES:
            extra = v73.top_by_rate(score, rate) & (~base)
            review = base | extra
            row = v73.evaluate("strict_external_exploratory", f"v75_plus_{name}_r{int(rate * 1000):03d}", ext, review, extra)
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
    selected = select(agg)
    ext_selected, ext_all = run_external(dev, ext, ext_scores, selected)

    pseudo.to_csv(OUT_DIR / "v76_fp_overlay_pseudodomain_all_results.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "v76_fp_overlay_pseudodomain_aggregate.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v76_fp_overlay_selected_policies.csv", index=False, encoding="utf-8-sig")
    ext_selected.to_csv(OUT_DIR / "v76_fp_overlay_selected_strict_external_eval.csv", index=False, encoding="utf-8-sig")
    ext_all.to_csv(OUT_DIR / "v76_fp_overlay_all_strict_external_exploratory.csv", index=False, encoding="utf-8-sig")

    print("Pseudo-domain FP overlay top 15:")
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
        .sort_values(["pseudo_min_bacc", "pseudo_total_fp", "pseudo_mean_control"], ascending=[False, True, True])
        .head(15)
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
