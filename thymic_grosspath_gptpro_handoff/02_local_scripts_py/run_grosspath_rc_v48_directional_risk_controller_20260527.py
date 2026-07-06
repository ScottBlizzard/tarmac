from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v48_directional_risk_controller_20260527"
FIG_DIR = OUT_DIR / "figures"
BUDGET_GRID = np.round(np.arange(0.0, 0.805, 0.025), 3)
TARGETS = [0.90, 0.93, 0.95, 0.97]
QUOTA_GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)


def error_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    y = df["label_idx"].to_numpy(dtype=int)
    p = df["p2_pred"].to_numpy(dtype=int)
    wrong = p != y
    return {
        "any_wrong": wrong,
        "fn_high_to_low": wrong & (y == 1) & (p == 0),
        "fp_low_to_high": wrong & (y == 0) & (p == 1),
        "pred_low": p == 0,
        "pred_high": p == 1,
    }


def generic_oof_external_scores(
    dev: pd.DataFrame,
    ext: pd.DataFrame,
    features: list[str],
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    y = np.asarray(target, dtype=int)
    model = v30.make_models(
        [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in features],
        [c for c in v30.CAT_FEATURES if c in features],
    )["hard_logistic"]
    oof = np.zeros(len(dev), dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260527)
    for tr, te in cv.split(dev[features], y):
        fold_model = clone(model)
        fold_model.fit(dev.iloc[tr][features], y[tr])
        oof[te] = fold_model.predict_proba(dev.iloc[te][features])[:, 1]
    final_model = clone(model)
    final_model.fit(dev[features], y)
    ext_score = final_model.predict_proba(ext[features])[:, 1]
    try:
        dev_auc = float(roc_auc_score(y, oof))
    except ValueError:
        dev_auc = float("nan")
    ext_masks = error_masks(ext)
    ext_target_name = "any_wrong"
    if int(y.sum()) == int(error_masks(dev)["fn_high_to_low"].sum()):
        ext_target_name = "fn_high_to_low"
    elif int(y.sum()) == int(error_masks(dev)["fp_low_to_high"].sum()):
        ext_target_name = "fp_low_to_high"
    try:
        ext_auc = float(roc_auc_score(ext_masks[ext_target_name].astype(int), ext_score))
    except ValueError:
        ext_auc = float("nan")
    return oof, ext_score, dev_auc, ext_auc


def percentile_rank_within_group(score: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros(len(score), dtype=float)
    for group_value in np.unique(groups):
        idx = np.flatnonzero(groups == group_value)
        if len(idx) <= 1:
            out[idx] = 0.0
            continue
        order = idx[np.argsort(score[idx], kind="mergesort")]
        ranks = np.empty(len(idx), dtype=float)
        ranks[np.arange(len(idx))] = np.linspace(0.0, 1.0, len(idx))
        out[order] = ranks
    return out


def top_budget(score: np.ndarray, budget: float) -> np.ndarray:
    return v30.top_budget(np.asarray(score, dtype=float), float(budget))


def top_quota_directional(
    df: pd.DataFrame,
    fn_score: np.ndarray,
    fp_score: np.ndarray,
    budget: float,
    fn_quota: float,
) -> np.ndarray:
    n = len(df)
    k = int(round(n * budget))
    review = np.zeros(n, dtype=bool)
    if k <= 0:
        return review
    p = df["p2_pred"].to_numpy(dtype=int)
    low_idx = np.flatnonzero(p == 0)
    high_idx = np.flatnonzero(p == 1)
    k_fn = int(round(k * fn_quota))
    k_fp = k - k_fn
    k_fn = min(k_fn, len(low_idx))
    k_fp = min(k_fp, len(high_idx))
    spare = k - k_fn - k_fp
    if spare > 0:
        if len(low_idx) - k_fn >= len(high_idx) - k_fp:
            k_fn = min(len(low_idx), k_fn + spare)
        else:
            k_fp = min(len(high_idx), k_fp + spare)
    if k_fn > 0 and len(low_idx):
        low_order = low_idx[np.argsort(-fn_score[low_idx], kind="mergesort")]
        review[low_order[:k_fn]] = True
    if k_fp > 0 and len(high_idx):
        high_order = high_idx[np.argsort(-fp_score[high_idx], kind="mergesort")]
        review[high_order[:k_fp]] = True
    return review


def evaluate_review(df: pd.DataFrame, review: np.ndarray) -> dict[str, float | int]:
    masks = error_masks(df)
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    final = p2.copy()
    final[review] = y[review]
    m = v30.metrics_binary(y, final)
    m.update(
        {
            "review_n": int(review.sum()),
            "review_rate": float(review.mean()),
            "captured_wrong_n": int((review & masks["any_wrong"]).sum()),
            "captured_wrong_rate": float((review & masks["any_wrong"]).sum() / masks["any_wrong"].sum()) if masks["any_wrong"].sum() else 0.0,
            "captured_fn_n": int((review & masks["fn_high_to_low"]).sum()),
            "captured_fn_rate": float((review & masks["fn_high_to_low"]).sum() / masks["fn_high_to_low"].sum()) if masks["fn_high_to_low"].sum() else 0.0,
            "captured_fp_n": int((review & masks["fp_low_to_high"]).sum()),
            "captured_fp_rate": float((review & masks["fp_low_to_high"]).sum() / masks["fp_low_to_high"].sum()) if masks["fp_low_to_high"].sum() else 0.0,
            "review_precision_vs_p2_error": float((review & masks["any_wrong"]).sum() / review.sum()) if review.sum() else 0.0,
        }
    )
    return m


def fixed_budget_tables(dev: pd.DataFrame, ext: pd.DataFrame, scores: dict[str, tuple[np.ndarray, np.ndarray]], fn_scores: tuple[np.ndarray, np.ndarray], fp_scores: tuple[np.ndarray, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed_rows = []
    quota_rows = []
    dev_fn, ext_fn = fn_scores
    dev_fp, ext_fp = fp_scores

    for policy, (dev_score, ext_score) in scores.items():
        for budget in BUDGET_GRID:
            dev_m = evaluate_review(dev, top_budget(dev_score, float(budget)))
            ext_m = evaluate_review(ext, top_budget(ext_score, float(budget)))
            fixed_rows.append({"policy": policy, "split": "development", "budget": float(budget), **dev_m})
            fixed_rows.append({"policy": policy, "split": "external", "budget": float(budget), **ext_m})

    for budget in BUDGET_GRID:
        best = None
        for quota in QUOTA_GRID:
            review = top_quota_directional(dev, dev_fn, dev_fp, float(budget), float(quota))
            m = evaluate_review(dev, review)
            row = {"budget": float(budget), "fn_quota": float(quota), **m}
            if best is None or (row["balanced_accuracy"], row["review_precision_vs_p2_error"]) > (best["balanced_accuracy"], best["review_precision_vs_p2_error"]):
                best = row
        assert best is not None
        ext_review = top_quota_directional(ext, ext_fn, ext_fp, float(budget), float(best["fn_quota"]))
        ext_m = evaluate_review(ext, ext_review)
        quota_rows.append({"split": "development", **best})
        quota_rows.append({"split": "external", "budget": float(budget), "fn_quota": float(best["fn_quota"]), **ext_m})

    return pd.DataFrame(fixed_rows), pd.DataFrame(quota_rows)


def choose_target_rows(fixed: pd.DataFrame, quota: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_fixed = fixed.copy()
    quota_named = quota.copy()
    quota_named["policy"] = "direction_quota_devopt"
    combined = pd.concat([all_fixed, quota_named], ignore_index=True, sort=False)
    for policy, group in combined.groupby("policy"):
        dev = group.loc[group["split"].eq("development")].sort_values("review_rate")
        ext = group.loc[group["split"].eq("external")]
        for target in TARGETS:
            ok = dev.loc[dev["balanced_accuracy"].ge(target)]
            if ok.empty:
                chosen = dev.iloc[-1]
            else:
                chosen = ok.iloc[0]
            matched = ext.loc[np.isclose(ext["budget"].astype(float), float(chosen["budget"]))]
            if "fn_quota" in chosen.index and not pd.isna(chosen.get("fn_quota", np.nan)):
                matched = matched.loc[np.isclose(matched["fn_quota"].astype(float), float(chosen["fn_quota"]))]
            ext_row = matched.iloc[0]
            rows.append(
                {
                    "policy": policy,
                    "target_dev_bacc": float(target),
                    "selected_budget": float(chosen["budget"]),
                    "selected_fn_quota": float(chosen.get("fn_quota", np.nan)) if "fn_quota" in chosen.index else np.nan,
                    **{f"dev_{k}": v for k, v in chosen.items() if k not in {"split", "policy"}},
                    **{f"external_{k}": v for k, v in ext_row.items() if k not in {"split", "policy"}},
                }
            )
    return pd.DataFrame(rows)


def make_plots(fixed: pd.DataFrame, quota: pd.DataFrame, target: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([fixed, quota.assign(policy="direction_quota_devopt")], ignore_index=True, sort=False)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for policy in [
        "any_error_hard_gate",
        "direction_conditional_raw",
        "direction_conditional_group_rank",
        "direction_quota_devopt",
    ]:
        sub = combined.loc[(combined["split"].eq("external")) & (combined["policy"].eq(policy))].sort_values("review_rate")
        if sub.empty:
            continue
        ax.plot(sub["review_rate"] * 100, sub["balanced_accuracy"] * 100, marker="o", linewidth=1.7, label=policy)
    ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(93, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("External review rate (%)")
    ax.set_ylabel("External workflow balanced accuracy (%)")
    ax.set_title("Direction-aware risk controllers")
    ax.set_xlim(-2, 82)
    ax.set_ylim(68, 98)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v48_directional_controller_external_bacc.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v48_directional_controller_external_bacc.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    focus = target.loc[target["target_dev_bacc"].isin([0.95, 0.97])].copy()
    if not focus.empty:
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        for target_value, marker in [(0.95, "o"), (0.97, "s")]:
            sub = focus.loc[focus["target_dev_bacc"].eq(target_value)]
            ax.scatter(sub["external_review_rate"] * 100, sub["external_balanced_accuracy"] * 100, s=72, marker=marker, label=f"dev target {int(target_value*100)}%")
            for _, row in sub.iterrows():
                ax.text(row["external_review_rate"] * 100 + 0.6, row["external_balanced_accuracy"] * 100, row["policy"], fontsize=7)
        ax.axhline(90, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(93, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_xlabel("External review rate (%)")
        ax.set_ylabel("External workflow balanced accuracy (%)")
        ax.set_title("Dev-selected transfer of directional risk controllers")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(-2, 82)
        ax.set_ylim(68, 98)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "v48_directional_controller_dev_target_transfer.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / "v48_directional_controller_dev_target_transfer.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = v30.load_development()
    ext = v30.load_external()
    numeric = [c for c in v30.PROB_FEATURES + v30.IMAGE_FEATURES + v30.BIN_FEATURES if c in dev.columns and c in ext.columns]
    categorical = [c for c in v30.CAT_FEATURES if c in dev.columns and c in ext.columns]
    features = numeric + categorical

    dev_masks = error_masks(dev)
    ext_masks = error_masks(ext)
    any_dev, any_ext, any_dev_auc, any_ext_auc = generic_oof_external_scores(dev, ext, features, dev_masks["any_wrong"])
    fn_dev, fn_ext, fn_dev_auc, fn_ext_auc = generic_oof_external_scores(dev, ext, features, dev_masks["fn_high_to_low"])
    fp_dev, fp_ext, fp_dev_auc, fp_ext_auc = generic_oof_external_scores(dev, ext, features, dev_masks["fp_low_to_high"])

    dev_pred_group = dev["p2_pred"].to_numpy(dtype=int)
    ext_pred_group = ext["p2_pred"].to_numpy(dtype=int)
    cond_dev = np.where(dev_pred_group == 0, fn_dev, fp_dev)
    cond_ext = np.where(ext_pred_group == 0, fn_ext, fp_ext)
    cond_rank_dev = percentile_rank_within_group(cond_dev, dev_pred_group)
    cond_rank_ext = percentile_rank_within_group(cond_ext, ext_pred_group)

    scores = {
        "any_error_hard_gate": (any_dev, any_ext),
        "direction_conditional_raw": (cond_dev, cond_ext),
        "direction_conditional_group_rank": (cond_rank_dev, cond_rank_ext),
        "direction_max_raw": (np.maximum(fn_dev, fp_dev), np.maximum(fn_ext, fp_ext)),
        "fn_ranker_only": (fn_dev, fn_ext),
        "fp_ranker_only": (fp_dev, fp_ext),
    }

    fixed, quota = fixed_budget_tables(dev, ext, scores, (fn_dev, fn_ext), (fp_dev, fp_ext))
    target = choose_target_rows(fixed, quota)
    auc = pd.DataFrame(
        [
            {"target": "any_wrong", "dev_auc": any_dev_auc, "external_auc": any_ext_auc, "dev_positive_n": int(dev_masks["any_wrong"].sum()), "external_positive_n": int(ext_masks["any_wrong"].sum())},
            {"target": "fn_high_to_low", "dev_auc": fn_dev_auc, "external_auc": fn_ext_auc, "dev_positive_n": int(dev_masks["fn_high_to_low"].sum()), "external_positive_n": int(ext_masks["fn_high_to_low"].sum())},
            {"target": "fp_low_to_high", "dev_auc": fp_dev_auc, "external_auc": fp_ext_auc, "dev_positive_n": int(dev_masks["fp_low_to_high"].sum()), "external_positive_n": int(ext_masks["fp_low_to_high"].sum())},
        ]
    )

    fixed.to_csv(OUT_DIR / "v48_fixed_budget_directional_policies.csv", index=False, encoding="utf-8-sig")
    quota.to_csv(OUT_DIR / "v48_direction_quota_devopt_fixed_budget.csv", index=False, encoding="utf-8-sig")
    target.to_csv(OUT_DIR / "v48_dev_selected_directional_policies.csv", index=False, encoding="utf-8-sig")
    auc.to_csv(OUT_DIR / "v48_directional_ranker_auc.csv", index=False, encoding="utf-8-sig")
    make_plots(fixed, quota, target)

    show = target.loc[target["target_dev_bacc"].isin([0.95, 0.97])].copy()
    show = show[
        [
            "policy",
            "target_dev_bacc",
            "selected_budget",
            "selected_fn_quota",
            "external_review_rate",
            "external_balanced_accuracy",
            "external_accuracy",
            "external_fn",
            "external_fp",
            "external_captured_fn_n",
            "external_captured_fp_n",
        ]
    ].sort_values(["target_dev_bacc", "external_balanced_accuracy"], ascending=[True, False])
    print("Directional ranker AUC:")
    print(auc.to_string(index=False))
    print("\nDev-selected transfer:")
    print(show.to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
