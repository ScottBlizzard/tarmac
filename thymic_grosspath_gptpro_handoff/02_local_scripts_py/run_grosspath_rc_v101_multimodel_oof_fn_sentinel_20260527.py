from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
UNIFIED_OOF = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_adaptation_runs" / "44_old_third_unified_feature_cv_20260523" / "unified_feature_cv_all_oof_predictions.csv"
SELECTED_UNIFIED = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_adaptation_runs" / "44_old_third_unified_feature_cv_20260523" / "selected_unified_feature_oof_predictions.csv"
SELECTED_DINOV3 = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_adaptation_runs" / "73_old_third_dinov3_convnextb_wpc352_feature_cv_20260523" / "selected_dinov3_feature_oof_predictions.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v101_multimodel_oof_fn_sentinel_20260527"
FIG_DIR = OUT_DIR / "figures"

BASE_POLICY = "adaptive_v50_to_v79_light"
MAX_CONTROL = 0.82
RANDOM_STATE = 20260527


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def normalize_domain(s: pd.Series) -> pd.Series:
    return s.astype(str).replace({"old_data": "old", "third_batch": "third"})


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["domain_key"] = normalize_domain(out["domain"])
    out["final_wrong"] = out["final_correct"].eq(0)
    out["main_robust_gap"] = (out["main_prob"] - out["robust_prob"]).abs()
    return out


def build_multimodel_features() -> pd.DataFrame:
    raw = pd.read_csv(UNIFIED_OOF)
    raw["domain_key"] = normalize_domain(raw["domain"])
    raw["config"] = (
        raw["variant"].astype(str)
        + "__"
        + raw["model"].astype(str)
        + "__"
        + raw["weight_mode"].astype(str)
    )
    pivot = raw.pivot_table(
        index=["domain_key", "original_case_id"],
        columns="config",
        values="oof_prob_high",
        aggfunc="mean",
    )
    prob_cols = list(pivot.columns)
    feats = pd.DataFrame(index=pivot.index)
    values = pivot.to_numpy(float)
    feats["mm_prob_mean"] = np.nanmean(values, axis=1)
    feats["mm_prob_median"] = np.nanmedian(values, axis=1)
    feats["mm_prob_max"] = np.nanmax(values, axis=1)
    feats["mm_prob_p75"] = np.nanquantile(values, 0.75, axis=1)
    feats["mm_vote_ge50"] = np.nanmean(values >= 0.50, axis=1)
    feats["mm_vote_ge60"] = np.nanmean(values >= 0.60, axis=1)
    feats["mm_vote_ge70"] = np.nanmean(values >= 0.70, axis=1)
    feats["mm_prob_std"] = np.nanstd(values, axis=1)
    feats["mm_config_n"] = np.sum(~np.isnan(values), axis=1)

    selected = pd.read_csv(SELECTED_UNIFIED)
    selected["domain_key"] = normalize_domain(selected["domain"])
    sel = selected.set_index(["domain_key", "original_case_id"])["oof_prob_high"].rename("selected_unified_prob")
    feats = feats.join(sel, how="left")

    dinov3 = pd.read_csv(SELECTED_DINOV3)
    dinov3["domain_key"] = normalize_domain(dinov3["domain"])
    dino = dinov3.set_index(["domain_key", "original_case_id"])["oof_prob_high"].rename("selected_dinov3_prob")
    feats = feats.join(dino, how="left")

    feats = feats.reset_index()
    feats.columns = [str(c) for c in feats.columns]
    return feats


def attach_multimodel(base: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    out = base.merge(feats, on=["domain_key", "original_case_id"], how="left", validate="many_to_one")
    out["mm_max_minus_core"] = out["mm_prob_max"] - out["prob_mean_core"]
    out["mm_selected_minus_core"] = out["selected_unified_prob"] - out["prob_mean_core"]
    out["mm_dinov3_minus_core"] = out["selected_dinov3_prob"] - out["prob_mean_core"]
    return out


def metrics(df: pd.DataFrame, review: pd.Series | np.ndarray) -> dict[str, float | int]:
    review = pd.Series(review, index=df.index).astype(bool)
    remaining = (~review) & df["final_wrong"]
    fn = int((remaining & df["label_idx"].eq(1) & df["final_pred"].eq(0)).sum())
    fp = int((remaining & df["label_idx"].eq(0) & df["final_pred"].eq(1)).sum())
    pos = int(df["label_idx"].eq(1).sum())
    neg = int(df["label_idx"].eq(0).sum())
    sens = (pos - fn) / pos if pos else np.nan
    spec = (neg - fp) / neg if neg else np.nan
    return {
        "n": len(df),
        "control_rate": float(review.mean()),
        "remaining_error_n": int(remaining.sum()),
        "fn": fn,
        "fp": fp,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
    }


def extra_review(df: pd.DataFrame, signal: str, threshold: float) -> pd.Series:
    base_review = df["review_or_control"].astype(bool)
    low_auto = (~base_review) & df["final_pred"].eq(0)
    return low_auto & pd.to_numeric(df[signal], errors="coerce").ge(threshold)


def evaluate_rule(df: pd.DataFrame, signal: str, threshold: float) -> dict[str, object]:
    extra = extra_review(df, signal, threshold)
    review = df["review_or_control"].astype(bool) | extra
    row = {
        "signal": signal,
        "threshold": threshold,
        "extra_review_n": int(extra.sum()),
        "extra_captured_error_n": int((extra & df["final_wrong"]).sum()),
    }
    row.update(metrics(df, review))
    return row


def search_rules(train: pd.DataFrame) -> pd.DataFrame:
    grids = {
        "mm_prob_max": np.arange(0.45, 0.901, 0.025),
        "mm_prob_p75": np.arange(0.35, 0.801, 0.025),
        "mm_prob_mean": np.arange(0.30, 0.751, 0.025),
        "mm_vote_ge50": np.arange(0.10, 0.901, 0.05),
        "mm_vote_ge60": np.arange(0.05, 0.801, 0.05),
        "mm_max_minus_core": np.arange(0.05, 0.501, 0.025),
        "selected_unified_prob": np.arange(0.25, 0.751, 0.025),
        "selected_dinov3_prob": np.arange(0.25, 0.751, 0.025),
        "mm_dinov3_minus_core": np.arange(0.05, 0.501, 0.025),
    }
    rows = []
    for signal, thresholds in grids.items():
        if signal not in train.columns:
            continue
        for threshold in thresholds:
            rows.append(evaluate_rule(train, signal, float(round(threshold, 3))))
    return pd.DataFrame(rows)


def select_rule(train: pd.DataFrame) -> dict[str, object]:
    search = search_rules(train)
    feasible = search.loc[search["control_rate"].le(MAX_CONTROL)].copy()
    if feasible.empty:
        feasible = search.copy()
    selected = feasible.sort_values(
        ["fn", "balanced_accuracy", "control_rate"],
        ascending=[True, False, True],
    ).iloc[0]
    return selected.to_dict()


def scope_rows(df: pd.DataFrame, review: pd.Series, workflow: str) -> list[dict[str, object]]:
    rows = [{"workflow": workflow, "scope": "internal_all", **metrics(df, review)}]
    for domain, sub in df.groupby("domain", sort=False):
        rows.append({"workflow": workflow, "scope": domain, **metrics(sub, review.loc[sub.index])})
    return rows


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    for col in ["threshold"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return out


def make_plots(fold_rules: pd.DataFrame, comparison: pd.DataFrame, top_rules: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    counts = fold_rules["rule_label"].value_counts()
    axes[0].bar(counts.index, counts.values, color="#4C78A8")
    axes[0].set_title("Nested selected multimodel rules")
    axes[0].set_ylabel("Fold count")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].grid(axis="y", alpha=0.25)

    focus = comparison.loc[comparison["scope"].isin(["internal_all", "third_batch"])].copy()
    for workflow, sub in focus.groupby("workflow", sort=False):
        axes[1].plot(sub["scope"], sub["balanced_accuracy"] * 100, marker="o", label=workflow)
    axes[1].set_title("Held-out performance")
    axes[1].set_ylabel("BAcc (%)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v101_multimodel_nested_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v101_multimodel_nested_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plot = top_rules.head(30).copy()
    ax.scatter(plot["control_rate"] * 100, plot["balanced_accuracy"] * 100, c=plot["fn"], cmap="Reds_r", s=65)
    ax.set_xlabel("Training-fold control rate (%)")
    ax.set_ylabel("Training-fold BAcc (%)")
    ax.set_title("Top multimodel FN sentinel candidates")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v101_top_candidate_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v101_top_candidate_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(fold_rules: pd.DataFrame, comparison: pd.DataFrame, matched: pd.DataFrame) -> None:
    base_third = comparison.loc[(comparison["workflow"] == "Batch-adaptive main") & (comparison["scope"] == "third_batch")].iloc[0]
    nested_third = comparison.loc[(comparison["workflow"] == "Nested multimodel FN sentinel") & (comparison["scope"] == "third_batch")].iloc[0]
    base_all = comparison.loc[(comparison["workflow"] == "Batch-adaptive main") & (comparison["scope"] == "internal_all")].iloc[0]
    nested_all = comparison.loc[(comparison["workflow"] == "Nested multimodel FN sentinel") & (comparison["scope"] == "internal_all")].iloc[0]
    rule_counts = fold_rules["rule_label"].value_counts().to_dict()
    missing = int(matched["mm_prob_max"].isna().sum())
    lines = [
        "# v101 Multimodel OOF FN Sentinel",
        "",
        "## Key Findings",
        "",
        f"- Matched multimodel OOF features for {len(matched) - missing}/{len(matched)} internal cases.",
        f"- Fold-selected rules: {rule_counts}.",
        f"- Internal all-domain BAcc changes from {pct(base_all['balanced_accuracy'])}, FN={int(base_all['fn'])}, FP={int(base_all['fp'])}, control={pct(base_all['control_rate'])} to {pct(nested_all['balanced_accuracy'])}, FN={int(nested_all['fn'])}, FP={int(nested_all['fp'])}, control={pct(nested_all['control_rate'])}.",
        f"- Third batch BAcc changes from {pct(base_third['balanced_accuracy'])}, FN={int(base_third['fn'])}, FP={int(base_third['fp'])} to {pct(nested_third['balanced_accuracy'])}, FN={int(nested_third['fn'])}, FP={int(nested_third['fp'])}.",
        "",
        "## Boundary",
        "",
        "This is an internal OOF-only analysis. It tests whether a family of older feature heads contains reusable high-risk evidence for cases that the current workflow would auto-pass as low risk. Because the aligned strict-external multimodel OOF feature table is not available, this module should be treated as an internal candidate until the same feature heads are extracted for external validation.",
        "",
    ]
    (OUT_DIR / "v101_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = add_base_features(pd.read_csv(ROUTES))
    base = routes.loc[routes["policy"].eq(BASE_POLICY) & routes["domain"].isin(["old_data", "third_batch"])].copy().reset_index(drop=True)
    feats = build_multimodel_features()
    matched = attach_multimodel(base, feats).reset_index(drop=True)
    matched.to_csv(OUT_DIR / "v101_internal_cases_with_multimodel_features.csv", index=False, encoding="utf-8-sig")

    y_strata = matched["domain"].astype(str) + "_" + matched["label_idx"].astype(str)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    nested_review = pd.Series(False, index=matched.index)
    fold_rows = []
    all_search_rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(matched, y_strata), start=1):
        train = matched.iloc[train_idx]
        test = matched.iloc[test_idx]
        search = search_rules(train)
        search.insert(0, "fold", fold)
        all_search_rows.append(search)
        selected = select_rule(train)
        signal = str(selected["signal"])
        threshold = float(selected["threshold"])
        test_extra = extra_review(test, signal, threshold)
        nested_review.iloc[test_idx] = test["review_or_control"].astype(bool).to_numpy() | test_extra.to_numpy()
        selected["fold"] = fold
        selected["rule_label"] = f"{signal}>={threshold:.3f}"
        fold_rows.append(selected)

    base_review = matched["review_or_control"].astype(bool)
    comparison_rows = []
    comparison_rows.extend(scope_rows(matched, base_review, "Batch-adaptive main"))
    comparison_rows.extend(scope_rows(matched, nested_review, "Nested multimodel FN sentinel"))
    comparison = pd.DataFrame(comparison_rows)
    fold_rules = pd.DataFrame(fold_rows)
    all_search = pd.concat(all_search_rows, ignore_index=True)

    all_search.to_csv(OUT_DIR / "v101_all_fold_rule_search.csv", index=False, encoding="utf-8-sig")
    fold_rules.to_csv(OUT_DIR / "v101_fold_selected_rules.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "v101_nested_internal_comparison.csv", index=False, encoding="utf-8-sig")
    format_table(fold_rules).to_csv(OUT_DIR / "v101_fold_selected_rules_formatted.csv", index=False, encoding="utf-8-sig")
    format_table(comparison).to_csv(OUT_DIR / "v101_nested_internal_comparison_formatted.csv", index=False, encoding="utf-8-sig")

    top_rules = all_search.sort_values(["fn", "balanced_accuracy", "control_rate"], ascending=[True, False, True]).head(100)
    top_rules.to_csv(OUT_DIR / "v101_top100_rule_search.csv", index=False, encoding="utf-8-sig")
    format_table(top_rules).to_csv(OUT_DIR / "v101_top100_rule_search_formatted.csv", index=False, encoding="utf-8-sig")
    make_plots(fold_rules, comparison, top_rules)
    write_summary(fold_rules, comparison, matched)

    print("Wrote", OUT_DIR)
    print(format_table(fold_rules[["fold", "rule_label", "control_rate", "balanced_accuracy", "fn", "fp", "extra_review_n", "extra_captured_error_n"]]).to_string(index=False))
    print()
    print(format_table(comparison).to_string(index=False))


if __name__ == "__main__":
    main()
