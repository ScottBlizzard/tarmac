from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
V101_CASES = ROOT / "outputs" / "grosspath_rc_v101_multimodel_oof_fn_sentinel_20260527" / "v101_internal_cases_with_multimodel_features.csv"
V101_COMP = ROOT / "outputs" / "grosspath_rc_v101_multimodel_oof_fn_sentinel_20260527" / "v101_nested_internal_comparison.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v104_multimodel_ab_filtered_sentinel_20260527"
FIG_DIR = OUT_DIR / "figures"
RANDOM_STATE = 20260527
MAX_CONTROL = 0.80


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["base_final_wrong"] = out["final_correct"].eq(0)
    out["base_review"] = out["review_or_control"].astype(bool)
    out["low_auto"] = (~out["base_review"]) & out["final_pred"].eq(0)
    out["base_fn"] = out["base_final_wrong"] & out["label_idx"].eq(1) & out["final_pred"].eq(0)
    out["base_fp"] = out["base_final_wrong"] & out["label_idx"].eq(0) & out["final_pred"].eq(1)
    return out


def metrics(df: pd.DataFrame, review: pd.Series | np.ndarray) -> dict[str, float | int]:
    review = pd.Series(review, index=df.index).astype(bool)
    rem = (~review) & df["base_final_wrong"]
    fn = int((rem & df["label_idx"].eq(1) & df["final_pred"].eq(0)).sum())
    fp = int((rem & df["label_idx"].eq(0) & df["final_pred"].eq(1)).sum())
    pos = int(df["label_idx"].eq(1).sum())
    neg = int(df["label_idx"].eq(0).sum())
    sens = (pos - fn) / pos if pos else np.nan
    spec = (neg - fp) / neg if neg else np.nan
    return {
        "n": len(df),
        "control_rate": float(review.mean()),
        "remaining_error_n": int(rem.sum()),
        "fn": fn,
        "fp": fp,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
    }


def rule_mask(df: pd.DataFrame, rule: dict[str, object]) -> pd.Series:
    kind = str(rule["kind"])
    low = df["low_auto"]
    if kind == "single":
        return low & pd.to_numeric(df[str(rule["signal"])], errors="coerce").ge(float(rule["threshold"]))
    if kind == "corele":
        return (
            low
            & pd.to_numeric(df[str(rule["signal"])], errors="coerce").ge(float(rule["threshold"]))
            & pd.to_numeric(df["prob_mean_core"], errors="coerce").le(float(rule["filter_value"]))
        )
    if kind == "diffge":
        return (
            low
            & pd.to_numeric(df[str(rule["signal"])], errors="coerce").ge(float(rule["threshold"]))
            & pd.to_numeric(df[str(rule["filter_signal"])], errors="coerce").ge(float(rule["filter_value"]))
        )
    if kind == "unified_dino":
        return (
            low
            & pd.to_numeric(df["selected_unified_prob"], errors="coerce").ge(float(rule["threshold"]))
            & pd.to_numeric(df["selected_dinov3_prob"], errors="coerce").ge(float(rule["filter_value"]))
        )
    raise ValueError(kind)


def candidate_rules() -> list[dict[str, object]]:
    rules: list[dict[str, object]] = []
    for signal, thresholds in {
        "selected_unified_prob": np.arange(0.25, 0.701, 0.05),
        "selected_dinov3_prob": np.arange(0.25, 0.701, 0.05),
        "mm_prob_p75": np.arange(0.35, 0.801, 0.05),
        "mm_prob_max": np.arange(0.45, 0.901, 0.05),
    }.items():
        for threshold in thresholds:
            rules.append({"kind": "single", "signal": signal, "threshold": round(float(threshold), 3), "filter_signal": "", "filter_value": np.nan})
            for coremax in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
                rules.append({"kind": "corele", "signal": signal, "threshold": round(float(threshold), 3), "filter_signal": "prob_mean_core", "filter_value": coremax})
            diff_signal = {
                "selected_unified_prob": "mm_selected_minus_core",
                "selected_dinov3_prob": "mm_dinov3_minus_core",
                "mm_prob_p75": "mm_max_minus_core",
                "mm_prob_max": "mm_max_minus_core",
            }[signal]
            for delta in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
                rules.append({"kind": "diffge", "signal": signal, "threshold": round(float(threshold), 3), "filter_signal": diff_signal, "filter_value": delta})
    for unified_t in np.arange(0.25, 0.701, 0.05):
        for dino_t in np.arange(0.20, 0.701, 0.05):
            rules.append({"kind": "unified_dino", "signal": "selected_unified_prob", "threshold": round(float(unified_t), 3), "filter_signal": "selected_dinov3_prob", "filter_value": round(float(dino_t), 3)})
    return rules


def evaluate_rule(df: pd.DataFrame, rule: dict[str, object]) -> dict[str, object]:
    extra = rule_mask(df, rule)
    review = df["base_review"] | extra
    row = dict(rule)
    row["rule_label"] = label_rule(rule)
    row["extra_review_n"] = int(extra.sum())
    row["extra_captured_error_n"] = int((extra & df["base_final_wrong"]).sum())
    row["extra_clean_review_n"] = int((extra & (~df["base_final_wrong"])).sum())
    row.update(metrics(df, review))
    return row


def label_rule(rule: dict[str, object]) -> str:
    kind = str(rule["kind"])
    if kind == "single":
        return f"{rule['signal']}>={float(rule['threshold']):.3f}"
    if kind == "corele":
        return f"{rule['signal']}>={float(rule['threshold']):.3f} & core<={float(rule['filter_value']):.2f}"
    if kind == "diffge":
        return f"{rule['signal']}>={float(rule['threshold']):.3f} & {rule['filter_signal']}>={float(rule['filter_value']):.2f}"
    if kind == "unified_dino":
        return f"unified>={float(rule['threshold']):.3f} & dino>={float(rule['filter_value']):.2f}"
    return kind


def select_rule(train: pd.DataFrame) -> dict[str, object]:
    rows = [evaluate_rule(train, rule) for rule in candidate_rules()]
    search = pd.DataFrame(rows)
    feasible = search.loc[search["control_rate"].le(MAX_CONTROL)].copy()
    if feasible.empty:
        feasible = search.copy()
    selected = feasible.sort_values(
        ["fn", "balanced_accuracy", "extra_clean_review_n", "control_rate"],
        ascending=[True, False, True, True],
    ).iloc[0]
    return selected.to_dict(), search


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
    for col in ["threshold", "filter_value"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    return out


def make_plot(fold_rules: pd.DataFrame, comparison: pd.DataFrame, cases: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    counts = fold_rules["rule_label"].value_counts()
    axes[0].bar(counts.index, counts.values, color="#4C78A8")
    axes[0].set_title("v104 selected filtered rules")
    axes[0].set_ylabel("Fold count")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].grid(axis="y", alpha=0.25)

    focus = comparison.loc[comparison["scope"].isin(["internal_all", "third_batch"])].copy()
    for workflow, sub in focus.groupby("workflow", sort=False):
        axes[1].plot(sub["scope"], sub["balanced_accuracy"] * 100, marker="o", label=workflow)
    axes[1].set_title("Held-out BAcc")
    axes[1].set_ylabel("BAcc (%)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v104_ab_filtered_nested_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v104_ab_filtered_nested_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    third = cases.loc[cases["domain"].eq("third_batch")].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = np.select(
        [third["v104_rescued_error"], third["v104_remaining_wrong"], third["v104_extra_clean_review"]],
        ["#2E7D32", "#C62828", "#F9A825"],
        default="#B0BEC5",
    )
    ax.scatter(third["prob_mean_core"], third["selected_unified_prob"], c=colors, s=68, edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="#666666", linestyle="--", linewidth=1)
    ax.set_xlabel("Current core high-risk probability")
    ax.set_ylabel("Selected unified high-risk probability")
    ax.set_title("v104 third-batch mechanism")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v104_third_batch_mechanism.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v104_third_batch_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(fold_rules: pd.DataFrame, comparison: pd.DataFrame, cases: pd.DataFrame) -> None:
    base_third = comparison.loc[(comparison["workflow"] == "Batch-adaptive main") & (comparison["scope"] == "third_batch")].iloc[0]
    v101_third = comparison.loc[(comparison["workflow"] == "v101 multimodel sentinel") & (comparison["scope"] == "third_batch")].iloc[0]
    v104_third = comparison.loc[(comparison["workflow"] == "v104 AB-filtered multimodel sentinel") & (comparison["scope"] == "third_batch")].iloc[0]
    v104_all = comparison.loc[(comparison["workflow"] == "v104 AB-filtered multimodel sentinel") & (comparison["scope"] == "internal_all")].iloc[0]
    third = cases.loc[cases["domain"].eq("third_batch")]
    rescued = third.loc[third["v104_rescued_error"], "original_case_id"].astype(str).tolist()
    extra_clean_ab = int((third["v104_extra_clean_review"] & third["task_l6_label"].eq("AB")).sum())
    lines = [
        "# v104 Multimodel Sentinel with AB-clean Review Filter",
        "",
        "## Key Findings",
        "",
        f"- v104 selected filtered rules across folds: {fold_rules['rule_label'].value_counts().to_dict()}.",
        f"- Third batch baseline: BAcc {pct(base_third['balanced_accuracy'])}, FN={int(base_third['fn'])}, FP={int(base_third['fp'])}, control {pct(base_third['control_rate'])}.",
        f"- v101 multimodel sentinel: BAcc {pct(v101_third['balanced_accuracy'])}, FN={int(v101_third['fn'])}, FP={int(v101_third['fp'])}, control {pct(v101_third['control_rate'])}.",
        f"- v104 filtered sentinel: BAcc {pct(v104_third['balanced_accuracy'])}, FN={int(v104_third['fn'])}, FP={int(v104_third['fp'])}, control {pct(v104_third['control_rate'])}; internal all-domain BAcc {pct(v104_all['balanced_accuracy'])}, control {pct(v104_all['control_rate'])}.",
        f"- v104 rescued third-batch residual error IDs: {', '.join(rescued)}.",
        f"- v104 third-batch extra clean AB reviews: {extra_clean_ab}.",
        "",
        "## Interpretation",
        "",
        "The filtered search tests whether the v101 high-risk sentinel can keep TC/B2 rescue while reducing clean AB review burden. It is still an internal nested analysis and should be treated as a candidate refinement rather than a locked workflow.",
        "",
    ]
    (OUT_DIR / "v104_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = prep(pd.read_csv(V101_CASES))
    v101_comp = pd.read_csv(V101_COMP)

    y_strata = cases["domain"].astype(str) + "_" + cases["label_idx"].astype(str)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    v104_extra = pd.Series(False, index=cases.index)
    fold_rows = []
    searches = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(cases, y_strata), start=1):
        train = cases.iloc[train_idx]
        test = cases.iloc[test_idx]
        selected, search = select_rule(train)
        search.insert(0, "fold", fold)
        searches.append(search)
        extra = rule_mask(test, selected)
        v104_extra.iloc[test_idx] = extra.to_numpy()
        selected["fold"] = fold
        fold_rows.append(selected)

    v104_review = cases["base_review"] | v104_extra
    cases["v104_extra_review"] = v104_extra
    cases["v104_review_or_control"] = v104_review
    cases["v104_remaining_wrong"] = (~v104_review) & cases["base_final_wrong"]
    cases["v104_rescued_error"] = v104_extra & cases["base_final_wrong"]
    cases["v104_extra_clean_review"] = v104_extra & (~cases["base_final_wrong"])

    comparison_rows = []
    comparison_rows.extend(scope_rows(cases, cases["base_review"], "Batch-adaptive main"))
    v101_nested = v101_comp.loc[v101_comp["workflow"].eq("Nested multimodel FN sentinel")].copy()
    for _, row in v101_nested.iterrows():
        comparison_rows.append(row.to_dict() | {"workflow": "v101 multimodel sentinel"})
    comparison_rows.extend(scope_rows(cases, v104_review, "v104 AB-filtered multimodel sentinel"))
    comparison = pd.DataFrame(comparison_rows)
    fold_rules = pd.DataFrame(fold_rows)
    all_search = pd.concat(searches, ignore_index=True)

    fold_rules.to_csv(OUT_DIR / "v104_fold_selected_rules.csv", index=False, encoding="utf-8-sig")
    all_search.to_csv(OUT_DIR / "v104_all_fold_rule_search.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "v104_nested_internal_comparison.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v104_internal_cases_with_flags.csv", index=False, encoding="utf-8-sig")
    format_table(fold_rules).to_csv(OUT_DIR / "v104_fold_selected_rules_formatted.csv", index=False, encoding="utf-8-sig")
    format_table(comparison).to_csv(OUT_DIR / "v104_nested_internal_comparison_formatted.csv", index=False, encoding="utf-8-sig")
    make_plot(fold_rules, comparison, cases)
    write_summary(fold_rules, comparison, cases)

    print("Wrote", OUT_DIR)
    print(format_table(fold_rules[["fold", "rule_label", "control_rate", "balanced_accuracy", "fn", "fp", "extra_review_n", "extra_clean_review_n"]]).to_string(index=False))
    print()
    print(format_table(comparison.loc[comparison["scope"].isin(["internal_all", "third_batch"])]).to_string(index=False))
    print()
    third = cases.loc[cases["domain"].eq("third_batch")]
    print("third rescued", third.loc[third["v104_rescued_error"], ["original_case_id", "task_l6_label", "prob_mean_core", "selected_unified_prob", "selected_dinov3_prob"]].to_string(index=False))
    print("third extra clean by label")
    print(third.loc[third["v104_extra_clean_review"]].groupby("task_l6_label").size().to_string())


if __name__ == "__main__":
    main()
