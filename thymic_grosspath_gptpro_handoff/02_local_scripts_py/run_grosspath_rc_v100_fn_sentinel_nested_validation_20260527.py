from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v100_fn_sentinel_nested_validation_20260527"
FIG_DIR = OUT_DIR / "figures"

BASE_POLICY = "adaptive_v50_to_v79_light"
MAX_CONTROL = 0.80
RANDOM_STATE = 20260527


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["final_wrong"] = out["final_correct"].eq(0)
    out["main_robust_gap"] = (out["main_prob"] - out["robust_prob"]).abs()
    return out


def candidate_extra(df: pd.DataFrame, feature: str, threshold: float) -> pd.Series:
    base_review = df["review_or_control"].astype(bool)
    low_auto = (~base_review) & df["final_pred"].eq(0)
    values = {
        "main_prob": df["main_prob"],
        "robust_prob": df["robust_prob"],
        "prob_mean_core": df["prob_mean_core"],
        "main_robust_gap": df["main_robust_gap"],
    }[feature]
    return low_auto & values.ge(threshold)


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


def evaluate_candidate(df: pd.DataFrame, feature: str, threshold: float) -> dict[str, object]:
    extra = candidate_extra(df, feature, threshold)
    review = df["review_or_control"].astype(bool) | extra
    m = metrics(df, review)
    m.update(
        {
            "feature": feature,
            "threshold": threshold,
            "extra_review_n": int(extra.sum()),
            "extra_captured_error_n": int((extra & df["final_wrong"]).sum()),
        }
    )
    return m


def select_rule(train: pd.DataFrame) -> dict[str, object]:
    rows = []
    thresholds = {
        "main_prob": np.arange(0.15, 0.451, 0.025),
        "robust_prob": np.arange(0.15, 0.451, 0.025),
        "prob_mean_core": np.arange(0.15, 0.451, 0.025),
        "main_robust_gap": np.arange(0.05, 0.201, 0.025),
    }
    for feature, values in thresholds.items():
        for threshold in values:
            rows.append(evaluate_candidate(train, feature, float(round(threshold, 3))))
    search = pd.DataFrame(rows)
    feasible = search.loc[search["control_rate"].le(MAX_CONTROL)].copy()
    if feasible.empty:
        feasible = search.copy()
    # High-risk safety first, then overall BAcc, then lower control.
    selected = feasible.sort_values(
        ["fn", "balanced_accuracy", "control_rate"],
        ascending=[True, False, True],
    ).iloc[0]
    return selected.to_dict()


def scope_rows(df: pd.DataFrame, review: pd.Series, workflow: str) -> list[dict[str, object]]:
    rows = []
    rows.append({"workflow": workflow, "scope": "internal_all", **metrics(df, review)})
    for domain, sub in df.groupby("domain", sort=False):
        rows.append({"workflow": workflow, "scope": domain, **metrics(sub, review.loc[sub.index])})
    return rows


def make_plots(fold_rules: pd.DataFrame, comparison: pd.DataFrame, strict: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    labels = fold_rules["rule_label"].value_counts()
    axes[0].bar(labels.index, labels.values, color="#4C78A8")
    axes[0].set_title("Selected rule stability")
    axes[0].set_ylabel("Fold count")
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[0].grid(axis="y", alpha=0.25)

    focus = comparison.loc[comparison["scope"].isin(["internal_all", "third_batch"])].copy()
    for workflow, sub in focus.groupby("workflow", sort=False):
        axes[1].plot(sub["scope"], sub["balanced_accuracy"] * 100, marker="o", label=workflow)
    axes[1].set_ylabel("BAcc (%)")
    axes[1].set_title("Nested internal held-out performance")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v100_nested_rule_stability.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v100_nested_rule_stability.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(strict))
    ax.bar(x - 0.18, strict["control_rate"] * 100, width=0.36, label="control", color="#9E9E9E")
    ax.bar(x + 0.18, strict["balanced_accuracy"] * 100, width=0.36, label="BAcc", color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {i}" for i in strict["fold"]])
    ax.set_ylabel("%")
    ax.set_title("Strict external performance of fold-selected rules")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v100_strict_external_fold_selected_rules.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v100_strict_external_fold_selected_rules.pdf", bbox_inches="tight")
    plt.close(fig)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    for col in ["threshold"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.3f}")
    return out


def write_summary(fold_rules: pd.DataFrame, comparison: pd.DataFrame, strict: pd.DataFrame) -> None:
    nested = comparison.loc[(comparison["workflow"] == "Nested FN sentinel") & (comparison["scope"] == "third_batch")].iloc[0]
    base = comparison.loc[(comparison["workflow"] == "Batch-adaptive main") & (comparison["scope"] == "third_batch")].iloc[0]
    strict_mean = strict[["control_rate", "balanced_accuracy", "fn", "fp"]].mean(numeric_only=True)
    top_rules = fold_rules["rule_label"].value_counts().to_dict()
    lines = [
        "# v100 FN Sentinel Nested Validation",
        "",
        "## Key Findings",
        "",
        f"- Fold-selected rules: {top_rules}.",
        f"- Third-batch held-out performance improves from BAcc {pct(base['balanced_accuracy'])}, FN={int(base['fn'])}, FP={int(base['fp'])} to BAcc {pct(nested['balanced_accuracy'])}, FN={int(nested['fn'])}, FP={int(nested['fp'])}.",
        f"- Applying the fold-selected rules to strict external gives mean control {pct(strict_mean['control_rate'])}, mean BAcc {pct(strict_mean['balanced_accuracy'])}, mean FN={strict_mean['fn']:.1f}, mean FP={strict_mean['fp']:.1f}.",
        "",
        "## Boundary",
        "",
        "This is still an internal nested analysis based on existing OOF predictions, not a prospective lock. It is stronger than a single post-hoc rule search because rule selection happens inside internal folds and strict external labels are not used for selection.",
        "",
    ]
    (OUT_DIR / "v100_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = add_features(pd.read_csv(ROUTES))
    base = routes.loc[routes["policy"].eq(BASE_POLICY)].copy().reset_index(drop=True)
    internal = base.loc[base["domain"].isin(["old_data", "third_batch"])].copy().reset_index(drop=True)
    strict = base.loc[base["domain"].eq("strict_external")].copy().reset_index(drop=True)

    y_strata = internal["domain"].astype(str) + "_" + internal["label_idx"].astype(str)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    nested_review = pd.Series(False, index=internal.index)
    fold_rows = []
    strict_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(internal, y_strata), start=1):
        train = internal.iloc[train_idx]
        test = internal.iloc[test_idx]
        selected = select_rule(train)
        feature = str(selected["feature"])
        threshold = float(selected["threshold"])
        test_extra = candidate_extra(test, feature, threshold)
        nested_review.iloc[test_idx] = test["review_or_control"].astype(bool).to_numpy() | test_extra.to_numpy()

        strict_extra = candidate_extra(strict, feature, threshold)
        strict_review = strict["review_or_control"].astype(bool) | strict_extra
        strict_m = metrics(strict, strict_review)
        strict_rows.append({"fold": fold, "feature": feature, "threshold": threshold, "rule_label": f"{feature}>={threshold:.3f}", **strict_m})
        fold_rows.append({"fold": fold, "feature": feature, "threshold": threshold, "rule_label": f"{feature}>={threshold:.3f}", **selected})

    base_review = internal["review_or_control"].astype(bool)
    comparison_rows = []
    comparison_rows.extend(scope_rows(internal, base_review, "Batch-adaptive main"))
    comparison_rows.extend(scope_rows(internal, nested_review, "Nested FN sentinel"))
    comparison = pd.DataFrame(comparison_rows)
    fold_rules = pd.DataFrame(fold_rows)
    strict_eval = pd.DataFrame(strict_rows)

    fold_rules.to_csv(OUT_DIR / "v100_fold_selected_rules.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "v100_nested_internal_comparison.csv", index=False, encoding="utf-8-sig")
    strict_eval.to_csv(OUT_DIR / "v100_strict_external_fold_selected_rules.csv", index=False, encoding="utf-8-sig")
    format_table(fold_rules).to_csv(OUT_DIR / "v100_fold_selected_rules_formatted.csv", index=False, encoding="utf-8-sig")
    format_table(comparison).to_csv(OUT_DIR / "v100_nested_internal_comparison_formatted.csv", index=False, encoding="utf-8-sig")
    format_table(strict_eval).to_csv(OUT_DIR / "v100_strict_external_fold_selected_rules_formatted.csv", index=False, encoding="utf-8-sig")
    make_plots(fold_rules, comparison, strict_eval)
    write_summary(fold_rules, comparison, strict_eval)

    print("Wrote", OUT_DIR)
    print(format_table(fold_rules[["fold", "rule_label", "control_rate", "balanced_accuracy", "fn", "fp"]]).to_string(index=False))
    print()
    print(format_table(comparison).to_string(index=False))
    print()
    print(format_table(strict_eval[["fold", "rule_label", "control_rate", "balanced_accuracy", "fn", "fp"]]).to_string(index=False))


if __name__ == "__main__":
    main()
