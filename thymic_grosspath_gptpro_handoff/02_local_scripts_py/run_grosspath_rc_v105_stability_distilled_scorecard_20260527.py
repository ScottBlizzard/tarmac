from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V101_CASES = ROOT / "outputs" / "grosspath_rc_v101_multimodel_oof_fn_sentinel_20260527" / "v101_internal_cases_with_multimodel_features.csv"
V104_COMP = ROOT / "outputs" / "grosspath_rc_v104_multimodel_ab_filtered_sentinel_20260527" / "v104_nested_internal_comparison.csv"
ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
EXT67 = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_external_runs" / "70_locked_536567_fullprob_external_eval_20260523" / "67_old_third_no64_meta_stack_plus_dinov3vitl_ft_20260523_external_predictions.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v105_stability_distilled_scorecard_20260527"
FIG_DIR = OUT_DIR / "figures"

BASE_POLICY = "adaptive_v50_to_v79_light"

SCORECARDS = [
    {
        "scorecard": "v105_majority_unified025_core045",
        "description": "selected_unified_prob >= 0.25 and prob_mean_core <= 0.45",
        "selected_unified_min": 0.25,
        "selected_dinov3_min": None,
        "core_max": 0.45,
    },
    {
        "scorecard": "v105_strict_unified030_core035",
        "description": "selected_unified_prob >= 0.30 and prob_mean_core <= 0.35",
        "selected_unified_min": 0.30,
        "selected_dinov3_min": None,
        "core_max": 0.35,
    },
    {
        "scorecard": "v105_dual_unified025_dino025_core045",
        "description": "selected_unified_prob >= 0.25 and selected_dinov3_prob >= 0.25 and prob_mean_core <= 0.45",
        "selected_unified_min": 0.25,
        "selected_dinov3_min": 0.25,
        "core_max": 0.45,
    },
]


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def prep_internal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["scope_group"] = out["domain"]
    out["base_final_wrong"] = out["final_correct"].eq(0)
    out["base_review"] = out["review_or_control"].astype(bool)
    out["low_auto"] = (~out["base_review"]) & out["final_pred"].eq(0)
    return out


def build_external_approx() -> pd.DataFrame:
    routes = pd.read_csv(ROUTES)
    base = routes.loc[routes["policy"].eq(BASE_POLICY) & routes["domain"].eq("strict_external")].copy().reset_index(drop=True)
    ext = pd.read_csv(EXT67)
    feats = ext[["case_id", "selected_base_prob", "dinov3vitl_ft_prob"]].copy()
    feats = feats.rename(columns={"selected_base_prob": "selected_unified_prob", "dinov3vitl_ft_prob": "selected_dinov3_prob"})
    out = base.merge(feats, on="case_id", how="left", validate="many_to_one")
    out["scope_group"] = "strict_external_approx"
    out["base_final_wrong"] = out["final_correct"].eq(0)
    out["base_review"] = out["review_or_control"].astype(bool)
    out["low_auto"] = (~out["base_review"]) & out["final_pred"].eq(0)
    out["approx_feature_source"] = "selected_base_prob + dinov3vitl_ft_prob from locked67 external predictions"
    return out


def scorecard_extra(df: pd.DataFrame, card: dict[str, object]) -> pd.Series:
    extra = df["low_auto"].astype(bool)
    extra &= pd.to_numeric(df["selected_unified_prob"], errors="coerce").ge(float(card["selected_unified_min"]))
    if card["selected_dinov3_min"] is not None:
        extra &= pd.to_numeric(df["selected_dinov3_prob"], errors="coerce").ge(float(card["selected_dinov3_min"]))
    extra &= pd.to_numeric(df["prob_mean_core"], errors="coerce").le(float(card["core_max"]))
    return extra


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


def evaluate_scorecards(df: pd.DataFrame, cards: list[dict[str, object]], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    flagged = df.copy()
    for card in cards:
        extra = scorecard_extra(df, card)
        review = df["base_review"] | extra
        col = card["scorecard"] + "_extra_review"
        flagged[col] = extra
        for scope, sub in [("all", df)] + list(df.groupby("scope_group", sort=False)):
            sub_review = review.loc[sub.index]
            row = {
                "setting": prefix,
                "scorecard": card["scorecard"],
                "description": card["description"],
                "scope": scope,
                "extra_review_n": int(extra.loc[sub.index].sum()),
                "extra_captured_error_n": int((extra.loc[sub.index] & sub["base_final_wrong"]).sum()),
                "extra_clean_review_n": int((extra.loc[sub.index] & (~sub["base_final_wrong"])).sum()),
            }
            row.update(metrics(sub, sub_review))
            rows.append(row)
    return pd.DataFrame(rows), flagged


def main_candidate_case_tables(internal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    card = SCORECARDS[0]
    extra = scorecard_extra(internal, card)
    review = internal["base_review"] | extra
    rescued = extra & internal["base_final_wrong"]
    remaining = (~review) & internal["base_final_wrong"]
    clean_extra = extra & (~internal["base_final_wrong"])
    cols = [
        "case_id",
        "original_case_id",
        "domain",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "final_pred",
        "prob_mean_core",
        "selected_unified_prob",
        "selected_dinov3_prob",
        "mm_prob_p75",
        "mm_prob_max",
    ]
    return (
        internal.loc[rescued, cols].sort_values(["domain", "task_l6_label", "original_case_id"]).reset_index(drop=True),
        internal.loc[remaining, cols].sort_values(["domain", "task_l6_label", "original_case_id"]).reset_index(drop=True),
        internal.loc[clean_extra, cols].sort_values(["domain", "task_l6_label", "original_case_id"]).reset_index(drop=True),
    )


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    return out


def make_plot(internal_summary: pd.DataFrame, external_summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    focus = internal_summary.loc[internal_summary["scope"].isin(["all", "third_batch"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for scorecard, sub in focus.groupby("scorecard", sort=False):
        axes[0].plot(sub["scope"], sub["balanced_accuracy"] * 100, marker="o", label=scorecard.replace("v105_", ""))
    axes[0].set_title("Internal scorecard BAcc")
    axes[0].set_ylabel("BAcc (%)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    ext = external_summary.loc[external_summary["scope"].eq("strict_external_approx")].copy()
    x = np.arange(len(ext))
    axes[1].bar(x - 0.18, ext["control_rate"] * 100, width=0.36, label="Control", color="#9E9E9E")
    axes[1].bar(x + 0.18, ext["balanced_accuracy"] * 100, width=0.36, label="BAcc", color="#2E7D32")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ext["scorecard"].str.replace("v105_", "", regex=False), rotation=25, ha="right")
    axes[1].set_title("Strict external approximate application")
    axes[1].set_ylabel("%")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v105_scorecard_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v105_scorecard_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(internal_summary: pd.DataFrame, external_summary: pd.DataFrame, rescued: pd.DataFrame, remaining: pd.DataFrame, clean: pd.DataFrame) -> None:
    main_all = internal_summary.loc[(internal_summary["scorecard"].eq(SCORECARDS[0]["scorecard"])) & (internal_summary["scope"].eq("all"))].iloc[0]
    main_third = internal_summary.loc[(internal_summary["scorecard"].eq(SCORECARDS[0]["scorecard"])) & (internal_summary["scope"].eq("third_batch"))].iloc[0]
    main_ext = external_summary.loc[(external_summary["scorecard"].eq(SCORECARDS[0]["scorecard"])) & (external_summary["scope"].eq("strict_external_approx"))].iloc[0]
    third_rescued = rescued.loc[rescued["domain"].eq("third_batch")]
    third_clean_ab = clean.loc[clean["domain"].eq("third_batch") & clean["task_l6_label"].eq("AB")]
    lines = [
        "# v105 Stability-distilled Multimodel Scorecard",
        "",
        "## Key Findings",
        "",
        f"- Main distilled rule: {SCORECARDS[0]['description']}.",
        f"- Internal all-domain: BAcc {pct(main_all['balanced_accuracy'])}, control {pct(main_all['control_rate'])}, FN={int(main_all['fn'])}, FP={int(main_all['fp'])}.",
        f"- Third batch: BAcc {pct(main_third['balanced_accuracy'])}, control {pct(main_third['control_rate'])}, FN={int(main_third['fn'])}, FP={int(main_third['fp'])}.",
        f"- Rescued third-batch residual errors: {', '.join(third_rescued['original_case_id'].astype(str).tolist())}.",
        f"- Third-batch extra clean AB reviews: {len(third_clean_ab)}.",
        f"- Strict external approximate application: BAcc {pct(main_ext['balanced_accuracy'])}, control {pct(main_ext['control_rate'])}, FN={int(main_ext['fn'])}, FP={int(main_ext['fp'])}.",
        "",
        "## Boundary",
        "",
        "The scorecard is distilled from stable v104 fold-selected rules, so it is simpler and more deployable, but it is still derived from internal OOF analyses. The strict external result uses approximate columns, not the exact v101/v104 feature family.",
        "",
    ]
    (OUT_DIR / "v105_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    internal = prep_internal(pd.read_csv(V101_CASES))
    external = build_external_approx()
    internal_summary, internal_flagged = evaluate_scorecards(internal, SCORECARDS, "internal_oof")
    external_summary, external_flagged = evaluate_scorecards(external, SCORECARDS, "strict_external_approx")
    rescued, remaining, clean = main_candidate_case_tables(internal)

    internal_summary.to_csv(OUT_DIR / "v105_internal_scorecard_summary.csv", index=False, encoding="utf-8-sig")
    external_summary.to_csv(OUT_DIR / "v105_strict_external_approx_scorecard_summary.csv", index=False, encoding="utf-8-sig")
    format_table(internal_summary).to_csv(OUT_DIR / "v105_internal_scorecard_summary_formatted.csv", index=False, encoding="utf-8-sig")
    format_table(external_summary).to_csv(OUT_DIR / "v105_strict_external_approx_scorecard_summary_formatted.csv", index=False, encoding="utf-8-sig")
    internal_flagged.to_csv(OUT_DIR / "v105_internal_cases_with_scorecard_flags.csv", index=False, encoding="utf-8-sig")
    external_flagged.to_csv(OUT_DIR / "v105_strict_external_approx_cases_with_scorecard_flags.csv", index=False, encoding="utf-8-sig")
    rescued.to_csv(OUT_DIR / "v105_main_scorecard_rescued_errors.csv", index=False, encoding="utf-8-sig")
    remaining.to_csv(OUT_DIR / "v105_main_scorecard_remaining_errors.csv", index=False, encoding="utf-8-sig")
    clean.to_csv(OUT_DIR / "v105_main_scorecard_extra_clean_reviews.csv", index=False, encoding="utf-8-sig")
    make_plot(internal_summary, external_summary)
    write_summary(internal_summary, external_summary, rescued, remaining, clean)

    print("Wrote", OUT_DIR)
    print(format_table(internal_summary.loc[internal_summary["scope"].isin(["all", "third_batch"])]).to_string(index=False))
    print()
    print(format_table(external_summary.loc[external_summary["scope"].eq("strict_external_approx")]).to_string(index=False))
    print()
    print("main rescued third")
    print(rescued.loc[rescued["domain"].eq("third_batch")][["original_case_id", "task_l6_label", "prob_mean_core", "selected_unified_prob", "selected_dinov3_prob"]].to_string(index=False))


if __name__ == "__main__":
    main()
