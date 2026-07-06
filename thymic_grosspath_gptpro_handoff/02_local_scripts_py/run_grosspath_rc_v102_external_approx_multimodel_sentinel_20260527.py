from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
V101_RULES = ROOT / "outputs" / "grosspath_rc_v101_multimodel_oof_fn_sentinel_20260527" / "v101_fold_selected_rules.csv"
EXT67 = ROOT / "outputs" / "batch1_batch2_task567_20260514" / "task7_external_runs" / "70_locked_536567_fullprob_external_eval_20260523" / "67_old_third_no64_meta_stack_plus_dinov3vitl_ft_20260523_external_predictions.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v102_external_approx_multimodel_sentinel_20260527"
FIG_DIR = OUT_DIR / "figures"

BASE_POLICY = "adaptive_v50_to_v79_light"
REFERENCE_POLICY = "v79_light_lowrisk_guard"


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["final_wrong"] = out["final_correct"].eq(0)
    return out


def build_external_approx_features() -> pd.DataFrame:
    ext = pd.read_csv(EXT67)
    prob_cols = [
        c
        for c in ext.columns
        if (
            c.startswith("oldonly_")
            or c.startswith("adapt_")
            or c in ["selected_base_prob", "dinov3whole_ft_prob", "dinov3vitl_ft_prob", "locked_prob_high"]
        )
    ]
    probs = ext[prob_cols].to_numpy(float)
    out = ext[["case_id", "original_case_id", "task_l6_label", "label_idx"]].copy()
    out["selected_unified_prob"] = ext["selected_base_prob"]
    out["selected_dinov3_prob"] = ext["dinov3vitl_ft_prob"]
    out["mm_prob_mean"] = np.nanmean(probs, axis=1)
    out["mm_prob_median"] = np.nanmedian(probs, axis=1)
    out["mm_prob_max"] = np.nanmax(probs, axis=1)
    out["mm_prob_p75"] = np.nanquantile(probs, 0.75, axis=1)
    out["mm_vote_ge50"] = np.nanmean(probs >= 0.50, axis=1)
    out["mm_vote_ge60"] = np.nanmean(probs >= 0.60, axis=1)
    out["mm_vote_ge70"] = np.nanmean(probs >= 0.70, axis=1)
    out["mm_prob_std"] = np.nanstd(probs, axis=1)
    out["approx_prob_columns"] = len(prob_cols)
    out["approx_feature_source"] = "external_locked67_fullprob_columns"
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
        "auto_n": int((~review).sum()),
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
    if signal not in df.columns:
        return pd.Series(False, index=df.index)
    return low_auto & pd.to_numeric(df[signal], errors="coerce").ge(threshold)


def evaluate_external(matched: pd.DataFrame, rules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    extra_cols = []
    base_review = matched["review_or_control"].astype(bool)
    rows.append({"workflow": "Batch-adaptive main", "rule_label": "base", **metrics(matched, base_review)})
    for _, rule in rules.iterrows():
        signal = str(rule["signal"])
        threshold = float(rule["threshold"])
        extra = extra_review(matched, signal, threshold)
        extra_cols.append(extra.rename(f"fold{int(rule['fold'])}_{signal}_ge_{threshold:.3f}"))
        review = base_review | extra
        row = {
            "workflow": "v102 approx fold-selected sentinel",
            "fold": int(rule["fold"]),
            "signal": signal,
            "threshold": threshold,
            "rule_label": f"{signal}>={threshold:.3f}",
            "extra_review_n": int(extra.sum()),
            "extra_captured_error_n": int((extra & matched["final_wrong"]).sum()),
        }
        row.update(metrics(matched, review))
        rows.append(row)

    if extra_cols:
        extra_mat = pd.concat(extra_cols, axis=1)
        union_extra = extra_mat.any(axis=1)
        vote2_extra = extra_mat.sum(axis=1).ge(2)
        for label, extra in [
            ("v102 approx sentinel union", union_extra),
            ("v102 approx sentinel vote>=2", vote2_extra),
        ]:
            review = base_review | extra
            row = {
                "workflow": label,
                "rule_label": label,
                "extra_review_n": int(extra.sum()),
                "extra_captured_error_n": int((extra & matched["final_wrong"]).sum()),
            }
            row.update(metrics(matched, review))
            rows.append(row)
        matched = pd.concat([matched, extra_mat], axis=1)
        matched["v102_union_extra_review"] = union_extra
        matched["v102_vote2_extra_review"] = vote2_extra
    return pd.DataFrame(rows), matched


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(pct)
    for col in ["threshold"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out


def make_plot(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot = summary.loc[summary["workflow"].ne("v102 approx fold-selected sentinel") | summary["fold"].isna()].copy()
    fold_rows = summary.loc[summary["workflow"].eq("v102 approx fold-selected sentinel")].copy()
    mean_fold = {
        "workflow": "v102 fold mean",
        "balanced_accuracy": fold_rows["balanced_accuracy"].mean(),
        "control_rate": fold_rows["control_rate"].mean(),
        "fn": fold_rows["fn"].mean(),
        "fp": fold_rows["fp"].mean(),
    }
    plot = pd.concat([plot, pd.DataFrame([mean_fold])], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(plot))
    ax.bar(x - 0.18, plot["control_rate"] * 100, width=0.36, label="Control", color="#9E9E9E")
    ax.bar(x + 0.18, plot["balanced_accuracy"] * 100, width=0.36, label="BAcc", color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["workflow"], rotation=25, ha="right")
    ax.set_ylabel("%")
    ax.set_title("Strict external approximate multimodel sentinel")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v102_external_approx_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v102_external_approx_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(summary: pd.DataFrame, matched: pd.DataFrame) -> None:
    base = summary.loc[summary["workflow"].eq("Batch-adaptive main")].iloc[0]
    fold = summary.loc[summary["workflow"].eq("v102 approx fold-selected sentinel")]
    union = summary.loc[summary["workflow"].eq("v102 approx sentinel union")].iloc[0]
    vote2 = summary.loc[summary["workflow"].eq("v102 approx sentinel vote>=2")].iloc[0]
    lines = [
        "# v102 External Approximate Multimodel Sentinel",
        "",
        "## Key Findings",
        "",
        f"- Strict external matched approximate multimodel features for {matched['selected_unified_prob'].notna().sum()}/{len(matched)} cases.",
        f"- Base Batch-adaptive main: BAcc {pct(base['balanced_accuracy'])}, control {pct(base['control_rate'])}, FN={int(base['fn'])}, FP={int(base['fp'])}.",
        f"- Applying the five v101 fold-selected rules with approximate external signals gives mean BAcc {pct(fold['balanced_accuracy'].mean())}, mean control {pct(fold['control_rate'].mean())}, mean FN={fold['fn'].mean():.1f}, mean FP={fold['fp'].mean():.1f}.",
        f"- Union of fold-selected approximate rules: BAcc {pct(union['balanced_accuracy'])}, control {pct(union['control_rate'])}, FN={int(union['fn'])}, FP={int(union['fp'])}.",
        f"- Vote>=2 approximate rule: BAcc {pct(vote2['balanced_accuracy'])}, control {pct(vote2['control_rate'])}, FN={int(vote2['fn'])}, FP={int(vote2['fp'])}.",
        "",
        "## Boundary",
        "",
        "This is not a strict external validation of v101 because the exact internal multimodel OOF feature family is not available for the strict external set. The mapping uses approximate columns from the locked67 external prediction table. It is useful as a compatibility stress test and for deciding whether it is worth extracting the exact same feature heads for the external cohort.",
        "",
    ]
    (OUT_DIR / "v102_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = add_base_features(pd.read_csv(ROUTES))
    base = routes.loc[routes["policy"].eq(BASE_POLICY) & routes["domain"].eq("strict_external")].copy()
    ext_feats = build_external_approx_features()
    matched = base.merge(
        ext_feats,
        on="case_id",
        how="left",
        suffixes=("", "_ext67"),
        validate="many_to_one",
    )
    matched["mm_max_minus_core"] = matched["mm_prob_max"] - matched["prob_mean_core"]
    matched["mm_selected_minus_core"] = matched["selected_unified_prob"] - matched["prob_mean_core"]
    matched["mm_dinov3_minus_core"] = matched["selected_dinov3_prob"] - matched["prob_mean_core"]
    rules = pd.read_csv(V101_RULES)
    summary, matched_aug = evaluate_external(matched, rules)

    matched_aug.to_csv(OUT_DIR / "v102_strict_external_approx_features_and_routes.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "v102_external_approx_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v102_external_approx_summary_formatted.csv", index=False, encoding="utf-8-sig")
    make_plot(summary)
    write_summary(summary, matched_aug)

    print("Wrote", OUT_DIR)
    print(format_table(summary).to_string(index=False))


if __name__ == "__main__":
    main()
