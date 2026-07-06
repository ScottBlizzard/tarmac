from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v73_pseudodomain_policy_search_20260527 as v73  # noqa: E402
import run_grosspath_rc_v74_dev_prespecified_quality_gate_20260527 as v74  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v89_risk_ranker_selective_diagnosis_20260527"
FIG_DIR = OUT_DIR / "figures"
CONTROL_RATES = np.round(np.arange(0.0, 0.91, 0.05), 2)
CAPTURE_RATES = [0.10, 0.20, 0.30, 0.50, 0.75, 0.80]


SCORE_LABELS = {
    "random_expected": "Random expected",
    "oracle_wrong": "Oracle",
    "uncertainty_min_margin": "Confidence uncertainty",
    "ensemble_disagreement": "Ensemble disagreement",
    "risk_any_oof": "OOF any-error risk",
    "risk_direction_oof": "OOF direction risk",
    "quality_proxy": "Quality proxy",
    "quality_plus_lowconf": "Quality + low confidence",
    "quality_plus_direction": "Quality + direction risk",
    "max_any_direction_quality": "Max(any, direction, quality)",
}


def rank01(x: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    if s.notna().sum() <= 1:
        return np.zeros(len(s), dtype=float)
    return s.rank(method="average", pct=True).fillna(0.0).to_numpy(float)


def minmax_clean(x: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float)).replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return np.zeros(len(s), dtype=float)
    return s.fillna(float(s.median())).to_numpy(float)


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source = out["source_folder"] if "source_folder" in out.columns else pd.Series([np.nan] * len(out), index=out.index)
    out["domain"] = np.where(source.notna(), "third_batch", "old_data")
    return out


def prepare_domains() -> dict[str, tuple[pd.DataFrame, dict[str, np.ndarray]]]:
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev = add_domain(dev)
    domains: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray]]] = {}
    for domain in ["old_data", "third_batch"]:
        mask = dev["domain"].eq(domain).to_numpy()
        df, scores = v73.subset(dev, dev_scores, mask)
        domains[domain] = (df, scores)
    domains["strict_external"] = (ext.reset_index(drop=True), {k: np.asarray(v) for k, v in ext_scores.items()})
    domains["development_all"] = (dev.reset_index(drop=True), {k: np.asarray(v) for k, v in dev_scores.items()})
    return domains


def build_scores(reference: pd.DataFrame, df: pd.DataFrame, risk_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    main_margin = minmax_clean(pd.to_numeric(df["main_margin_abs"], errors="coerce").to_numpy(float))
    robust_margin = minmax_clean(pd.to_numeric(df["robust_margin_abs"], errors="coerce").to_numpy(float))
    lowconf = 1.0 - np.minimum(main_margin, robust_margin)
    out["uncertainty_min_margin"] = rank01(lowconf)

    parts = []
    for col in ["core_prob_std", "core_prob_range", "main_robust_abs_diff"]:
        if col in df.columns:
            parts.append(rank01(pd.to_numeric(df[col], errors="coerce").to_numpy(float)))
    if not parts:
        probs = df[["main_prob", "robust_prob", "prob_mean_core"]].apply(pd.to_numeric, errors="coerce")
        parts.append(rank01(probs.std(axis=1).to_numpy(float)))
        parts.append(rank01((probs.max(axis=1) - probs.min(axis=1)).to_numpy(float)))
    disagreement = np.nanmean(np.vstack(parts), axis=0)
    out["ensemble_disagreement"] = rank01(disagreement)

    out["risk_any_oof"] = rank01(risk_scores["any"])
    out["risk_direction_oof"] = rank01(risk_scores["direction"])
    q = v74.quality_proxy_risk(reference, df)
    out["quality_proxy"] = rank01(q)
    out["quality_plus_lowconf"] = rank01((out["quality_proxy"] + out["uncertainty_min_margin"]) / 2)
    out["quality_plus_direction"] = rank01((out["quality_proxy"] + out["risk_direction_oof"]) / 2)
    out["max_any_direction_quality"] = np.maximum.reduce([out["risk_any_oof"], out["risk_direction_oof"], out["quality_proxy"]])
    return out


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def ap_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, score))


def top_mask(score: np.ndarray, rate: float) -> np.ndarray:
    n = len(score)
    k = int(round(n * rate))
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    mask[order[: min(k, n)]] = True
    return mask


def selective_curve(df: pd.DataFrame, score: np.ndarray, score_name: str, domain: str) -> pd.DataFrame:
    y = df["label_idx"].to_numpy(int)
    p = df["p2_pred"].to_numpy(int)
    wrong = p != y
    rows = []
    for rate in CONTROL_RATES:
        review = top_mask(score, float(rate))
        auto = ~review
        auto_n = int(auto.sum())
        remaining_wrong = int((auto & wrong).sum())
        final = p.copy()
        final[review] = y[review]
        tn = int(((y == 0) & (final == 0)).sum())
        fp = int(((y == 0) & (final == 1)).sum())
        fn = int(((y == 1) & (final == 0)).sum())
        tp = int(((y == 1) & (final == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        bacc = (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else np.nan
        rows.append(
            {
                "domain": domain,
                "score": score_name,
                "score_label": SCORE_LABELS.get(score_name, score_name),
                "control_rate": float(rate),
                "auto_coverage": auto_n / len(df),
                "auto_selective_risk": remaining_wrong / auto_n if auto_n else np.nan,
                "remaining_wrong_n": remaining_wrong,
                "captured_wrong_n": int((review & wrong).sum()),
                "captured_fn_n": int((review & (y == 1) & (p == 0)).sum()),
                "captured_fp_n": int((review & (y == 0) & (p == 1)).sum()),
                "final_bacc": bacc,
                "final_sensitivity": sens,
                "final_specificity": spec,
            }
        )
    return pd.DataFrame(rows)


def aurc(curve: pd.DataFrame) -> float:
    sub = curve.loc[curve["auto_coverage"].ge(0.10), ["auto_coverage", "auto_selective_risk"]].dropna().sort_values("auto_coverage")
    if len(sub) < 2:
        return np.nan
    return float(np.trapz(sub["auto_selective_risk"], sub["auto_coverage"]) / (sub["auto_coverage"].max() - sub["auto_coverage"].min()))


def summarize_score(domain: str, df: pd.DataFrame, score_name: str, score: np.ndarray, curve: pd.DataFrame) -> dict[str, object]:
    y = df["label_idx"].to_numpy(int)
    p = df["p2_pred"].to_numpy(int)
    wrong = (p != y).astype(int)
    fn_mask = p == 0
    fp_mask = p == 1
    row = {
        "domain": domain,
        "score": score_name,
        "score_label": SCORE_LABELS.get(score_name, score_name),
        "n": len(df),
        "wrong_n": int(wrong.sum()),
        "wrong_rate": float(wrong.mean()),
        "error_auc": auc_or_nan(wrong, score),
        "error_ap": ap_or_nan(wrong, score),
        "aurc": aurc(curve),
    }
    if fn_mask.sum():
        fn_target = ((y == 1) & (p == 0))[fn_mask].astype(int)
        row["fn_auc_in_pred_low"] = auc_or_nan(fn_target, score[fn_mask])
        row["fn_ap_in_pred_low"] = ap_or_nan(fn_target, score[fn_mask])
    else:
        row["fn_auc_in_pred_low"] = np.nan
        row["fn_ap_in_pred_low"] = np.nan
    if fp_mask.sum():
        fp_target = ((y == 0) & (p == 1))[fp_mask].astype(int)
        row["fp_auc_in_pred_high"] = auc_or_nan(fp_target, score[fp_mask])
        row["fp_ap_in_pred_high"] = ap_or_nan(fp_target, score[fp_mask])
    else:
        row["fp_auc_in_pred_high"] = np.nan
        row["fp_ap_in_pred_high"] = np.nan
    for rate in CAPTURE_RATES:
        review = top_mask(score, rate)
        captured = int((review & wrong.astype(bool)).sum())
        row[f"capture_at_{int(rate * 100)}"] = captured / int(wrong.sum()) if wrong.sum() else np.nan
        row[f"precision_at_{int(rate * 100)}"] = captured / int(review.sum()) if review.sum() else np.nan
        row[f"lift_at_{int(rate * 100)}"] = (captured / int(review.sum())) / wrong.mean() if review.sum() and wrong.mean() else np.nan
    return row


def random_expected_curve(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    y = df["label_idx"].to_numpy(int)
    p = df["p2_pred"].to_numpy(int)
    wrong = p != y
    rows = []
    for rate in CONTROL_RATES:
        n = len(df)
        k = int(round(n * rate))
        auto_n = n - k
        expected_remaining_wrong = int(wrong.sum()) * (auto_n / n) if n else np.nan
        final = p.copy()
        # Expected random review cannot define exact sensitivity/specificity, so keep BAcc blank here.
        rows.append(
            {
                "domain": domain,
                "score": "random_expected",
                "score_label": SCORE_LABELS["random_expected"],
                "control_rate": float(rate),
                "auto_coverage": auto_n / n,
                "auto_selective_risk": expected_remaining_wrong / auto_n if auto_n else np.nan,
                "remaining_wrong_n": expected_remaining_wrong,
                "captured_wrong_n": int(wrong.sum()) - expected_remaining_wrong,
                "captured_fn_n": np.nan,
                "captured_fp_n": np.nan,
                "final_bacc": np.nan,
                "final_sensitivity": np.nan,
                "final_specificity": np.nan,
            }
        )
    return pd.DataFrame(rows)


def random_summary(domain: str, df: pd.DataFrame, curve: pd.DataFrame) -> dict[str, object]:
    wrong = (df["p2_pred"].to_numpy(int) != df["label_idx"].to_numpy(int)).astype(int)
    row = {
        "domain": domain,
        "score": "random_expected",
        "score_label": SCORE_LABELS["random_expected"],
        "n": len(df),
        "wrong_n": int(wrong.sum()),
        "wrong_rate": float(wrong.mean()),
        "error_auc": 0.5 if len(np.unique(wrong)) > 1 else np.nan,
        "error_ap": float(wrong.mean()) if len(np.unique(wrong)) > 1 else np.nan,
        "aurc": aurc(curve),
        "fn_auc_in_pred_low": 0.5,
        "fn_ap_in_pred_low": np.nan,
        "fp_auc_in_pred_high": 0.5,
        "fp_ap_in_pred_high": np.nan,
    }
    for rate in CAPTURE_RATES:
        row[f"capture_at_{int(rate * 100)}"] = rate
        row[f"precision_at_{int(rate * 100)}"] = float(wrong.mean())
        row[f"lift_at_{int(rate * 100)}"] = 1.0
    return row


def run_domain(domain: str, reference: pd.DataFrame, df: pd.DataFrame, scores: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_map = build_scores(reference, df, scores)
    score_map["oracle_wrong"] = (df["p2_pred"].to_numpy(int) != df["label_idx"].to_numpy(int)).astype(float)
    curves = [random_expected_curve(df, domain)]
    summaries = [random_summary(domain, df, curves[0])]
    for name, score in score_map.items():
        curve = selective_curve(df, score, name, domain)
        curves.append(curve)
        summaries.append(summarize_score(domain, df, name, score, curve))
    return pd.DataFrame(summaries), pd.concat(curves, ignore_index=True)


def select_dev_score(summary: pd.DataFrame) -> pd.DataFrame:
    dev = summary.loc[summary["domain"].isin(["old_data", "third_batch"]) & ~summary["score"].isin(["random_expected", "oracle_wrong"])].copy()
    agg = (
        dev.groupby(["score", "score_label"], as_index=False)
        .agg(
            dev_mean_aurc=("aurc", "mean"),
            dev_max_aurc=("aurc", "max"),
            dev_mean_error_ap=("error_ap", "mean"),
            dev_min_error_auc=("error_auc", "min"),
            dev_mean_capture80=("capture_at_80", "mean"),
            dev_min_capture80=("capture_at_80", "min"),
        )
        .sort_values(["dev_mean_aurc", "dev_max_aurc", "dev_mean_error_ap"], ascending=[True, True, False])
    )
    agg["selected_by"] = ""
    if len(agg):
        agg.loc[agg.index[0], "selected_by"] = "min_dev_mean_aurc"
    return agg


def make_plots(summary: pd.DataFrame, curves: pd.DataFrame, selected_score: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    keep_scores = ["random_expected", "uncertainty_min_margin", "risk_any_oof", "risk_direction_oof", selected_score, "oracle_wrong"]
    keep_scores = list(dict.fromkeys([s for s in keep_scores if s in curves["score"].unique()]))
    color_map = {
        "random_expected": "#8c8c8c",
        "uncertainty_min_margin": "#1f77b4",
        "risk_any_oof": "#2ca02c",
        "risk_direction_oof": "#ff7f0e",
        selected_score: "#d62728",
        "oracle_wrong": "#000000",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8), sharey=True)
    for ax, domain in zip(axes, ["old_data", "third_batch", "strict_external"]):
        sub = curves.loc[curves["domain"].eq(domain) & curves["score"].isin(keep_scores)].copy()
        for score_name, g in sub.groupby("score", sort=False):
            g = g.sort_values("auto_coverage", ascending=False)
            ax.plot(
                g["auto_coverage"] * 100,
                g["auto_selective_risk"] * 100,
                label=SCORE_LABELS.get(score_name, score_name),
                linewidth=2.0 if score_name in [selected_score, "oracle_wrong"] else 1.4,
                color=color_map.get(score_name),
                linestyle="--" if score_name == "random_expected" else "-",
            )
        ax.set_title(domain)
        ax.set_xlabel("Auto-passed coverage (%)")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Residual error risk among auto-passed cases (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.11))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v89_selective_risk_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v89_selective_risk_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    plot_df = summary.loc[summary["score"].isin(keep_scores) & summary["domain"].isin(["old_data", "third_batch", "strict_external"])].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(plot_df["score_label"].unique()))
    score_order = keep_scores
    width = 0.24
    for idx, domain in enumerate(["old_data", "third_batch", "strict_external"]):
        vals = []
        labels = []
        for s in score_order:
            row = plot_df.loc[plot_df["domain"].eq(domain) & plot_df["score"].eq(s)]
            if row.empty:
                vals.append(np.nan)
                labels.append(SCORE_LABELS.get(s, s))
            else:
                vals.append(float(row["error_ap"].iloc[0]) * 100)
                labels.append(row["score_label"].iloc[0])
        ax.bar(x + (idx - 1) * width, vals, width=width, label=domain)
    ax.set_xticks(x)
    ax.set_xticklabels([SCORE_LABELS.get(s, s) for s in score_order], rotation=25, ha="right")
    ax.set_ylabel("AP for detecting pure-auto errors (%)")
    ax.set_title("Error-ranking quality across domains")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v89_error_detection_ap_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v89_error_detection_ap_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def write_messages(summary: pd.DataFrame, dev_select: pd.DataFrame, selected_score: str) -> None:
    ext = summary.loc[summary["domain"].eq("strict_external")]
    sel_ext = ext.loc[ext["score"].eq(selected_score)].iloc[0]
    rand_ext = ext.loc[ext["score"].eq("random_expected")].iloc[0]
    oracle_ext = ext.loc[ext["score"].eq("oracle_wrong")].iloc[0]
    sel_label = SCORE_LABELS.get(selected_score, selected_score)
    text = f"""# v89 风险排序与选择性诊断曲线

## 开发域选择

用 old data 和 third batch 两个开发域选择风险排序分数，按平均 AURC 最低作为主选择。当前选中：`{selected_score}`（{sel_label}）。

## 严格外部集表现

- 随机复核的 AP 期望为 {rand_ext['error_ap'] * 100:.2f}%，AURC 为 {rand_ext['aurc'] * 100:.2f}%。
- 开发域选中的 `{selected_score}` 在严格外部集上错误检测 AP 为 {sel_ext['error_ap'] * 100:.2f}%，AUROC 为 {sel_ext['error_auc']:.3f}，AURC 为 {sel_ext['aurc'] * 100:.2f}%。
- 作为理论上限，oracle AURC 为 {oracle_ext['aurc'] * 100:.2f}%。这说明现有风险排序仍有提升空间，但已经明显优于随机复核。
- 该实验的意义是把“复核触发器”从经验阈值提升为可量化的选择性诊断模块：我们能报告风险排序的 AP/AUC、选择性风险曲线和 AURC。

## 写作边界

`oracle_wrong` 只作为理论上限，不能作为可部署方法。严格外部集只用于最终评估，不用于选择风险分数；开发域选择文件见 `v89_dev_score_selection.csv`。
"""
    (OUT_DIR / "v89_key_messages.md").write_text(text, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    domains = prepare_domains()
    reference = domains["development_all"][0]
    summaries = []
    curves = []
    for domain in ["old_data", "third_batch", "strict_external"]:
        df, scores = domains[domain]
        s, c = run_domain(domain, reference, df, scores)
        summaries.append(s)
        curves.append(c)
    summary = pd.concat(summaries, ignore_index=True)
    curve_df = pd.concat(curves, ignore_index=True)
    dev_selection = select_dev_score(summary)
    selected_score = str(dev_selection.iloc[0]["score"])

    summary.to_csv(OUT_DIR / "v89_risk_ranker_summary.csv", index=False, encoding="utf-8-sig")
    curve_df.to_csv(OUT_DIR / "v89_selective_risk_curves.csv", index=False, encoding="utf-8-sig")
    dev_selection.to_csv(OUT_DIR / "v89_dev_score_selection.csv", index=False, encoding="utf-8-sig")
    make_plots(summary, curve_df, selected_score)
    write_messages(summary, dev_selection, selected_score)

    show = summary.loc[summary["score"].isin(["random_expected", selected_score, "risk_any_oof", "risk_direction_oof", "uncertainty_min_margin", "oracle_wrong"])].copy()
    print("Development score selection:")
    print(dev_selection.to_string(index=False))
    print("\nFocused risk-ranker summary:")
    print(
        show[
            [
                "domain",
                "score_label",
                "wrong_n",
                "error_auc",
                "error_ap",
                "aurc",
                "capture_at_20",
                "precision_at_20",
                "capture_at_80",
                "precision_at_80",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
