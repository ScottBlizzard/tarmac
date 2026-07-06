from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v73_pseudodomain_policy_search_20260527 as v73  # noqa: E402
import run_grosspath_rc_v89_risk_ranker_selective_diagnosis_20260527 as v89  # noqa: E402


V80_ROUTES = ROOT / "outputs" / "grosspath_rc_v80_tiered_lowrisk_guard_summary_20260527" / "v80_tiered_lowrisk_guard_case_routes.csv"
V82_DECISIONS = ROOT / "outputs" / "grosspath_rc_v82_unlabeled_adaptive_workflow_20260527" / "v82_unlabeled_batch_policy_decisions.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527"
FIG_DIR = OUT_DIR / "figures"


POLICY_LABELS = {
    "pure_auto": "Pure auto",
    "v50_main": "Fixed v50",
    "v79_light_lowrisk_guard": "Fixed v79-light",
    "v79_strict_lowrisk_guard": "Fixed v79-strict",
    "adaptive_v50_to_v79_light": "Batch-adaptive main",
    "adaptive_v50_to_v79_strict": "Batch-adaptive strict",
    "risk_any_shiftaware_target10": "Risk-rank target10 shift-aware",
    "risk_direction_uniform80": "Risk-direction uniform80",
    "quality_direction_uniform90": "Quality+direction uniform90",
}

POLICY_ORDER = [
    "pure_auto",
    "v50_main",
    "v79_light_lowrisk_guard",
    "v79_strict_lowrisk_guard",
    "adaptive_v50_to_v79_light",
    "adaptive_v50_to_v79_strict",
    "risk_any_shiftaware_target10",
    "risk_direction_uniform80",
    "quality_direction_uniform90",
]

DOMAIN_ORDER = ["old_data", "third_batch", "strict_external"]


def add_domain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source = out["source_folder"] if "source_folder" in out.columns else pd.Series([np.nan] * len(out), index=out.index)
    out["domain"] = np.where(source.notna(), "third_batch", "old_data")
    return out


def top_mask(score: np.ndarray, rate: float) -> np.ndarray:
    n = len(score)
    k = int(round(n * rate))
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    mask[order[: min(k, n)]] = True
    return mask


def metrics(df: pd.DataFrame) -> dict[str, object]:
    y = df["label_idx"].to_numpy(int)
    pred = df["final_pred"].to_numpy(int)
    control = df["review_or_control"].to_numpy(int).astype(bool)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    wrong = pred != y
    return {
        "n": int(len(df)),
        "control_n": int(control.sum()),
        "control_rate": float(control.mean()),
        "auto_n": int((~control).sum()),
        "auto_rate": float((~control).mean()),
        "accuracy": float((~wrong).mean()),
        "balanced_accuracy": float((sens + spec) / 2),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "remaining_error_n": int(wrong.sum()),
        "remaining_auto_risk": float(wrong[~control].mean()) if (~control).sum() else 0.0,
    }


def route_from_df(domain: str, policy: str, df: pd.DataFrame, review: np.ndarray, source_policy: str, shift_category: str = "") -> pd.DataFrame:
    y = df["label_idx"].to_numpy(int)
    p2 = df["p2_pred"].to_numpy(int)
    final = p2.copy()
    final[review] = y[review]
    cols = [
        c
        for c in [
            "case_id",
            "original_case_id",
            "task_l6_label",
            "task_l7_label",
            "source_folder",
            "view_type_final",
            "image_name",
            "main_prob",
            "robust_prob",
            "prob_mean_core",
            "core_agree_count",
            "quality_score",
            "quality_status",
        ]
        if c in df.columns
    ]
    out = df[cols].copy()
    out.insert(0, "domain", domain)
    out.insert(1, "policy", policy)
    out["source_policy"] = source_policy
    out["shift_category"] = shift_category
    out["review_or_control"] = review.astype(int)
    out["label_idx"] = y
    out["p2_pred"] = p2
    out["final_pred"] = final
    out["p2_wrong"] = (p2 != y).astype(int)
    out["final_correct"] = (final == y).astype(int)
    out["error_direction"] = np.select(
        [(y == 1) & (p2 == 0), (y == 0) & (p2 == 1)],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )
    return out


def fixed_routes() -> pd.DataFrame:
    routes = pd.read_csv(V80_ROUTES)
    keep = routes.loc[routes["policy"].isin(["v50_main", "v79_light_lowrisk_guard", "v79_strict_lowrisk_guard"])].copy()
    keep["source_policy"] = keep["policy"]
    keep["shift_category"] = ""
    return keep


def pure_and_risk_routes() -> pd.DataFrame:
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev = add_domain(dev)
    reference = dev.reset_index(drop=True)
    out_parts = []
    for domain in DOMAIN_ORDER:
        if domain == "strict_external":
            df = ext.reset_index(drop=True)
            scores = {k: np.asarray(v) for k, v in ext_scores.items()}
        else:
            mask = dev["domain"].eq(domain).to_numpy()
            df, scores = v73.subset(dev, dev_scores, mask)
        score_map = v89.build_scores(reference, df, scores)
        out_parts.append(route_from_df(domain, "pure_auto", df, np.zeros(len(df), dtype=bool), "pure_auto"))

        any_rate = 0.85 if domain == "strict_external" else 0.55
        out_parts.append(
            route_from_df(
                domain,
                "risk_any_shiftaware_target10",
                df,
                top_mask(score_map["risk_any_oof"], any_rate),
                f"risk_any_oof_top{int(any_rate * 100)}",
                "severe_shift" if domain == "strict_external" else "within_internal_shift",
            )
        )
        out_parts.append(
            route_from_df(
                domain,
                "risk_direction_uniform80",
                df,
                top_mask(score_map["risk_direction_oof"], 0.80),
                "risk_direction_oof_top80",
                "risk_rank_uniform",
            )
        )
        out_parts.append(
            route_from_df(
                domain,
                "quality_direction_uniform90",
                df,
                top_mask(score_map["quality_plus_direction"], 0.90),
                "quality_plus_direction_top90",
                "risk_rank_uniform",
            )
        )
    return pd.concat(out_parts, ignore_index=True)


def adaptive_routes(routes: pd.DataFrame) -> pd.DataFrame:
    decisions = pd.read_csv(V82_DECISIONS)
    rows = []
    for _, row in decisions.iterrows():
        domain = row["domain"]
        light_policy = row["selected_policy_adaptive_light"]
        strict_policy = row["selected_policy_adaptive_strict"]
        for out_policy, selected in [
            ("adaptive_v50_to_v79_light", light_policy),
            ("adaptive_v50_to_v79_strict", strict_policy),
        ]:
            sub = routes.loc[routes["domain"].eq(domain) & routes["policy"].eq(selected)].copy()
            sub["source_policy"] = selected
            sub["policy"] = out_policy
            sub["shift_category"] = row["shift_category"]
            rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def summarize(case_routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full = case_routes.copy()
    for (domain, policy), sub in full.groupby(["domain", "policy"], sort=False):
        row = {"scope": domain, "policy": policy, "policy_label": POLICY_LABELS.get(policy, policy)}
        row.update(metrics(sub))
        rows.append(row)
    pooled = full.copy()
    pooled["domain"] = "all_domains"
    for policy, sub in pooled.groupby("policy", sort=False):
        row = {"scope": "all_domains", "policy": policy, "policy_label": POLICY_LABELS.get(policy, policy)}
        row.update(metrics(sub))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["policy"] = pd.Categorical(out["policy"], categories=POLICY_ORDER, ordered=True)
    out["scope"] = pd.Categorical(out["scope"], categories=DOMAIN_ORDER + ["all_domains"], ordered=True)
    return out.sort_values(["scope", "policy"]).reset_index(drop=True)


def fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def make_plot(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot = summary.loc[summary["scope"].eq("all_domains")].copy()
    colors = {
        "pure_auto": "#666666",
        "v50_main": "#1f77b4",
        "v79_light_lowrisk_guard": "#ff7f0e",
        "v79_strict_lowrisk_guard": "#d62728",
        "adaptive_v50_to_v79_light": "#2ca02c",
        "adaptive_v50_to_v79_strict": "#9467bd",
        "risk_any_shiftaware_target10": "#17becf",
        "risk_direction_uniform80": "#8c564b",
        "quality_direction_uniform90": "#e377c2",
    }
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for _, row in plot.iterrows():
        ax.scatter(
            row["control_rate"] * 100,
            row["balanced_accuracy"] * 100,
            s=92,
            color=colors.get(row["policy"], "#333333"),
            label=row["policy_label"],
        )
    ax.set_xlabel("Overall review/control rate (%)")
    ax.set_ylabel("Overall balanced accuracy (%)")
    ax.set_title("Integrated batch-adaptive selective diagnosis framework")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v91_integrated_framework_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v91_integrated_framework_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)

    focus = summary.loc[summary["scope"].eq("strict_external") & summary["policy"].isin([
        "v50_main",
        "v79_light_lowrisk_guard",
        "adaptive_v50_to_v79_light",
        "risk_any_shiftaware_target10",
        "risk_direction_uniform80",
        "quality_direction_uniform90",
    ])].copy()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(focus))
    ax.bar(x, focus["remaining_error_n"], color=[colors.get(p, "#333333") for p in focus["policy"]])
    ax.set_xticks(x)
    ax.set_xticklabels(focus["policy_label"], rotation=25, ha="right")
    ax.set_ylabel("Strict external remaining errors")
    ax.set_title("Strict external residual errors by workflow")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v91_strict_external_remaining_errors.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v91_strict_external_remaining_errors.pdf", bbox_inches="tight")
    plt.close(fig)


def write_messages(summary: pd.DataFrame) -> None:
    all_dom = summary.loc[summary["scope"].eq("all_domains")].copy()
    strict = summary.loc[summary["scope"].eq("strict_external")].copy()

    def row(policy: str, scope_df: pd.DataFrame = all_dom) -> pd.Series:
        return scope_df.loc[scope_df["policy"].eq(policy)].iloc[0]

    main = row("adaptive_v50_to_v79_light")
    strict_main = row("adaptive_v50_to_v79_light", strict)
    fixed_light = row("v79_light_lowrisk_guard")
    risk_any = row("risk_any_shiftaware_target10")
    risk_dir = row("risk_direction_uniform80")
    qd = row("quality_direction_uniform90")
    lines = ["# v91 批次自适应选择性诊断总流程对照", ""]
    lines.append("## 核心结论")
    lines.append("")
    lines.append(
        f"- 当前最适合作为主推部署流程的仍是 Batch-adaptive main：全域控制率 {fmt_pct(main['control_rate'])}，全域 BAcc {fmt_pct(main['balanced_accuracy'])}，严格外部 BAcc {fmt_pct(strict_main['balanced_accuracy'])}，严格外部剩余错误 {int(strict_main['remaining_error_n'])} 例。"
    )
    lines.append(
        f"- 固定 v79-light 全域 BAcc 更高（{fmt_pct(fixed_light['balanced_accuracy'])}），但全域控制率也更高（{fmt_pct(fixed_light['control_rate'])}）；因此它更像高安全固定方案，不是最低复核部署方案。"
    )
    lines.append(
        f"- 风险排序选择性诊断可以单独工作：risk-direction uniform80 全域 BAcc {fmt_pct(risk_dir['balanced_accuracy'])}，quality+direction uniform90 全域 BAcc {fmt_pct(qd['balanced_accuracy'])}。但它们控制率较高，且不是当前最优主流程。"
    )
    lines.append(
        f"- risk_any shift-aware 证明了 v90 的批次升级逻辑可行：全域控制率 {fmt_pct(risk_any['control_rate'])}，全域 BAcc {fmt_pct(risk_any['balanced_accuracy'])}，但第三批敏感性不足，不能替代 v50/v79 主流程。"
    )
    lines.append("")
    lines.append("## 论文写法")
    lines.append("")
    lines.append(
        "v91 的结论不是要用风险排序策略替代 v79，而是把整体框架边界讲清楚：固定高安全流程给出性能上限，批次自适应流程给出部署折中，风险排序曲线解释复核触发器为什么有效，v90 解释为什么严重偏移批次需要安全升级。"
    )
    lines.append("")
    lines.append("## 全域对照")
    lines.append("")
    lines.append("| 流程 | 控制率 | BAcc | Sens | Spec | FN | FP | 剩余错误 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in all_dom.sort_values("policy").iterrows():
        lines.append(
            f"| {r['policy_label']} | {fmt_pct(r['control_rate'])} | {fmt_pct(r['balanced_accuracy'])} | {fmt_pct(r['sensitivity'])} | {fmt_pct(r['specificity'])} | {int(r['fn'])} | {int(r['fp'])} | {int(r['remaining_error_n'])} |"
        )
    (OUT_DIR / "v91_key_messages.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixed = fixed_routes()
    risk = pure_and_risk_routes()
    adaptive = adaptive_routes(fixed)
    routes = pd.concat([risk, fixed, adaptive], ignore_index=True)
    routes["policy"] = pd.Categorical(routes["policy"], categories=POLICY_ORDER, ordered=True)
    routes.to_csv(OUT_DIR / "v91_integrated_case_routes.csv", index=False, encoding="utf-8-sig")
    summary = summarize(routes)
    summary.to_csv(OUT_DIR / "v91_integrated_summary.csv", index=False, encoding="utf-8-sig")
    make_plot(summary)
    write_messages(summary)

    show = summary.loc[summary["scope"].isin(["all_domains", "strict_external"])].copy()
    print("Integrated framework summary:")
    print(
        show[
            [
                "scope",
                "policy_label",
                "control_rate",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "fn",
                "fp",
                "remaining_error_n",
                "remaining_auto_risk",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
