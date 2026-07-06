from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v95_selective_risk_confidence_bounds_20260527"
FIG_DIR = OUT_DIR / "figures"
Z95 = 1.959963984540054

POLICY_ORDER = [
    "pure_auto",
    "v50_main",
    "adaptive_v50_to_v79_light",
    "v79_light_lowrisk_guard",
    "v79_strict_lowrisk_guard",
    "risk_any_shiftaware_target10",
    "risk_direction_uniform80",
    "quality_direction_uniform90",
]

POLICY_LABELS = {
    "pure_auto": "Pure auto",
    "v50_main": "Fixed v50",
    "adaptive_v50_to_v79_light": "Batch-adaptive main",
    "v79_light_lowrisk_guard": "Fixed v79-light",
    "v79_strict_lowrisk_guard": "Fixed v79-strict",
    "risk_any_shiftaware_target10": "Risk-rank shift-aware",
    "risk_direction_uniform80": "Risk-direction uniform80",
    "quality_direction_uniform90": "Quality+direction uniform90",
}


def wilson_ci(success: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = success / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    radius = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return max(0.0, (centre - radius) / denom), min(1.0, (centre + radius) / denom)


def bacc_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    sens_l, sens_u = wilson_ci(tp, tp + fn)
    spec_l, spec_u = wilson_ci(tn, tn + fp)
    return {
        "balanced_accuracy": float((sens + spec) / 2),
        "balanced_accuracy_wilson_low": float(np.nanmean([sens_l, spec_l])),
        "balanced_accuracy_wilson_high": float(np.nanmean([sens_u, spec_u])),
        "sensitivity": float(sens),
        "sensitivity_wilson_low": sens_l,
        "sensitivity_wilson_high": sens_u,
        "specificity": float(spec),
        "specificity_wilson_low": spec_l,
        "specificity_wilson_high": spec_u,
        "fn": fn,
        "fp": fp,
    }


def summarize_group(scope: str, policy: str, df: pd.DataFrame) -> dict[str, object]:
    y = df["label_idx"].to_numpy(int)
    pred = df["final_pred"].to_numpy(int)
    review = df["review_or_control"].to_numpy(int).astype(bool)
    p2_wrong = df["p2_wrong"].to_numpy(int).astype(bool)
    final_wrong = pred != y
    auto = ~review
    n = len(df)
    control_n = int(review.sum())
    auto_n = int(auto.sum())
    auto_wrong_n = int((auto & final_wrong).sum())
    captured_wrong_n = int((review & p2_wrong).sum())
    total_p2_wrong_n = int(p2_wrong.sum())
    auto_risk_l, auto_risk_u = wilson_ci(auto_wrong_n, auto_n)
    control_l, control_u = wilson_ci(control_n, n)
    capture_l, capture_u = wilson_ci(captured_wrong_n, total_p2_wrong_n)
    yield_l, yield_u = wilson_ci(captured_wrong_n, control_n)
    bacc = bacc_metrics(y, pred)
    row: dict[str, object] = {
        "scope": scope,
        "policy": policy,
        "policy_label": POLICY_LABELS.get(policy, policy),
        "n": n,
        "control_n": control_n,
        "control_rate": control_n / n,
        "control_rate_wilson_low": control_l,
        "control_rate_wilson_high": control_u,
        "auto_n": auto_n,
        "auto_rate": auto_n / n,
        "auto_wrong_n": auto_wrong_n,
        "auto_pass_error_risk": auto_wrong_n / auto_n if auto_n else 0.0,
        "auto_pass_error_risk_wilson_low": auto_risk_l,
        "auto_pass_error_risk_wilson_high": auto_risk_u,
        "total_p2_wrong_n": total_p2_wrong_n,
        "captured_wrong_n": captured_wrong_n,
        "error_capture_rate": captured_wrong_n / total_p2_wrong_n if total_p2_wrong_n else np.nan,
        "error_capture_rate_wilson_low": capture_l,
        "error_capture_rate_wilson_high": capture_u,
        "review_yield": captured_wrong_n / control_n if control_n else np.nan,
        "review_yield_wilson_low": yield_l,
        "review_yield_wilson_high": yield_u,
        "remaining_error_n": int(final_wrong.sum()),
    }
    row.update(bacc)
    for target in [0.05, 0.10, 0.15, 0.20]:
        row[f"auto_risk_upper_le_{int(target * 100)}"] = bool(auto_risk_u <= target)
    return row


def build_summary(routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keep = routes.loc[routes["policy"].isin(POLICY_ORDER)].copy()
    keep["policy"] = pd.Categorical(keep["policy"], categories=POLICY_ORDER, ordered=True)
    for (domain, policy), sub in keep.groupby(["domain", "policy"], sort=False, observed=True):
        rows.append(summarize_group(domain, policy, sub))
    pooled = keep.copy()
    pooled["domain"] = "all_domains"
    for policy, sub in pooled.groupby("policy", sort=False, observed=True):
        rows.append(summarize_group("all_domains", str(policy), sub))
    out = pd.DataFrame(rows)
    out["policy"] = pd.Categorical(out["policy"], categories=POLICY_ORDER, ordered=True)
    scope_order = ["old_data", "third_batch", "strict_external", "all_domains"]
    out["scope"] = pd.Categorical(out["scope"], categories=scope_order, ordered=True)
    return out.sort_values(["scope", "policy"]).reset_index(drop=True)


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def format_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scope",
        "policy_label",
        "control_rate",
        "auto_n",
        "auto_pass_error_risk",
        "auto_pass_error_risk_wilson_high",
        "balanced_accuracy",
        "balanced_accuracy_wilson_low",
        "error_capture_rate",
        "error_capture_rate_wilson_low",
        "remaining_error_n",
        "fn",
        "fp",
    ]
    out = summary[cols].copy()
    for col in [
        "control_rate",
        "auto_pass_error_risk",
        "auto_pass_error_risk_wilson_high",
        "balanced_accuracy",
        "balanced_accuracy_wilson_low",
        "error_capture_rate",
        "error_capture_rate_wilson_low",
    ]:
        out[col] = out[col].map(pct)
    return out


def make_plot(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot = summary.loc[summary["scope"].isin(["all_domains", "strict_external"])].copy()
    plot = plot.loc[plot["policy"].isin(["v50_main", "adaptive_v50_to_v79_light", "v79_light_lowrisk_guard", "v79_strict_lowrisk_guard", "quality_direction_uniform90"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax, scope in zip(axes, ["all_domains", "strict_external"]):
        sub = plot.loc[plot["scope"].eq(scope)].sort_values("policy")
        x = np.arange(len(sub))
        point = sub["auto_pass_error_risk"].to_numpy(float) * 100
        low = sub["auto_pass_error_risk_wilson_low"].to_numpy(float) * 100
        high = sub["auto_pass_error_risk_wilson_high"].to_numpy(float) * 100
        yerr = np.vstack([np.maximum(0, point - low), np.maximum(0, high - point)])
        ax.errorbar(x, point, yerr=yerr, fmt="o", capsize=4, color="#1f77b4", ecolor="#6baed6")
        ax.axhline(10, color="#9b1c1c", linestyle="--", linewidth=1, label="10% risk")
        ax.axhline(20, color="#7a5a00", linestyle=":", linewidth=1, label="20% risk")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["policy_label"], rotation=25, ha="right")
        ax.set_title(scope)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Auto-passed residual error risk with Wilson 95% CI (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Selective-risk confidence bounds", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "v95_selective_risk_confidence_bounds.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v95_selective_risk_confidence_bounds.pdf", bbox_inches="tight")
    plt.close(fig)


def write_messages(summary: pd.DataFrame) -> None:
    all_dom = summary.loc[summary["scope"].eq("all_domains")]
    ext = summary.loc[summary["scope"].eq("strict_external")]

    def row(scope_df: pd.DataFrame, policy: str) -> pd.Series:
        return scope_df.loc[scope_df["policy"].eq(policy)].iloc[0]

    main_all = row(all_dom, "adaptive_v50_to_v79_light")
    light_ext = row(ext, "v79_light_lowrisk_guard")
    strict_ext = row(ext, "v79_strict_lowrisk_guard")
    qd_all = row(all_dom, "quality_direction_uniform90")
    qd_ext = row(ext, "quality_direction_uniform90")
    lines = ["# v95 选择性风险置信边界", ""]
    lines.append("## 关键发现")
    lines.append("")
    lines.append(
        f"- Batch-adaptive main 全域放行残余错误率点估计为 {pct(main_all['auto_pass_error_risk'])}，Wilson 95% 上界为 {pct(main_all['auto_pass_error_risk_wilson_high'])}。"
    )
    lines.append(
        f"- Fixed v79-light 严格外部放行残余错误率点估计为 {pct(light_ext['auto_pass_error_risk'])}，Wilson 95% 上界为 {pct(light_ext['auto_pass_error_risk_wilson_high'])}；点估计很好，但外部自动放行数只有 {int(light_ext['auto_n'])}，所以置信上界仍偏宽。"
    )
    lines.append(
        f"- Fixed v79-strict 严格外部放行 0 错误，但自动放行数只有 {int(strict_ext['auto_n'])}，Wilson 上界仍为 {pct(strict_ext['auto_pass_error_risk_wilson_high'])}。"
    )
    lines.append(
        f"- Quality+direction uniform90 全域放行残余错误率为 {pct(qd_all['auto_pass_error_risk'])}，Wilson 上界为 {pct(qd_all['auto_pass_error_risk_wilson_high'])}；严格外部上界为 {pct(qd_ext['auto_pass_error_risk_wilson_high'])}，说明高复核上限仍需要更多外部样本验证。"
    )
    lines.append("")
    lines.append("## 论文边界")
    lines.append("")
    lines.append(
        "这一步让风险控制声明更严谨：当前结果可以报告很强的点估计和错误减少，但不能把单个外部集的小样本点估计写成严格统计保证。"
        "主文可以写“risk-controlled selective diagnosis with confidence-bounded safety reporting”，并把 Wilson 上界作为未来前瞻性验证的依据。"
    )
    lines.append("")
    lines.append("## Focus Table")
    focus = summary.loc[
        summary["scope"].isin(["all_domains", "strict_external"])
        & summary["policy"].isin(["v50_main", "adaptive_v50_to_v79_light", "v79_light_lowrisk_guard", "v79_strict_lowrisk_guard", "quality_direction_uniform90"])
    ]
    lines.append("")
    lines.append("| Scope | Workflow | Control | Auto n | Auto risk | Auto risk upper | BAcc | BAcc lower | Capture rate | Remaining errors |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in focus.iterrows():
        lines.append(
            f"| {r['scope']} | {r['policy_label']} | {pct(r['control_rate'])} | {int(r['auto_n'])} | {pct(r['auto_pass_error_risk'])} | "
            f"{pct(r['auto_pass_error_risk_wilson_high'])} | {pct(r['balanced_accuracy'])} | {pct(r['balanced_accuracy_wilson_low'])} | "
            f"{pct(r['error_capture_rate'])} | {int(r['remaining_error_n'])} |"
        )
    (OUT_DIR / "v95_key_messages.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(ROUTES)
    summary = build_summary(routes)
    formatted = format_table(summary)
    summary.to_csv(OUT_DIR / "v95_selective_risk_confidence_bounds.csv", index=False, encoding="utf-8-sig")
    formatted.to_csv(OUT_DIR / "v95_selective_risk_confidence_bounds_formatted.csv", index=False, encoding="utf-8-sig")
    make_plot(summary)
    write_messages(summary)

    print("v95 focus summary:")
    focus = formatted.loc[
        formatted["scope"].isin(["all_domains", "strict_external"])
        & formatted["policy_label"].isin(["Fixed v50", "Batch-adaptive main", "Fixed v79-light", "Fixed v79-strict", "Quality+direction uniform90"])
    ]
    print(focus.to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
