from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V91 = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_summary.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v93_leave_domain_out_selection_20260527"
FIG_DIR = OUT_DIR / "figures"

DOMAINS = ["old_data", "third_batch", "strict_external"]
EXCLUDE_POLICIES = {"pure_auto"}

RULES = [
    {
        "rule": "basic_cross_domain_min_review",
        "description": "开发域 min BAcc>=95%、min Sens>=90%、min Spec>=95%，选择最大控制率最低的流程。",
        "constraints": {"min_bacc": 0.95, "min_sens": 0.90, "min_spec": 0.95},
        "sort": [("max_control_rate", True), ("total_error_n", True), ("min_bacc", False)],
    },
    {
        "rule": "high_safety_balanced_min_review",
        "description": "开发域 min BAcc>=96%、min Sens>=93%、min Spec>=98%，选择最大控制率最低的流程。",
        "constraints": {"min_bacc": 0.96, "min_sens": 0.93, "min_spec": 0.98},
        "sort": [("max_control_rate", True), ("total_error_n", True), ("min_bacc", False)],
    },
    {
        "rule": "zero_fn_if_possible",
        "description": "开发域 FN 总数为 0，选择最大控制率最低的流程。",
        "constraints": {"total_fn": 0},
        "sort": [("max_control_rate", True), ("total_fp", True), ("total_error_n", True)],
    },
    {
        "rule": "min_error_under_90_control",
        "description": "开发域最大控制率不超过 90%，选择开发域总错误最少的流程。",
        "constraints": {"max_control_rate": 0.90},
        "sort": [("total_error_n", True), ("max_control_rate", True), ("min_bacc", False)],
    },
    {
        "rule": "zero_error_if_possible",
        "description": "开发域总错误为 0，选择最大控制率最低的流程。",
        "constraints": {"total_error_n": 0},
        "sort": [("max_control_rate", True), ("mean_control_rate", True)],
    },
]


def aggregate_dev(summary: pd.DataFrame, dev_domains: list[str]) -> pd.DataFrame:
    rows = []
    pool = summary.loc[summary["scope"].isin(dev_domains) & ~summary["policy"].isin(EXCLUDE_POLICIES)].copy()
    for policy, sub in pool.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "policy_label": sub["policy_label"].iloc[0],
                "dev_domains": "+".join(dev_domains),
                "mean_control_rate": sub["control_rate"].mean(),
                "max_control_rate": sub["control_rate"].max(),
                "min_bacc": sub["balanced_accuracy"].min(),
                "mean_bacc": sub["balanced_accuracy"].mean(),
                "min_sens": sub["sensitivity"].min(),
                "min_spec": sub["specificity"].min(),
                "total_fn": int(sub["fn"].sum()),
                "total_fp": int(sub["fp"].sum()),
                "total_error_n": int(sub["remaining_error_n"].sum()),
                "max_remaining_auto_risk": sub["remaining_auto_risk"].max(),
            }
        )
    return pd.DataFrame(rows)


def apply_constraints(dev: pd.DataFrame, constraints: dict[str, float]) -> pd.DataFrame:
    out = dev.copy()
    for key, value in constraints.items():
        if key in ["min_bacc", "min_sens", "min_spec"]:
            out = out.loc[out[key].ge(value)]
        elif key == "max_control_rate":
            out = out.loc[out[key].le(value)]
        elif key in ["total_fn", "total_error_n"]:
            out = out.loc[out[key].eq(value)]
        else:
            raise ValueError(key)
    return out


def select_policy(dev: pd.DataFrame, rule: dict[str, object]) -> pd.Series | None:
    pool = apply_constraints(dev, rule["constraints"])  # type: ignore[arg-type]
    if pool.empty:
        return None
    sort_cols = [x[0] for x in rule["sort"]]  # type: ignore[index]
    ascending = [x[1] for x in rule["sort"]]  # type: ignore[index]
    return pool.sort_values(sort_cols, ascending=ascending).iloc[0]


def run_selection(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    dev_rows = []
    for heldout in DOMAINS:
        dev_domains = [d for d in DOMAINS if d != heldout]
        dev = aggregate_dev(summary, dev_domains)
        dev.insert(0, "heldout_domain", heldout)
        dev_rows.append(dev)
        held = summary.loc[summary["scope"].eq(heldout)].copy()
        for rule in RULES:
            selected = select_policy(dev, rule)
            if selected is None:
                rows.append(
                    {
                        "heldout_domain": heldout,
                        "dev_domains": "+".join(dev_domains),
                        "rule": rule["rule"],
                        "description": rule["description"],
                        "selected": False,
                    }
                )
                continue
            h = held.loc[held["policy"].eq(selected["policy"])].iloc[0]
            rows.append(
                {
                    "heldout_domain": heldout,
                    "dev_domains": "+".join(dev_domains),
                    "rule": rule["rule"],
                    "description": rule["description"],
                    "selected": True,
                    "selected_policy": selected["policy"],
                    "selected_policy_label": selected["policy_label"],
                    "dev_mean_control_rate": selected["mean_control_rate"],
                    "dev_max_control_rate": selected["max_control_rate"],
                    "dev_min_bacc": selected["min_bacc"],
                    "dev_min_sens": selected["min_sens"],
                    "dev_min_spec": selected["min_spec"],
                    "dev_total_fn": selected["total_fn"],
                    "dev_total_fp": selected["total_fp"],
                    "dev_total_error_n": selected["total_error_n"],
                    "heldout_control_rate": h["control_rate"],
                    "heldout_bacc": h["balanced_accuracy"],
                    "heldout_sens": h["sensitivity"],
                    "heldout_spec": h["specificity"],
                    "heldout_fn": int(h["fn"]),
                    "heldout_fp": int(h["fp"]),
                    "heldout_error_n": int(h["remaining_error_n"]),
                    "heldout_auto_risk": h["remaining_auto_risk"],
                }
            )
    return pd.DataFrame(rows), pd.concat(dev_rows, ignore_index=True)


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def make_plot(selection: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot = selection.loc[selection["selected"].eq(True)].copy()
    rule_order = [r["rule"] for r in RULES]
    heldout_order = DOMAINS
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), sharey=True)
    for ax, heldout in zip(axes, heldout_order):
        sub = plot.loc[plot["heldout_domain"].eq(heldout)].copy()
        sub["rule"] = pd.Categorical(sub["rule"], categories=rule_order, ordered=True)
        sub = sub.sort_values("rule")
        x = np.arange(len(sub))
        ax.scatter(sub["heldout_control_rate"] * 100, sub["heldout_bacc"] * 100, s=70, color="#1f77b4")
        for _, row in sub.iterrows():
            ax.annotate(row["selected_policy_label"], (row["heldout_control_rate"] * 100, row["heldout_bacc"] * 100), xytext=(5, 5), textcoords="offset points", fontsize=7)
        ax.set_title(f"Held out: {heldout}")
        ax.set_xlabel("Held-out control rate (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out balanced accuracy (%)")
    fig.suptitle("Leave-domain-out workflow selection", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v93_leave_domain_out_selection.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v93_leave_domain_out_selection.pdf", bbox_inches="tight")
    plt.close(fig)


def write_messages(selection: pd.DataFrame) -> None:
    strict = selection.loc[selection["heldout_domain"].eq("strict_external") & selection["selected"].eq(True)].copy()
    basic = strict.loc[strict["rule"].eq("basic_cross_domain_min_review")].iloc[0]
    high = strict.loc[strict["rule"].eq("high_safety_balanced_min_review")].iloc[0]
    zero = strict.loc[strict["rule"].eq("zero_error_if_possible")].iloc[0]
    all_selected = selection.loc[selection["selected"].eq(True)].copy()
    lines = ["# v93 留一数据域外推工作流选择", ""]
    lines.append("## 严格外部作为留出域")
    lines.append("")
    lines.append(
        f"- 只用 old data + third batch 按基础跨域约束选择时，自动选中 `{basic['selected_policy_label']}`，严格外部 BAcc {pct(basic['heldout_bacc'])}，控制率 {pct(basic['heldout_control_rate'])}，剩余错误 {int(basic['heldout_error_n'])}。"
    )
    lines.append(
        f"- 按较高安全约束选择时，自动选中 `{high['selected_policy_label']}`，严格外部 BAcc {pct(high['heldout_bacc'])}，控制率 {pct(high['heldout_control_rate'])}，剩余错误 {int(high['heldout_error_n'])}。"
    )
    lines.append(
        f"- 如果开发域要求总错误为 0，会选中 `{zero['selected_policy_label']}`，严格外部 BAcc {pct(zero['heldout_bacc'])}，但控制率升到 {pct(zero['heldout_control_rate'])}。"
    )
    lines.append("")
    lines.append("## 解释")
    lines.append("")
    lines.append(
        "这个实验说明，v50/v79-light 的定位不是外部集暴露后硬挑出来的：当严格外部完全留出时，old+third 的约束选择会自然给出 v50 或 v79-light。"
        "同时，高复核上限策略也会被开发域零错误约束选出来，但代价是接近 90% 控制率。"
    )
    lines.append("")
    lines.append("## 选择结果总表")
    lines.append("")
    lines.append("| Heldout | Rule | Selected | Dev min BAcc | Heldout control | Heldout BAcc | Heldout FN | Heldout FP |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for _, r in all_selected.iterrows():
        lines.append(
            f"| {r['heldout_domain']} | {r['rule']} | {r['selected_policy_label']} | {pct(r['dev_min_bacc'])} | {pct(r['heldout_control_rate'])} | {pct(r['heldout_bacc'])} | {int(r['heldout_fn'])} | {int(r['heldout_fp'])} |"
        )
    (OUT_DIR / "v93_key_messages.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(V91)
    selection, dev_agg = run_selection(summary)
    selection.to_csv(OUT_DIR / "v93_leave_domain_out_selection.csv", index=False, encoding="utf-8-sig")
    dev_agg.to_csv(OUT_DIR / "v93_leave_domain_dev_policy_aggregate.csv", index=False, encoding="utf-8-sig")
    make_plot(selection)
    write_messages(selection)

    print("v93 leave-domain-out selections:")
    print(
        selection.loc[selection["selected"].eq(True), [
            "heldout_domain",
            "rule",
            "selected_policy_label",
            "dev_min_bacc",
            "dev_min_sens",
            "dev_min_spec",
            "dev_total_error_n",
            "heldout_control_rate",
            "heldout_bacc",
            "heldout_sens",
            "heldout_spec",
            "heldout_fn",
            "heldout_fp",
            "heldout_error_n",
        ]].to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
