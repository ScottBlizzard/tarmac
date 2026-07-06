from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V87 = ROOT / "outputs" / "grosspath_rc_v87_triage_efficiency_curve_20260527" / "v87_triage_efficiency_summary.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v88_constraint_operating_points_20260527"
FIG_DIR = OUT_DIR / "figures"

POLICY_ORDER = ["P2_pure_auto", "v50_main", "v75_quality_lowconf", "v79_light_lowrisk_guard", "v79_strict_lowrisk_guard"]
POLICY_LABELS = {
    "P2_pure_auto": "Pure auto",
    "v50_main": "v50",
    "v75_quality_lowconf": "v75",
    "v79_light_lowrisk_guard": "v79-light",
    "v79_strict_lowrisk_guard": "v79-strict",
}


SCENARIOS = [
    {
        "scenario": "basic_cross_domain_screen",
        "description": "三域最终 BAcc >=95%，敏感性 >=90%，特异性 >=95%；适合低门槛跨域可用性展示。",
        "min_final_bacc": 0.95,
        "min_final_sensitivity": 0.90,
        "min_final_specificity": 0.95,
        "min_error_capture": 0.0,
        "max_remaining_error_n": np.inf,
        "require_external_zero_fn": False,
        "require_all_domain_zero_fp": False,
    },
    {
        "scenario": "high_safety_balanced",
        "description": "三域最终 BAcc >=96%，敏感性 >=93%，特异性 >=98%，原始错误捕获率 >=90%；强调高安全和可解释复核收益。",
        "min_final_bacc": 0.96,
        "min_final_sensitivity": 0.93,
        "min_final_specificity": 0.98,
        "min_error_capture": 0.90,
        "max_remaining_error_n": np.inf,
        "require_external_zero_fn": False,
        "require_all_domain_zero_fp": False,
    },
    {
        "scenario": "no_external_highrisk_miss",
        "description": "在 high_safety_balanced 基础上，严格外部集高危漏诊为 0；对应医生最关心的漏诊安全版本。",
        "min_final_bacc": 0.96,
        "min_final_sensitivity": 0.93,
        "min_final_specificity": 0.98,
        "min_error_capture": 0.90,
        "max_remaining_error_n": np.inf,
        "require_external_zero_fn": True,
        "require_all_domain_zero_fp": False,
    },
    {
        "scenario": "zero_lowrisk_overcall_all_domains",
        "description": "在 high_safety_balanced 基础上，三域低危误升级为 0；适合强调避免低危过度升级。",
        "min_final_bacc": 0.96,
        "min_final_sensitivity": 0.93,
        "min_final_specificity": 0.98,
        "min_error_capture": 0.90,
        "max_remaining_error_n": np.inf,
        "require_external_zero_fn": True,
        "require_all_domain_zero_fp": True,
    },
    {
        "scenario": "strictest_current_candidate",
        "description": "三域最终 BAcc >=96%，严格外部剩余错误为 0；作为当前高安全候选上限，不作为最终泛化声明。",
        "min_final_bacc": 0.96,
        "min_final_sensitivity": 0.93,
        "min_final_specificity": 0.98,
        "min_error_capture": 0.90,
        "max_remaining_error_n": np.inf,
        "require_external_zero_fn": True,
        "require_all_domain_zero_fp": True,
        "require_external_zero_error": True,
    },
]


def policy_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, sub in summary.groupby("policy", sort=False):
        strict = sub.loc[sub["domain"].eq("strict_external")].iloc[0]
        rows.append(
            {
                "policy": policy,
                "policy_label": POLICY_LABELS.get(policy, policy),
                "mean_control_rate": sub["control_rate"].mean(),
                "max_control_rate": sub["control_rate"].max(),
                "min_final_bacc": sub["final_balanced_accuracy"].min(),
                "min_final_sensitivity": sub["final_sensitivity"].min(),
                "min_final_specificity": sub["final_specificity"].min(),
                "min_error_capture_rate": sub["error_capture_rate"].replace(0, np.nan).min() if policy != "P2_pure_auto" else 0.0,
                "max_remaining_error_n": sub["remaining_error_n"].max(),
                "total_remaining_error_n": sub["remaining_error_n"].sum(),
                "max_remaining_fn_n": sub["remaining_fn_n"].max(),
                "max_remaining_fp_n": sub["remaining_fp_n"].max(),
                "all_domain_zero_fp": bool(sub["remaining_fp_n"].sum() == 0),
                "strict_external_remaining_error_n": int(strict["remaining_error_n"]),
                "strict_external_remaining_fn_n": int(strict["remaining_fn_n"]),
                "strict_external_remaining_fp_n": int(strict["remaining_fp_n"]),
            }
        )
    out = pd.DataFrame(rows)
    out["policy"] = pd.Categorical(out["policy"], categories=POLICY_ORDER, ordered=True)
    return out.sort_values("policy").reset_index(drop=True)


def scenario_pass(row: pd.Series, sc: dict[str, object]) -> bool:
    ok = (
        row["min_final_bacc"] >= sc["min_final_bacc"]
        and row["min_final_sensitivity"] >= sc["min_final_sensitivity"]
        and row["min_final_specificity"] >= sc["min_final_specificity"]
        and row["min_error_capture_rate"] >= sc["min_error_capture"]
        and row["max_remaining_error_n"] <= sc["max_remaining_error_n"]
    )
    if sc.get("require_external_zero_fn"):
        ok = ok and row["strict_external_remaining_fn_n"] == 0
    if sc.get("require_all_domain_zero_fp"):
        ok = ok and bool(row["all_domain_zero_fp"])
    if sc.get("require_external_zero_error"):
        ok = ok and row["strict_external_remaining_error_n"] == 0
    return bool(ok)


def select_operating_points(frontier: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        tmp = frontier.copy()
        tmp["pass"] = tmp.apply(lambda r: scenario_pass(r, sc), axis=1)
        passed = tmp.loc[tmp["pass"]].sort_values(["max_control_rate", "mean_control_rate", "total_remaining_error_n"])
        selected = passed.iloc[0] if len(passed) else None
        for _, row in tmp.iterrows():
            rows.append(
                {
                    "scenario": sc["scenario"],
                    "description": sc["description"],
                    "policy": row["policy"],
                    "policy_label": row["policy_label"],
                    "pass": bool(row["pass"]),
                    "selected_min_control": bool(selected is not None and row["policy"] == selected["policy"]),
                    "mean_control_rate": row["mean_control_rate"],
                    "max_control_rate": row["max_control_rate"],
                    "min_final_bacc": row["min_final_bacc"],
                    "min_final_sensitivity": row["min_final_sensitivity"],
                    "min_final_specificity": row["min_final_specificity"],
                    "min_error_capture_rate": row["min_error_capture_rate"],
                    "max_remaining_error_n": row["max_remaining_error_n"],
                    "total_remaining_error_n": row["total_remaining_error_n"],
                    "strict_external_remaining_error_n": row["strict_external_remaining_error_n"],
                    "strict_external_remaining_fn_n": row["strict_external_remaining_fn_n"],
                    "all_domain_zero_fp": row["all_domain_zero_fp"],
                }
            )
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def write_markdown(frontier: pd.DataFrame, selections: pd.DataFrame) -> None:
    selected = selections.loc[selections["selected_min_control"]].copy()
    lines = ["# v88 临床约束下的操作点选择", ""]
    lines.append("## 最小控制率选择结果")
    lines.append("")
    lines.append("| 场景 | 选中流程 | 最大控制率 | 最差域 BAcc | 最差域 Sens | 最差域 Spec | 最差错误捕获率 | 三域剩余错误 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in selected.iterrows():
        lines.append(
            f"| {r['scenario']} | {r['policy_label']} | {pct(r['max_control_rate'])} | {pct(r['min_final_bacc'])} | "
            f"{pct(r['min_final_sensitivity'])} | {pct(r['min_final_specificity'])} | {pct(r['min_error_capture_rate'])} | {int(r['total_remaining_error_n'])} |"
        )
    lines.append("")
    lines.append("## 解释")
    lines.append("")
    lines.append(
        "这一步把流程选择从“看哪个分数高”改成“先定义临床安全约束，再选控制率最低的流程”。"
        "在高安全平衡约束下，v79-light 是最小控制率可行解；如果要求三域低危误升级全部为 0，同时严格外部无高危漏诊，则会自然切到 v79-strict。"
    )
    lines.append("")
    lines.append("## 策略边界")
    lines.append("")
    lines.append(
        "v79-light 更适合作为主推版本，因为它在三域约束下已经达到较高安全性，同时控制率低于 strict；"
        "v79-strict 是高安全候选上限，适合复核资源充足或对低危误升级非常敏感的场景，但仍需要更多外部批次验证。"
    )
    (OUT_DIR / "v88_constraint_operating_points.md").write_text("\n".join(lines), encoding="utf-8-sig")


def make_plot(frontier: pd.DataFrame, selections: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(frontier["max_control_rate"] * 100, frontier["min_final_bacc"] * 100, s=70, color="#1f77b4")
    offsets = {
        "Pure auto": (8, 10),
        "v50": (8, 10),
        "v75": (-18, 12),
        "v79-light": (8, 8),
        "v79-strict": (-28, -18),
    }
    for _, r in frontier.iterrows():
        ax.annotate(
            r["policy_label"],
            (r["max_control_rate"] * 100, r["min_final_bacc"] * 100),
            xytext=offsets.get(r["policy_label"], (5, 5)),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(96, color="#999999", linestyle="--", linewidth=1, label="BAcc 96%")
    ax.set_xlim(-4, 94)
    ax.set_ylim(69, 98.2)
    ax.set_xlabel("Worst-domain review/control rate (%)")
    ax.set_ylabel("Worst-domain final BAcc (%)")
    ax.set_title("Constraint-oriented operating frontier")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v88_worst_domain_operating_frontier.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v88_worst_domain_operating_frontier.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(V87)
    frontier = policy_frontier(summary)
    selections = select_operating_points(frontier)
    frontier.to_csv(OUT_DIR / "v88_policy_frontier_summary.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(OUT_DIR / "v88_constraint_operating_points.csv", index=False, encoding="utf-8-sig")
    write_markdown(frontier, selections)
    make_plot(frontier, selections)

    print("Selected operating points:")
    print(
        selections.loc[selections["selected_min_control"], [
            "scenario",
            "policy_label",
            "max_control_rate",
            "min_final_bacc",
            "min_final_sensitivity",
            "min_final_specificity",
            "min_error_capture_rate",
            "total_remaining_error_n",
        ]].to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
