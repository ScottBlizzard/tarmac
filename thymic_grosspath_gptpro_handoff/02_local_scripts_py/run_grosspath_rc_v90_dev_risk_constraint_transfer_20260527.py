from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V89_DIR = ROOT / "outputs" / "grosspath_rc_v89_risk_ranker_selective_diagnosis_20260527"
CURVES = V89_DIR / "v89_selective_risk_curves.csv"
DEV_SELECTION = V89_DIR / "v89_dev_score_selection.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v90_dev_risk_constraint_transfer_20260527"
FIG_DIR = OUT_DIR / "figures"

TARGET_RISKS = [0.15, 0.10, 0.05]
EXCLUDE = {"random_expected", "oracle_wrong"}


def pick_dev_rate(curves: pd.DataFrame, score: str, target: float) -> dict[str, object] | None:
    sub = curves.loc[curves["score"].eq(score) & curves["domain"].isin(["old_data", "third_batch"])].copy()
    pivot = sub.pivot(index="control_rate", columns="domain", values="auto_selective_risk")
    ok = pivot.loc[(pivot["old_data"] <= target) & (pivot["third_batch"] <= target)]
    if ok.empty:
        return None
    rate = float(ok.index.min())
    row = pivot.loc[rate]
    return {
        "dev_selected_rate": rate,
        "dev_old_risk": float(row["old_data"]),
        "dev_third_risk": float(row["third_batch"]),
        "dev_max_risk": float(max(row["old_data"], row["third_batch"])),
    }


def eval_at_rate(curves: pd.DataFrame, score: str, domain: str, rate: float, prefix: str) -> dict[str, object]:
    sub = curves.loc[curves["score"].eq(score) & curves["domain"].eq(domain)].copy()
    idx = (sub["control_rate"] - rate).abs().idxmin()
    row = sub.loc[idx]
    return {
        f"{prefix}_rate": float(row["control_rate"]),
        f"{prefix}_auto_risk": float(row["auto_selective_risk"]),
        f"{prefix}_remaining_wrong_n": int(round(float(row["remaining_wrong_n"]))),
        f"{prefix}_captured_wrong_n": int(round(float(row["captured_wrong_n"]))),
        f"{prefix}_final_bacc": float(row["final_bacc"]) if not pd.isna(row["final_bacc"]) else np.nan,
        f"{prefix}_final_sensitivity": float(row["final_sensitivity"]) if not pd.isna(row["final_sensitivity"]) else np.nan,
        f"{prefix}_final_specificity": float(row["final_specificity"]) if not pd.isna(row["final_specificity"]) else np.nan,
    }


def external_min_feasible(curves: pd.DataFrame, score: str, target: float) -> dict[str, object]:
    sub = curves.loc[curves["score"].eq(score) & curves["domain"].eq("strict_external")].copy().sort_values("control_rate")
    ok = sub.loc[sub["auto_selective_risk"].le(target)]
    if ok.empty:
        return {
            "external_min_feasible_rate": np.nan,
            "external_min_feasible_risk": np.nan,
            "external_min_feasible_remaining_wrong_n": np.nan,
        }
    row = ok.iloc[0]
    return {
        "external_min_feasible_rate": float(row["control_rate"]),
        "external_min_feasible_risk": float(row["auto_selective_risk"]),
        "external_min_feasible_remaining_wrong_n": int(round(float(row["remaining_wrong_n"]))),
    }


def build_transfer_table(curves: pd.DataFrame, selected_score: str) -> pd.DataFrame:
    rows = []
    scores = [s for s in curves["score"].drop_duplicates().tolist() if s not in EXCLUDE]
    for target in TARGET_RISKS:
        for score in scores:
            picked = pick_dev_rate(curves, score, target)
            if picked is None:
                rows.append(
                    {
                        "target_auto_risk": target,
                        "score": score,
                        "is_v89_selected_score": score == selected_score,
                        "dev_feasible": False,
                    }
                )
                continue
            rate = float(picked["dev_selected_rate"])
            inflated = min(0.90, round(rate + 0.30, 2))
            row = {
                "target_auto_risk": target,
                "score": score,
                "is_v89_selected_score": score == selected_score,
                "dev_feasible": True,
                **picked,
                **eval_at_rate(curves, score, "strict_external", rate, "external_dev_rate"),
                "severe_shift_inflated_rate_rule": "+30pp capped at 90%",
                **eval_at_rate(curves, score, "strict_external", inflated, "external_inflated"),
                **external_min_feasible(curves, score, target),
            }
            row["external_dev_rate_meets_target"] = bool(row["external_dev_rate_auto_risk"] <= target)
            row["external_inflated_meets_target"] = bool(row["external_inflated_auto_risk"] <= target)
            rows.append(row)
    return pd.DataFrame(rows)


def select_scenarios(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, sub in table.loc[table["dev_feasible"].eq(True)].groupby("target_auto_risk"):
        min_control = sub.sort_values(["dev_selected_rate", "external_dev_rate_auto_risk"]).iloc[0]
        best_transfer = sub.sort_values(["external_dev_rate_auto_risk", "dev_selected_rate"]).iloc[0]
        best_inflated = sub.sort_values(["external_inflated_auto_risk", "external_inflated_rate"]).iloc[0]
        for name, row in [
            ("min_dev_control", min_control),
            ("best_external_transfer_analysis", best_transfer),
            ("best_severe_shift_inflated_analysis", best_inflated),
        ]:
            out = row.to_dict()
            out["selection_view"] = name
            rows.append(out)
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def write_messages(table: pd.DataFrame, selected: pd.DataFrame, v89_score: str) -> None:
    focus = table.loc[table["score"].eq(v89_score) & table["target_auto_risk"].eq(0.10) & table["dev_feasible"].eq(True)]
    lines = ["# v90 开发域选择性风险约束的外部转移", ""]
    if not focus.empty:
        r = focus.iloc[0]
        lines.append("## v89 主风险分数在 10% 放行风险目标下")
        lines.append("")
        lines.append(
            f"- 开发域为了让 old/third 放行残余错误率都 <=10%，`{v89_score}` 需要控制率 {pct(r['dev_selected_rate'])}，开发域最差风险 {pct(r['dev_max_risk'])}。"
        )
        lines.append(
            f"- 直接转到严格外部时，放行残余错误率为 {pct(r['external_dev_rate_auto_risk'])}，未达到 10% 目标。"
        )
        lines.append(
            f"- 如果无标签批次审计判定 severe shift 后按预设 +30pp 安全升级，控制率到 {pct(r['external_inflated_rate'])}，严格外部放行残余错误率降到 {pct(r['external_inflated_auto_risk'])}。"
        )
    lines.append("")
    lines.append("## 主要结论")
    lines.append("")
    lines.append(
        "开发域选择性风险约束不能直接保证严重外部偏移批次仍达标；这支持我们保留 v77/v82 的无标签批次审计和安全升级机制。"
        "换句话说，风险排序器解决“哪些病例更危险”，批次审计解决“这个批次是否需要整体提高复核强度”。"
    )
    lines.append("")
    lines.append("## 最优转移分析")
    lines.append("")
    lines.append("| 目标放行风险 | 视角 | 分数 | 开发选择控制率 | 外部直接风险 | 外部升级风险 |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, r in selected.iterrows():
        lines.append(
            f"| {pct(r['target_auto_risk'])} | {r['selection_view']} | {r['score']} | {pct(r.get('dev_selected_rate', np.nan))} | "
            f"{pct(r.get('external_dev_rate_auto_risk', np.nan))} | {pct(r.get('external_inflated_auto_risk', np.nan))} |"
        )
    (OUT_DIR / "v90_key_messages.md").write_text("\n".join(lines), encoding="utf-8-sig")


def make_plot(table: pd.DataFrame, v89_score: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot = table.loc[table["dev_feasible"].eq(True) & table["target_auto_risk"].eq(0.10)].copy()
    if plot.empty:
        return
    plot = plot.sort_values("external_dev_rate_auto_risk")
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    x = np.arange(len(plot))
    ax.bar(x - 0.18, plot["external_dev_rate_auto_risk"] * 100, width=0.36, label="Direct transfer", color="#1f77b4")
    ax.bar(x + 0.18, plot["external_inflated_auto_risk"] * 100, width=0.36, label="Severe-shift +30pp", color="#ff7f0e")
    ax.axhline(10, color="#9b1c1c", linestyle="--", linewidth=1.2, label="10% target")
    labels = [f"{s}{'*' if s == v89_score else ''}" for s in plot["score"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Strict external auto-passed residual risk (%)", fontsize=10)
    ax.set_title("Transfer of development-set selective-risk constraint")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0.04, 0.02, 1, 1))
    fig.savefig(FIG_DIR / "v90_target10_external_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v90_target10_external_transfer.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(CURVES)
    dev_selection = pd.read_csv(DEV_SELECTION)
    v89_score = str(dev_selection.loc[dev_selection["selected_by"].eq("min_dev_mean_aurc"), "score"].iloc[0])
    table = build_transfer_table(curves, v89_score)
    selected = select_scenarios(table)
    table.to_csv(OUT_DIR / "v90_dev_constraint_transfer_all_scores.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v90_dev_constraint_transfer_selected_views.csv", index=False, encoding="utf-8-sig")
    write_messages(table, selected, v89_score)
    make_plot(table, v89_score)

    show = table.loc[table["target_auto_risk"].eq(0.10) & table["dev_feasible"].eq(True)].copy()
    print("Target <=10% residual auto risk transfer:")
    print(
        show[
            [
                "score",
                "is_v89_selected_score",
                "dev_selected_rate",
                "dev_max_risk",
                "external_dev_rate_auto_risk",
                "external_dev_rate_remaining_wrong_n",
                "external_inflated_rate",
                "external_inflated_auto_risk",
                "external_inflated_remaining_wrong_n",
                "external_min_feasible_rate",
            ]
        ]
        .sort_values(["external_dev_rate_auto_risk", "dev_selected_rate"])
        .to_string(index=False)
    )
    print("\nScenario views:")
    print(
        selected[
            [
                "target_auto_risk",
                "selection_view",
                "score",
                "dev_selected_rate",
                "external_dev_rate_auto_risk",
                "external_inflated_rate",
                "external_inflated_auto_risk",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
