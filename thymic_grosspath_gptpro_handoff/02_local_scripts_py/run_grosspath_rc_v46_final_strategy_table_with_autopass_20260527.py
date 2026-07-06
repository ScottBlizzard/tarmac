from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V43 = ROOT / "outputs" / "grosspath_rc_v43_final_strategy_table_20260527" / "v43_final_strategy_table.csv"
V44 = ROOT / "outputs" / "grosspath_rc_v44_selective_auto_pass_20260527" / "v44_dev_selected_auto_pass_scenarios_external_eval.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v46_final_strategy_table_with_autopass_20260527"


def pct(x: float) -> float:
    return round(float(x) * 100, 2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(V43)
    v44 = pd.read_csv(V44)
    auto_map = {
        "v37 rank controller dev95": "workflow_bacc95",
        "v37 rank controller dev97": "workflow_bacc97",
    }
    table["external_auto_pass_rate_pct"] = ""
    table["external_auto_pass_accuracy_pct"] = ""
    table["external_auto_pass_fn"] = ""
    table["external_auto_pass_fp"] = ""
    table["deployment_boundary"] = ""

    for idx, row in table.iterrows():
        strategy = row["strategy"]
        if strategy in auto_map:
            r = v44.loc[v44["scenario"].eq(auto_map[strategy])].iloc[0]
            table.loc[idx, "external_auto_pass_rate_pct"] = pct(r["external_auto_rate"])
            table.loc[idx, "external_auto_pass_accuracy_pct"] = pct(r["external_auto_accuracy"])
            table.loc[idx, "external_auto_pass_fn"] = int(r["external_auto_fn_n"])
            table.loc[idx, "external_auto_pass_fp"] = int(r["external_auto_fp_n"])
            table.loc[idx, "deployment_boundary"] = "工作流可用；自动放行准确率未达到90%，需保留复核环节。"
        elif str(strategy).startswith("P2"):
            table.loc[idx, "external_auto_pass_rate_pct"] = 100.0
            table.loc[idx, "external_auto_pass_accuracy_pct"] = row["external_accuracy_pct"]
            table.loc[idx, "external_auto_pass_fn"] = row["external_fn"]
            table.loc[idx, "external_auto_pass_fp"] = row["external_fp"]
            table.loc[idx, "deployment_boundary"] = "纯自动基线，外部准确率不足。"
        elif str(strategy).startswith("P0"):
            table.loc[idx, "deployment_boundary"] = "纯自动参考，不推荐部署。"
        elif str(strategy).startswith("P7"):
            table.loc[idx, "deployment_boundary"] = "高复核量工作流，上阶段对照。"
        else:
            table.loc[idx, "deployment_boundary"] = "排序效率指标，不是单独部署策略。"

    table.to_csv(OUT_DIR / "v46_final_strategy_table_with_autopass.csv", index=False, encoding="utf-8-sig")

    md = [
        "# v46 最终策略表：区分工作流表现和自动放行表现",
        "",
        "| strategy | review % | workflow BAcc % | workflow Acc % | FN | FP | auto-pass % | auto-pass Acc % | auto-pass FN | auto-pass FP | boundary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in table.iterrows():
        md.append(
            f"| {r['strategy']} | {r['external_review_rate_pct']} | {r['external_bacc_pct']} | {r['external_accuracy_pct']} | "
            f"{r['external_fn']} | {r['external_fp']} | {r['external_auto_pass_rate_pct']} | {r['external_auto_pass_accuracy_pct']} | "
            f"{r['external_auto_pass_fn']} | {r['external_auto_pass_fp']} | {r['deployment_boundary']} |"
        )
    (OUT_DIR / "v46_final_strategy_table_with_autopass.md").write_text("\n".join(md), encoding="utf-8")

    print(table.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
