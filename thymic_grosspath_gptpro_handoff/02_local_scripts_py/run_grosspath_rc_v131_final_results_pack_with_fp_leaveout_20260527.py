from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v131_final_results_pack_with_fp_leaveout_20260527"
V129_DIR = ROOT / "outputs" / "grosspath_rc_v129_stability_results_pack_20260527"
V130_SUMMARY = ROOT / "outputs" / "grosspath_rc_v130_leave_one_fp_out_validation_20260527" / "v130_leave_one_fp_out_summary.csv"
V130_DETAIL = ROOT / "outputs" / "grosspath_rc_v130_leave_one_fp_out_validation_20260527" / "v130_leave_one_fp_out_detail.csv"


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["heldout_capture_rate", "min_full_bacc", "full_bacc", "full_control_rate"]:
            out[col] = out[col].map(pct)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    for src in V129_DIR.glob("*.csv"):
        shutil.copy2(src, OUT_DIR / src.name)
    for src in (V129_DIR / "figures").glob("*"):
        shutil.copy2(src, fig_dir / src.name)

    summary = pd.read_csv(V130_SUMMARY)
    detail = pd.read_csv(V130_DETAIL)
    summary.to_csv(OUT_DIR / "v131_leave_one_fp_out_summary.csv", index=False, encoding="utf-8-sig")
    format_metrics(summary).to_csv(OUT_DIR / "v131_leave_one_fp_out_summary_formatted.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT_DIR / "v131_leave_one_fp_out_detail.csv", index=False, encoding="utf-8-sig")
    format_metrics(detail).to_csv(OUT_DIR / "v131_leave_one_fp_out_detail_formatted.csv", index=False, encoding="utf-8-sig")

    min_row = summary[summary["selector"].eq("min_review")].iloc[0]
    cap_row = summary[summary["selector"].eq("capped_stable_envelope")].iloc[0]
    lines = [
        "# v131 Final Results Pack with Leave-one-FP-out Evidence",
        "",
        "本包在 v129 基础上补入 v130 留一 FP 错例验证，作为当前最完整的 Results 入口。",
        "",
        "## v130 稳定性补充",
        "",
        f"- min-review：held-out FP 抓回 {int(min_row['heldout_captured_n'])}/{int(min_row['heldout_cases'])}，平均控制率 {pct(min_row['mean_full_control_rate'])}，最差 FP={int(min_row['max_full_fp'])}。",
        f"- capped stable-envelope：held-out FP 抓回 {int(cap_row['heldout_captured_n'])}/{int(cap_row['heldout_cases'])}，平均控制率 {pct(cap_row['mean_full_control_rate'])}，最差 FP={int(cap_row['max_full_fp'])}。",
        "",
        "## 当前写作边界",
        "",
        "v118/v119 仍是主写的高安全候选；v128/v130 作为稳定性补充，说明带上限稳定包络能够在内部留域和留一错例场景中保持 FP=0。由于 held-out FP 只有 3 个，该证据不能替代真正多中心前瞻验证。",
    ]
    (OUT_DIR / "v131_final_results_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")

    print("Wrote", OUT_DIR)
    print(format_metrics(summary).to_string(index=False))


if __name__ == "__main__":
    main()
