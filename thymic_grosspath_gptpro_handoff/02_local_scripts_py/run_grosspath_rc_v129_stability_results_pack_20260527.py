from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v129_stability_results_pack_20260527"

V125_DIR = ROOT / "outputs" / "grosspath_rc_v125_paper_results_pack_20260527"
V126_SUMMARY = ROOT / "outputs" / "grosspath_rc_v126_leave_domain_two_signal_validation_20260527" / "v126_leave_domain_summary.csv"
V127_SUMMARY = ROOT / "outputs" / "grosspath_rc_v127_leave_domain_stable_envelope_20260527" / "v127_leave_domain_stable_summary.csv"
V128_SUMMARY = ROOT / "outputs" / "grosspath_rc_v128_capped_leave_domain_envelope_20260527" / "v128_capped_leave_domain_summary.csv"
V128_RULES = ROOT / "outputs" / "grosspath_rc_v128_capped_leave_domain_envelope_20260527" / "v128_capped_leave_domain_rules.csv"


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy", "holdout_bacc"]:
            out[col] = out[col].map(pct)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Keep v125 as the core Results pack and add stability-specific material.
    for src in V125_DIR.glob("*.csv"):
        shutil.copy2(src, OUT_DIR / src.name)
    for src in (V125_DIR / "figures").glob("*"):
        shutil.copy2(src, fig_dir / src.name)

    v126 = pd.read_csv(V126_SUMMARY)
    v127 = pd.read_csv(V127_SUMMARY)
    v128 = pd.read_csv(V128_SUMMARY)
    v128_rules = pd.read_csv(V128_RULES)

    stability_rows = []
    specs = [
        ("v126_min_leave_domain", v126, "leave_domain_selected_two_signal"),
        ("v127_unbounded_stable_envelope", v127, "leave_domain_stable_envelope"),
        ("v128_capped_stable_envelope", v128, "capped_leave_domain_envelope"),
    ]
    for label, frame, workflow in specs:
        row = frame[(frame["workflow"].eq(workflow)) & (frame["scope"].eq("all_domains"))].iloc[0].to_dict()
        stability_rows.append({"stability_workflow": label, **row})
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(OUT_DIR / "v129_leave_domain_stability_table.csv", index=False, encoding="utf-8-sig")
    format_metrics(stability).to_csv(OUT_DIR / "v129_leave_domain_stability_table_formatted.csv", index=False, encoding="utf-8-sig")
    v128_rules.to_csv(OUT_DIR / "v129_capped_envelope_rules.csv", index=False, encoding="utf-8-sig")
    format_metrics(v128_rules).to_csv(OUT_DIR / "v129_capped_envelope_rules_formatted.csv", index=False, encoding="utf-8-sig")

    v126_row = stability[stability["stability_workflow"].eq("v126_min_leave_domain")].iloc[0]
    v127_row = stability[stability["stability_workflow"].eq("v127_unbounded_stable_envelope")].iloc[0]
    v128_row = stability[stability["stability_workflow"].eq("v128_capped_stable_envelope")].iloc[0]
    lines = [
        "# v129 Stability Results Pack",
        "",
        "本包在 v125 Results 固定入口基础上，补入 v126-v128 内部留域稳定性证据。",
        "",
        "## 留域稳定性",
        "",
        f"- v126 最窄留域规则：BAcc {pct(v126_row['balanced_accuracy'])}，控制率 {pct(v126_row['control_rate'])}，FN={int(v126_row['fn'])}，FP={int(v126_row['fp'])}。",
        f"- v127 无上限稳定包络：BAcc {pct(v127_row['balanced_accuracy'])}，控制率 {pct(v127_row['control_rate'])}，FN={int(v127_row['fn'])}，FP={int(v127_row['fp'])}；能清 FP，但复核过高。",
        f"- v128 带安全平台上限包络：BAcc {pct(v128_row['balanced_accuracy'])}，控制率 {pct(v128_row['control_rate'])}，FN={int(v128_row['fn'])}，FP={int(v128_row['fp'])}。",
        "",
        "## 写作建议",
        "",
        "v118/v119 仍是主写的高安全候选；v128 可作为稳定性补充，说明 two-signal FP guard 在内部留域下也能通过“安全平台 + 包络选择”维持 FP=0，但代价是控制率略高于 v118。",
    ]
    (OUT_DIR / "v129_stability_results_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")

    print("Wrote", OUT_DIR)
    print(format_metrics(stability).to_string(index=False))
    print()
    print(format_metrics(v128_rules).to_string(index=False))


if __name__ == "__main__":
    main()
