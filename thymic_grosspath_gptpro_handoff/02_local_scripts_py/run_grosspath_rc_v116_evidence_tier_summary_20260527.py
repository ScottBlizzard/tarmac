from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v116_evidence_tier_summary_20260527"
FIG_DIR = OUT_DIR / "figures"


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def add_rows(rows: list[dict[str, object]], source: pd.DataFrame, workflow: str, label: str, tier: str, source_file: str) -> None:
    for scope in ["all_domains", "internal_all", "old_data", "third_batch", "strict_external"]:
        sub = source[(source["workflow"].eq(workflow) if "workflow" in source.columns else source["policy_label"].eq(workflow)) & source["scope"].eq(scope)]
        if sub.empty:
            continue
        r = sub.iloc[0].to_dict()
        rows.append(
            {
                "workflow_label": label,
                "evidence_tier": tier,
                "scope": scope,
                "control_rate": float(r["control_rate"]),
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "remaining_error_n": int(r["remaining_error_n"]),
                "fn": int(r["fn"]),
                "fp": int(r["fp"]),
                "source_file": source_file,
            }
        )


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["control_rate", "balanced_accuracy"]:
        out[col] = out[col].map(pct)
    return out


def make_plot(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    focus = df[df["scope"].eq("all_domains")].copy()
    order = [
        "v91 main locked",
        "v111 branch high-safety",
        "v115 nested stable candidate",
        "v113 post-hoc upper bound",
    ]
    focus["workflow_label"] = pd.Categorical(focus["workflow_label"], categories=order, ordered=True)
    focus = focus.sort_values("workflow_label")

    fig, ax1 = plt.subplots(figsize=(9.2, 4.8))
    x = range(len(focus))
    ax1.bar([i - 0.18 for i in x], focus["balanced_accuracy"] * 100, width=0.36, color="#1B5E20", label="BAcc")
    ax1.bar([i + 0.18 for i in x], focus["control_rate"] * 100, width=0.36, color="#78909C", label="Control")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(focus["workflow_label"], rotation=20, ha="right")
    ax1.set_ylabel("%")
    ax1.set_ylim(70, 102)
    ax1.set_title("Task7 evidence tiers: accuracy-control tradeoff")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v116_evidence_tier_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v116_evidence_tier_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def write_key_messages(df: pd.DataFrame) -> None:
    def row(label: str, scope: str = "all_domains") -> pd.Series:
        return df[(df["workflow_label"].eq(label)) & (df["scope"].eq(scope))].iloc[0]

    v91 = row("v91 main locked")
    v115 = row("v115 nested stable candidate")
    v113 = row("v113 post-hoc upper bound")
    lines = [
        "# v116 Evidence-tier Summary",
        "",
        "- v91 remains the clean locked main workflow.",
        f"- v91 all-domain: BAcc {pct(v91['balanced_accuracy'])}, control {pct(v91['control_rate'])}, FN={int(v91['fn'])}, FP={int(v91['fp'])}.",
        "- v115 is the strongest fold-wise supported high-safety candidate.",
        f"- v115 all-domain: BAcc {pct(v115['balanced_accuracy'])}, control {pct(v115['control_rate'])}, FN={int(v115['fn'])}, FP={int(v115['fp'])}.",
        "- v113 is the post-hoc upper bound and should not be written as the unbiased main result.",
        f"- v113 all-domain: BAcc {pct(v113['balanced_accuracy'])}, control {pct(v113['control_rate'])}, FN={int(v113['fn'])}, FP={int(v113['fp'])}.",
        "",
        "Recommended writing hierarchy: v91 as locked primary workflow, v115 as nested high-safety candidate, v113 as upper-bound/ablation evidence.",
    ]
    (OUT_DIR / "v116_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    v91 = read_csv(ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_summary.csv")
    add_rows(rows, v91, "Batch-adaptive main", "v91 main locked", "locked primary", "v91_integrated_summary.csv")
    add_rows(rows, v91, "Fixed v79-strict", "v79 strict fixed", "locked high-review reference", "v91_integrated_summary.csv")

    v111 = read_csv(ROOT / "outputs" / "grosspath_rc_v111_branch_specific_v109_v79strict_20260527" / "v111_branch_specific_summary.csv")
    add_rows(rows, v111, "internal_v109_external_v79strict", "v111 branch high-safety", "post-v109 candidate", "v111_branch_specific_summary.csv")

    v112 = read_csv(ROOT / "outputs" / "grosspath_rc_v112_symmetric_lowrisk_guard_20260527" / "v112_lowrisk_guard_summary.csv")
    add_rows(rows, v112, "v111_plus_v112_symmetric_lowrisk_guard", "v112 post-hoc symmetric guard", "post-hoc internal candidate", "v112_lowrisk_guard_summary.csv")

    v113 = read_csv(ROOT / "outputs" / "grosspath_rc_v113_triple_guard_zero_error_candidate_20260527" / "v113_triple_guard_summary.csv")
    add_rows(rows, v113, "v111_v112_plus_v113_crop_rescue", "v113 post-hoc upper bound", "post-hoc upper bound", "v113_triple_guard_summary.csv")

    v114 = read_csv(ROOT / "outputs" / "grosspath_rc_v114_nested_directional_guards_20260527" / "v114_nested_directional_guard_summary.csv")
    add_rows(rows, v114, "nested_v112_plus_nested_v113", "v114 nested narrow guard", "fold-wise validation", "v114_nested_directional_guard_summary.csv")

    v115 = read_csv(ROOT / "outputs" / "grosspath_rc_v115_nested_stable_envelope_fp_guard_20260527" / "v115_nested_stable_envelope_summary.csv")
    add_rows(rows, v115, "nested_stable_envelope_fp_guard", "v115 nested stable candidate", "fold-wise validation", "v115_nested_stable_envelope_summary.csv")

    df = pd.DataFrame(rows)
    order = [
        "v91 main locked",
        "v79 strict fixed",
        "v111 branch high-safety",
        "v112 post-hoc symmetric guard",
        "v113 post-hoc upper bound",
        "v114 nested narrow guard",
        "v115 nested stable candidate",
    ]
    scope_order = ["all_domains", "internal_all", "old_data", "third_batch", "strict_external"]
    df["workflow_label"] = pd.Categorical(df["workflow_label"], categories=order, ordered=True)
    df["scope"] = pd.Categorical(df["scope"], categories=scope_order, ordered=True)
    df = df.sort_values(["workflow_label", "scope"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "v116_evidence_tier_summary.csv", index=False, encoding="utf-8-sig")
    format_table(df).to_csv(OUT_DIR / "v116_evidence_tier_summary_formatted.csv", index=False, encoding="utf-8-sig")
    make_plot(df)
    write_key_messages(df)

    print("Wrote", OUT_DIR)
    print(format_table(df[df["scope"].eq("all_domains")]).to_string(index=False))


if __name__ == "__main__":
    main()
