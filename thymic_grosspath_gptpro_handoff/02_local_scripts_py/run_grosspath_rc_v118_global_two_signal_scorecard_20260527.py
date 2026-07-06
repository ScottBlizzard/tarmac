from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v114_nested_directional_guards_20260527 as v114  # noqa: E402
import run_grosspath_rc_v117_two_signal_bidirectional_scorecard_20260527 as v117  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v118_global_two_signal_scorecard_20260527"


def select_global_rule(internal: pd.DataFrame) -> pd.Series:
    base_review = internal["v111_review_or_control"].astype(bool)
    scan = v117.scan_two_signal_rules(internal, base_review)
    candidates = scan[(scan["fp"].eq(0)) & (scan["fn"].le(1))].copy()
    if candidates.empty:
        candidates = scan.copy()
    return candidates.sort_values(
        ["extra_review_n", "clean_review_n", "remaining_error_n", "fp", "fn", "wc_max", "core_max"],
        ascending=[True, True, True, True, True, True, True],
    ).iloc[0]


def evaluate_scope_rows(df: pd.DataFrame, review_col: str, workflow: str) -> pd.DataFrame:
    rows = []
    for scope, sub in [
        ("all_domains", df),
        ("internal_all", df[df["domain"].isin(["old_data", "third_batch"])]),
        ("old_data", df[df["domain"].eq("old_data")]),
        ("third_batch", df[df["domain"].eq("third_batch")]),
        ("strict_external", df[df["domain"].eq("strict_external")]),
    ]:
        row = {"workflow": workflow, "scope": scope}
        row.update(v114.metrics(sub, sub[review_col].astype(bool)))
        rows.append(row)
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("_rate") or col in ["sensitivity", "specificity", "balanced_accuracy"]:
            out[col] = out[col].map(v114.pct)
    return out


def plot_decision_boundary(df: pd.DataFrame, wc_max: float, core_max: float) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    base_review = df["v111_review_or_control"].astype(bool)
    auto_high = df["domain"].isin(["old_data", "third_batch"]) & (~base_review) & df["final_pred"].eq(1)
    sub = df[auto_high].copy()
    sub["plot_group"] = np.where(sub["final_correct"].eq(0), "rescued FP", "clean auto-high")
    colors = {"clean auto-high": "#8fb3d9", "rescued FP": "#d14b3a"}

    plt.figure(figsize=(7.2, 5.4))
    for group, part in sub.groupby("plot_group"):
        plt.scatter(
            part["wholecrop_prob"],
            part["prob_mean_core"],
            s=46 if group == "rescued FP" else 30,
            c=colors[group],
            label=f"{group} (n={len(part)})",
            edgecolors="black" if group == "rescued FP" else "none",
            linewidths=0.8,
            alpha=0.9,
        )
    plt.axvline(wc_max, color="#2d2d2d", linestyle="--", linewidth=1.2)
    plt.axhline(core_max, color="#2d2d2d", linestyle="--", linewidth=1.2)
    plt.fill_between([0, wc_max], [0, 0], [core_max, core_max], color="#d14b3a", alpha=0.08, label="review zone")
    plt.xlabel("Whole-crop high-risk probability")
    plt.ylabel("Core-model mean high-risk probability")
    plt.title("v118 two-signal low-risk FP guard")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.22)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "v118_two_signal_decision_boundary.png", dpi=220)
    plt.savefig(fig_dir / "v118_two_signal_decision_boundary.pdf")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(v114.V112_CASES, dtype={"case_id": str, "original_case_id": str})
    for col in ["label_idx", "final_pred", "final_correct"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(int)
    raw["v111_review_or_control"] = v114.as_bool(raw["v111_review_or_control"])
    df = v114.attach_fold_and_crop(raw)

    internal = df[df["domain"].isin(["old_data", "third_batch"])].copy()
    selected = select_global_rule(internal)
    wc_max = float(selected["wc_max"])
    core_max = float(selected["core_max"])

    df["v118_extra_review"] = v117.fp_guard_two_signal(df, df["v111_review_or_control"].astype(bool), wc_max, core_max)
    df["v118_review_or_control"] = df["v111_review_or_control"].astype(bool) | df["v118_extra_review"].astype(bool)
    df["v118_rescued_error"] = df["v118_extra_review"].astype(bool) & df["final_correct"].eq(0)
    df["v118_extra_clean_review"] = df["v118_extra_review"].astype(bool) & df["final_correct"].eq(1)

    summary = pd.concat(
        [
            evaluate_scope_rows(df, "v111_review_or_control", "v111_base"),
            evaluate_scope_rows(df, "v118_review_or_control", "global_two_signal_scorecard"),
        ],
        ignore_index=True,
    )
    rule = pd.DataFrame(
        [
            {
                "wc_max": wc_max,
                "core_max": core_max,
                "train_extra_review_n": int(selected["extra_review_n"]),
                "train_captured_error_n": int(selected["captured_error_n"]),
                "train_clean_review_n": int(selected["clean_review_n"]),
                "train_remaining_error_n": int(selected["remaining_error_n"]),
                "train_fn": int(selected["fn"]),
                "train_fp": int(selected["fp"]),
            }
        ]
    )

    summary.to_csv(OUT_DIR / "v118_global_two_signal_summary.csv", index=False, encoding="utf-8-sig")
    format_table(summary).to_csv(OUT_DIR / "v118_global_two_signal_summary_formatted.csv", index=False, encoding="utf-8-sig")
    rule.to_csv(OUT_DIR / "v118_global_two_signal_selected_rule.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT_DIR / "v118_global_two_signal_cases.csv", index=False, encoding="utf-8-sig")
    plot_decision_boundary(df, wc_max, core_max)

    all_row = summary[(summary["workflow"].eq("global_two_signal_scorecard")) & (summary["scope"].eq("all_domains"))].iloc[0]
    third = summary[(summary["workflow"].eq("global_two_signal_scorecard")) & (summary["scope"].eq("third_batch"))].iloc[0]
    rescued = df[df["v118_rescued_error"]][["domain", "fold_id", "original_case_id", "task_l6_label"]].astype(str)
    rescued_text = "; ".join(
        f"{r.domain}:fold{r.fold_id}:{r.original_case_id}/{r.task_l6_label}" for r in rescued.itertuples(index=False)
    )
    lines = [
        "# v118 Global Two-signal Scorecard",
        "",
        f"Selected fixed low-risk FP guard: wholecrop_prob <= {wc_max:.3f} and prob_mean_core <= {core_max:.3f}.",
        "",
        f"- All-domain: BAcc {v114.pct(all_row['balanced_accuracy'])}, control {v114.pct(all_row['control_rate'])}, FN={int(all_row['fn'])}, FP={int(all_row['fp'])}.",
        f"- Third batch: BAcc {v114.pct(third['balanced_accuracy'])}, control {v114.pct(third['control_rate'])}, FN={int(third['fn'])}, FP={int(third['fp'])}.",
        f"- Rescued FP cases: {rescued_text}.",
        "",
        "This is the deployable fixed-rule version after v117 nested validation. It is simpler than v115 and slightly lower-control than the fold-wise v117 envelope on the current data.",
    ]
    (OUT_DIR / "v118_key_messages.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote", OUT_DIR)
    print(format_table(summary[summary["scope"].isin(["all_domains", "internal_all", "third_batch", "strict_external"])]).to_string(index=False))
    print()
    print(rule.to_string(index=False))
    print()
    print("rescued:")
    print(rescued.to_string(index=False))


if __name__ == "__main__":
    main()
