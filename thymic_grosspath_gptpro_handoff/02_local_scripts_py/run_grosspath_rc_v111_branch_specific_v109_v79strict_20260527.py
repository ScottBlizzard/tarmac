from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v110_high_safety_branch_with_v109_scorecard_20260527 as v110  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v111_branch_specific_v109_v79strict_20260527"


def evaluate_custom(policy_name: str, frame: pd.DataFrame, extra: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    review = frame["review_or_control"].astype(bool) | extra
    flagged = frame.copy()
    flagged["v111_extra_review"] = extra
    flagged["v111_review_or_control"] = review
    flagged["v111_rescued_error"] = extra & frame["final_correct"].eq(0)
    flagged["v111_extra_clean_review"] = extra & frame["final_correct"].eq(1)
    rows = []
    scopes = [("all_domains", frame)] + list(frame.groupby("domain", sort=False))
    internal = frame[frame["domain"].isin(["old_data", "third_batch"])]
    if not internal.empty:
        scopes.insert(1, ("internal_all", internal))
    for scope, sub in scopes:
        sub_extra = extra.loc[sub.index]
        row = {
            "workflow": policy_name,
            "scope": scope,
            "prob_threshold": v110.PROB_THRESHOLD,
            "core_max": v110.CORE_MAX,
            "extra_review_n": int(sub_extra.sum()),
            "extra_captured_error_n": int((sub_extra & sub["final_correct"].eq(0)).sum()),
            "extra_clean_review_n": int((sub_extra & sub["final_correct"].eq(1)).sum()),
        }
        row.update(v110.metrics(sub, review.loc[sub.index]))
        rows.append(row)
    return pd.DataFrame(rows), flagged


def write_key_messages(summary: pd.DataFrame) -> None:
    all_row = summary[summary["scope"].eq("all_domains")].iloc[0]
    internal = summary[summary["scope"].eq("internal_all")].iloc[0]
    third = summary[summary["scope"].eq("third_batch")].iloc[0]
    external = summary[summary["scope"].eq("strict_external")].iloc[0]
    lines = [
        "# v111 Branch-specific v109 + v79-strict",
        "",
        "Internal-like batches use v109 whole-crop scorecard; severe-shift strict external uses v79-strict without the extra scorecard.",
        "",
        f"- All domains: BAcc {v110.pct(all_row['balanced_accuracy'])}, control {v110.pct(all_row['control_rate'])}, FN={int(all_row['fn'])}, FP={int(all_row['fp'])}.",
        f"- Internal all: BAcc {v110.pct(internal['balanced_accuracy'])}, control {v110.pct(internal['control_rate'])}, FN={int(internal['fn'])}, FP={int(internal['fp'])}.",
        f"- Third batch: BAcc {v110.pct(third['balanced_accuracy'])}, control {v110.pct(third['control_rate'])}, FN={int(third['fn'])}, FP={int(third['fp'])}.",
        f"- Strict external: BAcc {v110.pct(external['balanced_accuracy'])}, control {v110.pct(external['control_rate'])}, FN={int(external['fn'])}, FP={int(external['fp'])}.",
        "",
        "This is cleaner than applying the scorecard again on the severe-shift branch, because v79-strict already removes the remaining strict-external FP.",
    ]
    (OUT_DIR / "v111_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(v110.ROUTES, dtype={"case_id": str, "original_case_id": str})
    frame = v110.attach_wholecrop_prob(routes[routes["policy"].eq(v110.STRICT_POLICY)].copy())

    extra = v110.apply_scorecard(frame)
    extra.loc[frame["domain"].eq("strict_external")] = False
    summary, cases = evaluate_custom("internal_v109_external_v79strict", frame, extra)

    summary.to_csv(OUT_DIR / "v111_branch_specific_summary.csv", index=False, encoding="utf-8-sig")
    v110.format_table(summary).to_csv(OUT_DIR / "v111_branch_specific_summary_formatted.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v111_branch_specific_cases_with_flags.csv", index=False, encoding="utf-8-sig")
    write_key_messages(summary)

    focus = summary[summary["scope"].isin(["all_domains", "internal_all", "third_batch", "strict_external"])]
    print("Wrote", OUT_DIR)
    print(v110.format_table(focus).to_string(index=False))


if __name__ == "__main__":
    main()
