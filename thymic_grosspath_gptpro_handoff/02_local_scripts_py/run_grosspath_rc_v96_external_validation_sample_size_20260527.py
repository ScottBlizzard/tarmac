from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V95 = ROOT / "outputs" / "grosspath_rc_v95_selective_risk_confidence_bounds_20260527" / "v95_selective_risk_confidence_bounds.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v96_external_validation_sample_size_20260527"
FIG_DIR = OUT_DIR / "figures"
Z95 = 1.959963984540054

FOCUS_POLICIES = [
    "v50_main",
    "adaptive_v50_to_v79_light",
    "v79_light_lowrisk_guard",
    "v79_strict_lowrisk_guard",
    "quality_direction_uniform90",
]

POLICY_LABELS = {
    "v50_main": "Fixed v50",
    "adaptive_v50_to_v79_light": "Batch-adaptive main",
    "v79_light_lowrisk_guard": "Fixed v79-light",
    "v79_strict_lowrisk_guard": "Fixed v79-strict",
    "quality_direction_uniform90": "Quality+direction uniform90",
}


def wilson_high(errors: int, n: int, z: float = Z95) -> float:
    if n <= 0:
        return np.nan
    p = errors / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    radius = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return min(1.0, (centre + radius) / denom)


def min_n_for_upper(target: float, errors: int, max_n: int = 100_000) -> int:
    n = max(errors, 1)
    while n <= max_n:
        if wilson_high(errors, n) <= target:
            return n
        n += 1
    raise RuntimeError(f"No feasible n up to {max_n} for target={target}, errors={errors}")


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:.2f}%"


def build_fresh_requirements() -> pd.DataFrame:
    rows = []
    for target in [0.10, 0.05]:
        for errors in range(0, 6):
            n = min_n_for_upper(target, errors)
            rows.append(
                {
                    "target_upper": target,
                    "allowed_auto_errors": errors,
                    "required_auto_passed_cases": n,
                    "observed_error_rate": errors / n,
                    "wilson_upper": wilson_high(errors, n),
                }
            )
    return pd.DataFrame(rows)


def build_workflow_shortfall(v95: pd.DataFrame) -> pd.DataFrame:
    ext = v95.loc[(v95["scope"] == "strict_external") & (v95["policy"].isin(FOCUS_POLICIES))].copy()
    rows = []
    for _, row in ext.iterrows():
        auto_rate = float(row["auto_rate"])
        current_auto_n = int(row["auto_n"])
        current_errors = int(row["auto_wrong_n"])
        for target in [0.10, 0.05]:
            required_auto_n_if_errors_hold = min_n_for_upper(target, current_errors)
            additional_auto_if_no_more_errors = max(0, required_auto_n_if_errors_hold - current_auto_n)
            estimated_total_external_cases = ceil(required_auto_n_if_errors_hold / auto_rate) if auto_rate > 0 else np.nan
            estimated_additional_external_cases = ceil(additional_auto_if_no_more_errors / auto_rate) if auto_rate > 0 else np.nan
            rows.append(
                {
                    "policy": row["policy"],
                    "policy_label": row["policy_label"],
                    "target_upper": target,
                    "current_auto_n": current_auto_n,
                    "current_auto_errors": current_errors,
                    "current_auto_error_rate": float(row["auto_pass_error_risk"]),
                    "current_wilson_upper": float(row["auto_pass_error_risk_wilson_high"]),
                    "strict_external_auto_rate": auto_rate,
                    "required_auto_n_if_errors_hold": required_auto_n_if_errors_hold,
                    "additional_auto_if_no_more_errors": additional_auto_if_no_more_errors,
                    "estimated_total_external_cases_at_current_auto_rate": estimated_total_external_cases,
                    "estimated_additional_external_cases_at_current_auto_rate": estimated_additional_external_cases,
                }
            )
    out = pd.DataFrame(rows)
    out["policy"] = pd.Categorical(out["policy"], categories=FOCUS_POLICIES, ordered=True)
    return out.sort_values(["target_upper", "policy"]).reset_index(drop=True)


def format_outputs(fresh: pd.DataFrame, shortfall: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fresh_fmt = fresh.copy()
    for col in ["target_upper", "observed_error_rate", "wilson_upper"]:
        fresh_fmt[col] = fresh_fmt[col].map(pct)

    short_fmt = shortfall.copy()
    for col in [
        "target_upper",
        "current_auto_error_rate",
        "current_wilson_upper",
        "strict_external_auto_rate",
    ]:
        short_fmt[col] = short_fmt[col].map(pct)
    return fresh_fmt, short_fmt


def make_plots(fresh: pd.DataFrame, shortfall: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for target, sub in fresh.groupby("target_upper"):
        ax.plot(
            sub["allowed_auto_errors"],
            sub["required_auto_passed_cases"],
            marker="o",
            label=f"Wilson upper <= {target * 100:.0f}%",
        )
    ax.set_xlabel("Allowed auto-pass errors in a fresh validation cohort")
    ax.set_ylabel("Required auto-passed cases")
    ax.set_title("Prospective selective-safety validation size")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v96_fresh_validation_auto_n_requirements.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v96_fresh_validation_auto_n_requirements.pdf", bbox_inches="tight")
    plt.close(fig)

    plot = shortfall.loc[shortfall["target_upper"] == 0.10].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = plot["policy_label"].tolist()
    vals = plot["estimated_additional_external_cases_at_current_auto_rate"].to_numpy(float)
    ax.bar(labels, vals, color="#4C78A8")
    ax.set_ylabel("Estimated additional external cases")
    ax.set_title("Extra external cases needed if no more auto-pass errors occur\nTarget: Wilson upper <= 10%")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v96_additional_external_cases_target10.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v96_additional_external_cases_target10.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(fresh: pd.DataFrame, shortfall: pd.DataFrame) -> None:
    zero_10 = fresh.loc[(fresh["target_upper"] == 0.10) & (fresh["allowed_auto_errors"] == 0)].iloc[0]
    zero_5 = fresh.loc[(fresh["target_upper"] == 0.05) & (fresh["allowed_auto_errors"] == 0)].iloc[0]
    one_10 = fresh.loc[(fresh["target_upper"] == 0.10) & (fresh["allowed_auto_errors"] == 1)].iloc[0]
    one_5 = fresh.loc[(fresh["target_upper"] == 0.05) & (fresh["allowed_auto_errors"] == 1)].iloc[0]

    light_10 = shortfall.loc[
        (shortfall["policy"] == "v79_light_lowrisk_guard") & (shortfall["target_upper"] == 0.10)
    ].iloc[0]
    strict_10 = shortfall.loc[
        (shortfall["policy"] == "v79_strict_lowrisk_guard") & (shortfall["target_upper"] == 0.10)
    ].iloc[0]
    qd_10 = shortfall.loc[
        (shortfall["policy"] == "quality_direction_uniform90") & (shortfall["target_upper"] == 0.10)
    ].iloc[0]

    lines = [
        "# v96 External Validation Sample-size Planning",
        "",
        "## Key Numbers",
        "",
        f"- If a fresh prospective external validation cohort has 0 auto-pass errors, at least {int(zero_10['required_auto_passed_cases'])} auto-passed cases are needed for Wilson upper <=10%, and {int(zero_5['required_auto_passed_cases'])} auto-passed cases for Wilson upper <=5%.",
        f"- If 1 auto-pass error is allowed, the requirement rises to {int(one_10['required_auto_passed_cases'])} auto-passed cases for <=10%, and {int(one_5['required_auto_passed_cases'])} for <=5%.",
        f"- Fixed v79-light currently has 19 strict-external auto-passed cases with 1 error. If no additional auto-pass errors occur, it needs {int(light_10['additional_auto_if_no_more_errors'])} more auto-passed cases, about {int(light_10['estimated_additional_external_cases_at_current_auto_rate'])} more total external cases at the current auto-pass rate, to bring the cumulative Wilson upper below 10%.",
        f"- Fixed v79-strict currently has 14 strict-external auto-passed cases with 0 errors. It needs {int(strict_10['additional_auto_if_no_more_errors'])} more auto-passed cases, about {int(strict_10['estimated_additional_external_cases_at_current_auto_rate'])} more total external cases at the current auto-pass rate, to bring the cumulative Wilson upper below 10%.",
        f"- Quality+direction uniform90 has strong all-domain safety but only 11 strict-external auto-passed cases with 1 error; under the current strict-external auto-pass rate, the estimated additional total cases for <=10% is {int(qd_10['estimated_additional_external_cases_at_current_auto_rate'])}.",
        "",
        "## Interpretation",
        "",
        "This is a validation-design analysis, not a new tuned model result. It shows that the current strict external set is enough to demonstrate a strong trend, but not enough to provide a tight statistical guarantee for auto-pass safety. A paper can report the confidence-bounded selective-diagnosis framework now, while positioning a larger prospective external cohort as the next validation step.",
        "",
    ]
    (OUT_DIR / "v96_key_messages.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v95 = pd.read_csv(V95)
    fresh = build_fresh_requirements()
    shortfall = build_workflow_shortfall(v95)
    fresh_fmt, short_fmt = format_outputs(fresh, shortfall)

    fresh.to_csv(OUT_DIR / "v96_fresh_validation_auto_n_requirements.csv", index=False, encoding="utf-8-sig")
    shortfall.to_csv(OUT_DIR / "v96_workflow_validation_shortfall.csv", index=False, encoding="utf-8-sig")
    fresh_fmt.to_csv(OUT_DIR / "v96_fresh_validation_auto_n_requirements_formatted.csv", index=False, encoding="utf-8-sig")
    short_fmt.to_csv(OUT_DIR / "v96_workflow_validation_shortfall_formatted.csv", index=False, encoding="utf-8-sig")
    make_plots(fresh, shortfall)
    write_summary(fresh, shortfall)

    print("Wrote", OUT_DIR)
    print(fresh_fmt.to_string(index=False))
    print()
    print(short_fmt.to_string(index=False))


if __name__ == "__main__":
    main()
