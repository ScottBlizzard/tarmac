from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v48_directional_risk_controller_20260527 as v48  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v65_rank_normalized_conformal_autopass_20260527 as v65  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v66_unified_three_layer_workflow_20260527"
FIG_DIR = OUT_DIR / "figures"
V65_FULL = ROOT / "outputs" / "grosspath_rc_v65_rank_normalized_conformal_autopass_20260527" / "v65_full_dev_rank_calibrated_external_eval.csv"
V65_SUMMARY = ROOT / "outputs" / "grosspath_rc_v65_rank_normalized_conformal_autopass_20260527" / "v65_rank_rcps_split_summary.csv"


def configure_matplotlib_font() -> None:
    for font_path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def v50_review_mask(df: pd.DataFrame, scores: dict[str, np.ndarray]) -> np.ndarray:
    base = v30.top_budget(scores["any"], 0.525)
    return v50.add_top_candidates(df, base, scores["direction"], 0.200, "all_direction")


def auto_by_coverage(score: np.ndarray, coverage: float) -> np.ndarray:
    return v65.auto_by_coverage(score, coverage)


def build_safe_auto_specs() -> list[dict[str, object]]:
    full = pd.read_csv(V65_FULL)
    summary = pd.read_csv(V65_SUMMARY)
    specs: list[dict[str, object]] = []

    # Development-calibrated conservative policy: target 5%, maximize full-dev calibrated coverage.
    row = full.loc[(full["target_auto_error"].eq(0.05)) & (full["selector"].eq("safe_by_direction_risk"))].iloc[0]
    specs.append(
        {
            "safe_name": "v65_t05_direction_devmax",
            "selector": row["selector"],
            "coverage": float(row["cal_auto_rate"]),
            "selection_rule": "full-dev target<=5%, max calibrated coverage",
        }
    )

    # Strictest low-error policy: target 5%, any-risk had zero full-dev auto errors and fewer external FPs.
    row = full.loc[(full["target_auto_error"].eq(0.05)) & (full["selector"].eq("safe_by_any_risk"))].iloc[0]
    specs.append(
        {
            "safe_name": "v65_t05_any_zeroerr",
            "selector": row["selector"],
            "coverage": float(row["cal_auto_rate"]),
            "selection_rule": "full-dev target<=5%, zero calibrated auto error",
        }
    )

    # Stability-driven policy chosen only from split summary: target 10%, require usable splits and choose lowest dev-test error.
    candidates = summary.loc[
        summary["target_auto_error"].eq(0.10)
        & summary["no_auto_rate"].lt(0.25)
        & summary["dev_test_auto_rate_median"].ge(0.10)
    ].copy()
    chosen = candidates.sort_values(
        ["dev_test_auto_error_rate_median", "dev_test_auto_rate_median"],
        ascending=[True, False],
    ).iloc[0]
    row = full.loc[(full["target_auto_error"].eq(0.10)) & (full["selector"].eq(chosen["selector"]))].iloc[0]
    specs.append(
        {
            "safe_name": "v65_t10_splitstable",
            "selector": row["selector"],
            "coverage": float(row["cal_auto_rate"]),
            "selection_rule": "split-stable target<=10%, lowest dev-test error",
        }
    )
    return specs


def evaluate_policy(
    df: pd.DataFrame,
    review_mask: np.ndarray,
    policy: str,
    route: np.ndarray | None = None,
    safe_auto: np.ndarray | None = None,
) -> dict[str, object]:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    final = p2.copy()
    final[review_mask] = y[review_mask]
    m = v30.metrics_binary(y, final)
    wrong = p2 != y
    auto = ~review_mask
    row: dict[str, object] = {
        "policy": policy,
        "n": int(len(df)),
        "control_n": int(review_mask.sum()),
        "control_rate": float(review_mask.mean()),
        "auto_n": int(auto.sum()),
        "auto_rate": float(auto.mean()),
        "auto_wrong_n": int((auto & wrong).sum()),
        "auto_fn_n": int((auto & (y == 1) & (p2 == 0)).sum()),
        "auto_fp_n": int((auto & (y == 0) & (p2 == 1)).sum()),
        "captured_wrong_n": int((review_mask & wrong).sum()),
        "remaining_error_n": int((final != y).sum()),
    }
    row.update(m)
    if safe_auto is not None:
        row["safe_auto_n"] = int(safe_auto.sum())
        row["safe_auto_rate"] = float(safe_auto.mean())
        row["safe_auto_inside_v50_review_n"] = int((safe_auto & review_mask).sum())
    if route is not None:
        vc = pd.Series(route).value_counts()
        for key in ["auto_pass", "model_review", "quality_control"]:
            row[f"route_{key}_n"] = int(vc.get(key, 0))
            row[f"route_{key}_rate"] = float(vc.get(key, 0) / len(df))
    return row


def route_cases(
    df: pd.DataFrame,
    review_mask: np.ndarray,
    policy: str,
    route: np.ndarray,
    safe_auto: np.ndarray | None = None,
) -> pd.DataFrame:
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    final = p2.copy()
    final[review_mask] = y[review_mask]
    cols = [
        c
        for c in [
            "case_id",
            "original_case_id",
            "source_folder",
            "task_l6_label",
            "task_l7_label",
            "image_name",
            "quality_status",
            "quality_score",
            "p2_pred",
            "main_prob",
            "robust_prob",
            "prob_mean_core",
        ]
        if c in df.columns
    ]
    out = df[cols].copy()
    out["policy"] = policy
    out["route"] = route
    out["review_or_control"] = review_mask.astype(int)
    if safe_auto is not None:
        out["safe_auto"] = safe_auto.astype(int)
    out["label_idx"] = y
    out["final_pred"] = final
    out["p2_wrong"] = (p2 != y).astype(int)
    out["final_correct"] = (final == y).astype(int)
    out["error_direction"] = np.select(
        [(y == 1) & (p2 == 0), (y == 0) & (p2 == 1)],
        ["FN_high_to_low", "FP_low_to_high"],
        default="correct",
    )
    return out


def build_policy_set(
    df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    v50_review = v50_review_mask(df, scores)
    y = df["label_idx"].to_numpy(dtype=int)
    p2 = df["p2_pred"].to_numpy(dtype=int)
    quality = pd.to_numeric(df["quality_score"], errors="coerce").to_numpy(dtype=float) if "quality_score" in df.columns else np.full(len(df), np.nan)
    q82 = np.isfinite(quality) & (quality <= 82)
    q88 = np.isfinite(quality) & (quality <= 88)

    rows = []
    case_tables = []

    base_route = np.where(v50_review, "model_review", "auto_pass")
    rows.append(evaluate_policy(df, v50_review, "v50_main_high_safety", base_route))
    case_tables.append(route_cases(df, v50_review, "v50_main_high_safety", base_route))

    for qname, qmask in [("quality82", q82), ("quality88", q88)]:
        review = v50_review | qmask
        route = np.where(qmask, "quality_control", np.where(v50_review, "model_review", "auto_pass"))
        rows.append(evaluate_policy(df, review, f"v50_plus_{qname}_exploratory", route))
        case_tables.append(route_cases(df, review, f"v50_plus_{qname}_exploratory", route))

    selector_scores = {
        "safe_by_any_risk": -scores["any"],
        "safe_by_direction_risk": -scores["direction"],
        "safe_by_fn_risk": -scores["fn"],
    }
    # Include these only when the required model scores are available. The selector list intentionally mirrors v65.
    for spec in build_safe_auto_specs():
        name = str(spec["safe_name"])
        selector = str(spec["selector"])
        coverage = float(spec["coverage"])
        if selector not in selector_scores:
            continue
        safe = auto_by_coverage(selector_scores[selector], coverage)

        # Strict mode: only consensus-safe cases auto-pass; all other cases are controlled.
        strict_review = ~safe
        strict_route = np.where(safe, "auto_pass", "model_review")
        rows.append(evaluate_policy(df, strict_review, f"{name}_safe_auto_only", strict_route, safe))
        case_tables.append(route_cases(df, strict_review, f"{name}_safe_auto_only", strict_route, safe))

        # Override mode: v65 safe cases can skip v50 review, reducing review burden.
        override_review = v50_review & (~safe)
        override_route = np.where(~override_review, "auto_pass", "model_review")
        rows.append(evaluate_policy(df, override_review, f"v50_override_by_{name}", override_route, safe))
        case_tables.append(route_cases(df, override_review, f"v50_override_by_{name}", override_route, safe))

        # Guarded mode: v50 auto-pass cases must also be in the calibrated safe set.
        guarded_review = v50_review | (~safe)
        guarded_route = np.where(guarded_review, "model_review", "auto_pass")
        rows.append(evaluate_policy(df, guarded_review, f"v50_guarded_by_{name}", guarded_route, safe))
        case_tables.append(route_cases(df, guarded_review, f"v50_guarded_by_{name}", guarded_route, safe))

        if split == "external":
            # External quality overlay is descriptive/exploratory because the threshold needs prospective validation.
            combo_review = (v50_review & (~safe)) | q82
            combo_route = np.where(q82, "quality_control", np.where(combo_review, "model_review", "auto_pass"))
            rows.append(evaluate_policy(df, combo_review, f"v50_override_by_{name}_plus_quality82_exploratory", combo_route, safe))
            case_tables.append(route_cases(df, combo_review, f"v50_override_by_{name}_plus_quality82_exploratory", combo_route, safe))

    summary = pd.DataFrame(rows)
    summary.insert(1, "split", split)
    cases = pd.concat(case_tables, ignore_index=True)
    cases.insert(1, "split", split)
    return summary, cases


def make_plot(summary: pd.DataFrame) -> None:
    configure_matplotlib_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ext = summary.loc[summary["split"].eq("external")].copy()
    label_map = {
        "v50_main_high_safety": "v50 main",
        "v50_plus_quality82_exploratory": "v50 + quality82",
        "v50_plus_quality88_exploratory": "v50 + quality88",
        "v65_t05_any_zeroerr_safe_auto_only": "v65 safe-only",
    }
    focus = ext.loc[ext["policy"].isin(label_map)].copy()
    focus = focus.sort_values("control_rate")

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    colors = np.where(focus["policy"].str.contains("quality"), "#8e44ad", np.where(focus["policy"].str.contains("guarded|safe_auto_only"), "#c0392b", "#1f618d"))
    ax.scatter(focus["control_rate"] * 100, focus["balanced_accuracy"] * 100, s=84, c=colors, edgecolor="white", linewidth=0.7)
    for _, row in focus.iterrows():
        ax.text(row["control_rate"] * 100 + 0.35, row["balanced_accuracy"] * 100 + 0.04, label_map[row["policy"]], fontsize=8)
    ax.axhline(95, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(97, color="#7d6608", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("External review / control rate (%)")
    ax.set_ylabel("External workflow BAcc (%)")
    ax.set_title("Unified three-layer workflow candidates")
    ax.set_xlim(70, 91)
    ax.set_ylim(96.8, 100.2)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v66_unified_workflow_tradeoff.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v66_unified_workflow_tradeoff.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev, ext, dev_scores, ext_scores = v50.get_scores()
    dev_summary, dev_cases = build_policy_set(dev, dev_scores, "development")
    ext_summary, ext_cases = build_policy_set(ext, ext_scores, "external")
    summary = pd.concat([dev_summary, ext_summary], ignore_index=True)
    cases = pd.concat([dev_cases, ext_cases], ignore_index=True)

    summary.to_csv(OUT_DIR / "v66_unified_workflow_summary.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(OUT_DIR / "v66_unified_workflow_case_routes.csv", index=False, encoding="utf-8-sig")
    make_plot(summary)

    show = summary.loc[summary["split"].eq("external"), [
        "policy",
        "control_rate",
        "auto_rate",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "fn",
        "fp",
        "auto_wrong_n",
        "auto_fn_n",
        "auto_fp_n",
    ]].sort_values(["balanced_accuracy", "control_rate"], ascending=[False, True])
    print(show.to_string(index=False))
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
