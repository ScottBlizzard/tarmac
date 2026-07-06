from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    HAS_MPL = True
except ModuleNotFoundError:
    plt = None
    PdfPages = None
    HAS_MPL = False


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v152_framework_evidence_pack_20260527"
FIG_DIR = OUT_DIR / "figures"

V141 = ROOT / "outputs" / "grosspath_rc_v141_image_concept_fusion_probe_20260527" / "image_concept_fusion_oof_summary.csv"
V142_RISK = ROOT / "outputs" / "grosspath_rc_v142_concept_guided_error_router_20260527" / "v142_error_risk_auc_summary.csv"
V143_RISK = ROOT / "outputs" / "grosspath_rc_v143_image_feature_error_router_20260527" / "v143_image_feature_error_risk_summary.csv"
V144_RISK = ROOT / "outputs" / "grosspath_rc_v144_distilled_concept_router_20260527" / "v144_distilled_concept_router_risk_summary.csv"
V144_CONCEPT = ROOT / "outputs" / "grosspath_rc_v144_distilled_concept_router_20260527" / "v144_predicted_concept_performance.csv"
V147 = ROOT / "outputs" / "grosspath_rc_v147_unlabeled_shift_aware_router_card_20260527" / "v147_shift_aware_policy_budget_summary.csv"
V148_COMP = ROOT / "outputs" / "grosspath_rc_v148_router_card_comparative_evidence_20260527" / "v148_policy_vs_low_conf_paired_comparison.csv"
V148_RAND = ROOT / "outputs" / "grosspath_rc_v148_router_card_comparative_evidence_20260527" / "v148_random_review_baseline.csv"
V149 = ROOT / "outputs" / "grosspath_rc_v149_directional_auto_flip_corrector_20260527" / "v149_auto_flip_corrector_summary.csv"
V150 = ROOT / "outputs" / "grosspath_rc_v150_severe_shift_hybrid_correct_review_20260527" / "v150_hybrid_correct_review_summary.csv"
V151 = ROOT / "outputs" / "grosspath_rc_v151_no_leak_autoflip_budget_selection_20260527" / "v151_no_leak_selected_autoflip_external_check.csv"
V77 = ROOT / "outputs" / "grosspath_rc_v77_batch_shift_audit_policy_switch_20260527" / "v77_unlabeled_batch_shift_audit.csv"


def pct(x: float | int | None, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100 * float(x):.{digits}f}%"


def pp(x: float | int | None, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100 * float(x):+.{digits}f} pp"


def pick(df: pd.DataFrame, **conds: object) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for k, v in conds.items():
        if isinstance(v, float):
            mask &= np.isclose(df[k].astype(float), v)
        else:
            mask &= df[k].eq(v)
    sub = df.loc[mask]
    if sub.empty:
        raise ValueError(f"missing row for {conds}")
    return sub.iloc[0]


def build_main_results() -> pd.DataFrame:
    v147 = pd.read_csv(V147)
    v148 = pd.read_csv(V148_COMP)
    v148_rand = pd.read_csv(V148_RAND)
    v150 = pd.read_csv(V150)
    rows: list[dict[str, object]] = []

    for policy, status, note in [
        ("baseline_low_conf_dev_selected", "baseline", "low-confidence selective review selected on old/third"),
        ("dev_stable_router_all_domains", "dev-supported", "router selected by old/third internal stability"),
        ("shift_aware_image_directional_candidate", "candidate", "severe-shift image-feature branch; needs prospective severe-shift validation"),
        ("shift_aware_concept_directional_candidate", "candidate", "severe-shift predicted-concept branch; needs prospective severe-shift validation"),
    ]:
        for domain in ["old_data", "third_batch", "strict_external", "all_three_domains"]:
            r = pick(v147, policy=policy, eval_domain=domain, review_budget=0.3)
            rows.append(
                {
                    "block": "router_selective_review",
                    "policy": policy,
                    "evidence_status": status,
                    "domain": domain,
                    "n": int(r["system_n"]),
                    "review_rate": float(r["review_rate"]),
                    "auto_rate": float(r["auto_rate"]),
                    "system_accuracy": float(r["system_accuracy"]),
                    "system_bacc": float(r["system_balanced_accuracy"]),
                    "system_sensitivity_high": float(r["system_sensitivity_high"]),
                    "system_specificity_low": float(r["system_specificity_low"]),
                    "fn": int(r["system_fn"]),
                    "fp": int(r["system_fp"]),
                    "note": note,
                }
            )

    for domain in ["old_data", "third_batch", "strict_external", "all_three_domains"]:
        r = v148_rand.loc[v148_rand["domain"].eq(domain)].iloc[0]
        rows.append(
            {
                "block": "random_review_reference",
                "policy": "random_review_30pct",
                "evidence_status": "baseline",
                "domain": domain,
                "n": int(r["n"]),
                "review_rate": int(r["random_review_n"]) / int(r["n"]),
                "auto_rate": 1.0 - int(r["random_review_n"]) / int(r["n"]),
                "system_accuracy": np.nan,
                "system_bacc": float(r["random_system_bacc_mean"]),
                "system_sensitivity_high": np.nan,
                "system_specificity_low": np.nan,
                "fn": np.nan,
                "fp": np.nan,
                "note": "mean over random review simulations",
            }
        )

    for domain in ["strict_external_hybrid", "all_three_domains_hybrid"]:
        best = v150.loc[v150["scope"].eq(domain)].sort_values("final_balanced_accuracy", ascending=False).iloc[0]
        rows.append(
            {
                "block": "auto_correct_plus_review_upper_bound",
                "policy": "severe_shift_auto_flip_plus_review_best_scan",
                "evidence_status": "exploratory upper bound",
                "domain": domain,
                "n": int(best["n"]),
                "review_rate": float(best["doctor_review_rate"]),
                "auto_rate": 1.0 - float(best["doctor_review_rate"]),
                "system_accuracy": float(best["final_accuracy"]),
                "system_bacc": float(best["final_balanced_accuracy"]),
                "system_sensitivity_high": float(best["final_sensitivity_high"]),
                "system_specificity_low": float(best["final_specificity_low"]),
                "fn": int(best["final_fn"]),
                "fp": int(best["final_fp"]),
                "note": f"external-label scan: auto_flip={best['severe_auto_flip_budget']}, extra_review={best['severe_extra_review_budget']}; not a locked policy",
            }
        )

    return pd.DataFrame(rows)


def build_module_evidence() -> pd.DataFrame:
    v141 = pd.read_csv(V141)
    v142 = pd.read_csv(V142_RISK)
    v143 = pd.read_csv(V143_RISK)
    v144 = pd.read_csv(V144_RISK)
    v144_concept = pd.read_csv(V144_CONCEPT)
    v148 = pd.read_csv(V148_COMP)
    v151 = pd.read_csv(V151)
    audit = pd.read_csv(V77)

    image_only = v141.loc[v141["feature_set"].eq("image_probs_only")].sort_values("balanced_accuracy", ascending=False).iloc[0]
    image_concept = v141.loc[v141["feature_set"].eq("image_plus_concept")].sort_values("balanced_accuracy", ascending=False).iloc[0]
    fp_concept = v142.loc[v142["target"].eq("fp_error")].sort_values("auroc", ascending=False).iloc[0]
    strict_v143 = v143.loc[
        v143["scope"].eq("strict_external_locked")
        & v143["base_model"].eq("robust_prob")
        & v143["target"].eq("any_error")
    ].sort_values("auroc", ascending=False).iloc[0]
    strict_v144 = v144.loc[
        v144["scope"].eq("strict_external_locked")
        & v144["base_model"].eq("prob_mean_core")
        & v144["target"].eq("any_error")
    ].sort_values("auroc", ascending=False).iloc[0]
    best_concepts = v144_concept.sort_values("oof_auroc", ascending=False).head(4)
    external_comp = v148.loc[
        v148["domain"].eq("strict_external")
        & v148["policy"].eq("shift_aware_image_directional_candidate")
    ].iloc[0]
    v151_strict = v151.loc[
        v151["policy"].eq("shift_aware_concept_directional_candidate")
        & v151["eval_domain"].eq("strict_external")
    ].iloc[0]
    strict_audit = audit.loc[audit["target"].eq("strict_external")].iloc[0]

    rows = [
        {
            "module": "Concepts as direct classification input",
            "main_result": f"BAcc {pct(image_only['balanced_accuracy'])} -> {pct(image_concept['balanced_accuracy'])}",
            "evidence": "v141 image probability stack vs image+concept fusion",
            "status": "negative/control",
            "interpretation": "Direct tabular concept fusion is not the performance breakthrough.",
        },
        {
            "module": "Concept-guided direction-aware routing",
            "main_result": f"FP-error AUROC {float(fp_concept['auroc']):.3f}",
            "evidence": f"v142 best FP-error router: {fp_concept['feature_set']} / {fp_concept['router_model']}",
            "status": "dev-supported direction signal",
            "interpretation": "Concepts are more useful for low-risk overcalling protection than for generic error detection.",
        },
        {
            "module": "Frozen image-feature router",
            "main_result": f"Strict external any-error AUROC {float(strict_v143['auroc']):.3f}",
            "evidence": f"v143 best strict external robust_prob router: {strict_v143['router']}",
            "status": "candidate",
            "interpretation": "Image features help under severe shift, but internal OOF selection alone is not enough.",
        },
        {
            "module": "Image-distilled concept router",
            "main_result": f"Strict external any-error AUROC {float(strict_v144['auroc']):.3f}",
            "evidence": f"v144 router: {strict_v144['router']}; top predicted concepts: "
            + ", ".join(f"{r.concept}({float(r.oof_auroc):.2f})" for r in best_concepts.itertuples()),
            "status": "candidate",
            "interpretation": "Doctor concepts can be used as training-time intermediate supervision without test-time text lookup.",
        },
        {
            "module": "Unlabeled batch-shift audit",
            "main_result": f"Strict external domain AUC {float(strict_audit['domain_auc_cv']):.3f}, shift index {float(strict_audit['batch_shift_index']):.3f}",
            "evidence": "v77/v97 audit labels strict external as severe_shift without using labels",
            "status": "supported",
            "interpretation": "Incoming batches can be triaged before labels are available.",
        },
        {
            "module": "Shift-aware selective review",
            "main_result": f"Strict external Delta BAcc vs low-conf {pp(external_comp['delta_bacc'])}",
            "evidence": "v147/v148 matched 30% review budget comparison",
            "status": "trend; not statistically significant",
            "interpretation": "The router improves external point estimate, but CI crosses zero.",
        },
        {
            "module": "Automatic correction lockability",
            "main_result": f"Internal no-harm selected auto-flip {pct(v151_strict['selected_by_internal_no_harm'], 0)}",
            "evidence": "v151 old/third no-harm auto-flip budget selection",
            "status": "not lockable yet",
            "interpretation": "Auto-correction should remain a severe-shift candidate until another severe-shift development batch exists.",
        },
    ]
    return pd.DataFrame(rows)


def build_claim_map() -> pd.DataFrame:
    rows = [
        {
            "claim": "The framework should be domain-aware rather than a fixed confidence threshold.",
            "evidence": "v77/v97 identify strict external as severe shift; v145 shows internal selection collapses to low confidence; v147 improves strict external point estimate with router switching.",
            "status": "supported as design claim",
            "risk": "Needs more external batches for threshold stability.",
        },
        {
            "claim": "Doctor concepts are useful as intermediate supervision, not as a direct test-time table.",
            "evidence": "v141 direct fusion gains only about +0.14 pp BAcc; v142 FP-risk AUROC about 0.924; v144 predicted concepts improve severe-shift routing.",
            "status": "supported with nuanced scope",
            "risk": "Need image-level concept head or prospective concept validation for stronger computer-science claim.",
        },
        {
            "claim": "Direction-aware routing is better than random review and has external improvement trend over low-confidence selection.",
            "evidence": "v148 strict external random review mean BAcc about 0.674 vs low-conf 0.781 and image-directional 0.806.",
            "status": "supported vs random; trend vs low-confidence",
            "risk": "Bootstrap CI vs low-confidence crosses zero.",
        },
        {
            "claim": "The current automatic corrector should not be deployed as a locked all-domain module.",
            "evidence": "v149 direct flip harms old/third; v151 internal no-harm rule selects 0% auto-flip.",
            "status": "supported safety boundary",
            "risk": "Severe-shift auto-correction remains attractive but not lockable.",
        },
        {
            "claim": "Automatic correction plus review has meaningful severe-shift upper-bound potential.",
            "evidence": "v150 strict external best scan BAcc about 0.888, all-domain about 0.895.",
            "status": "exploratory upper bound",
            "risk": "Budget scanned on external labels; cannot be formal main result.",
        },
    ]
    return pd.DataFrame(rows)


def build_next_steps() -> pd.DataFrame:
    rows = [
        {
            "gap": "Severe-shift branch is not lockable from old/third only.",
            "next_experiment": "Create pseudo-severe shifts from old/third using quality/background/scale corruptions, then select auto/review budgets on those synthetic shifts.",
            "success_gate": "Selected budgets must not harm old/third and must improve held-out pseudo-severe shifts before checking strict external.",
        },
        {
            "gap": "Predicted concepts are frozen-feature probes, not a true image concept head.",
            "next_experiment": "Train a lightweight multi-task concept head on DINO/WPC features or finetuned backbone with Task7 + concept auxiliary losses.",
            "success_gate": "Concept AUROC improves for capsule/boundary/cystic concepts and strict external router improves without external tuning.",
        },
        {
            "gap": "External advantage over low-confidence is not statistically significant.",
            "next_experiment": "Add another external or severe-quality batch, or perform leave-batch/pseudo-shift validation with locked routing rules.",
            "success_gate": "Paired BAcc delta vs low-confidence CI excludes or mostly exceeds 0 under matched review budgets.",
        },
        {
            "gap": "Automatic correction currently unsafe all-domain.",
            "next_experiment": "Convert auto-correction into calibrated correction-with-abstention: only flip when direction score and concept score agree, otherwise review.",
            "success_gate": "Internal no-harm rule selects >0% flip while keeping old/third BAcc non-decreasing.",
        },
    ]
    return pd.DataFrame(rows)


def make_pipeline_figure() -> None:
    if not HAS_MPL:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.axis("off")

    boxes = [
        ("Gross pathology image", 0.04, 0.66, 0.15, 0.18, "#E8F3F1"),
        ("Stage 1\nbase classifier\nDINO/WPC probabilities", 0.25, 0.66, 0.18, 0.18, "#EAF0FA"),
        ("Unlabeled batch audit\nquality + domain shift", 0.49, 0.66, 0.19, 0.18, "#FFF4D8"),
        ("within-internal\nselective review router", 0.75, 0.78, 0.20, 0.14, "#E9F7EF"),
        ("severe-shift\nimage/concept router", 0.75, 0.52, 0.20, 0.14, "#FDEDEC"),
        ("Auto-pass\nwhen low risk", 0.18, 0.23, 0.16, 0.14, "#F5F5F5"),
        ("Candidate auto-correction\nonly if safely lockable", 0.42, 0.23, 0.21, 0.14, "#FCE4EC"),
        ("Doctor review / reject\nfor unresolved risk", 0.71, 0.23, 0.20, 0.14, "#EDE7F6"),
    ]
    for text, x, y, w, h, color in boxes:
        patch = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, fontweight="bold")

    arrows = [
        ((0.19, 0.75), (0.25, 0.75)),
        ((0.43, 0.75), (0.49, 0.75)),
        ((0.68, 0.76), (0.75, 0.85)),
        ((0.68, 0.69), (0.75, 0.59)),
        ((0.34, 0.66), (0.27, 0.37)),
        ((0.85, 0.78), (0.81, 0.37)),
        ((0.85, 0.52), (0.52, 0.37)),
        ((0.63, 0.30), (0.71, 0.30)),
        ((0.34, 0.30), (0.42, 0.30)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))

    ax.text(
        0.04,
        0.05,
        "Evidence boundary: selective review is the current lockable main line; auto-correction remains a severe-shift candidate until another severe-shift development batch validates it.",
        fontsize=10,
        color="#8A1F11",
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v152_framework_pipeline.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v152_framework_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def make_external_bar_figure(main_results: pd.DataFrame) -> None:
    if not HAS_MPL:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    focus = main_results.loc[
        main_results["domain"].isin(["strict_external", "strict_external_hybrid"])
        & main_results["block"].isin(["router_selective_review", "random_review_reference", "auto_correct_plus_review_upper_bound"])
    ].copy()
    focus["label"] = focus["policy"].replace(
        {
            "random_review_30pct": "Random review",
            "baseline_low_conf_dev_selected": "Low-conf review",
            "dev_stable_router_all_domains": "Dev-stable router",
            "shift_aware_image_directional_candidate": "Severe image router",
            "shift_aware_concept_directional_candidate": "Severe concept router",
            "severe_shift_auto_flip_plus_review_best_scan": "Hybrid upper bound",
        }
    )
    order = [
        "Random review",
        "Low-conf review",
        "Dev-stable router",
        "Severe image router",
        "Severe concept router",
        "Hybrid upper bound",
    ]
    focus["order"] = focus["label"].map({k: i for i, k in enumerate(order)})
    focus = focus.sort_values("order")
    colors = ["#BDBDBD", "#90CAF9", "#64B5F6", "#FFCC80", "#CE93D8", "#EF9A9A"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(focus["label"], focus["system_bacc"] * 100, color=colors[: len(focus)], edgecolor="#333333")
    ax.set_ylabel("BAcc on strict external / severe-shift set (%)")
    ax.set_ylim(55, max(92, float((focus["system_bacc"] * 100).max()) + 4))
    ax.set_title("External severe-shift evidence: locked vs candidate modules")
    ax.tick_params(axis="x", labelrotation=25)
    for i, r in enumerate(focus.itertuples()):
        ax.text(i, r.system_bacc * 100 + 0.8, f"{r.system_bacc * 100:.1f}", ha="center", fontsize=9)
    ax.text(
        0.02,
        0.02,
        "Hybrid upper bound scans external labels and is not a locked policy.",
        transform=ax.transAxes,
        fontsize=9,
        color="#8A1F11",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v152_external_evidence_bar.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v152_external_evidence_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown(main_results: pd.DataFrame, module: pd.DataFrame, claims: pd.DataFrame, next_steps: pd.DataFrame) -> None:
    lines = [
        "# Task7 Framework Evidence Pack v152",
        "",
        "## One-sentence takeaway",
        "",
        "The current lockable contribution is a domain-aware selective-review framework; concept-distilled automatic correction is promising under severe shift but cannot yet be locked without another severe-shift development batch.",
        "",
        "## Main External Readout",
        "",
    ]
    external = main_results.loc[main_results["domain"].isin(["strict_external", "strict_external_hybrid"])].copy()
    for r in external.itertuples():
        lines.append(
            f"- {r.policy}: BAcc {pct(r.system_bacc)}, review {pct(r.review_rate)}, FN={'' if pd.isna(r.fn) else int(r.fn)}, FP={'' if pd.isna(r.fp) else int(r.fp)}; status={r.evidence_status}."
        )
    lines += [
        "",
        "## Module Evidence",
        "",
    ]
    for r in module.itertuples():
        lines.append(f"- {r.module}: {r.main_result}. Evidence: {r.evidence}. Status: {r.status}.")
    lines += [
        "",
        "## Claim-Evidence Map",
        "",
    ]
    for r in claims.itertuples():
        lines.append(f"- Claim: {r.claim} Evidence: {r.evidence} Status: {r.status} Risk: {r.risk}")
    lines += [
        "",
        "## Next Experiments",
        "",
    ]
    for r in next_steps.itertuples():
        lines.append(f"- Gap: {r.gap} Next: {r.next_experiment} Gate: {r.success_gate}")
    (OUT_DIR / "v152_framework_evidence_pack.md").write_text("\n".join(lines), encoding="utf-8")


def make_pdf_summary(main_results: pd.DataFrame, module: pd.DataFrame, claims: pd.DataFrame) -> None:
    if not HAS_MPL:
        return
    pdf_path = OUT_DIR / "v152_framework_evidence_pack.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.text(0.03, 0.94, "Task7 Framework Evidence Pack v152", fontsize=18, fontweight="bold")
        ax.text(
            0.03,
            0.88,
            "Main lockable line: domain-aware selective review. Auto-correction: severe-shift candidate, not yet lockable.",
            fontsize=11,
        )
        table = main_results.loc[
            main_results["domain"].isin(["strict_external", "strict_external_hybrid"])
        ][["policy", "evidence_status", "review_rate", "system_bacc", "fn", "fp"]].copy()
        table["review_rate"] = table["review_rate"].map(lambda x: pct(x))
        table["system_bacc"] = table["system_bacc"].map(lambda x: pct(x))
        cell_text = table.values.tolist()
        col_labels = ["Policy", "Status", "Review", "BAcc", "FN", "FP"]
        t = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="left", loc="center", bbox=[0.03, 0.18, 0.94, 0.62])
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        ax.text(0.03, 0.08, "Note: Hybrid upper bound scans strict external labels and is not a no-leak locked workflow.", fontsize=9, color="#8A1F11")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.text(0.03, 0.94, "Module Evidence and Claim Boundaries", fontsize=16, fontweight="bold")
        y = 0.87
        for r in module.itertuples():
            ax.text(0.04, y, f"{r.module}: {r.main_result}", fontsize=10, fontweight="bold")
            y -= 0.035
            ax.text(0.06, y, f"Evidence: {r.evidence}", fontsize=8.5)
            y -= 0.03
            ax.text(0.06, y, f"Status: {r.status}. {r.interpretation}", fontsize=8.5)
            y -= 0.055
            if y < 0.12:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                ax.axis("off")
                y = 0.92
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    main_results = build_main_results()
    module = build_module_evidence()
    claims = build_claim_map()
    next_steps = build_next_steps()

    main_results.to_csv(OUT_DIR / "v152_main_framework_results.csv", index=False, encoding="utf-8-sig")
    module.to_csv(OUT_DIR / "v152_module_evidence_table.csv", index=False, encoding="utf-8-sig")
    claims.to_csv(OUT_DIR / "v152_claim_evidence_map.csv", index=False, encoding="utf-8-sig")
    next_steps.to_csv(OUT_DIR / "v152_next_experiment_gaps.csv", index=False, encoding="utf-8-sig")
    make_pipeline_figure()
    make_external_bar_figure(main_results)
    write_markdown(main_results, module, claims, next_steps)
    make_pdf_summary(main_results, module, claims)

    output_files = [
        "v152_main_framework_results.csv",
        "v152_module_evidence_table.csv",
        "v152_claim_evidence_map.csv",
        "v152_next_experiment_gaps.csv",
        "v152_framework_evidence_pack.md",
    ]
    if HAS_MPL:
        output_files.extend(
            [
                "v152_framework_evidence_pack.pdf",
                "figures/v152_framework_pipeline.png",
                "figures/v152_external_evidence_bar.png",
            ]
        )
    report = {
        "outputs": output_files,
        "matplotlib_available": HAS_MPL,
        "main_boundary": "domain-aware selective review is lockable; auto-correction is candidate/upper bound until severe-shift validation exists.",
    }
    (OUT_DIR / "v152_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v152] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
