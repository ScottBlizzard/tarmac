from __future__ import annotations

from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "outputs" / "grosspath_rc_v91_integrated_batch_adaptive_framework_20260527" / "v91_integrated_case_routes.csv"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v92_integrated_paired_stats_20260527"
SEED = 20260527
N_BOOT = 3000

POLICY_LABELS = {
    "v50_main": "Fixed v50",
    "v79_light_lowrisk_guard": "Fixed v79-light",
    "v79_strict_lowrisk_guard": "Fixed v79-strict",
    "adaptive_v50_to_v79_light": "Batch-adaptive main",
    "adaptive_v50_to_v79_strict": "Batch-adaptive strict",
    "risk_direction_uniform80": "Risk-direction uniform80",
    "quality_direction_uniform90": "Quality+direction uniform90",
}

COMPARISONS = [
    ("all_domains", "quality_direction_uniform90", "v79_light_lowrisk_guard"),
    ("all_domains", "quality_direction_uniform90", "v79_strict_lowrisk_guard"),
    ("all_domains", "quality_direction_uniform90", "adaptive_v50_to_v79_light"),
    ("all_domains", "quality_direction_uniform90", "risk_direction_uniform80"),
    ("all_domains", "adaptive_v50_to_v79_light", "v50_main"),
    ("all_domains", "v79_light_lowrisk_guard", "v50_main"),
    ("strict_external", "quality_direction_uniform90", "v79_light_lowrisk_guard"),
    ("strict_external", "quality_direction_uniform90", "v50_main"),
    ("strict_external", "adaptive_v50_to_v79_light", "v50_main"),
]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = y.astype(int)
    pred = pred.astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float((sens + spec) / 2),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "fn": fn,
        "fp": fp,
        "error_n": int((pred != y).sum()),
    }


def mcnemar_exact(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return float(min(1.0, p))


def paired_frame(routes: pd.DataFrame, scope: str, candidate: str, baseline: str) -> pd.DataFrame:
    df = routes.copy()
    if scope != "all_domains":
        df = df.loc[df["domain"].eq(scope)].copy()
    keep = df.loc[df["policy"].isin([candidate, baseline])].copy()
    keep["pair_id"] = keep["domain"].astype(str) + "::" + keep["case_id"].astype(str)
    piv = keep.pivot_table(index=["pair_id", "domain"], columns="policy", values="final_pred", aggfunc="first").reset_index()
    labels = keep.drop_duplicates("pair_id").set_index("pair_id")["label_idx"]
    piv["label_idx"] = piv["pair_id"].map(labels).astype(int)
    missing = piv[[candidate, baseline]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"missing paired predictions for {candidate} vs {baseline}: {missing.sum()}")
    return piv


def bootstrap_delta(y: np.ndarray, cand: np.ndarray, base: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(y)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        m_c = metrics(y[idx], cand[idx])
        m_b = metrics(y[idx], base[idx])
        vals.append(
            {
                "delta_bacc": m_c["balanced_accuracy"] - m_b["balanced_accuracy"],
                "delta_acc": m_c["accuracy"] - m_b["accuracy"],
                "delta_sens": m_c["sensitivity"] - m_b["sensitivity"],
                "delta_spec": m_c["specificity"] - m_b["specificity"],
            }
        )
    return pd.DataFrame(vals)


def run_comparison(routes: pd.DataFrame, scope: str, candidate: str, baseline: str, rng: np.random.Generator) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    paired = paired_frame(routes, scope, candidate, baseline)
    y = paired["label_idx"].to_numpy(int)
    cand = paired[candidate].to_numpy(int)
    base = paired[baseline].to_numpy(int)
    cand_m = metrics(y, cand)
    base_m = metrics(y, base)
    cand_correct = cand == y
    base_correct = base == y
    better = int((cand_correct & ~base_correct).sum())
    worse = int((~cand_correct & base_correct).sum())
    boot = bootstrap_delta(y, cand, base, rng)
    row = {
        "scope": scope,
        "candidate": candidate,
        "candidate_label": POLICY_LABELS.get(candidate, candidate),
        "baseline": baseline,
        "baseline_label": POLICY_LABELS.get(baseline, baseline),
        "n": int(len(paired)),
        "candidate_bacc": cand_m["balanced_accuracy"],
        "baseline_bacc": base_m["balanced_accuracy"],
        "delta_bacc": cand_m["balanced_accuracy"] - base_m["balanced_accuracy"],
        "delta_bacc_ci_low": float(boot["delta_bacc"].quantile(0.025)),
        "delta_bacc_ci_high": float(boot["delta_bacc"].quantile(0.975)),
        "candidate_acc": cand_m["accuracy"],
        "baseline_acc": base_m["accuracy"],
        "delta_acc": cand_m["accuracy"] - base_m["accuracy"],
        "delta_acc_ci_low": float(boot["delta_acc"].quantile(0.025)),
        "delta_acc_ci_high": float(boot["delta_acc"].quantile(0.975)),
        "candidate_fn": cand_m["fn"],
        "baseline_fn": base_m["fn"],
        "delta_fn": int(cand_m["fn"] - base_m["fn"]),
        "candidate_fp": cand_m["fp"],
        "baseline_fp": base_m["fp"],
        "delta_fp": int(cand_m["fp"] - base_m["fp"]),
        "candidate_error_n": cand_m["error_n"],
        "baseline_error_n": base_m["error_n"],
        "delta_error_n": int(cand_m["error_n"] - base_m["error_n"]),
        "candidate_better_case_n": better,
        "candidate_worse_case_n": worse,
        "mcnemar_p": mcnemar_exact(better, worse),
    }
    changed = paired.loc[cand != base, ["pair_id", "domain", "label_idx", candidate, baseline]].copy()
    changed["candidate"] = candidate
    changed["baseline"] = baseline
    changed["scope"] = scope
    changed["candidate_correct"] = (changed[candidate] == changed["label_idx"]).astype(int)
    changed["baseline_correct"] = (changed[baseline] == changed["label_idx"]).astype(int)
    return row, boot.assign(scope=scope, candidate=candidate, baseline=baseline), changed


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def write_messages(summary: pd.DataFrame) -> None:
    q_light = summary.loc[
        summary["scope"].eq("all_domains")
        & summary["candidate"].eq("quality_direction_uniform90")
        & summary["baseline"].eq("v79_light_lowrisk_guard")
    ].iloc[0]
    q_strict = summary.loc[
        summary["scope"].eq("all_domains")
        & summary["candidate"].eq("quality_direction_uniform90")
        & summary["baseline"].eq("v79_strict_lowrisk_guard")
    ].iloc[0]
    ext = summary.loc[
        summary["scope"].eq("strict_external")
        & summary["candidate"].eq("quality_direction_uniform90")
        & summary["baseline"].eq("v79_light_lowrisk_guard")
    ].iloc[0]
    lines = ["# v92 v91 候选流程配对统计检验", ""]
    lines.append("## 核心结果")
    lines.append("")
    lines.append(
        f"- 全域上，Quality+direction uniform90 相比 Fixed v79-light，BAcc +{pct(q_light['delta_bacc'])}，错误数 {int(q_light['baseline_error_n'])}->{int(q_light['candidate_error_n'])}，FN {int(q_light['baseline_fn'])}->{int(q_light['candidate_fn'])}，FP {int(q_light['baseline_fp'])}->{int(q_light['candidate_fp'])}，McNemar p={q_light['mcnemar_p']:.3f}。"
    )
    lines.append(
        f"- 全域上，Quality+direction uniform90 相比 Fixed v79-strict，BAcc +{pct(q_strict['delta_bacc'])}，错误数 {int(q_strict['baseline_error_n'])}->{int(q_strict['candidate_error_n'])}，McNemar p={q_strict['mcnemar_p']:.3f}。"
    )
    lines.append(
        f"- 严格外部集上，Quality+direction uniform90 与 Fixed v79-light 同为 1 个错误，BAcc 差值 {pct(ext['delta_bacc'])}；因此它的主要增益来自旧数据和第三批的高复核纠错，而不是严格外部额外提升。"
    )
    lines.append("")
    lines.append("## 写作边界")
    lines.append("")
    lines.append(
        "Quality+direction uniform90 可以作为高复核高安全候选上限写入补充实验；主文更稳妥的主推仍是 Batch-adaptive main 和 Fixed v79-light。"
        "如果写它的 99.88% 全域 BAcc，必须同时写控制率 89.84%，避免被理解成低成本自动诊断。"
    )
    lines.append("")
    lines.append("## 对照表")
    lines.append("")
    lines.append("| Scope | Candidate | Baseline | ΔBAcc | 95%CI | 错误数 | FN | FP | McNemar p |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['scope']} | {r['candidate_label']} | {r['baseline_label']} | {pct(r['delta_bacc'])} | "
            f"{pct(r['delta_bacc_ci_low'])} to {pct(r['delta_bacc_ci_high'])} | "
            f"{int(r['baseline_error_n'])}->{int(r['candidate_error_n'])} | "
            f"{int(r['baseline_fn'])}->{int(r['candidate_fn'])} | {int(r['baseline_fp'])}->{int(r['candidate_fp'])} | {r['mcnemar_p']:.3f} |"
        )
    (OUT_DIR / "v92_key_messages.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(ROUTES)
    rng = np.random.default_rng(SEED)
    rows = []
    boots = []
    changed = []
    for scope, candidate, baseline in COMPARISONS:
        row, boot, chg = run_comparison(routes, scope, candidate, baseline, rng)
        rows.append(row)
        boots.append(boot)
        changed.append(chg)
    summary = pd.DataFrame(rows)
    boot_df = pd.concat(boots, ignore_index=True)
    changed_df = pd.concat(changed, ignore_index=True)
    summary.to_csv(OUT_DIR / "v92_integrated_paired_stats_summary.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(OUT_DIR / "v92_integrated_paired_bootstrap_samples.csv", index=False, encoding="utf-8-sig")
    changed_df.to_csv(OUT_DIR / "v92_integrated_changed_cases.csv", index=False, encoding="utf-8-sig")
    write_messages(summary)

    print("v92 paired summary:")
    print(
        summary[
            [
                "scope",
                "candidate_label",
                "baseline_label",
                "delta_bacc",
                "delta_bacc_ci_low",
                "delta_bacc_ci_high",
                "baseline_error_n",
                "candidate_error_n",
                "delta_fn",
                "delta_fp",
                "candidate_better_case_n",
                "candidate_worse_case_n",
                "mcnemar_p",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
