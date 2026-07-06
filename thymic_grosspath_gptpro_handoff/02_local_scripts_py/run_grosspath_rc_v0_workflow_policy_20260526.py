from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "grosspath_rc_v0_20260526"
REPORT_DIR = ROOT / "汇报"


def safe_bacc(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y, pred))


def safe_f1(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(f1_score(y, pred, zero_division=0))


def subset_metric(y: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    if len(y) == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "sensitivity_high": float("nan"),
            "specificity_low": float("nan"),
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": safe_bacc(y, pred),
        "f1": safe_f1(y, pred),
        "sensitivity_high": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity_low": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def policy_masks(df: pd.DataFrame, policy: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return auto, review, retake, auto_pred."""
    index = df.index
    false = pd.Series(False, index=index)
    true = pd.Series(True, index=index)
    pred162 = df["pred_for_prob162_blend"].astype(int)
    pred103 = df["pred_for_prob103_vitl"].astype(int)
    pred107 = df["pred_for_prob107_qkvb"].astype(int)
    all3 = pred162.eq(pred103) & pred162.eq(pred107)
    agree_162_103 = pred162.eq(pred103)
    agree_162_107 = pred162.eq(pred107)

    if policy == "forced_162_all":
        return true, false, false, pred162
    if policy == "forced_dinov3_all":
        return true, false, false, pred103
    if policy == "auto_if_162_103_agree":
        auto = agree_162_103
        return auto, ~auto, false, pred162
    if policy == "auto_if_162_107_agree":
        auto = agree_162_107
        return auto, ~auto, false, pred162
    if policy == "auto_if_all3_agree":
        auto = all3
        return auto, ~auto, false, pred162

    quality = df.get("manual_quality_status_v1")
    if quality is None:
        return false, true, false, pred162
    q = quality.fillna("").astype(str)
    retake = q.eq("reject_retake")
    borderline = q.eq("borderline_review")
    readable = q.eq("pass_readable")
    if policy == "quality_readable_force_162":
        auto = readable
        review = borderline | (~readable & ~retake)
        return auto, review, retake, pred162
    if policy == "quality_readable_162_103_agree":
        auto = readable & agree_162_103
        review = (readable & ~agree_162_103) | borderline | (~readable & ~retake)
        return auto, review, retake, pred162
    if policy == "quality_readable_all3_agree":
        auto = readable & all3
        review = (readable & ~all3) | borderline | (~readable & ~retake)
        return auto, review, retake, pred162

    raise ValueError(policy)


def eval_policy(df: pd.DataFrame, group: str, policy: str) -> dict[str, object]:
    auto, review, retake, auto_pred = policy_masks(df, policy)
    y = df["label_idx"].astype(int)
    auto_y = y[auto].to_numpy(int)
    auto_p = auto_pred[auto].to_numpy(int)
    m_auto = subset_metric(auto_y, auto_p)

    base_pred = df["pred_for_prob162_blend"].astype(int)
    review_error = (
        float((base_pred[review].to_numpy(int) != y[review].to_numpy(int)).mean()) if int(review.sum()) else float("nan")
    )
    retake_error = (
        float((base_pred[retake].to_numpy(int) != y[retake].to_numpy(int)).mean()) if int(retake.sum()) else float("nan")
    )
    total = len(df)
    auto_low = auto & auto_pred.eq(0)
    auto_high = auto & auto_pred.eq(1)
    auto_low_high_miss = (
        float((y[auto_low].to_numpy(int) == 1).mean()) if int(auto_low.sum()) else float("nan")
    )
    auto_high_ppv = (
        float((y[auto_high].to_numpy(int) == 1).mean()) if int(auto_high.sum()) else float("nan")
    )
    high_total = int((y == 1).sum())
    high_auto_low = int(((y == 1) & auto_low).sum())
    high_auto_high = int(((y == 1) & auto_high).sum())
    high_review_or_retake = int(((y == 1) & (review | retake)).sum())

    out = {
        "group": group,
        "policy": policy,
        "total_n": int(total),
        "auto_n": int(auto.sum()),
        "review_n": int(review.sum()),
        "retake_n": int(retake.sum()),
        "auto_coverage": float(auto.sum() / total) if total else float("nan"),
        "review_rate": float(review.sum() / total) if total else float("nan"),
        "retake_rate": float(retake.sum() / total) if total else float("nan"),
        "review_error_rate_base162": review_error,
        "retake_error_rate_base162": retake_error,
        "auto_low_n": int(auto_low.sum()),
        "auto_high_n": int(auto_high.sum()),
        "auto_low_high_miss_rate": auto_low_high_miss,
        "auto_high_ppv": auto_high_ppv,
        "high_total": high_total,
        "high_auto_low_missed": high_auto_low,
        "high_auto_high": high_auto_high,
        "high_review_or_retake": high_review_or_retake,
        "high_risk_not_auto_low_rate": float((high_total - high_auto_low) / high_total) if high_total else float("nan"),
    }
    for k, v in m_auto.items():
        out[f"auto_{k}"] = v
    return out


def format_float(x: object) -> str:
    if isinstance(x, float):
        if np.isnan(x):
            return ""
        return f"{x:.4f}"
    return str(x)


def df_to_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_float(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def group_masks_dev(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "dev_all_old_plus_third": pd.Series(True, index=df.index),
        "old": df["domain"].eq("old"),
        "third_all": df["domain"].eq("third"),
        "third_holdout234": df["third_split"].eq("holdout234"),
    }


def group_masks_external(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "external_all": pd.Series(True, index=df.index),
        "external_strict": df["strict_task7_eval"].astype(int).eq(1),
        "external_readable_auto": df["manual_quality_status_v1"].eq("pass_readable"),
    }


def make_plot(policy_df: pd.DataFrame, group: str, out_path: Path) -> None:
    sub = policy_df[policy_df["group"].eq(group)].copy()
    if sub.empty:
        return
    keep = [
        "forced_162_all",
        "forced_dinov3_all",
        "auto_if_162_103_agree",
        "auto_if_all3_agree",
        "quality_readable_162_103_agree",
        "quality_readable_all3_agree",
    ]
    sub = sub[sub["policy"].isin(keep)].copy()
    labels = sub["policy"].str.replace("_", "\n", regex=False).tolist()
    x = np.arange(len(sub))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(11, 5.8))
    ax1.bar(x - width / 2, sub["auto_coverage"], width=width, label="auto coverage", color="#6aa6b8")
    ax1.bar(x + width / 2, sub["auto_accuracy"], width=width, label="auto accuracy", color="#c57b57")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Proportion")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, fontsize=8)
    ax1.set_title(f"GrossPath-RC v0 workflow policies: {group}")
    ax1.legend(loc="lower left")
    ax1.grid(axis="y", alpha=0.25)

    for i, row in sub.reset_index(drop=True).iterrows():
        ax1.text(i - width / 2, row["auto_coverage"] + 0.015, f"{row['auto_coverage']:.2f}", ha="center", fontsize=8)
        ax1.text(i + width / 2, row["auto_accuracy"] + 0.015, f"{row['auto_accuracy']:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    dev = pd.read_csv(OUT / "dev_model_behavior_table.csv")
    ext = pd.read_csv(OUT / "external_model_behavior_table.csv")

    base_policies = [
        "forced_162_all",
        "forced_dinov3_all",
        "auto_if_162_103_agree",
        "auto_if_162_107_agree",
        "auto_if_all3_agree",
    ]
    quality_policies = [
        "quality_readable_force_162",
        "quality_readable_162_103_agree",
        "quality_readable_all3_agree",
    ]

    rows = []
    for group, mask in group_masks_dev(dev).items():
        sub = dev[mask].copy()
        for policy in base_policies:
            rows.append(eval_policy(sub, group, policy))
    dev_policy = pd.DataFrame(rows)
    dev_policy.to_csv(OUT / "workflow_policy_metrics_dev.csv", index=False, encoding="utf-8-sig")

    rows = []
    for group, mask in group_masks_external(ext).items():
        sub = ext[mask].copy()
        for policy in base_policies + quality_policies:
            rows.append(eval_policy(sub, group, policy))
    ext_policy = pd.DataFrame(rows)
    ext_policy.to_csv(OUT / "workflow_policy_metrics_external_stress.csv", index=False, encoding="utf-8-sig")

    make_plot(ext_policy, "external_strict", OUT / "workflow_policy_external_strict.png")
    make_plot(ext_policy, "external_readable_auto", OUT / "workflow_policy_external_readable_auto.png")
    make_plot(dev_policy, "dev_all_old_plus_third", OUT / "workflow_policy_dev_all.png")

    selected_cols = [
        "group",
        "policy",
        "total_n",
        "auto_n",
        "review_n",
        "retake_n",
        "auto_coverage",
        "auto_accuracy",
        "auto_balanced_accuracy",
        "auto_sensitivity_high",
        "auto_specificity_low",
        "auto_low_high_miss_rate",
        "auto_high_ppv",
        "review_error_rate_base162",
        "high_auto_low_missed",
        "high_review_or_retake",
    ]
    ext_focus = ext_policy[selected_cols].copy()
    dev_focus = dev_policy[selected_cols].copy()

    strict_rank = ext_focus[ext_focus["group"].eq("external_strict")].sort_values(
        ["auto_accuracy", "auto_coverage"], ascending=False
    )
    readable_rank = ext_focus[ext_focus["group"].eq("external_readable_auto")].sort_values(
        ["auto_accuracy", "auto_coverage"], ascending=False
    )

    summary = {
        "dev_best_by_auto_accuracy": dev_focus.sort_values(["auto_accuracy", "auto_coverage"], ascending=False)
        .head(10)
        .to_dict("records"),
        "external_strict_best_by_auto_accuracy": strict_rank.head(10).to_dict("records"),
        "external_readable_best_by_auto_accuracy": readable_rank.head(10).to_dict("records"),
    }
    (OUT / "workflow_policy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# GrossPath-RC v0 完整 workflow policy 实验",
        "",
        "## 实验目标",
        "",
        "把第一轮结果收敛成正式工作流：强制分类、模型一致性自动放行、质量门控、复核、重拍。",
        "",
        "## 开发集策略对照",
        "",
        df_to_md(dev_focus.sort_values(["group", "auto_accuracy"], ascending=[True, False]), max_rows=24),
        "",
        "## 外部严格集策略对照",
        "",
        df_to_md(ext_focus[ext_focus["group"].eq("external_strict")].sort_values("auto_accuracy", ascending=False)),
        "",
        "## 外部 readable_auto 策略对照",
        "",
        df_to_md(ext_focus[ext_focus["group"].eq("external_readable_auto")].sort_values("auto_accuracy", ascending=False)),
        "",
        "## 阶段结论",
        "",
        "1. `forced_162_all` 仍是全量强制分类基线，但外部严格集 Acc 只有约 0.638。",
        "2. 模型一致性是当前最稳定的风险控制信号，尤其是 `auto_if_all3_agree` 和 `auto_if_162_103_agree`。",
        "3. 在外部 readable_auto 子集，`auto_if_all3_agree` / `auto_if_162_103_agree` 的自动准确率约 0.78，覆盖率约 0.52-0.58。",
        "4. 质量门控 + consensus 是更接近临床工作流的策略，但会进一步降低全体覆盖率；它适合写成安全 workflow，而不是刷全量 accuracy。",
        "5. 当前最适合继续强化的主线是 selective prediction / defer-to-review，而不是概念直接提分。",
    ]
    report_path = REPORT_DIR / "2026-05-26_GrossPath-RC_v0完整workflow实验报告.md"
    report_path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
