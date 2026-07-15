from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ORIGINAL = "original_standard6"
BACKGROUND_ONLY = "background_only_standard6"
PRIMARY_MODEL = "risk_balanced"
CANDIDATE_PRIORITY = (
    "evidence_top4",
    "scale_normalized_standard6",
    "neutral_background_standard6",
    "tight_standard6",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze locked H11 counterfactual predictions.")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--reproduction-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return math.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    positive_rank_sum = float(ranks[positive].sum())
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )


def metric_record(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["label_idx"].astype(int).to_numpy()
    probability = frame["prob_high"].astype(float).to_numpy()
    predictions = probability >= 0.5
    low = labels == 0
    high = labels == 1
    specificity = float((predictions[low] == 0).mean()) if low.any() else math.nan
    sensitivity = float((predictions[high] == 1).mean()) if high.any() else math.nan
    return {
        "n": int(len(frame)),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(np.nanmean([specificity, sensitivity])),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "auc": float(auc_score(labels, probability)),
    }


def add_baseline_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions[predictions["variant"] == ORIGINAL][
        ["case_id", "model_name", "prob_high", "correct", "true_class_probability"]
    ].rename(
        columns={
            "prob_high": "baseline_prob_high",
            "correct": "baseline_correct",
            "true_class_probability": "baseline_true_class_probability",
        }
    )
    if baseline.duplicated(["case_id", "model_name"]).any():
        raise ValueError("Duplicate original predictions")
    merged = predictions.merge(baseline, on=["case_id", "model_name"], how="left", validate="many_to_one")
    if merged["baseline_prob_high"].isna().any():
        raise ValueError("Missing original predictions")
    merged["correct"] = merged["correct"].astype(str).str.lower().map({"true": True, "false": False}).fillna(merged["correct"]).astype(bool)
    merged["baseline_correct"] = merged["baseline_correct"].astype(str).str.lower().map({"true": True, "false": False}).fillna(merged["baseline_correct"]).astype(bool)
    merged["true_probability_delta"] = (
        merged["true_class_probability"] - merged["baseline_true_class_probability"]
    )
    merged["rescued"] = ~merged["baseline_correct"] & merged["correct"]
    merged["harmed"] = merged["baseline_correct"] & ~merged["correct"]
    return merged


def scope_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    image_match = frame["image_name_match"].astype(str).str.lower().eq("true")
    audited = frame["audit_code"].fillna("").astype(str).ne("")
    return {
        "all117": pd.Series(True, index=frame.index),
        "image_concordant": image_match,
        "audited_image_concordant": image_match & audited,
        "stable_m3_image_concordant": image_match & frame["model_correct_count"].astype(int).eq(3),
        "recoverable_hard": image_match & frame["post_unblind_attribution"].fillna("").eq("recoverable_visual_model_miss"),
        "mimic_hard": image_match & frame["post_unblind_attribution"].fillna("").eq("phenotype_label_mimic"),
        "evidence_limited_hard": image_match & frame["post_unblind_attribution"].fillna("").eq("evidence_limited_or_ambiguous"),
    }


def build_metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, variant), group in frame.groupby(["model_name", "variant"], sort=True):
        for scope, mask in scope_masks(group).items():
            subset = group[mask]
            if subset.empty:
                continue
            metrics = metric_record(subset)
            rows.append(
                {
                    "model_name": model_name,
                    "variant": variant,
                    "scope": scope,
                    **metrics,
                    "rescued": int(subset["rescued"].sum()),
                    "harmed": int(subset["harmed"].sum()),
                    "net_gain": int(subset["rescued"].sum() - subset["harmed"].sum()),
                    "median_true_probability_delta": float(subset["true_probability_delta"].median()),
                }
            )
        for subtype, subset in group[group["image_name_match"].astype(str).str.lower().eq("true")].groupby("task_l6_label"):
            metrics = metric_record(subset)
            rows.append(
                {
                    "model_name": model_name,
                    "variant": variant,
                    "scope": f"subtype_{subtype}_image_concordant",
                    **metrics,
                    "rescued": int(subset["rescued"].sum()),
                    "harmed": int(subset["harmed"].sum()),
                    "net_gain": int(subset["rescued"].sum() - subset["harmed"].sum()),
                    "median_true_probability_delta": float(subset["true_probability_delta"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["model_name", "variant", "scope"]).reset_index(drop=True)


def decision_for_variant(frame: pd.DataFrame, variant: str) -> dict[str, Any]:
    candidate = frame[frame["variant"] == variant]
    primary = candidate[candidate["model_name"] == PRIMARY_MODEL]
    image_match = primary["image_name_match"].astype(str).str.lower().eq("true")
    main = primary[image_match]
    recoverable = main[
        main["post_unblind_attribution"].fillna("").eq("recoverable_visual_model_miss")
    ]
    recoverable_large_or_correct = int(
        ((recoverable["true_probability_delta"] >= 0.15) | recoverable["correct"]).sum()
    )
    recoverable_median = float(recoverable["true_probability_delta"].median())
    anchors = main[main["model_correct_count"].astype(int).eq(3)]
    anchor_harms = int(anchors["harmed"].sum())
    anchor_flip_rate = float(anchor_harms / len(anchors)) if len(anchors) else math.nan
    net_gain = int(main["rescued"].sum() - main["harmed"].sum())
    subtype_net: dict[str, int] = {}
    for subtype in ("B1", "B2"):
        subset = main[main["task_l6_label"] == subtype]
        subtype_net[subtype] = int(subset["rescued"].sum() - subset["harmed"].sum())
    support_nets: dict[str, int] = {}
    for model_name, group in candidate.groupby("model_name"):
        group = group[group["image_name_match"].astype(str).str.lower().eq("true")]
        support_nets[str(model_name)] = int(group["rescued"].sum() - group["harmed"].sum())
    support_values = list(support_nets.values())
    criteria = {
        "recoverable_count": recoverable_large_or_correct >= 2,
        "recoverable_median": recoverable_median >= 0.10,
        "anchor_protection": anchor_flip_rate <= 0.05,
        "main_net_gain": net_gain >= 3,
        "b1_nonnegative": subtype_net["B1"] >= 0,
        "b2_nonnegative": subtype_net["B2"] >= 0,
        "support_median_nonnegative": float(np.median(support_values)) >= 0,
        "support_worst_not_below_minus3": min(support_values) >= -3,
    }
    return {
        "variant": variant,
        "passed": bool(all(criteria.values())),
        "criteria": criteria,
        "recoverable_n": int(len(recoverable)),
        "recoverable_large_shift_or_correct_count": recoverable_large_or_correct,
        "recoverable_median_true_probability_delta": recoverable_median,
        "anchor_n": int(len(anchors)),
        "anchor_harms": anchor_harms,
        "anchor_flip_rate": anchor_flip_rate,
        "image_concordant_n": int(len(main)),
        "rescued": int(main["rescued"].sum()),
        "harmed": int(main["harmed"].sum()),
        "net_gain": net_gain,
        "subtype_net_gain": subtype_net,
        "all_model_net_gain": support_nets,
    }


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def build_markdown(
    reproduction: dict[str, Any], decisions: list[dict[str, Any]], background: dict[str, Any], selected: str | None
) -> str:
    lines = [
        "# H11 锁定视觉证据解耦诊断结果",
        "",
        "日期：2026-07-15",
        "",
        "## 复现闸门",
        "",
        f"- 通过：`{reproduction['passed']}`",
        f"- 分类一致：`{reproduction['classification_match']}`",
        f"- 最大概率绝对误差：`{fmt(float(reproduction['max_absolute_probability_error']), 6)}`",
        f"- 容差：`{fmt(float(reproduction['tolerance']), 6)}`",
        "",
        "## 固定候选闸门",
        "",
        "| 候选 | 可恢复满足数/3 | 可恢复真类概率中位变化 | 锚点伤害率 | 救回 | 伤害 | 净增益 | B1净增益 | B2净增益 | 通过 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in decisions:
        lines.append(
            f"| `{item['variant']}` | {item['recoverable_large_shift_or_correct_count']}/{item['recoverable_n']} "
            f"| {fmt(item['recoverable_median_true_probability_delta'])} | {fmt(item['anchor_flip_rate'])} "
            f"| {item['rescued']} | {item['harmed']} | {item['net_gain']} "
            f"| {item['subtype_net_gain']['B1']} | {item['subtype_net_gain']['B2']} | `{item['passed']}` |"
        )
    lines.extend(
        [
            "",
            "## 背景负对照",
            "",
            f"- BAcc：`{fmt(background['balanced_accuracy'])}`",
            f"- AUC：`{fmt(background['auc'])}`",
            f"- 明显背景捷径信号：`{background['shortcut_signal']}`",
            "",
            "## 预注册决策",
            "",
            (
                f"H11 GO。按预先固定优先级，H12 使用 `{selected}` 作为候选证据分支。"
                if selected
                else "H11 NO-GO。没有固定视图方案同时满足可恢复漏读、锚点保护、净增益和 B1/B2 保护闸门；不启动同一路径的 H12。"
            ),
            "",
            "该结论只适用于机制诊断子集。模型能力是否真正提升，仍须由完整 591 例嵌套交叉验证以及 source-LODO/严格外部数据验证。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reproduction = json.loads(Path(args.reproduction_json).read_text(encoding="utf-8"))
    if not reproduction.get("passed"):
        raise RuntimeError("Cannot analyze counterfactuals because reproduction gate failed")
    predictions = pd.read_csv(args.predictions_csv, dtype={"case_id": str})
    frame = add_baseline_columns(predictions)
    expected_variants = {ORIGINAL, BACKGROUND_ONLY, *CANDIDATE_PRIORITY}
    missing = sorted(expected_variants - set(frame["variant"]))
    if missing:
        raise ValueError(f"Missing H11 variants: {missing}")

    metric_summary = build_metric_summary(frame)
    metric_summary.to_csv(output_dir / "h11_metric_summary.csv", index=False, encoding="utf-8")
    decisions = [decision_for_variant(frame, variant) for variant in CANDIDATE_PRIORITY]
    selected = next((item["variant"] for item in decisions if item["passed"]), None)

    background_frame = frame[
        (frame["variant"] == BACKGROUND_ONLY)
        & (frame["model_name"] == PRIMARY_MODEL)
        & frame["image_name_match"].astype(str).str.lower().eq("true")
    ]
    background_metrics = metric_record(background_frame)
    background = {
        **background_metrics,
        "shortcut_signal": bool(
            background_metrics["balanced_accuracy"] >= 0.65 or background_metrics["auc"] >= 0.70
        ),
    }
    gate_payload = {
        "experiment": "H11_LOCKED_VISUAL_EVIDENCE_DISENTANGLEMENT_20260715",
        "reproduction": reproduction,
        "primary_model": PRIMARY_MODEL,
        "candidate_priority": list(CANDIDATE_PRIORITY),
        "candidate_decisions": decisions,
        "background_only": background,
        "h12_go": selected is not None,
        "selected_by_preregistered_priority": selected,
    }
    write_json(output_dir / "h11_gate_decisions.json", gate_payload)
    markdown = build_markdown(reproduction, decisions, background, selected)
    (output_dir / "H11_LOCKED_VISUAL_EVIDENCE_DISENTANGLEMENT_RESULTS_20260715.md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown, flush=True)


if __name__ == "__main__":
    main()

