from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def metrics(sub: pd.DataFrame, pred_col: str = "locked162_blend_pred_idx") -> dict[str, object]:
    if sub.empty:
        return {"n": 0}
    y = sub["label_idx"].astype(int)
    p = pd.to_numeric(sub[pred_col], errors="coerce").astype(int)
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    n = int(len(sub))
    acc = (tn + tp) / n
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    bacc = float(np.nanmean([sens, spec]))
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else np.nan
    return {
        "n": n,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "sensitivity_high": sens,
        "specificity_low": spec,
        "precision_high": prec,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:
    outdir = Path("outputs/external_quality_gate_20260525")
    merged_path = outdir / "external_quality_predictions_merged_manual_v1.csv"
    df = pd.read_csv(merged_path)
    df["original_case_id"] = df["original_case_id"].astype(str)

    retake = {
        "1603320": "非标准灰白背景，碎片散在且主体暗，建议补充标准近景切面图",
        "2035384": "主体偏小且原图分辨率偏低，切面细节不足",
        "2115368": "主体远小于画面，背景占比过大，建议重拍近景",
        "2207087": "主体过小且暗色，切面细节不足",
        "2238125": "主体过小，拍摄距离过远，背景信息占比过高",
        "2304562": "主体过小且水渍/反光背景干扰明显",
        "2344315": "主体偏小并伴大面积反光，画面边缘干扰明显",
        "2414243": "主体很小，远景图，切面信息不足",
        "2560943": "主体很小，远景图，建议补充更标准切面近照",
    }

    borderline = {
        "1709499": "主体偏小，但仍可初步判读",
        "1726670": "主体偏小，背景占比较高",
        "1821773": "器械接触主体，局部信息可能受遮挡",
        "1905040": "局部偏暗，细节对比一般",
        "1913309": "主体偏小且背景较多",
        "2016282": "器械遮挡局部组织，建议复核",
        "2032976": "碎片化取材，整体结构连续性较弱",
        "2102321": "多碎片取材，组织成分混杂，建议复核",
        "2130975": "主体偏小且暗色区域较多",
        "2137954": "主体偏小，背景反光干扰",
        "2219959": "主体长条状且局部细节有限",
        "2305325": "主体偏小，背景占比较高",
        "2316953": "主体偏小且以外观红褐色区域为主",
        "2323861": "主体偏小偏暗，背景水渍较多",
        "2329253": "主体偏小，建议低置信复核",
        "2350704": "主体偏小且局部反光明显",
        "2400934": "主体偏小，背景占比较高",
        "2404139": "主体偏小，建议复核",
        "2411691-2": "主体呈细长状，切面信息有限",
        "2440727": "主体偏小，背景反光较明显",
        "2443514": "主体偏暗，切面细节一般",
        "2451754": "主体偏小，背景占比较高",
        "2452239": "主体偏小，建议低置信复核",
        "2510854": "主体偏小且远景感明显",
        "2518212": "局部偏亮/反光，切面细节一般",
        "2520297": "主体偏小，背景占比较高",
        "2613517": "水渍和反光较明显，建议复核",
    }

    def label(cid: str) -> str:
        if cid in retake:
            return "retake_required"
        if cid in borderline:
            return "readable_review"
        return "readable_auto"

    def reason(cid: str) -> str:
        if cid in retake:
            return retake[cid]
        if cid in borderline:
            return borderline[cid]
        return "主体和切面信息基本可见，可进入自动判读"

    def gate_action(label_name: str) -> str:
        if label_name == "retake_required":
            return "不输出诊断；建议重拍或补充标准切面图"
        if label_name == "readable_review":
            return "可进入模型，但结果建议标记低置信或人工复核"
        return "可进入模型自动判读"

    df["gpt_quality_label_v1"] = df["original_case_id"].map(label)
    df["gpt_quality_reason_v1"] = df["original_case_id"].map(reason)
    df["gpt_quality_gate_action_v1"] = df["gpt_quality_label_v1"].map(gate_action)
    df["gpt_quality_review_rounds"] = "round1_blind_global;round2_recheck_retake_and_borderline;round3_metrics_audit_no_label_change"

    strict = df[df["strict_task7_eval"] == 1].copy()
    prob = pd.to_numeric(strict["locked162_blend_prob_high"], errors="coerce")
    strict["confidence_margin"] = (prob - 0.5).abs()

    eval_rows = []
    subsets = [
        ("strict_all", strict),
        ("gpt_auto_only", strict[strict["gpt_quality_label_v1"] == "readable_auto"]),
        ("gpt_auto_plus_review", strict[strict["gpt_quality_label_v1"] != "retake_required"]),
        ("gpt_review_only", strict[strict["gpt_quality_label_v1"] == "readable_review"]),
        ("gpt_retake_required", strict[strict["gpt_quality_label_v1"] == "retake_required"]),
    ]
    for name, sub in subsets:
        row = metrics(sub)
        row["subset"] = name
        row["coverage_vs_strict"] = len(sub) / len(strict)
        eval_rows.append(row)

    for margin in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        sub = strict[(strict["gpt_quality_label_v1"] != "retake_required") & (strict["confidence_margin"] >= margin)]
        row = metrics(sub)
        row["subset"] = f"gpt_auto_plus_review_margin_ge_{margin:.2f}"
        row["coverage_vs_strict"] = len(sub) / len(strict)
        eval_rows.append(row)

    labels_path = outdir / "external_gpt_quality_labels_v1.csv"
    eval_path = outdir / "external_gpt_quality_gate_eval_v1.csv"
    df.to_csv(labels_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(eval_rows).to_csv(eval_path, index=False, encoding="utf-8-sig")

    retake_path = outdir / "external_gpt_quality_retake_required_v1.csv"
    review_path = outdir / "external_gpt_quality_readable_review_v1.csv"
    keep_cols = [
        "case_id",
        "original_case_id",
        "image_name",
        "task_l6_label",
        "task_l7_label",
        "gpt_quality_label_v1",
        "gpt_quality_reason_v1",
        "gpt_quality_gate_action_v1",
        "locked162_blend_prob_high",
        "locked162_blend_pred_idx",
        "locked162_blend_correct",
        "local_path",
    ]
    df[df["gpt_quality_label_v1"] == "retake_required"][keep_cols].to_csv(retake_path, index=False, encoding="utf-8-sig")
    df[df["gpt_quality_label_v1"] == "readable_review"][keep_cols].to_csv(review_path, index=False, encoding="utf-8-sig")

    print("labels:", labels_path)
    print("eval:", eval_path)
    print(df["gpt_quality_label_v1"].value_counts().to_string())
    print(pd.DataFrame(eval_rows).to_string(index=False))


if __name__ == "__main__":
    main()
