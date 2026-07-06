from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(".").resolve()
OUT = ROOT / "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/20_old_guardrail_summary_20260522"


def first_row(path: str) -> dict[str, object] | None:
    p = ROOT / path
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def report_json(path: str) -> dict[str, object] | None:
    p = ROOT / path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def metric_row(
    experiment: str,
    model_policy: str,
    old_accuracy: float | None,
    old_bacc: float | None,
    third_accuracy: float | None,
    third_bacc: float | None,
    notes: str,
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "model_policy": model_policy,
        "old_accuracy": old_accuracy,
        "old_balanced_accuracy": old_bacc,
        "third_accuracy": third_accuracy,
        "third_balanced_accuracy": third_bacc,
        "old_guardrail_0p90_pass": bool(old_accuracy is not None and old_accuracy >= 0.90),
        "old_guardrail_0p88_pass": bool(old_accuracy is not None and old_accuracy >= 0.88),
        "notes": notes,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    # Official old-data No.64-style nested route result before third-batch adaptation.
    old64 = first_row(
        "outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521/nested_route_summary.csv"
    )
    if old64:
        rows.append(
            metric_row(
                "old_internal_no64_reference",
                "No.64 image-only hardcore reviewer nested route",
                float(old64["accuracy"]),
                float(old64["balanced_accuracy"]),
                None,
                None,
                "旧数据内部 OOF 最高参照线；后续第三批适配不能牺牲这条线。",
            )
        )

    # Frozen 64-style reimplementation used for the first third-batch external probe.
    ext64 = report_json(
        "outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521/external_64style_report.json"
    )
    if ext64:
        old_m = ext64["old_oof_selected_policy"]
        third_m = ext64["third_external_final_metrics"]
        rows.append(
            metric_row(
                "third_external_old_only_wpc64_retrain",
                "old-only frozen whole+crop 64-style retraining",
                float(old_m["accuracy"]),
                float(old_m["balanced_accuracy"]),
                float(third_m["accuracy"]),
                float(third_m["balanced_accuracy"]),
                "这是无第三批训练的外部测试基线，但不是原 No.64 完整流程。",
            )
        )

    # Adapt72 logreg: best external holdout but old OOF is modest.
    adapt72 = first_row(
        "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/02_third_logreg_fast_20260521/third_adaptation_logreg_summary.csv"
    )
    if adapt72:
        rows.append(
            metric_row(
                "adapt72_logreg_fast",
                "old + third adapt72 high-focus, frozen whole+crop logreg",
                float(adapt72.get("old_285_oof_accuracy", adapt72.get("old_oof_accuracy", float("nan")))),
                float(adapt72.get("old_285_oof_balanced_accuracy", adapt72.get("old_oof_balanced_accuracy", float("nan")))),
                float(adapt72.get("holdout_accuracy", float("nan"))),
                float(adapt72.get("holdout_balanced_accuracy", float("nan"))),
                "第三批 holdout 有提升，但旧数据 OOF 没达到旧模型守门线。",
            )
        )

    # Profile scan best rows, keep both OOF-selected and holdout-best references.
    profile_report = report_json(
        "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/17_adaptation_profile_scan_20260521/adaptation_profile_scan_report.json"
    )
    if profile_report:
        for key, label in [
            ("selected_by_oof", "profile_scan_selected_by_training_oof"),
            ("best_holdout_accuracy_for_reference", "profile_scan_best_holdout_accuracy_reference"),
            ("best_holdout_balanced_accuracy_for_reference", "profile_scan_best_holdout_bacc_reference"),
        ]:
            m = profile_report[key]
            rows.append(
                metric_row(
                    label,
                    f"{m['profile']} / {m['model_kind']} / repeat {m['repeat_adapt']} / {m['threshold_objective']}",
                    float(m["old_oof_accuracy"]),
                    float(m["old_oof_balanced_accuracy"]),
                    float(m["holdout_accuracy"]),
                    float(m["holdout_balanced_accuracy"]),
                    "第三批划分扫描结果；holdout-best 只能作为参考，不能当最终模型选择依据。",
                )
            )

    # Adapt180 repeated stability mean.
    stability = report_json(
        "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/18_adapt180_stability_20260521/adapt180_stability_report.json"
    )
    if stability:
        agg = {r["index"]: r for r in stability["aggregate"]}
        mean = agg["mean"]
        rows.append(
            metric_row(
                "adapt180_repeated_holdout_mean",
                "adapt180 logreg C=0.0003 repeat4, 10 deterministic splits mean",
                float(mean["old_oof_accuracy"]),
                None,
                float(mean["holdout_accuracy"]),
                float(mean["holdout_balanced_accuracy"]),
                "多次划分平均结果，比单次 0.90 更可信；旧数据 OOF 仍明显低于 No.64。",
            )
        )
        rep = stability.get("repeated_holdout_majority_vote", {})
        rows.append(
            metric_row(
                "adapt180_repeated_case_majority",
                "adapt180 repeated holdout per-case majority vote",
                None,
                None,
                float(rep.get("accuracy")),
                float(rep.get("balanced_accuracy")),
                "300 个第三批病例中反复作为 holdout 后聚合；没有旧数据指标，不可替代旧模型。",
            )
        )

    # Third outer CV with old always in training.
    outer = first_row(
        "outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/19_third_outer_cv_with_old_20260521/third_outer_cv_summary.csv"
    )
    if outer:
        rows.append(
            metric_row(
                "third_outer_cv_with_old_best_acc",
                f"{outer['kind']} repeat {outer['repeat_adapt']} / {outer['threshold_objective']}",
                None,
                None,
                float(outer["third_oof_accuracy"]),
                float(outer["third_oof_balanced_accuracy"]),
                "第三批 5 折外层 CV；旧数据始终参与训练，但这个表本身不证明旧数据不掉。",
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "task7_old_guardrail_comparison_20260522.csv", index=False, encoding="utf-8-sig")

    summary = {
        "old_guardrail_policy": {
            "primary_guardrail": "旧数据 No.64 image-only hardcore reviewer nested route OOF accuracy >= 0.90",
            "soft_guardrail": "若为了第三批泛化短期探索，可临时看 >=0.88，但不能作为最终替代模型。",
            "selection_rule": "任何第三批适配模型必须同时报告旧数据 OOF 和第三批外部/CV；旧数据不过线时只能作为实验分支，不能替换主流程。",
        },
        "rows": rows,
        "output_csv": str(OUT / "task7_old_guardrail_comparison_20260522.csv"),
    }
    (OUT / "task7_old_guardrail_summary_20260522.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Task7 旧数据守门线记录",
        "",
        "我们后续不再只按第三批表现选择模型。旧数据 No.64 流程的 OOF 约 0.926 是主守门线；第三批适配模型如果旧数据不过线，只能作为实验分支。",
        "",
        "| 实验 | 模型/策略 | 旧数据ACC | 旧数据BACC | 第三批ACC | 第三批BACC | 结论 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        def fmt(v: object) -> str:
            return "" if v is None or pd.isna(v) else f"{float(v):.4f}"

        verdict = "可作为主线候选" if r["old_guardrail_0p90_pass"] else "不能替换旧数据主模型"
        md.append(
            f"| {r['experiment']} | {r['model_policy']} | {fmt(r['old_accuracy'])} | {fmt(r['old_balanced_accuracy'])} | "
            f"{fmt(r['third_accuracy'])} | {fmt(r['third_balanced_accuracy'])} | {verdict} |"
        )
    md.append("")
    md.append("下一步所有新实验都按这个表补充：第三批收益必须和旧数据损失一起看。")
    (OUT / "Task7旧数据守门线记录_2026-05-22.md").write_text("\n".join(md), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
