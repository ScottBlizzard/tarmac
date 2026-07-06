from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "batch1_batch2_task567_20260514"
OUT_DIR = ROOT / "reports" / "ThymicGross" / "batch1_batch2_task67_reports_20260514"


@dataclass(frozen=True)
class RunSpec:
    code: str
    name: str
    path: Path


TASK5_CLASSES = ["A_AB", "B123", "TC"]
TASK6_CLASSES = ["A", "AB", "B1", "B2", "B3", "TC"]
TASK7_CLASSES = ["low_risk_group", "high_risk_group"]


TASK5_RUNS = [
    RunSpec("01_srx50_whole", "SE-ResNeXt50 + whole", RUN_ROOT / "7model_task567_runs" / "task5_threeclass" / "01_srx50_whole"),
    RunSpec("02_srx50_whole_plus_crop", "SE-ResNeXt50 + whole+crop", RUN_ROOT / "7model_task567_runs" / "task5_threeclass" / "02_srx50_whole_plus_crop"),
    RunSpec("03_dino_vitb14_whole", "DINOv2 vitb14 + whole", RUN_ROOT / "7model_task567_runs" / "task5_threeclass" / "03_dino_vitb14_whole"),
    RunSpec("04_dino_vits14_whole", "DINOv2 vits14 + whole", RUN_ROOT / "7model_task567_runs" / "task5_threeclass" / "04_dino_vits14_whole"),
    RunSpec("05_dino_vitl14_whole", "DINOv2 vitl14 + whole", RUN_ROOT / "7model_task567_runs" / "task5_threeclass" / "05_dino_vitl14_whole"),
    RunSpec("06_dino_vits_vitb_concat", "DINOv2 vits14+vitb14 concat", RUN_ROOT / "7model_task567_runs" / "task5_threeclass" / "06_dino_vits_vitb_concat"),
    RunSpec("07_dino_vits_vitb_experience_aux", "DINOv2 vits14+vitb14 + experience-aux", RUN_ROOT / "7model_task567_runs" / "task5_threeclass" / "07_dino_vits_vitb_experience_aux"),
]


TASK6_RUNS = [
    RunSpec("01_srx50_whole", "SE-ResNeXt50 + whole", RUN_ROOT / "7model_task567_runs" / "task6_sixclass" / "01_srx50_whole"),
    RunSpec("02_srx50_whole_plus_crop", "SE-ResNeXt50 + whole+crop", RUN_ROOT / "7model_task567_runs" / "task6_sixclass" / "02_srx50_whole_plus_crop"),
    RunSpec("03_dino_vitb14_whole", "DINOv2 vitb14 + whole", RUN_ROOT / "dino_task67_runs" / "task6_sixclass" / "03_dino_vitb14_whole"),
    RunSpec("04_dino_vits14_whole", "DINOv2 vits14 + whole", RUN_ROOT / "dino_task67_runs" / "task6_sixclass" / "04_dino_vits14_whole"),
    RunSpec("05_dino_vitl14_whole", "DINOv2 vitl14 + whole", RUN_ROOT / "dino_task67_runs" / "task6_sixclass" / "05_dino_vitl14_whole"),
    RunSpec("06_dino_vits_vitb_concat", "DINOv2 vits14+vitb14 concat", RUN_ROOT / "dino_task67_runs" / "task6_sixclass" / "06_dino_vits_vitb_concat"),
    RunSpec("07_dino_vits_vitb_experience_aux", "DINOv2 vits14+vitb14 + experience-aux", RUN_ROOT / "dino_task67_runs" / "task6_sixclass" / "07_dino_vits_vitb_experience_aux"),
]


TASK7_RUNS = [
    RunSpec("01_srx50_whole", "SE-ResNeXt50 + whole", RUN_ROOT / "7model_task567_runs" / "task7_lowhigh_tc" / "01_srx50_whole"),
    RunSpec("02_srx50_whole_plus_crop", "SE-ResNeXt50 + whole+crop", RUN_ROOT / "7model_task567_runs" / "task7_lowhigh_tc" / "02_srx50_whole_plus_crop"),
    RunSpec("03_dino_vitb14_whole", "DINOv2 vitb14 + whole", RUN_ROOT / "dino_task67_runs" / "task7_lowhigh_tc" / "03_dino_vitb14_whole"),
    RunSpec("04_dino_vits14_whole", "DINOv2 vits14 + whole", RUN_ROOT / "dino_task67_runs" / "task7_lowhigh_tc" / "04_dino_vits14_whole"),
    RunSpec("05_dino_vitl14_whole", "DINOv2 vitl14 + whole", RUN_ROOT / "dino_task67_runs" / "task7_lowhigh_tc" / "05_dino_vitl14_whole"),
    RunSpec("06_dino_vits_vitb_concat", "DINOv2 vits14+vitb14 concat", RUN_ROOT / "dino_task67_runs" / "task7_lowhigh_tc" / "06_dino_vits_vitb_concat"),
    RunSpec("07_dino_vits_vitb_experience_aux", "DINOv2 vits14+vitb14 + experience-aux", RUN_ROOT / "dino_task67_runs" / "task7_lowhigh_tc" / "07_dino_vits_vitb_experience_aux"),
]


def fmt(x: float | int | str) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def load_metric(run: RunSpec) -> pd.Series:
    df = pd.read_csv(run.path / "oof_metrics.csv")
    row = df[(df["split"] == "test_oof") & (df["level"] == "case") & (df["aggregation"] == "mean")]
    if row.empty:
        row = df.iloc[[0]]
    return row.iloc[0]


def load_predictions(run: RunSpec) -> pd.DataFrame:
    return pd.read_csv(run.path / "oof_case_predictions_mean.csv", dtype={"case_id": str})


def class_report_frame(preds: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    report = classification_report(
        preds["label_idx"],
        preds["pred_idx"],
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls in classes:
        rows.append(
            {
                "类别": cls,
                "例数": int(report[cls]["support"]),
                "Precision": report[cls]["precision"],
                "Recall": report[cls]["recall"],
                "F1": report[cls]["f1-score"],
            }
        )
    return pd.DataFrame(rows)


def confusion_frame(preds: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(preds["label_idx"], preds["pred_idx"], labels=list(range(len(classes))))
    out = pd.DataFrame(cm, columns=classes)
    out.insert(0, "真值 \\ 预测", classes)
    return out


def confusion_row_norm_frame(preds: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(preds["label_idx"], preds["pred_idx"], labels=list(range(len(classes)))).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = cm / row_sums.clip(min=1)
    out = pd.DataFrame(norm, columns=classes)
    out.insert(0, "真值 \\ 预测", classes)
    return out


def save_artifacts(task_key: str, run: RunSpec, preds: pd.DataFrame, classes: list[str]) -> None:
    task_dir = OUT_DIR / task_key
    task_dir.mkdir(parents=True, exist_ok=True)
    confusion_frame(preds, classes).to_csv(task_dir / f"{run.code}_confusion_counts.csv", index=False, encoding="utf-8-sig")
    confusion_row_norm_frame(preds, classes).to_csv(task_dir / f"{run.code}_confusion_row_norm.csv", index=False, encoding="utf-8-sig")
    class_report_frame(preds, classes).to_csv(task_dir / f"{run.code}_per_class_metrics.csv", index=False, encoding="utf-8-sig")


def task6_summary_rows() -> pd.DataFrame:
    rows = []
    for run in TASK6_RUNS:
        m = load_metric(run)
        rows.append(
            {
                "模型": run.name,
                "Accuracy": float(m["accuracy"]),
                "Balanced Accuracy": float(m["balanced_accuracy"]),
                "Macro-Precision": float(m["macro_precision"]),
                "Macro-Recall": float(m["macro_recall"]),
                "Macro-F1": float(m["macro_f1"]),
                "Macro-AUC": float(m["macro_auc"]),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["Macro-F1", "Macro-AUC"], ascending=False).reset_index(drop=True)


def task5_summary_rows() -> pd.DataFrame:
    rows = []
    for run in TASK5_RUNS:
        m = load_metric(run)
        rows.append(
            {
                "模型": run.name,
                "Accuracy": float(m["accuracy"]),
                "Balanced Accuracy": float(m["balanced_accuracy"]),
                "Macro-Precision": float(m["macro_precision"]),
                "Macro-Recall": float(m["macro_recall"]),
                "Macro-F1": float(m["macro_f1"]),
                "Macro-AUC": float(m["macro_auc"]),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["Macro-F1", "Macro-AUC"], ascending=False).reset_index(drop=True)


def task7_summary_rows() -> pd.DataFrame:
    rows = []
    for run in TASK7_RUNS:
        m = load_metric(run)
        rows.append(
            {
                "模型": run.name,
                "Accuracy": float(m["accuracy"]),
                "Balanced Accuracy": float(m["balanced_accuracy"]),
                "AUC": float(m["auc"]),
                "Sensitivity": float(m["sensitivity"]),
                "Specificity": float(m["specificity"]),
                "Precision": float(m["precision"]),
                "F1": float(m["f1"]),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["Accuracy", "AUC"], ascending=False).reset_index(drop=True)


def build_task5_report() -> str:
    summary = task5_summary_rows()
    best = summary.iloc[0]
    lines = [
        "# Task5 三分类模型结果与混淆矩阵报告",
        "",
        "日期：2026-05-14",
        "",
        "## 1. 任务定义",
        "",
        "`Task5`：`A_AB` vs `B123` vs `TC`。",
        "",
        "本轮使用第一批与第二批合并后的冻结数据，病例级样本量为 `285` 例；所有多图病例统一取该病例第二张图作为训练/评估输入。",
        "",
        "类别分布：`A_AB=94`, `B123=134`, `TC=57`。",
        "",
        "评估协议：`5-fold patient-level cross-validation`；结果采用 `case-level mean probability` 的 OOF 指标。",
        "",
        "## 2. 结果概览",
        "",
        md_table(summary),
        "",
        "当前最好结果：",
        "",
        f"- `{best['模型']}`",
        f"- `Accuracy = {best['Accuracy']:.4f}`",
        f"- `Balanced Accuracy = {best['Balanced Accuracy']:.4f}`",
        f"- `Macro-F1 = {best['Macro-F1']:.4f}`",
        f"- `Macro-AUC = {best['Macro-AUC']:.4f}`",
        "",
        "## 3. 各模型混淆矩阵与每类表现",
        "",
        "混淆矩阵行表示真实标签，列表示预测标签。",
        "",
    ]
    for run in TASK5_RUNS:
        preds = load_predictions(run)
        save_artifacts("task5_threeclass", run, preds, TASK5_CLASSES)
        m = load_metric(run)
        lines += [
            f"### {run.name}",
            "",
            f"- `Accuracy = {float(m['accuracy']):.4f}`",
            f"- `Balanced Accuracy = {float(m['balanced_accuracy']):.4f}`",
            f"- `Macro-F1 = {float(m['macro_f1']):.4f}`",
            f"- `Macro-AUC = {float(m['macro_auc']):.4f}`",
            "",
            "confusion matrix：",
            "",
            md_table(confusion_frame(preds, TASK5_CLASSES)),
            "",
            "每类表现：",
            "",
            md_table(class_report_frame(preds, TASK5_CLASSES)),
            "",
        ]
    lines += [
        "## 4. 阶段性判断",
        "",
        "在当前合并数据上，Task5 三分类的最优模型为 `DINOv2 vits14+vitb14 concat`，整体表现优于 SE-ResNeXt50 系列。",
        "",
        "`A_AB` 与 `B123` 均属于胸腺瘤内部类别，二者之间仍存在主要混淆；`TC` 与胸腺瘤类别的区分相对更稳定，但样本量仍小于 `A_AB` 与 `B123`，后续扩充数据后需要重新评估稳定性。",
        "",
    ]
    return "\n".join(lines)


def build_task6_report() -> str:
    summary = task6_summary_rows()
    best = summary.iloc[0]
    lines = [
        "# Task6 六分类模型结果与混淆矩阵报告",
        "",
        "日期：2026-05-14",
        "",
        "## 1. 任务定义",
        "",
        "`Task6`：`A` vs `AB` vs `B1` vs `B2` vs `B3` vs `TC`。",
        "",
        "本轮使用第一批与第二批合并后的冻结数据，病例级样本量为 `285` 例；所有多图病例统一取该病例第二张图作为训练/评估输入。",
        "",
        "类别分布：`A=44`, `AB=50`, `B1=50`, `B2=60`, `B3=24`, `TC=57`。",
        "",
        "评估协议：`5-fold patient-level cross-validation`；结果采用 `case-level mean probability` 的 OOF 指标。",
        "",
        "## 2. 结果概览",
        "",
        md_table(summary),
        "",
        "当前最好结果：",
        "",
        f"- `{best['模型']}`",
        f"- `Accuracy = {best['Accuracy']:.4f}`",
        f"- `Balanced Accuracy = {best['Balanced Accuracy']:.4f}`",
        f"- `Macro-F1 = {best['Macro-F1']:.4f}`",
        f"- `Macro-AUC = {best['Macro-AUC']:.4f}`",
        "",
        "## 3. 各模型混淆矩阵与每类表现",
        "",
        "混淆矩阵行表示真实标签，列表示预测标签。",
        "",
    ]
    for run in TASK6_RUNS:
        preds = load_predictions(run)
        save_artifacts("task6_sixclass", run, preds, TASK6_CLASSES)
        m = load_metric(run)
        lines += [
            f"### {run.name}",
            "",
            f"- `Accuracy = {float(m['accuracy']):.4f}`",
            f"- `Balanced Accuracy = {float(m['balanced_accuracy']):.4f}`",
            f"- `Macro-F1 = {float(m['macro_f1']):.4f}`",
            f"- `Macro-AUC = {float(m['macro_auc']):.4f}`",
            "",
            "confusion matrix：",
            "",
            md_table(confusion_frame(preds, TASK6_CLASSES)),
            "",
            "每类表现：",
            "",
            md_table(class_report_frame(preds, TASK6_CLASSES)),
            "",
        ]
    lines += [
        "## 4. 阶段性判断",
        "",
        "在当前合并数据上，`DINOv2 vits14+vitb14 + experience-aux` 的 Macro-F1 最高，但绝对值仍偏低，提示六分类仍明显受细粒度类别相似性与样本量限制影响。",
        "",
        "`B2/B3/TC` 以及 `A/AB/B1` 的邻近类别混淆仍是后续主要误差来源；其中 `B3` 样本量最少，是影响稳定性的关键因素之一。",
        "",
    ]
    return "\n".join(lines)


def build_task7_report() -> str:
    summary = task7_summary_rows()
    best_acc = summary.iloc[0]
    best_auc = summary.sort_values(["AUC", "Accuracy"], ascending=False).iloc[0]
    lines = [
        "# Task7 低危/高危二分类模型结果与混淆矩阵报告",
        "",
        "日期：2026-05-14",
        "",
        "## 1. 任务定义",
        "",
        "`Task7`：`low_risk_group` vs `high_risk_group`。",
        "",
        "本轮使用第一批与第二批合并后的冻结数据，病例级样本量为 `285` 例；所有多图病例统一取该病例第二张图作为训练/评估输入。",
        "",
        "类别分布：`low_risk_group=144`, `high_risk_group=141`。",
        "",
        "评估协议：`5-fold patient-level cross-validation`；结果采用 `case-level mean probability` 的 OOF 指标。",
        "",
        "## 2. 结果概览",
        "",
        md_table(summary),
        "",
        "当前最好结果：",
        "",
        f"- Accuracy 最好：`{best_acc['模型']}`，`Accuracy = {best_acc['Accuracy']:.4f}`, `AUC = {best_acc['AUC']:.4f}`",
        f"- AUC 最好：`{best_auc['模型']}`，`AUC = {best_auc['AUC']:.4f}`, `Accuracy = {best_auc['Accuracy']:.4f}`",
        "",
        "## 3. 各模型混淆矩阵与每类表现",
        "",
        "混淆矩阵行表示真实标签，列表示预测标签。",
        "",
    ]
    for run in TASK7_RUNS:
        preds = load_predictions(run)
        save_artifacts("task7_lowhigh_tc", run, preds, TASK7_CLASSES)
        m = load_metric(run)
        lines += [
            f"### {run.name}",
            "",
            f"- `Accuracy = {float(m['accuracy']):.4f}`",
            f"- `Balanced Accuracy = {float(m['balanced_accuracy']):.4f}`",
            f"- `AUC = {float(m['auc']):.4f}`",
            f"- `Sensitivity = {float(m['sensitivity']):.4f}`",
            f"- `Specificity = {float(m['specificity']):.4f}`",
            f"- `F1 = {float(m['f1']):.4f}`",
            "",
            "confusion matrix：",
            "",
            md_table(confusion_frame(preds, TASK7_CLASSES)),
            "",
            "每类表现：",
            "",
            md_table(class_report_frame(preds, TASK7_CLASSES)),
            "",
        ]
    lines += [
        "## 4. 阶段性判断",
        "",
        "在当前合并数据上，`DINOv2 vits14+vitb14 concat` 取得最高 Accuracy 和 F1，`DINOv2 vits14+vitb14 + experience-aux` 取得最高 AUC。",
        "",
        "SE-ResNeXt50 的 `whole+crop` 未带来稳定收益，说明当前二分类主增益主要来自更强的 DINOv2 表征，而不是简单增加 crop 输入。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    task5_summary_rows().to_csv(OUT_DIR / "task5_model_metrics_summary.csv", index=False, encoding="utf-8-sig")
    task6_summary_rows().to_csv(OUT_DIR / "task6_model_metrics_summary.csv", index=False, encoding="utf-8-sig")
    task7_summary_rows().to_csv(OUT_DIR / "task7_model_metrics_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "2026-05-14_Task5三分类模型结果与混淆矩阵报告.md").write_text(build_task5_report(), encoding="utf-8")
    (OUT_DIR / "2026-05-14_Task6六分类模型结果与混淆矩阵报告.md").write_text(build_task6_report(), encoding="utf-8")
    (OUT_DIR / "2026-05-14_Task7低危高危模型结果与混淆矩阵报告.md").write_text(build_task7_report(), encoding="utf-8")
    print(f"Wrote reports to {OUT_DIR}")


if __name__ == "__main__":
    main()
