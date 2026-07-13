from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score


SUBTYPE_ORDER = ("A", "AB", "B1", "B2", "B3", "TC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the locked sequential AB/TC/fallback experiment.")
    parser.add_argument("--fivefold-dir", required=True)
    parser.add_argument("--lodo-dir", required=True)
    parser.add_argument("--c1-oof", required=True)
    parser.add_argument("--c1-lodo", required=True)
    parser.add_argument("--c2-oof", required=True)
    parser.add_argument("--c2-lodo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(y_true: Iterable[int], probability: Iterable[float]) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_prediction(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    label_column = "label_idx" if "label_idx" in frame else "risk_label"
    required = {"case_id", label_column, "source_dataset", "task_l6_label", "prob_high"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    result = frame.copy()
    numeric_label = pd.to_numeric(result[label_column], errors="coerce")
    if numeric_label.isna().any():
        mapped_label = result[label_column].astype(str).str.lower().map(
            {
                "low_risk_group": 0,
                "low": 0,
                "high_risk_group": 1,
                "high": 1,
            }
        )
        numeric_label = numeric_label.fillna(mapped_label)
    if numeric_label.isna().any():
        values = sorted(result.loc[numeric_label.isna(), label_column].astype(str).unique().tolist())
        raise ValueError(f"Could not normalize labels in {path}: {values}")
    result["label_idx"] = numeric_label.astype(int)
    result["prob_high"] = pd.to_numeric(result["prob_high"], errors="raise")
    result["model"] = name
    if result["case_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated case IDs")
    return result


def align(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    columns = ["case_id", "label_idx", "source_dataset", "task_l6_label", "prob_high"]
    left = reference[columns].rename(columns={"prob_high": "prob_reference"})
    right = candidate[["case_id", "label_idx", "prob_high"]].rename(
        columns={"label_idx": "label_candidate", "prob_high": "prob_candidate"}
    )
    merged = left.merge(right, on="case_id", how="inner", validate="one_to_one")
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError("Prediction cohorts do not align")
    if not np.array_equal(merged["label_idx"], merged["label_candidate"]):
        raise ValueError("Prediction labels do not align")
    return merged


def stratified_bootstrap_difference(
    aligned: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strata = [group.index.to_numpy(dtype=int) for _, group in aligned.groupby(["source_dataset", "label_idx"])]
    rows = []
    for repetition in range(repetitions):
        sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
        boot = aligned.loc[sampled]
        reference = metrics(boot["label_idx"], boot["prob_reference"])
        candidate = metrics(boot["label_idx"], boot["prob_candidate"])
        rows.append(
            {
                "repetition": repetition,
                "delta_accuracy": candidate["accuracy"] - reference["accuracy"],
                "delta_balanced_accuracy": candidate["balanced_accuracy"] - reference["balanced_accuracy"],
                "delta_auc": candidate["auc"] - reference["auc"],
                "delta_sensitivity": candidate["sensitivity"] - reference["sensitivity"],
                "delta_specificity": candidate["specificity"] - reference["specificity"],
            }
        )
    return pd.DataFrame(rows)


def interval_summary(samples: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for column in samples.columns:
        if not column.startswith("delta_"):
            continue
        values = samples[column].to_numpy(dtype=float)
        result[column] = {
            "mean": float(np.mean(values)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    return result


def comparison_rows(protocol: str, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows = []
    for model, frame in frames.items():
        row = metrics(frame["label_idx"], frame["prob_high"])
        row.update({"protocol": protocol, "model": model})
        rows.append(row)
    return rows


def subtype_rows(protocol: str, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows = []
    for model, frame in frames.items():
        for subtype in SUBTYPE_ORDER:
            subset = frame[frame["task_l6_label"] == subtype]
            prediction = (subset["prob_high"].to_numpy() >= 0.5).astype(int)
            rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "subtype": subtype,
                    "n": int(len(subset)),
                    "risk_accuracy": float(np.mean(prediction == subset["label_idx"].to_numpy())),
                    "mean_prob_high": float(subset["prob_high"].mean()),
                }
            )
    return rows


def source_rows(protocol: str, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows = []
    for model, frame in frames.items():
        for source, subset in frame.groupby("source_dataset", sort=False):
            row = metrics(subset["label_idx"], subset["prob_high"])
            row.update({"protocol": protocol, "model": model, "source_dataset": source})
            rows.append(row)
    return rows


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = {
        "C1": load_prediction(Path(args.c1_oof), "C1"),
        "C2": load_prediction(Path(args.c2_oof), "C2"),
        "Sequential": load_prediction(Path(args.fivefold_dir) / "oof_predictions.csv", "Sequential"),
    }
    lodo = {
        "C1": load_prediction(Path(args.c1_lodo), "C1"),
        "C2": load_prediction(Path(args.c2_lodo), "C2"),
        "Sequential": load_prediction(Path(args.lodo_dir) / "oof_predictions.csv", "Sequential"),
    }
    for protocol_frames in (oof, lodo):
        align(protocol_frames["C2"], protocol_frames["Sequential"])

    comparison = pd.DataFrame(
        comparison_rows("fivefold_oof", oof) + comparison_rows("source_lodo", lodo)
    )
    subtype = pd.DataFrame(subtype_rows("fivefold_oof", oof) + subtype_rows("source_lodo", lodo))
    source = pd.DataFrame(source_rows("fivefold_oof", oof) + source_rows("source_lodo", lodo))
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    subtype.to_csv(output_dir / "subtype_comparison.csv", index=False)
    source.to_csv(output_dir / "source_comparison.csv", index=False)

    bootstrap_outputs = {}
    for protocol, protocol_frames in (("fivefold_oof", oof), ("source_lodo", lodo)):
        aligned = align(protocol_frames["C2"], protocol_frames["Sequential"])
        samples = stratified_bootstrap_difference(
            aligned, args.bootstrap_repetitions, args.seed + (0 if protocol == "fivefold_oof" else 1)
        )
        samples.to_csv(output_dir / f"bootstrap_{protocol}.csv", index=False)
        bootstrap_outputs[protocol] = interval_summary(samples)
    write_json(output_dir / "bootstrap_summary.json", bootstrap_outputs)

    sequential_lodo = comparison[
        (comparison["protocol"] == "source_lodo") & (comparison["model"] == "Sequential")
    ].iloc[0]
    lodo_source = source[(source["protocol"] == "source_lodo") & (source["model"] == "Sequential")]
    lodo_subtype = subtype[(subtype["protocol"] == "source_lodo") & (subtype["model"] == "Sequential")]
    subtype_lookup = lodo_subtype.set_index("subtype")["risk_accuracy"].to_dict()
    gates = {
        "oof_bacc_gt_c2": bool(
            comparison.loc[
                (comparison["protocol"] == "fivefold_oof") & (comparison["model"] == "Sequential"),
                "balanced_accuracy",
            ].iloc[0]
            > comparison.loc[
                (comparison["protocol"] == "fivefold_oof") & (comparison["model"] == "C2"),
                "balanced_accuracy",
            ].iloc[0]
        ),
        "lodo_bacc_at_least_0_764": bool(sequential_lodo["balanced_accuracy"] >= 0.764),
        "lodo_sensitivity_at_least_0_7354": bool(sequential_lodo["sensitivity"] >= 0.7354),
        "minimum_source_bacc_at_least_0_7083": bool(lodo_source["balanced_accuracy"].min() >= 0.7083),
        "b1_accuracy_at_least_0_500": bool(subtype_lookup.get("B1", 0.0) >= 0.500),
        "b2_accuracy_at_least_0_663": bool(subtype_lookup.get("B2", 0.0) >= 0.663),
    }
    decision = "GO" if all(gates.values()) else "NO-GO"
    decision_record = {"decision": decision, "gates": gates}
    write_json(output_dir / "decision.json", decision_record)

    routing_oof = pd.read_csv(Path(args.fivefold_dir) / "routing_metrics.csv")
    routing_lodo = pd.read_csv(Path(args.lodo_dir) / "routing_metrics.csv")
    report = [
        "# AB/TC 顺序专家与六亚型覆盖二分类兜底实验结果",
        "",
        f"正式判定：**{decision}**",
        "",
        "## 总体结果",
        "",
        markdown_table(
            comparison,
            ["protocol", "model", "accuracy", "balanced_accuracy", "auc", "sensitivity", "specificity"],
        ),
        "",
        "## Source-LODO 亚型结果",
        "",
        markdown_table(lodo_subtype, ["model", "subtype", "n", "risk_accuracy", "mean_prob_high"]),
        "",
        "## Source-LODO 来源结果",
        "",
        markdown_table(lodo_source, ["model", "source_dataset", "n", "balanced_accuracy", "sensitivity", "specificity"]),
        "",
        "## 路由结果",
        "",
        "五折 OOF：",
        "",
        markdown_table(routing_oof, list(routing_oof.columns)),
        "",
        "Source-LODO：",
        "",
        markdown_table(routing_lodo, list(routing_lodo.columns)),
        "",
        "## 预注册门槛",
        "",
    ]
    report.extend([f"- {'PASS' if passed else 'FAIL'}：`{name}`" for name, passed in gates.items()])
    report.extend(
        [
            "",
            "## Bootstrap",
            "",
            "完整配对 bootstrap 结果见 `bootstrap_summary.json` 和对应 CSV。",
            "",
            "本报告只使用内部五折和来源留一结果。历史外部数据未参与训练、选择或本次分析。",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
