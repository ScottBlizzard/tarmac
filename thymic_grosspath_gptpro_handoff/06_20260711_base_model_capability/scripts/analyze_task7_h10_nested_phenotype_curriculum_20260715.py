from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze H10 Stage-2 nested curriculum.")
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    required = {
        "case_id",
        "source_dataset",
        "task_l6_label",
        "label_idx",
        "fold_id",
        "prob_high",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing {name} columns: {sorted(required - set(frame.columns))}")
    if len(frame) != 591 or frame["case_id"].duplicated().any():
        raise ValueError(f"{name} must contain 591 unique cases")
    frame = frame.sort_values("case_id").reset_index(drop=True)
    frame["label_idx"] = pd.to_numeric(frame["label_idx"], errors="raise").astype(int)
    frame["prob_high"] = pd.to_numeric(frame["prob_high"], errors="raise").astype(float)
    frame["pred_idx"] = (frame["prob_high"] >= 0.5).astype(int)
    frame["correct"] = frame["pred_idx"].eq(frame["label_idx"])
    return frame


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(probability) >= 0.5
    labels = np.asarray(labels, dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "n": int(len(labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "auc": float(roc_auc_score(labels, probability)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bacc(labels: np.ndarray, probability: np.ndarray) -> float:
    prediction = np.asarray(probability) >= 0.5
    low = np.asarray(labels) == 0
    return float((np.mean(~prediction[low]) + np.mean(prediction[~low])) / 2)


def bootstrap(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    baseline: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = frame["label_idx"].to_numpy(int)
    strata = [
        group.index.to_numpy(int)
        for _, group in frame.groupby("task_l6_label", sort=True)
    ]
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = np.concatenate(
            [rng.choice(rows, size=len(rows), replace=True) for rows in strata]
        )
        values[index] = bacc(labels[sampled], candidate[sampled]) - bacc(
            labels[sampled], baseline[sampled]
        )
    return {
        "delta_balanced_accuracy": bacc(labels, candidate) - bacc(labels, baseline),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_delta_gt_zero": float(np.mean(values > 0)),
        "replicates": int(replicates),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = load(Path(args.candidate_predictions), "candidate")
    baseline = load(Path(args.baseline_predictions), "baseline")
    identity = ["case_id", "label_idx", "task_l6_label", "source_dataset", "fold_id"]
    if not candidate[identity].equals(baseline[identity]):
        raise ValueError("Candidate and H10 baseline identities differ")

    candidate_probability = candidate["prob_high"].to_numpy(float)
    baseline_probability = baseline["prob_high"].to_numpy(float)
    overall = pd.DataFrame(
        [
            {"method": "H10_NESTED_PHENOTYPE_CURRICULUM", **metrics(candidate["label_idx"], candidate_probability)},
            {"method": "H10_INTERNAL_RISK_BALANCED", **metrics(baseline["label_idx"], baseline_probability)},
        ]
    )
    bootstrap_result = bootstrap(
        candidate,
        candidate_probability,
        baseline_probability,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )

    subtype_rows = []
    rescue_rows = []
    for subtype, rows in candidate.groupby("task_l6_label", sort=True).groups.items():
        rows = np.asarray(list(rows), dtype=int)
        for name, frame in [("candidate", candidate), ("baseline", baseline)]:
            group = frame.iloc[rows]
            subtype_rows.append(
                {
                    "method": name,
                    "task_l6_label": subtype,
                    "n": int(len(group)),
                    "correct_n": int(group["correct"].sum()),
                    "accuracy": float(group["correct"].mean()),
                }
            )
        candidate_ok = candidate.iloc[rows]["correct"].to_numpy(bool)
        baseline_ok = baseline.iloc[rows]["correct"].to_numpy(bool)
        rescue_rows.append(
            {
                "task_l6_label": subtype,
                "rescue": int(np.sum(candidate_ok & ~baseline_ok)),
                "harm": int(np.sum(~candidate_ok & baseline_ok)),
                "net": int(np.sum(candidate_ok) - np.sum(baseline_ok)),
            }
        )
    subtype = pd.DataFrame(subtype_rows)
    rescue = pd.DataFrame(rescue_rows)

    source_rows = []
    for source, rows in candidate.groupby("source_dataset", sort=True).groups.items():
        rows = np.asarray(list(rows), dtype=int)
        candidate_metrics = metrics(
            candidate.iloc[rows]["label_idx"], candidate_probability[rows]
        )
        baseline_metrics = metrics(
            baseline.iloc[rows]["label_idx"], baseline_probability[rows]
        )
        source_rows.append(
            {
                "source_dataset": source,
                "candidate_bacc": candidate_metrics["balanced_accuracy"],
                "baseline_bacc": baseline_metrics["balanced_accuracy"],
                "delta_bacc": candidate_metrics["balanced_accuracy"]
                - baseline_metrics["balanced_accuracy"],
                "candidate_sensitivity": candidate_metrics["sensitivity"],
                "candidate_specificity": candidate_metrics["specificity"],
            }
        )
    source = pd.DataFrame(source_rows)

    candidate_metrics = overall.iloc[0].to_dict()
    baseline_subtype = subtype[subtype["method"] == "baseline"].set_index("task_l6_label")
    candidate_subtype = subtype[subtype["method"] == "candidate"].set_index("task_l6_label")
    b1 = int(candidate_subtype.loc["B1", "correct_n"])
    b2 = int(candidate_subtype.loc["B2", "correct_n"])
    b1_delta = int(b1 - baseline_subtype.loc["B1", "correct_n"])
    b2_delta = int(b2 - baseline_subtype.loc["B2", "correct_n"])
    gates = [
        {
            "gate": "significant_bacc_gain",
            "value": bootstrap_result,
            "passed": bootstrap_result["delta_balanced_accuracy"] >= 0.0100
            and bootstrap_result["ci_low"] > 0,
        },
        {
            "gate": "accuracy_floor",
            "value": float(candidate_metrics["accuracy"]),
            "passed": float(candidate_metrics["accuracy"]) >= 0.7986,
        },
        {
            "gate": "tp_tn_floor",
            "value": {"tp": int(candidate_metrics["tp"]), "tn": int(candidate_metrics["tn"])},
            "passed": int(candidate_metrics["tp"]) >= 167 and int(candidate_metrics["tn"]) >= 299,
        },
        {
            "gate": "absolute_b1_b2",
            "value": {"B1": b1, "B2": b2},
            "passed": b1 >= 40 and b2 >= 59,
        },
        {
            "gate": "b1_b2_net_gain",
            "value": {"B1_delta": b1_delta, "B2_delta": b2_delta, "combined": b1_delta + b2_delta},
            "passed": b1_delta >= 0 and b2_delta >= 0 and b1_delta + b2_delta >= 8,
        },
        {
            "gate": "source_audit_stability",
            "value": {
                "minimum_candidate_bacc": float(source["candidate_bacc"].min()),
                "minimum_delta": float(source["delta_bacc"].min()),
            },
            "passed": float(source["candidate_bacc"].min()) >= 0.7419
            and float(source["delta_bacc"].min()) >= -0.0200,
        },
        {
            "gate": "complete_fixed_threshold",
            "value": {"n": int(candidate_metrics["n"]), "threshold": 0.5},
            "passed": int(candidate_metrics["n"]) == 591,
        },
    ]
    decision = "GO_NESTED_PHENOTYPE_CURRICULUM" if all(gate["passed"] for gate in gates) else "NO_GO_NESTED_PHENOTYPE_CURRICULUM"

    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    subtype.to_csv(output_dir / "subtype_metrics.csv", index=False)
    rescue.to_csv(output_dir / "rescue_harm_by_subtype.csv", index=False)
    source.to_csv(output_dir / "source_audit.csv", index=False)
    write_json(output_dir / "paired_bootstrap.json", bootstrap_result)
    write_json(output_dir / "gate_decision.json", {"decision": decision, "gates": gates})
    write_json(
        output_dir / "summary.json",
        {
            "experiment": "H10_STAGE2_NESTED_PHENOTYPE_CURRICULUM_20260715",
            "decision": decision,
            "candidate": candidate_metrics,
            "bootstrap": bootstrap_result,
            "B1": b1,
            "B2": b2,
            "B1_delta": b1_delta,
            "B2_delta": b2_delta,
        },
    )
    (output_dir / "RUN.status").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "gates": gates}, indent=2), flush=True)
    print("\nOVERALL\n" + overall.to_string(index=False), flush=True)
    print("\nSUBTYPE\n" + subtype.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
