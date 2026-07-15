from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGES = ("anchor_warmup", "boundary_bridge", "targeted_replay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the failed H10 nested curriculum.")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_peak(history: pd.DataFrame, stage: str) -> tuple[float, int]:
    rows = history[history["stage"] == stage]
    position = rows["val_bacc"].idxmax()
    return float(history.loc[position, "val_bacc"]), int(history.loc[position, "epoch"])


def consensus_role(values: pd.Series) -> str:
    counts = Counter(values.astype(str))
    role, count = counts.most_common(1)[0]
    return role if count >= 2 else "unstable_all_different"


def main() -> None:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    histories = []
    teachers = []
    roles = []
    for fold_id in range(1, 6):
        candidate_fold = candidate_dir / f"fold_{fold_id}"
        baseline_fold = baseline_dir / f"fold_{fold_id}"
        candidate_summary = read_json(candidate_fold / "fold_summary.json")
        baseline_summary = read_json(baseline_fold / "fold_summary.json")
        history = pd.read_csv(candidate_fold / "training_history.csv")
        history["fold_id"] = fold_id
        histories.append(history)
        teacher = pd.read_csv(candidate_fold / "nested_teacher_fold_metrics.csv")
        teachers.append(teacher)
        role = pd.read_csv(
            candidate_fold / "nested_training_roles_server_only.csv",
            dtype={"case_id": str},
            encoding="utf-8-sig",
        )
        role["outer_fold"] = fold_id
        roles.append(role)

        peaks = {stage: stage_peak(history, stage) for stage in STAGES}
        any_position = history["val_bacc"].idxmax()
        eligible = history[history["checkpoint_eligible"].astype(bool)]
        eligible_position = eligible["val_bacc"].idxmax()
        fold_rows.append(
            {
                "fold_id": fold_id,
                "anchor_peak_bacc": peaks["anchor_warmup"][0],
                "anchor_peak_epoch": peaks["anchor_warmup"][1],
                "bridge_peak_bacc": peaks["boundary_bridge"][0],
                "bridge_peak_epoch": peaks["boundary_bridge"][1],
                "targeted_peak_bacc": peaks["targeted_replay"][0],
                "targeted_peak_epoch": peaks["targeted_replay"][1],
                "any_peak_bacc": float(history.loc[any_position, "val_bacc"]),
                "any_peak_epoch": int(history.loc[any_position, "epoch"]),
                "eligible_peak_bacc": float(history.loc[eligible_position, "val_bacc"]),
                "eligible_peak_epoch": int(history.loc[eligible_position, "epoch"]),
                "eligible_minus_any_peak": float(
                    history.loc[eligible_position, "val_bacc"]
                    - history.loc[any_position, "val_bacc"]
                ),
                "candidate_test_bacc": float(
                    candidate_summary["test_metrics"]["balanced_accuracy"]
                ),
                "baseline_test_bacc": float(
                    baseline_summary["test_metrics"]["balanced_accuracy"]
                ),
                "test_delta_bacc": float(
                    candidate_summary["test_metrics"]["balanced_accuracy"]
                    - baseline_summary["test_metrics"]["balanced_accuracy"]
                ),
                "candidate_best_epoch": int(candidate_summary["best_epoch"]),
                "final_training_seconds": float(
                    candidate_summary["elapsed_seconds_final_training"]
                ),
            }
        )

    fold = pd.DataFrame(fold_rows)
    history = pd.concat(histories, ignore_index=True)
    teacher = pd.concat(teachers, ignore_index=True)
    role = pd.concat(roles, ignore_index=True)
    if len(role) != 591 * 3 or not role.groupby("case_id").size().eq(3).all():
        raise ValueError("Every case must have three outer-training role assignments")

    teacher_rows = []
    for sampler, rows in teacher.groupby("sampler", sort=True):
        teacher_rows.append(
            {
                "sampler": sampler,
                "inner_predictions_n": int(rows["test_n"].sum()),
                "weighted_inner_bacc": float(
                    np.average(rows["inner_test_bacc"], weights=rows["test_n"])
                ),
                "mean_inner_bacc": float(rows["inner_test_bacc"].mean()),
                "minimum_inner_bacc": float(rows["inner_test_bacc"].min()),
                "maximum_inner_bacc": float(rows["inner_test_bacc"].max()),
            }
        )
    teacher_summary = pd.DataFrame(teacher_rows)

    role_count = (
        role.groupby(["training_role", "task_l6_label"], sort=True)
        .size()
        .rename("assignment_n")
        .reset_index()
    )
    role_by_fold = (
        role.groupby(["outer_fold", "training_role"], sort=True)
        .size()
        .rename("assignment_n")
        .reset_index()
    )
    stability = (
        role.groupby("case_id", sort=False)["training_role"]
        .agg(unique_role_n="nunique", consensus_role=consensus_role)
        .reset_index()
    )
    agreement_pairs = 0
    for values in role.groupby("case_id", sort=False)["training_role"]:
        assignments = values[1].astype(str).tolist()
        agreement_pairs += sum(
            assignments[left] == assignments[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
    stability_summary = pd.DataFrame(
        [
            {
                "unique_role_n": int(unique_n),
                "case_n": int(count),
                "case_fraction": float(count / 591),
            }
            for unique_n, count in stability["unique_role_n"].value_counts().sort_index().items()
        ]
    )

    candidate = pd.read_csv(
        candidate_dir / "oof_predictions.csv", dtype={"case_id": str}, encoding="utf-8-sig"
    )
    baseline = pd.read_csv(
        baseline_dir / "oof_predictions.csv", dtype={"case_id": str}, encoding="utf-8-sig"
    )
    prediction = candidate[["case_id", "label_idx", "prob_high"]].rename(
        columns={"prob_high": "candidate_probability"}
    )
    prediction = prediction.merge(
        baseline[["case_id", "prob_high"]].rename(
            columns={"prob_high": "baseline_probability"}
        ),
        on="case_id",
        how="inner",
        validate="one_to_one",
    ).merge(stability[["case_id", "consensus_role"]], on="case_id", validate="one_to_one")
    prediction["candidate_correct"] = (
        (prediction["candidate_probability"] >= 0.5).astype(int)
        == prediction["label_idx"].astype(int)
    )
    prediction["baseline_correct"] = (
        (prediction["baseline_probability"] >= 0.5).astype(int)
        == prediction["label_idx"].astype(int)
    )
    consensus_rows = []
    for name, rows in prediction.groupby("consensus_role", sort=True):
        candidate_ok = rows["candidate_correct"].to_numpy(bool)
        baseline_ok = rows["baseline_correct"].to_numpy(bool)
        consensus_rows.append(
            {
                "consensus_role": name,
                "case_n": int(len(rows)),
                "candidate_accuracy": float(candidate_ok.mean()),
                "baseline_accuracy": float(baseline_ok.mean()),
                "rescue_n": int(np.sum(candidate_ok & ~baseline_ok)),
                "harm_n": int(np.sum(~candidate_ok & baseline_ok)),
                "net_correct": int(candidate_ok.sum() - baseline_ok.sum()),
            }
        )
    consensus_outcome = pd.DataFrame(consensus_rows)

    fold.to_csv(output_dir / "fold_stage_and_test_diagnostics.csv", index=False)
    teacher_summary.to_csv(output_dir / "nested_teacher_summary.csv", index=False)
    role_count.to_csv(output_dir / "role_assignment_by_subtype.csv", index=False)
    role_by_fold.to_csv(output_dir / "role_assignment_by_fold.csv", index=False)
    stability_summary.to_csv(output_dir / "role_stability_summary.csv", index=False)
    consensus_outcome.to_csv(output_dir / "consensus_role_outcomes.csv", index=False)
    summary = {
        "mean_anchor_peak_bacc": float(fold["anchor_peak_bacc"].mean()),
        "mean_bridge_peak_bacc": float(fold["bridge_peak_bacc"].mean()),
        "mean_targeted_peak_bacc": float(fold["targeted_peak_bacc"].mean()),
        "mean_eligible_peak_bacc": float(fold["eligible_peak_bacc"].mean()),
        "mean_eligible_minus_any_peak": float(fold["eligible_minus_any_peak"].mean()),
        "folds_candidate_better": int((fold["test_delta_bacc"] > 0).sum()),
        "folds_candidate_worse": int((fold["test_delta_bacc"] < 0).sum()),
        "role_assignment_pairwise_agreement": float(agreement_pairs / (591 * 3)),
        "stable_same_role_cases": int((stability["unique_role_n"] == 1).sum()),
        "unstable_all_different_cases": int((stability["unique_role_n"] == 3).sum()),
    }
    (output_dir / "failure_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nFOLD DIAGNOSTICS\n" + fold.to_string(index=False))
    print("\nTEACHERS\n" + teacher_summary.to_string(index=False))
    print("\nROLE STABILITY\n" + stability_summary.to_string(index=False))
    print("\nCONSENSUS ROLE OUTCOMES\n" + consensus_outcome.to_string(index=False))


if __name__ == "__main__":
    main()
