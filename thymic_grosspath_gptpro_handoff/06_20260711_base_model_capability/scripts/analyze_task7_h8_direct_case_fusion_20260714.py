from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


EXPERIMENT = "H8_C1_H3_DIRECT_CASE_EMBEDDING_FUSION_20260714"
EXACT = "C1_H3_EXACT"
C1_ONLY = "C1_ONLY_PADDED"
H3_ONLY = "H3_ONLY_PADDED"
DERANGED = "C1_H3_SAME_SOURCE_DERANGED"
CONFIGURATIONS = (C1_ONLY, H3_ONLY, EXACT, DERANGED)
STOP_DECISION = "STOP CURRENT-COHORT CLASSIFIER DEVELOPMENT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the locked H8 gates.")
    parser.add_argument("--stage", choices=("source_lodo", "fivefold", "confirmation"), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--c1-predictions", required=True)
    parser.add_argument("--c2-predictions", required=True)
    parser.add_argument("--h3-predictions", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--primary-run-dir", default="")
    parser.add_argument("--enforce-gates", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def canonical_source(value: Any) -> str:
    text = str(value)
    if text in {"batch1", "batch2", "third_batch"}:
        return text
    if text.startswith("third_batch"):
        return "third_batch"
    raise ValueError(f"Unexpected source: {value!r}")


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str}, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    required = {"case_id", "label_idx", "prob_high"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(frame.columns))}")
    if len(frame) != 591 or frame["case_id"].duplicated().any():
        raise ValueError(f"Expected 591 unique predictions in {path}")
    frame["label_idx"] = pd.to_numeric(frame["label_idx"], errors="raise").astype(int)
    frame["prob_high"] = pd.to_numeric(frame["prob_high"], errors="raise").astype(float)
    if not np.isfinite(frame["prob_high"]).all():
        raise ValueError(f"Nonfinite probabilities in {path}")
    if "source_dataset" in frame.columns:
        frame["source_dataset"] = frame["source_dataset"].map(canonical_source)
    return frame


def metric_record(labels: Iterable[int], probabilities: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=int)
    probability = np.asarray(list(probabilities), dtype=float)
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def align_reference(base: pd.DataFrame, path: Path, name: str) -> np.ndarray:
    reference = load_predictions(path)[["case_id", "label_idx", "prob_high"]].rename(
        columns={"label_idx": f"{name}_label", "prob_high": f"{name}_prob"}
    )
    aligned = base[["case_id", "label_idx"]].merge(reference, on="case_id", how="left", validate="one_to_one")
    if aligned[f"{name}_prob"].isna().any():
        raise ValueError(f"Reference {name} is missing cases")
    if not np.array_equal(aligned["label_idx"].to_numpy(int), aligned[f"{name}_label"].to_numpy(int)):
        raise ValueError(f"Reference label mismatch for {name}")
    return aligned[f"{name}_prob"].to_numpy(float)


def bacc(labels: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    low = labels == 0
    high = labels == 1
    return float((np.mean(~prediction[low]) + np.mean(prediction[high])) / 2)


def paired_stratified_bootstrap(
    frame: pd.DataFrame,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = frame["label_idx"].to_numpy(int)
    strata = [
        group.index.to_numpy(int)
        for _, group in frame.reset_index(drop=True).groupby(["source_dataset", "label_idx"], sort=True)
    ]
    if len(strata) != 6 or any(len(indices) == 0 for indices in strata):
        raise ValueError("Bootstrap requires six nonempty source-by-risk strata")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        deltas[replicate] = bacc(labels[sampled], probability_a[sampled]) - bacc(
            labels[sampled], probability_b[sampled]
        )
    point = bacc(labels, probability_a) - bacc(labels, probability_b)
    return {
        "point_delta_bacc": float(point),
        "bootstrap_mean_delta_bacc": float(deltas.mean()),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "probability_delta_gt_zero": float(np.mean(deltas > 0)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def comparator_seed(base_seed: int, name: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")
    return int((base_seed + suffix) % (2**32 - 1))


def subtype_table(frame: pd.DataFrame, probability: np.ndarray, method: str) -> pd.DataFrame:
    work = frame[["task_l6_label", "label_idx"]].copy()
    work["prob_high"] = probability
    work["correct"] = (probability >= 0.5).astype(int) == work["label_idx"].to_numpy(int)
    rows = []
    for subtype, group in work.groupby("task_l6_label", sort=False):
        rows.append(
            {
                "method": method,
                "subtype": subtype,
                "n": int(len(group)),
                "correct_n": int(group["correct"].sum()),
                "accuracy": float(group["correct"].mean()),
                "mean_prob_high": float(group["prob_high"].mean()),
            }
        )
    return pd.DataFrame(rows)


def source_table(frame: pd.DataFrame, probability: np.ndarray, method: str) -> pd.DataFrame:
    rows = []
    for source, group in frame.reset_index(drop=True).groupby("source_dataset", sort=False):
        indices = group.index.to_numpy(int)
        rows.append({"method": method, "source_dataset": source, **metric_record(group["label_idx"], probability[indices])})
    return pd.DataFrame(rows)


def source_subtype_table(frame: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    work = frame[["source_dataset", "task_l6_label", "label_idx"]].copy().reset_index(drop=True)
    work["prob_high"] = probability
    work["correct"] = (probability >= 0.5).astype(int) == work["label_idx"].to_numpy(int)
    return (
        work.groupby(["source_dataset", "task_l6_label"], sort=False)
        .agg(n=("correct", "size"), correct_n=("correct", "sum"), accuracy=("correct", "mean"), mean_prob_high=("prob_high", "mean"))
        .reset_index()
    )


def rescue_record(
    frame: pd.DataFrame,
    candidate_probability: np.ndarray,
    reference_probability: np.ndarray,
    reference: str,
    subset_name: str,
    mask: np.ndarray,
) -> dict[str, Any]:
    labels = frame["label_idx"].to_numpy(int)
    candidate_correct = (candidate_probability >= 0.5).astype(int) == labels
    reference_correct = (reference_probability >= 0.5).astype(int) == labels
    rescue = candidate_correct & ~reference_correct & mask
    harm = ~candidate_correct & reference_correct & mask
    return {
        "reference": reference,
        "subset": subset_name,
        "n": int(mask.sum()),
        "rescue_n": int(rescue.sum()),
        "harm_n": int(harm.sum()),
        "net_rescue": int(rescue.sum() - harm.sum()),
        "mcnemar_candidate_only_correct": int(rescue.sum()),
        "mcnemar_reference_only_correct": int(harm.sum()),
    }


def gate(name: str, requirement: str, value: Any, passed: bool) -> dict[str, Any]:
    return {"gate": name, "requirement": requirement, "value": value, "passed": bool(passed)}


def load_method_predictions(run_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = {configuration: load_predictions(run_dir / configuration / "oof_predictions.csv") for configuration in CONFIGURATIONS}
    base = frames[EXACT].copy().sort_values("feature_row").reset_index(drop=True)
    required_metadata = {"feature_row", "source_dataset", "task_l6_label", "fold_id"}
    if not required_metadata.issubset(base.columns):
        raise ValueError("H8 exact predictions lack required private metadata")
    base["source_dataset"] = base["source_dataset"].map(canonical_source)
    probabilities = {}
    for configuration, frame in frames.items():
        aligned = base[["case_id"]].merge(
            frame[["case_id", "label_idx", "prob_high"]], on="case_id", how="left", validate="one_to_one"
        )
        if aligned["prob_high"].isna().any() or not np.array_equal(
            base["label_idx"].to_numpy(int), aligned["label_idx"].to_numpy(int)
        ):
            raise ValueError(f"H8 configuration misalignment: {configuration}")
        probabilities[configuration] = aligned["prob_high"].to_numpy(float)
    return base, probabilities


def primary_gates(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    h3_probability: np.ndarray,
    bootstrap: dict[str, dict[str, Any]],
    source_metrics: pd.DataFrame,
    boundary_net_rescue: int,
) -> list[dict[str, Any]]:
    exact_metrics = metric_record(frame["label_idx"], probabilities[EXACT])
    exact_prediction = probabilities[EXACT] >= 0.5
    subtype = frame["task_l6_label"].astype(str).to_numpy()
    source = frame["source_dataset"].astype(str).to_numpy()
    labels = frame["label_idx"].to_numpy(int)
    correct = exact_prediction.astype(int) == labels
    b1_correct = int(correct[subtype == "B1"].sum())
    b2_correct = int(correct[subtype == "B2"].sum())
    third_b2_correct = int(correct[(source == "third_batch") & (subtype == "B2")].sum())
    exact_source = source_metrics[source_metrics["method"] == EXACT].set_index("source_dataset")
    h3_source = source_metrics[source_metrics["method"] == "LOCKED_H3"].set_index("source_dataset")
    source_delta = exact_source["balanced_accuracy"] - h3_source["balanced_accuracy"]
    deranged_source = source_metrics[source_metrics["method"] == DERANGED].set_index("source_dataset")
    deranged_delta = exact_source["balanced_accuracy"] - deranged_source["balanced_accuracy"]
    branch_best = max(
        metric_record(labels, probabilities[C1_ONLY])["balanced_accuracy"],
        metric_record(labels, probabilities[H3_ONLY])["balanced_accuracy"],
    )
    return [
        gate("P1", "591/591 at threshold 0.5", len(frame), len(frame) == 591),
        gate("P2", "BAcc >= 0.7739", exact_metrics["balanced_accuracy"], exact_metrics["balanced_accuracy"] >= 0.7739),
        gate("P3", "TP >= 164", exact_metrics["tp"], exact_metrics["tp"] >= 164),
        gate("P4", "TN >= 299", exact_metrics["tn"], exact_metrics["tn"] >= 299),
        gate("P5", "B1 correct >= 40/62", b1_correct, b1_correct >= 40),
        gate("P6", "B2 correct >= 59/89", b2_correct, b2_correct >= 59),
        gate("P7", "third-batch B2 correct >= 18/29", third_b2_correct, third_b2_correct >= 18),
        gate("P8", "positive H3 BAcc delta in >=2/3 sources", source_delta.to_dict(), int((source_delta > 0).sum()) >= 2),
        gate(
            "P9",
            "no source delta < -0.0200 and minimum source BAcc >= 0.7381",
            {"delta": source_delta.to_dict(), "minimum_bacc": float(exact_source["balanced_accuracy"].min())},
            bool((source_delta >= -0.0200).all() and exact_source["balanced_accuracy"].min() >= 0.7381),
        ),
        gate(
            "P10",
            "exact >= best padded branch + 0.0100",
            exact_metrics["balanced_accuracy"] - branch_best,
            exact_metrics["balanced_accuracy"] - branch_best >= 0.0100,
        ),
        gate(
            "P11",
            "exact-deranged >=0.0100, positive >=2/3 sources, CI low >0",
            {"bootstrap": bootstrap[DERANGED], "source_delta": deranged_delta.to_dict()},
            bool(
                bootstrap[DERANGED]["point_delta_bacc"] >= 0.0100
                and int((deranged_delta > 0).sum()) >= 2
                and bootstrap[DERANGED]["ci95_low"] > 0
            ),
        ),
        gate(
            "P12",
            "exact-H3 >=0.0200 and CI low >0",
            bootstrap["LOCKED_H3"],
            bool(bootstrap["LOCKED_H3"]["point_delta_bacc"] >= 0.0200 and bootstrap["LOCKED_H3"]["ci95_low"] > 0),
        ),
        gate(
            "P13",
            "B1+B2 net correct gain vs H3 >=7 with P5/P6 retained",
            boundary_net_rescue,
            bool(boundary_net_rescue >= 7 and b1_correct >= 40 and b2_correct >= 59),
        ),
    ]


def fivefold_gates(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    bootstrap: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    exact = metric_record(frame["label_idx"], probabilities[EXACT])
    predicted = probabilities[EXACT] >= 0.5
    labels = frame["label_idx"].to_numpy(int)
    subtype = frame["task_l6_label"].astype(str).to_numpy()
    correct = predicted.astype(int) == labels
    fold_bacc = {
        int(fold): metric_record(group["label_idx"], probabilities[EXACT][group.index.to_numpy(int)])["balanced_accuracy"]
        for fold, group in frame.reset_index(drop=True).groupby("fold_id", sort=True)
    }
    branch_best = max(
        metric_record(labels, probabilities[C1_ONLY])["balanced_accuracy"],
        metric_record(labels, probabilities[H3_ONLY])["balanced_accuracy"],
    )
    values = {
        "coverage": len(frame),
        "bacc": exact["balanced_accuracy"],
        "tp": exact["tp"],
        "tn": exact["tn"],
        "b1_correct": int(correct[subtype == "B1"].sum()),
        "b2_correct": int(correct[subtype == "B2"].sum()),
        "fold_bacc": fold_bacc,
        "branch_delta": exact["balanced_accuracy"] - branch_best,
        "deranged_bootstrap": bootstrap[DERANGED],
    }
    passed = bool(
        len(frame) == 591
        and exact["balanced_accuracy"] >= 0.7903
        and exact["tp"] >= 176
        and exact["tn"] >= 285
        and values["b1_correct"] >= 40
        and values["b2_correct"] >= 60
        and min(fold_bacc.values()) >= 0.7000
        and values["branch_delta"] >= 0.0100
        and bootstrap[DERANGED]["point_delta_bacc"] >= 0.0100
        and bootstrap[DERANGED]["ci95_low"] > 0
    )
    return [gate("S1-S9", "all locked secondary five-fold requirements", values, passed)]


def render_report(
    stage: str,
    overall: pd.DataFrame,
    source: pd.DataFrame,
    subtype: pd.DataFrame,
    rescue: pd.DataFrame,
    bootstrap: pd.DataFrame,
    gates: list[dict[str, Any]],
    decision: str,
    run_summary: dict[str, Any],
) -> str:
    exact = overall[overall["method"] == EXACT].iloc[0]
    lines = [
        "# H8 C1-H3 Direct Case-Fusion Results",
        "",
        f"Stage: `{stage}`",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "## Exact fusion",
        "",
        f"- BAcc: {exact['balanced_accuracy']:.4f}",
        f"- AUC: {exact['auc']:.4f}",
        f"- sensitivity: {exact['sensitivity']:.4f} ({int(exact['tp'])}/{int(exact['tp'] + exact['fn'])})",
        f"- specificity: {exact['specificity']:.4f} ({int(exact['tn'])}/{int(exact['tn'] + exact['fp'])})",
        "- threshold: 0.5; coverage: 591/591",
        "",
        "## Gate table",
        "",
        "| Gate | Passed | Requirement |",
        "| --- | --- | --- |",
    ]
    for item in gates:
        lines.append(f"| {item['gate']} | {'PASS' if item['passed'] else 'FAIL'} | {item['requirement']} |")
    lines.extend(
        [
            "",
            "## Source BAcc",
            "",
            "| Method | Source | BAcc | Sensitivity | Specificity |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in source.iterrows():
        lines.append(
            f"| {row['method']} | {row['source_dataset']} | {row['balanced_accuracy']:.4f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary subtypes",
            "",
            "| Method | Subtype | Correct/N | Accuracy | Mean p(high) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in subtype[subtype["subtype"].isin(["B1", "B2"])].iterrows():
        lines.append(
            f"| {row['method']} | {row['subtype']} | {int(row['correct_n'])}/{int(row['n'])} | "
            f"{row['accuracy']:.4f} | {row['mean_prob_high']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Stability comparisons",
            "",
            "Paired percentile intervals use 20,000 source-by-risk-stratified bootstrap replicates. "
            "They are repeated-cohort stability diagnostics, not independent confirmatory intervals.",
            "",
            "| Comparator | Point delta BAcc | 95% CI | P(delta>0) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in bootstrap.iterrows():
        lines.append(
            f"| {row['comparator']} | {row['point_delta_bacc']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['probability_delta_gt_zero']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- train/evaluation wall time: {run_summary.get('elapsed_seconds', 'NA')} seconds",
            f"- peak GPU allocation: {run_summary.get('peak_gpu_allocated_bytes', 'NA')} bytes",
            f"- peak resident memory: {run_summary.get('peak_resident_kib', 'NA')} KiB",
            "- trainable parameters per learned configuration: 6,226",
            "",
            "All images, paths, identifiers, embeddings, checkpoints, per-case predictions, and "
            "derangement maps remain server-only.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> bool:
    if args.bootstrap_replicates != 20000:
        raise ValueError("H8 bootstrap count is locked to 20,000")
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base, probabilities = load_method_predictions(run_dir)
    h3_probability = align_reference(base, Path(args.h3_predictions), "h3")
    c1_probability = align_reference(base, Path(args.c1_predictions), "c1")
    c2_probability = align_reference(base, Path(args.c2_predictions), "c2")
    average_probability = 0.5 * c1_probability + 0.5 * h3_probability
    all_probability = {
        **probabilities,
        "LOCKED_C1": c1_probability,
        "LOCKED_C2": c2_probability,
        "LOCKED_H3": h3_probability,
        "FIXED_C1_H3_MEAN": average_probability,
    }

    overall_rows = []
    source_frames = []
    subtype_frames = []
    for method, probability in all_probability.items():
        overall_rows.append({"method": method, **metric_record(base["label_idx"], probability)})
        source_frames.append(source_table(base, probability, method))
        subtype_frames.append(subtype_table(base, probability, method))
    overall = pd.DataFrame(overall_rows)
    source = pd.concat(source_frames, ignore_index=True)
    subtype = pd.concat(subtype_frames, ignore_index=True)
    source_subtype = source_subtype_table(base, probabilities[EXACT])

    boundary_mask = base["task_l6_label"].isin(["B1", "B2"]).to_numpy()
    all_mask = np.ones(len(base), dtype=bool)
    rescue = pd.DataFrame(
        [
            rescue_record(base, probabilities[EXACT], reference, name, subset, mask)
            for name, reference in (("LOCKED_H3", h3_probability), ("LOCKED_C2", c2_probability))
            for subset, mask in (("all", all_mask), ("B1+B2", boundary_mask))
        ]
    )
    boundary_net = int(
        rescue[(rescue["reference"] == "LOCKED_H3") & (rescue["subset"] == "B1+B2")]["net_rescue"].iloc[0]
    )

    comparator_probabilities = {
        C1_ONLY: probabilities[C1_ONLY],
        H3_ONLY: probabilities[H3_ONLY],
        DERANGED: probabilities[DERANGED],
        "LOCKED_H3": h3_probability,
        "LOCKED_C2": c2_probability,
    }
    bootstrap_records = {}
    for comparator, probability in comparator_probabilities.items():
        bootstrap_records[comparator] = paired_stratified_bootstrap(
            base,
            probabilities[EXACT],
            probability,
            args.bootstrap_replicates,
            comparator_seed(args.bootstrap_seed, comparator),
        )
    bootstrap = pd.DataFrame(
        [{"comparator": comparator, **record} for comparator, record in bootstrap_records.items()]
    )

    if args.stage in {"source_lodo", "confirmation"}:
        gates = primary_gates(base, probabilities, h3_probability, bootstrap_records, source, boundary_net)
        if args.stage == "confirmation":
            if not args.primary_run_dir:
                raise ValueError("Confirmation analysis requires --primary-run-dir")
            primary = load_predictions(Path(args.primary_run_dir) / EXACT / "oof_predictions.csv")
            primary_probability = align_reference(base, Path(args.primary_run_dir) / EXACT / "oof_predictions.csv", "primary")
            del primary
            primary_delta = bacc(base["label_idx"].to_numpy(int), primary_probability) - bacc(
                base["label_idx"].to_numpy(int), h3_probability
            )
            confirmation_delta = bootstrap_records["LOCKED_H3"]["point_delta_bacc"]
            mean_delta = float((primary_delta + confirmation_delta) / 2)
            gates.append(gate("C1", "mean H3 BAcc delta across two seeds >=0.0200", mean_delta, mean_delta >= 0.0200))
    else:
        gates = fivefold_gates(base, probabilities, bootstrap_records)

    all_pass = all(item["passed"] for item in gates)
    if not all_pass:
        decision = STOP_DECISION
    elif args.stage == "source_lodo":
        decision = "ADVANCE TO LOCKED FIVE-FOLD H8 STAGE"
    elif args.stage == "fivefold":
        decision = "ADVANCE TO LOCKED H8 CONFIRMATION SEED"
    else:
        decision = "H8 ALL PREREGISTERED GATES PASS"

    overall.to_csv(output_dir / "overall_metrics.csv", index=False)
    source.to_csv(output_dir / "source_metrics.csv", index=False)
    subtype.to_csv(output_dir / "subtype_metrics.csv", index=False)
    source_subtype.to_csv(output_dir / "source_subtype_metrics.csv", index=False)
    rescue.to_csv(output_dir / "rescue_harm_mcnemar.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    run_summary_path = run_dir / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    gate_payload = {
        "experiment": EXPERIMENT,
        "stage": args.stage,
        "all_primary_gates_pass": bool(all_pass) if args.stage == "source_lodo" else None,
        "all_secondary_gates_pass": bool(all_pass) if args.stage == "fivefold" else None,
        "all_confirmation_gates_pass": bool(all_pass) if args.stage == "confirmation" else None,
        "gates": gates,
        "decision": decision,
    }
    gate_path = output_dir / "gate_decision.json"
    atomic_write_text(
        gate_path,
        json.dumps(gate_payload, ensure_ascii=False, indent=2, default=json_default),
    )
    atomic_write_text(
        gate_path.with_suffix(gate_path.suffix + ".sha256"),
        f"{sha256_file(gate_path)}  {gate_path.name}\n",
    )
    atomic_write_text(output_dir / "FINAL_DECISION.txt", decision + "\n")
    report = render_report(args.stage, overall, source, subtype, rescue, bootstrap, gates, decision, run_summary)
    atomic_write_text(output_dir / "H8_C1_H3_DIRECT_CASE_FUSION_RESULTS_20260714.md", report)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "decision": decision,
                "exact_metrics": overall[overall["method"] == EXACT].iloc[0].to_dict(),
                "failed_gates": [item["gate"] for item in gates if not item["passed"]],
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    )
    return all_pass


def main() -> None:
    args = parse_args()
    passed = run(args)
    if args.enforce_gates and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
