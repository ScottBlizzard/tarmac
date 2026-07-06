from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROFILE = {"AB": 110, "B1": 8, "B2": 22, "TC": 40}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stability check for adapt180 Task7 profile.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--old-feature-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521",
    )
    parser.add_argument(
        "--third-feature-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/18_adapt180_stability_20260521",
    )
    parser.add_argument("--base-seed", type=int, default=20260521)
    return parser.parse_args()


def stable_key(case_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:adapt180_mixed:{case_id}".encode("utf-8")).hexdigest()


def split_profile(third: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    chosen: list[int] = []
    for subtype, n in PROFILE.items():
        g = third[third["task_l6_label"].eq(subtype)].copy()
        g["_key"] = g["case_id"].map(lambda x: stable_key(str(x), seed))
        chosen.extend(g.sort_values("_key").head(n).index.tolist())
    mask = np.zeros(len(third), dtype=bool)
    mask[np.array(chosen, dtype=int)] = True
    return np.where(mask)[0], np.where(~mask)[0]


def read_old(root: Path, rel: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel
    table = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    feat = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    return table.reset_index(drop=True), feat[table["feature_idx"].astype(int).to_numpy()]


def read_third(root: Path, rel: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel
    table = pd.read_csv(d / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    registry = pd.read_csv(d / "third_batch_task7_registry.csv", dtype={"case_id": str, "original_case_id": str})
    feat = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    table["feature_idx"] = table["feature_idx"].astype(int)
    frame = registry.merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    frame["feature_idx"] = frame["feature_idx"].astype(int)
    return frame.reset_index(drop=True), feat[frame["feature_idx"].to_numpy()]


def repeat_indices(is_adapt: np.ndarray, repeat_adapt: int) -> np.ndarray:
    old_idx = np.where(~is_adapt)[0]
    adapt_idx = np.where(is_adapt)[0]
    return np.concatenate([old_idx, np.tile(adapt_idx, repeat_adapt)])


def make_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.0003, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed),
    )


def metric_dict(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, object] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if score is not None:
        out["auc"] = float(roc_auc_score(y, score))
    return out


def best_accuracy_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    best = (float("-inf"), 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        score = accuracy_score(y, (prob >= t).astype(int))
        cand = (float(score), -abs(float(t) - 0.5))
        if cand > (best[0], -abs(best[1] - 0.5)):
            best = (float(score), float(t))
    return best[1]


def run_one(
    seed: int,
    old: pd.DataFrame,
    x_old: np.ndarray,
    third: pd.DataFrame,
    x_third: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    adapt_idx, hold_idx = split_profile(third, seed)
    adapt = third.iloc[adapt_idx].reset_index(drop=True)
    hold = third.iloc[hold_idx].reset_index(drop=True)
    x_train = np.concatenate([x_old, x_third[adapt_idx]], axis=0)
    y_train = np.concatenate([old["label_idx"].to_numpy(int), adapt["label_idx"].to_numpy(int)])
    x_hold = x_third[hold_idx]
    y_hold = hold["label_idx"].to_numpy(int)
    is_old = np.concatenate([np.ones(len(old), dtype=bool), np.zeros(len(adapt), dtype=bool)])
    is_adapt = ~is_old

    oof = np.zeros(len(y_train), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(x_train, y_train)):
        idx = repeat_indices(is_adapt[tr], 4)
        model = make_model(seed + fold)
        model.fit(x_train[tr][idx], y_train[tr][idx])
        oof[va] = model.predict_proba(x_train[va])[:, 1]
    threshold = best_accuracy_threshold(y_train, oof)

    full_idx = repeat_indices(is_adapt, 4)
    model = make_model(seed + 999)
    model.fit(x_train[full_idx], y_train[full_idx])
    hold_prob = model.predict_proba(x_hold)[:, 1]
    train_pred = (oof >= threshold).astype(int)
    hold_pred = (hold_prob >= threshold).astype(int)

    row = {
        "seed": seed,
        "threshold": threshold,
        "adapt_n": int(len(adapt)),
        "holdout_n": int(len(hold)),
    }
    row.update({f"train_{k}": v for k, v in metric_dict(y_train, train_pred, oof).items()})
    row.update({f"old_oof_{k}": v for k, v in metric_dict(y_train[is_old], train_pred[is_old], oof[is_old]).items()})
    row.update({f"adapt_oof_{k}": v for k, v in metric_dict(y_train[is_adapt], train_pred[is_adapt], oof[is_adapt]).items()})
    row.update({f"holdout_{k}": v for k, v in metric_dict(y_hold, hold_pred, hold_prob).items()})

    subtype = hold.assign(pred=hold_pred, correct=(hold_pred == y_hold).astype(int)).groupby("task_l6_label").agg(
        n=("case_id", "size"), correct=("correct", "sum"), accuracy=("correct", "mean")
    )
    subtype = subtype.reset_index().assign(seed=seed)
    pred_case = hold[
        ["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]
    ].copy()
    pred_case["seed"] = seed
    pred_case["threshold"] = threshold
    pred_case["prob_high"] = hold_prob
    pred_case["pred_idx"] = hold_pred
    pred_case["correct"] = (hold_pred == y_hold).astype(int)
    return row, subtype, pred_case


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    old, x_old = read_old(root, args.old_feature_dir)
    third, x_third = read_third(root, args.third_feature_dir)

    seeds = [args.base_seed + i * 101 for i in range(10)]
    rows: list[dict[str, object]] = []
    subtypes: list[pd.DataFrame] = []
    pred_cases: list[pd.DataFrame] = []
    for seed in seeds:
        row, subtype, pred_case = run_one(seed, old, x_old, third, x_third)
        rows.append(row)
        subtypes.append(subtype)
        pred_cases.append(pred_case)

    df = pd.DataFrame(rows)
    subtype_df = pd.concat(subtypes, ignore_index=True)
    pred_df = pd.concat(pred_cases, ignore_index=True)
    df.to_csv(out / "adapt180_stability_summary.csv", index=False, encoding="utf-8-sig")
    subtype_df.to_csv(out / "adapt180_stability_subtype_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out / "adapt180_stability_holdout_predictions.csv", index=False, encoding="utf-8-sig")

    case_agg = (
        pred_df.groupby(["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx"], as_index=False)
        .agg(
            n_holdout_votes=("seed", "size"),
            mean_prob_high=("prob_high", "mean"),
            mean_pred_vote=("pred_idx", "mean"),
            majority_pred=("pred_idx", lambda s: int(s.mean() >= 0.5)),
        )
    )
    case_agg["majority_correct"] = (case_agg["majority_pred"] == case_agg["label_idx"]).astype(int)
    case_agg["mean_prob_pred"] = (case_agg["mean_prob_high"] >= 0.5).astype(int)
    case_agg["mean_prob_correct"] = (case_agg["mean_prob_pred"] == case_agg["label_idx"]).astype(int)
    case_agg.to_csv(out / "adapt180_repeated_holdout_case_aggregate.csv", index=False, encoding="utf-8-sig")

    repeated_majority = metric_dict(
        case_agg["label_idx"].to_numpy(int),
        case_agg["majority_pred"].to_numpy(int),
        case_agg["mean_pred_vote"].to_numpy(float),
    )
    repeated_mean_prob = metric_dict(
        case_agg["label_idx"].to_numpy(int),
        case_agg["mean_prob_pred"].to_numpy(int),
        case_agg["mean_prob_high"].to_numpy(float),
    )
    metric_cols = ["holdout_accuracy", "holdout_balanced_accuracy", "holdout_f1", "holdout_auc", "old_oof_accuracy", "adapt_oof_accuracy"]
    aggregate = df[metric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    aggregate.to_csv(out / "adapt180_stability_aggregate.csv", index=False, encoding="utf-8-sig")
    report = {
        "profile": PROFILE,
        "model": "whole+crop DINO frozen features + logreg C=0.0003 + adapt repeat 4 + train-OOF accuracy threshold",
        "seeds": seeds,
        "aggregate": aggregate.to_dict("records"),
        "repeated_holdout_case_coverage_n": int(len(case_agg)),
        "repeated_holdout_majority_vote": repeated_majority,
        "repeated_holdout_mean_prob_t05": repeated_mean_prob,
        "best_seed_by_holdout_accuracy": df.sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
        .iloc[0]
        .to_dict(),
        "output_dir": str(out),
    }
    (out / "adapt180_stability_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
