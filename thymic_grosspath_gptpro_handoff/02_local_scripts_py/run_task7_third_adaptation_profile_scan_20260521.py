from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROFILES = {
    "adapt72_high_focus": {"AB": 24, "B1": 4, "B2": 22, "TC": 22},
    "adapt96_high_focus": {"AB": 32, "B1": 6, "B2": 24, "TC": 34},
    "adapt120_high_focus": {"AB": 50, "B1": 6, "B2": 22, "TC": 42},
    "adapt150_high_focus": {"AB": 80, "B1": 6, "B2": 22, "TC": 42},
    "adapt180_mixed": {"AB": 110, "B1": 8, "B2": 22, "TC": 40},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan third-batch adaptation sizes for unified Task7 models.")
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
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/17_adaptation_profile_scan_20260521",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def stable_key(case_id: str, seed: int, profile: str) -> str:
    return hashlib.sha1(f"{seed}:{profile}:{case_id}".encode("utf-8")).hexdigest()


def split_profile(third: pd.DataFrame, profile: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    chosen: list[int] = []
    spec = PROFILES[profile]
    for subtype, n in spec.items():
        g = third[third["task_l6_label"].eq(subtype)].copy()
        if len(g) < n:
            raise ValueError(f"Profile {profile} requests {n} {subtype}, only {len(g)} available")
        g["_key"] = g["case_id"].map(lambda x: stable_key(str(x), seed, profile))
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
    if repeat_adapt <= 1:
        return np.arange(len(is_adapt))
    old_idx = np.where(~is_adapt)[0]
    adapt_idx = np.where(is_adapt)[0]
    return np.concatenate([old_idx, np.tile(adapt_idx, repeat_adapt)])


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
        out["auc"] = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan
    return out


def make_model(kind: str, seed: int):
    if kind.startswith("logreg"):
        c = float(kind.split("_")[1])
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed),
        )
    if kind.startswith("extratrees"):
        depth = int(kind.split("_")[1])
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(kind)


def score_model(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    s = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))


def best_threshold(y: np.ndarray, prob: np.ndarray, objective: str) -> float:
    best = (float("-inf"), 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        score = accuracy_score(y, pred) if objective == "accuracy" else balanced_accuracy_score(y, pred)
        cand = (float(score), -abs(float(t) - 0.5))
        if cand > (best[0], -abs(best[1] - 0.5)):
            best = (float(score), float(t))
    return best[1]


def run_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    is_adapt: np.ndarray,
    x_hold: np.ndarray,
    kind: str,
    repeat_adapt: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y_train), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(x_train, y_train)):
        idx = repeat_indices(is_adapt[tr], repeat_adapt)
        model = make_model(kind, seed + fold)
        model.fit(x_train[tr][idx], y_train[tr][idx])
        oof[va] = score_model(model, x_train[va])
    full_idx = repeat_indices(is_adapt, repeat_adapt)
    model = make_model(kind, seed + 999)
    model.fit(x_train[full_idx], y_train[full_idx])
    return oof, score_model(model, x_hold)


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    old, x_old = read_old(root, args.old_feature_dir)
    third, x_third = read_third(root, args.third_feature_dir)
    y_old = old["label_idx"].to_numpy(int)

    kinds = ["logreg_0.0003", "logreg_0.001", "logreg_0.003", "extratrees_3", "extratrees_5"]
    repeats = [1, 2, 4]
    rows: list[dict[str, object]] = []
    subtype_rows: list[dict[str, object]] = []

    for profile in PROFILES:
        adapt_idx, hold_idx = split_profile(third, profile, args.seed)
        adapt = third.iloc[adapt_idx].reset_index(drop=True)
        hold = third.iloc[hold_idx].reset_index(drop=True)
        x_train = np.concatenate([x_old, x_third[adapt_idx]], axis=0)
        y_train = np.concatenate([y_old, adapt["label_idx"].to_numpy(int)])
        is_old = np.concatenate([np.ones(len(old), dtype=bool), np.zeros(len(adapt), dtype=bool)])
        is_adapt = ~is_old
        x_hold = x_third[hold_idx]
        y_hold = hold["label_idx"].to_numpy(int)

        for kind in kinds:
            for repeat in repeats:
                oof, hold_prob = run_candidate(
                    x_train, y_train, is_adapt, x_hold, kind, repeat, args.seed + len(rows) * 19
                )
                for objective in ["balanced_accuracy", "accuracy"]:
                    threshold = best_threshold(y_train, oof, objective)
                    train_pred = (oof >= threshold).astype(int)
                    hold_pred = (hold_prob >= threshold).astype(int)
                    row = {
                        "profile": profile,
                        "adapt_n": int(len(adapt)),
                        "holdout_n": int(len(hold)),
                        "model_kind": kind,
                        "repeat_adapt": repeat,
                        "threshold_objective": objective,
                        "threshold": threshold,
                    }
                    row.update({f"train_{k}": v for k, v in metric_dict(y_train, train_pred, oof).items()})
                    row.update({f"old_oof_{k}": v for k, v in metric_dict(y_train[is_old], train_pred[is_old], oof[is_old]).items()})
                    row.update(
                        {f"adapt_oof_{k}": v for k, v in metric_dict(y_train[is_adapt], train_pred[is_adapt], oof[is_adapt]).items()}
                    )
                    row.update({f"holdout_{k}": v for k, v in metric_dict(y_hold, hold_pred, hold_prob).items()})
                    row["selection_score"] = (
                        min(float(row["old_oof_balanced_accuracy"]), float(row["adapt_oof_balanced_accuracy"]))
                        + 0.12 * float(row["train_balanced_accuracy"])
                        + 0.05 * min(float(row["old_oof_accuracy"]), float(row["adapt_oof_accuracy"]))
                    )
                    rows.append(row)
                    subtype = hold.assign(pred=hold_pred, correct=(hold_pred == y_hold).astype(int)).groupby("task_l6_label").agg(
                        n=("case_id", "size"), correct=("correct", "sum"), accuracy=("correct", "mean")
                    )
                    for subtype_name, srow in subtype.reset_index().iterrows():
                        subtype_rows.append(
                            {
                                "profile": profile,
                                "model_kind": kind,
                                "repeat_adapt": repeat,
                                "threshold_objective": objective,
                                "threshold": threshold,
                                "task_l6_label": srow["task_l6_label"],
                                "n": int(srow["n"]),
                                "correct": int(srow["correct"]),
                                "accuracy": float(srow["accuracy"]),
                            }
                        )

    df = pd.DataFrame(rows).sort_values(
        ["selection_score", "old_oof_balanced_accuracy", "adapt_oof_balanced_accuracy"], ascending=False
    )
    df.to_csv(out / "adaptation_profile_scan_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(subtype_rows).to_csv(out / "adaptation_profile_scan_subtype_metrics.csv", index=False, encoding="utf-8-sig")

    report = {
        "selection_uses_holdout": False,
        "profiles": PROFILES,
        "selected_by_oof": df.iloc[0].to_dict(),
        "best_holdout_accuracy_for_reference": df.sort_values(["holdout_accuracy", "holdout_balanced_accuracy"], ascending=False)
        .iloc[0]
        .to_dict(),
        "best_holdout_balanced_accuracy_for_reference": df.sort_values(["holdout_balanced_accuracy", "holdout_accuracy"], ascending=False)
        .iloc[0]
        .to_dict(),
        "output_dir": str(out),
    }
    (out / "adaptation_profile_scan_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    cols = [
        "profile",
        "adapt_n",
        "holdout_n",
        "model_kind",
        "repeat_adapt",
        "threshold_objective",
        "threshold",
        "old_oof_accuracy",
        "old_oof_balanced_accuracy",
        "adapt_oof_accuracy",
        "adapt_oof_balanced_accuracy",
        "holdout_accuracy",
        "holdout_balanced_accuracy",
        "holdout_f1",
        "holdout_tn",
        "holdout_fp",
        "holdout_fn",
        "holdout_tp",
    ]
    print(df[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
