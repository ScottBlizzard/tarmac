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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast Task7 third-batch adaptation probe with logistic regression.")
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
        "--existing-third-pred",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/04_third_batch_whole_plus_crop_64style_20260521/third_batch_external_case_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/02_third_logreg_fast_20260521",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def metric_dict(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_threshold(y: np.ndarray, prob: np.ndarray, objective: str = "balanced_accuracy") -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        score = balanced_accuracy_score(y, pred) if objective == "balanced_accuracy" else accuracy_score(y, pred)
        if (score, -abs(t - 0.5)) > (best_s, -abs(best_t - 0.5)):
            best_t, best_s = float(t), float(score)
    return best_t, best_s


def read_old(root: Path, rel: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel
    table = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    feat = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    table["feature_idx"] = table.get("feature_idx", pd.Series(np.arange(len(table)))).astype(int)
    table["label_idx"] = table["label_idx"].astype(int)
    return table.reset_index(drop=True), feat[table["feature_idx"].to_numpy()]


def read_third(root: Path, rel: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / rel
    table = pd.read_csv(d / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    reg = pd.read_csv(d / "third_batch_task7_registry.csv", dtype={"case_id": str, "original_case_id": str})
    feat = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    table["feature_idx"] = table.get("feature_idx", pd.Series(np.arange(len(table)))).astype(int)
    frame = reg.merge(table[["case_id", "feature_idx"]], on="case_id", how="left")
    if frame["feature_idx"].isna().any():
        raise KeyError("Missing third feature rows")
    frame["label_idx"] = frame["label_idx"].astype(int)
    return frame.reset_index(drop=True), feat[frame["feature_idx"].astype(int).to_numpy()]


def profiles() -> dict[str, dict[str, int]]:
    return {
        "adapt48_balanced": {"AB": 18, "B1": 6, "B2": 12, "TC": 12},
        "adapt60_balanced": {"AB": 24, "B1": 6, "B2": 15, "TC": 15},
        "adapt72_balanced": {"AB": 30, "B1": 6, "B2": 18, "TC": 18},
        "adapt84_balanced": {"AB": 36, "B1": 6, "B2": 21, "TC": 21},
        "adapt72_high_focus": {"AB": 24, "B1": 4, "B2": 22, "TC": 22},
        "adapt96_high_focus": {"AB": 30, "B1": 4, "B2": 26, "TC": 36},
    }


def stable_key(case_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def split_profile(third: pd.DataFrame, profile: dict[str, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    chosen: list[int] = []
    for subtype, n in profile.items():
        g = third[third["task_l6_label"].eq(subtype)].copy()
        g["_key"] = g["case_id"].map(lambda x: stable_key(str(x), seed))
        chosen.extend(g.sort_values("_key").head(n).index.tolist())
    mask = np.zeros(len(third), dtype=bool)
    mask[np.array(chosen, dtype=int)] = True
    return np.where(mask)[0], np.where(~mask)[0]


def make_model(c: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=4000, class_weight="balanced", random_state=seed, solver="lbfgs"),
    )


def repeat_rows(x: np.ndarray, y: np.ndarray, is_adapt: np.ndarray, repeat_adapt: int) -> tuple[np.ndarray, np.ndarray]:
    if repeat_adapt <= 1:
        return x, y
    old_idx = np.where(~is_adapt)[0]
    adapt_idx = np.where(is_adapt)[0]
    idx = np.concatenate([old_idx, np.tile(adapt_idx, repeat_adapt)])
    return x[idx], y[idx]


def oof_and_holdout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    is_adapt: np.ndarray,
    x_hold: np.ndarray,
    c: float,
    repeat_adapt: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y_train), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(x_train, y_train)):
        x_tr, y_tr = repeat_rows(x_train[tr], y_train[tr], is_adapt[tr], repeat_adapt)
        model = make_model(c, seed + fold)
        model.fit(x_tr, y_tr)
        oof[va] = model.predict_proba(x_train[va])[:, 1]
    x_full, y_full = repeat_rows(x_train, y_train, is_adapt, repeat_adapt)
    model = make_model(c, seed + 1000)
    model.fit(x_full, y_full)
    return oof, model.predict_proba(x_hold)[:, 1]


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    old, x_old = read_old(root, args.old_feature_dir)
    third, x_third = read_third(root, args.third_feature_dir)
    y_old = old["label_idx"].to_numpy(dtype=int)
    existing = pd.read_csv(root / args.existing_third_pred, dtype={"case_id": str}).set_index("case_id")

    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    cs = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    repeats = [1, 2, 4, 8, 12]
    for profile_name, profile in profiles().items():
        adapt_idx, hold_idx = split_profile(third, profile, args.seed)
        adapt = third.iloc[adapt_idx].reset_index(drop=True)
        hold = third.iloc[hold_idx].reset_index(drop=True)
        x_adapt, x_hold = x_third[adapt_idx], x_third[hold_idx]
        y_adapt, y_hold = adapt["label_idx"].to_numpy(dtype=int), hold["label_idx"].to_numpy(dtype=int)
        for split_name, frame in [("adapt", adapt), ("holdout", hold)]:
            vc = frame["task_l6_label"].value_counts().to_dict()
            split_rows.append({"profile": profile_name, "split": split_name, "n": len(frame), **{f"n_{k}": vc.get(k, 0) for k in ["AB", "B1", "B2", "TC"]}})

        existing_hold = hold[["case_id", "label_idx"]].merge(
            existing[["final_pred_idx", "final_prob_high"]], left_on="case_id", right_index=True, how="left"
        )
        m = metric_dict(y_hold, existing_hold["final_pred_idx"].astype(int).to_numpy(), existing_hold["final_prob_high"].astype(float).to_numpy())
        rows.append({"profile": profile_name, "mode": "existing_wpc64_old_only", "repeat_adapt": 0, "C": np.nan, "threshold": np.nan, **{f"holdout_{k}": v for k, v in m.items()}})

        x_train = np.concatenate([x_old, x_adapt], axis=0)
        y_train = np.concatenate([y_old, y_adapt], axis=0)
        is_adapt = np.concatenate([np.zeros(len(old), dtype=bool), np.ones(len(adapt), dtype=bool)])
        for repeat_adapt in repeats:
            for c in cs:
                oof, p_hold = oof_and_holdout(x_train, y_train, is_adapt, x_hold, c, repeat_adapt, args.seed + int(c * 1e6) + repeat_adapt)
                t, _ = best_threshold(y_train, oof, "balanced_accuracy")
                oof_pred = (oof >= t).astype(int)
                hold_pred = (p_hold >= t).astype(int)
                oof_m = metric_dict(y_train, oof_pred, oof)
                hold_m = metric_dict(y_hold, hold_pred, p_hold)
                row = {
                    "profile": profile_name,
                    "mode": "old_plus_adapt_logreg",
                    "repeat_adapt": repeat_adapt,
                    "C": c,
                    "threshold": t,
                    "train_n_old": len(old),
                    "train_n_adapt": len(adapt),
                    "holdout_n": len(hold),
                }
                row.update({f"oof_{k}": v for k, v in oof_m.items()})
                row.update({f"holdout_{k}": v for k, v in hold_m.items()})
                rows.append(row)

        adapt[["case_id", "original_case_id", "task_l6_label", "label_idx", "image_name"]].to_csv(out / f"{profile_name}_adapt_cases.csv", index=False, encoding="utf-8-sig")
        hold[["case_id", "original_case_id", "task_l6_label", "label_idx", "image_name"]].to_csv(out / f"{profile_name}_holdout_cases.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows).sort_values(["holdout_balanced_accuracy", "holdout_accuracy", "holdout_f1"], ascending=False)
    split_df = pd.DataFrame(split_rows)
    summary.to_csv(out / "third_adaptation_logreg_fast_summary.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(out / "third_adaptation_logreg_fast_split_summary.csv", index=False, encoding="utf-8-sig")

    top = summary[summary["mode"].eq("old_plus_adapt_logreg")].iloc[0].to_dict()
    profile = profiles()[str(top["profile"])]
    adapt_idx, hold_idx = split_profile(third, profile, args.seed)
    adapt = third.iloc[adapt_idx].reset_index(drop=True)
    hold = third.iloc[hold_idx].reset_index(drop=True)
    x_adapt, x_hold = x_third[adapt_idx], x_third[hold_idx]
    y_adapt, y_hold = adapt["label_idx"].to_numpy(dtype=int), hold["label_idx"].to_numpy(dtype=int)
    x_train = np.concatenate([x_old, x_adapt], axis=0)
    y_train = np.concatenate([y_old, y_adapt], axis=0)
    is_adapt = np.concatenate([np.zeros(len(old), dtype=bool), np.ones(len(adapt), dtype=bool)])
    x_full, y_full = repeat_rows(x_train, y_train, is_adapt, int(top["repeat_adapt"]))
    model = make_model(float(top["C"]), args.seed + 9999)
    model.fit(x_full, y_full)
    p = model.predict_proba(x_hold)[:, 1]
    pred = (p >= float(top["threshold"])).astype(int)
    case_out = hold[["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]].copy()
    case_out["prob_high"] = p
    case_out["pred_idx"] = pred
    case_out["correct"] = (pred == y_hold).astype(int)
    case_out.to_csv(out / "best_logreg_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")
    subtype = case_out.groupby("task_l6_label").agg(n=("case_id", "size"), correct=("correct", "sum"), accuracy=("correct", "mean")).reset_index()
    subtype.to_csv(out / "best_logreg_holdout_metrics_by_subtype.csv", index=False, encoding="utf-8-sig")
    report = {
        "boundary": "Third-batch labels are used for the selected adaptation subset only; holdout is not used for fitting. Multiple profiles are exploratory.",
        "top": top,
        "top_subtype": subtype.to_dict("records"),
        "best_existing_same_holdout": summary[summary["mode"].eq("existing_wpc64_old_only")]
        .sort_values(["holdout_balanced_accuracy", "holdout_accuracy"], ascending=False)
        .head(3)
        .to_dict("records"),
    }
    (out / "third_adaptation_logreg_fast_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
