from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_task7_gross_feature_probe_20260520 import extract_gross_features
from run_task7_hardcore_gross_calibrator_20260520 import add_manual_gross_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 automatic stage-2 correction after a strict stage-1 gate.")
    parser.add_argument(
        "--case-scores-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/12_highrisk_review_policy_20260520/case_review_scores_all.csv",
    )
    parser.add_argument(
        "--gate-decisions-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/13_stage1_direct_pass_gate_20260520/stage1_gate_nested_case_decisions.csv",
    )
    parser.add_argument(
        "--registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/15_stage2_auto_corrector_20260520",
    )
    parser.add_argument("--gate-scores", default="upper_low_conf,hybrid_max_allrange_fn")
    parser.add_argument("--gate-targets", default="0.9")
    parser.add_argument("--feature-sets", default="raw_outputs,raw_plus_review,raw_plus_gross,raw_plus_review_gross")
    parser.add_argument("--models", default="logreg,extra")
    parser.add_argument("--train-scopes", default="all_train,review_train")
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "specificity_low": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "sensitivity_high": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def build_features(df: pd.DataFrame, registry_csv: Path | None = None) -> dict[str, pd.DataFrame]:
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    pred_cols = [c for c in df.columns if c.startswith("pred_") and c != "pred_upper"]
    review_cols = [c for c in df.columns if c.startswith("review_score_")]
    blocks: list[pd.DataFrame] = []
    raw = pd.DataFrame(index=df.index)
    for col in prob_cols:
        p = pd.to_numeric(df[col], errors="coerce").fillna(0.5).to_numpy(dtype=float)
        name = col[2:]
        raw[f"p_{name}"] = p
        raw[f"margin_{name}"] = np.abs(p - 0.5)
        raw[f"entropy_{name}"] = entropy_binary(p)
    for col in pred_cols:
        raw[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    probs = df[prob_cols].astype(float).fillna(0.5).to_numpy()
    votes = (probs >= 0.5).astype(float)
    p_upper = pd.to_numeric(df["p_upper"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    pred_upper = pd.to_numeric(df["pred_upper"], errors="coerce").fillna(0).to_numpy(dtype=float)
    raw["prob_mean"] = probs.mean(axis=1)
    raw["prob_median"] = np.median(probs, axis=1)
    raw["prob_std"] = probs.std(axis=1)
    raw["prob_range"] = probs.max(axis=1) - probs.min(axis=1)
    raw["vote_frac"] = votes.mean(axis=1)
    raw["vote_disagree"] = ((votes.sum(axis=1) > 0) & (votes.sum(axis=1) < votes.shape[1])).astype(float)
    raw["upper_conf"] = np.where(pred_upper >= 0.5, p_upper, 1.0 - p_upper)
    raw["upper_pred_is_high"] = pred_upper
    raw["image_count"] = pd.to_numeric(df.get("image_count", 1), errors="coerce").fillna(1.0)
    if "selection_rule" in df.columns:
        raw = pd.concat([raw, pd.get_dummies(df["selection_rule"].fillna(""), prefix="selection", dtype=float)], axis=1)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    review = df[review_cols].astype(float).fillna(0.0)
    out = {
        "raw_outputs": raw,
        "raw_plus_review": pd.concat([raw, review], axis=1),
    }
    if registry_csv is not None and registry_csv.exists():
        registry = pd.read_csv(registry_csv, dtype={"case_id": str, "original_case_id": str})
        aligned = df[["case_id"]].merge(registry, on="case_id", how="left")
        if "肉眼所见" in aligned.columns:
            aligned["肉眼所见"] = aligned["肉眼所见"].fillna("")
        gross = add_manual_gross_scores(extract_gross_features(aligned))
        gross = gross.add_prefix("gross_").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["raw_plus_gross"] = pd.concat([raw, gross], axis=1)
        out["raw_plus_review_gross"] = pd.concat([raw, review, gross], axis=1)
    return out


def make_models(seed: int) -> dict[str, object]:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed),
        ),
        "rf": RandomForestClassifier(
            n_estimators=260,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra": ExtraTreesClassifier(
            n_estimators=320,
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(max_depth=2, learning_rate=0.035, n_estimators=80, random_state=seed),
    }


def fit_predict_label(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clf = clone(model)
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    return (prob >= 0.5).astype(int), prob


def load_gate_options(gate_csv: Path, gate_scores: set[str], gate_targets: set[float]) -> pd.DataFrame:
    gate = pd.read_csv(gate_csv, dtype={"case_id": str})
    keep = gate[
        gate["score"].isin(gate_scores)
        & gate["selection_mode"].isin(["raw"])
        & gate["target_accept_acc"].isin(gate_targets)
    ].copy()
    return keep


def evaluate_stage2(
    df: pd.DataFrame,
    gate_df: pd.DataFrame,
    feature_name: str,
    x_df: pd.DataFrame,
    model_name: str,
    model: object,
    train_scope: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    base_pred = df["pred_upper"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    x = x_df.to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    case_outputs: list[pd.DataFrame] = []

    option_cols = ["score", "target_accept_acc", "selection_mode"]
    for option, g in gate_df.groupby(option_cols, sort=False):
        score_name, target_acc, selection_mode = option
        route = df[["case_id"]].merge(
            g[["case_id", "stage1_accept", "stage1_review"]],
            on="case_id",
            how="left",
        )
        accept = route["stage1_accept"].fillna(0).astype(int).to_numpy().astype(bool)
        review = ~accept
        final_pred = base_pred.copy()
        final_prob = df["p_upper"].astype(float).to_numpy().copy()

        for fold in sorted(np.unique(folds)):
            test = (folds == fold) & review
            if not test.any():
                continue
            if train_scope == "all_train":
                train = folds != fold
            elif train_scope == "review_train":
                train = (folds != fold) & review
            elif train_scope == "wrong_enriched_train":
                # Use reviewed cases plus a small amount of high-confidence accepted cases to stabilize the boundary.
                train = (folds != fold) & (review | (df["upper_wrong"].astype(int).to_numpy() == 1))
            else:
                raise ValueError(train_scope)
            if train.sum() < 15 or len(np.unique(y[train])) < 2:
                continue
            pred, prob = fit_predict_label(model, x[train], y[train], x[test])
            final_pred[test] = pred
            final_prob[test] = prob

        base_metrics = metric_row(y, base_pred)
        final_metrics = metric_row(y, final_pred)
        accept_metrics = metric_row(y[accept], base_pred[accept]) if accept.any() else {}
        review_metrics = metric_row(y[review], final_pred[review]) if review.any() else {}
        review_base_metrics = metric_row(y[review], base_pred[review]) if review.any() else {}
        hard_core = df["hard_core"].astype(int).to_numpy().astype(bool)
        upper_fn = df["upper_fn"].astype(int).to_numpy().astype(bool)
        base_wrong = base_pred != y
        final_wrong = final_pred != y
        rescued = base_wrong & (~final_wrong) & review
        hurt = (~base_wrong) & final_wrong & review
        rows.append(
            {
                "gate_score": score_name,
                "gate_target_accept_acc": float(target_acc),
                "gate_selection_mode": selection_mode,
                "feature_set": feature_name,
                "model": model_name,
                "train_scope": train_scope,
                "accept_n": int(accept.sum()),
                "accept_frac": float(accept.mean()),
                "review_n": int(review.sum()),
                "review_frac": float(review.mean()),
                "base_accuracy": base_metrics["accuracy"],
                "base_balanced_accuracy": base_metrics["balanced_accuracy"],
                "final_accuracy": final_metrics["accuracy"],
                "final_balanced_accuracy": final_metrics["balanced_accuracy"],
                "final_specificity_low": final_metrics["specificity_low"],
                "final_sensitivity_high": final_metrics["sensitivity_high"],
                "final_tn": final_metrics["tn"],
                "final_fp": final_metrics["fp"],
                "final_fn": final_metrics["fn"],
                "final_tp": final_metrics["tp"],
                "accept_accuracy": accept_metrics.get("accuracy", np.nan),
                "review_base_accuracy": review_base_metrics.get("accuracy", np.nan),
                "review_final_accuracy": review_metrics.get("accuracy", np.nan),
                "review_final_sensitivity_high": review_metrics.get("sensitivity_high", np.nan),
                "review_final_specificity_low": review_metrics.get("specificity_low", np.nan),
                "rescued_total": int(rescued.sum()),
                "hurt_total": int(hurt.sum()),
                "rescued_fn": int((rescued & upper_fn).sum()),
                "hardcore_final_accuracy": float(accuracy_score(y[hard_core], final_pred[hard_core])) if hard_core.any() else np.nan,
                "hardcore_rescued": int((rescued & hard_core).sum()),
                "hardcore_hurt": int((hurt & hard_core).sum()),
            }
        )
        out = df[
            [
                "case_id",
                "original_case_id",
                "fold_id",
                "label_idx",
                "pred_upper",
                "p_upper",
                "difficulty_fine",
                "hard_core",
                "upper_wrong",
                "upper_fn",
                "upper_fp",
            ]
        ].copy()
        out["gate_score"] = score_name
        out["gate_target_accept_acc"] = float(target_acc)
        out["feature_set"] = feature_name
        out["model"] = model_name
        out["train_scope"] = train_scope
        out["stage1_accept"] = accept.astype(int)
        out["stage2_pred"] = final_pred
        out["stage2_prob_high"] = final_prob
        out["final_wrong"] = (final_pred != y).astype(int)
        out["rescued"] = rescued.astype(int)
        out["hurt"] = hurt.astype(int)
        case_outputs.append(out)
    return rows, pd.concat(case_outputs, ignore_index=True)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    gate_scores = {x.strip() for x in args.gate_scores.split(",") if x.strip()}
    gate_targets = {float(x.strip()) for x in args.gate_targets.split(",") if x.strip()}
    gate_options = load_gate_options(Path(args.gate_decisions_csv), gate_scores, gate_targets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = build_features(df, Path(args.registry_csv))
    models = make_models(args.seed)
    feature_keep = {x.strip() for x in args.feature_sets.split(",") if x.strip()}
    model_keep = {x.strip() for x in args.models.split(",") if x.strip()}
    train_scopes = [x.strip() for x in args.train_scopes.split(",") if x.strip()]
    feature_sets = {k: v for k, v in feature_sets.items() if k in feature_keep}
    models = {k: v for k, v in models.items() if k in model_keep}

    rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    for feature_name, x_df in feature_sets.items():
        for model_name, model in models.items():
            for train_scope in train_scopes:
                r, c = evaluate_stage2(df, gate_options, feature_name, x_df, model_name, model, train_scope)
                rows.extend(r)
                case_frames.append(c)

    summary = pd.DataFrame(rows).sort_values(
        ["final_balanced_accuracy", "final_accuracy", "final_sensitivity_high"], ascending=False
    )
    cases = pd.concat(case_frames, ignore_index=True)
    summary.to_csv(out_dir / "stage2_auto_corrector_summary.csv", index=False)
    cases.to_csv(out_dir / "stage2_auto_corrector_case_outputs.csv", index=False)
    print(summary.head(30).to_string(index=False))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
