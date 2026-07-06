from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from run_task7_stage2_auto_corrector_20260520 import build_features, load_gate_options, make_models, metric_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 stage-2 selective flipper: keep base prediction unless correction is confident.")
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
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/16_stage2_selective_flipper_20260520",
    )
    parser.add_argument("--gate-scores", default="upper_low_conf,hybrid_max_allrange_fn,all_range")
    parser.add_argument("--gate-targets", default="0.9")
    parser.add_argument("--feature-sets", default="raw_outputs,raw_plus_review,raw_plus_gross,raw_plus_review_gross")
    parser.add_argument("--models", default="logreg,extra")
    parser.add_argument("--train-scopes", default="all_train,review_train")
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def predict_prob(model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    clf = clone(model)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def inner_probs(model: object, x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan, dtype=float)
    for fold in sorted(np.unique(folds)):
        tr = folds != fold
        va = folds == fold
        if len(np.unique(y[tr])) < 2:
            out[va] = float(np.mean(y[tr]))
        else:
            out[va] = predict_prob(model, x[tr], y[tr], x[va])
    return np.nan_to_num(out, nan=float(np.nanmean(out)))


def apply_flip(base_pred: np.ndarray, prob: np.ndarray, t_l2h: float, t_h2l: float) -> np.ndarray:
    out = base_pred.copy()
    out[(base_pred == 0) & (prob >= t_l2h)] = 1
    out[(base_pred == 1) & (prob <= 1.0 - t_h2l)] = 0
    return out


def choose_flip_thresholds(
    y: np.ndarray,
    base_pred: np.ndarray,
    prob: np.ndarray,
    review_mask: np.ndarray,
    objective: str,
) -> tuple[float, float, float]:
    grid = np.round(np.linspace(0.50, 0.95, 10), 2)
    best = (float(accuracy_score(y, base_pred)), 0.99, 0.99)
    for t_l2h in grid:
        for t_h2l in grid:
            candidate = base_pred.copy()
            flipped = apply_flip(base_pred[review_mask], prob[review_mask], float(t_l2h), float(t_h2l))
            candidate[review_mask] = flipped
            if objective == "balanced_accuracy":
                score = float(balanced_accuracy_score(y, candidate))
            else:
                score = float(accuracy_score(y, candidate))
            # Prefer fewer flips when scores tie, to reduce harm.
            n_flip = int((candidate != base_pred).sum())
            key = (score, -n_flip, t_l2h + t_h2l)
            if key > (best[0], -10**9, 0.0):
                best = (score, float(t_l2h), float(t_h2l))
    return best[1], best[2], best[0]


def run_combo(
    df: pd.DataFrame,
    gate_df: pd.DataFrame,
    feature_name: str,
    x_df: pd.DataFrame,
    model_name: str,
    model: object,
    train_scope: str,
    objective: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    y = df["label_idx"].astype(int).to_numpy()
    base_pred = df["pred_upper"].astype(int).to_numpy()
    folds = df["fold_id"].astype(int).to_numpy()
    x = x_df.to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    case_frames: list[pd.DataFrame] = []
    for option, g in gate_df.groupby(["score", "target_accept_acc", "selection_mode"], sort=False):
        score_name, target_acc, selection_mode = option
        route = df[["case_id"]].merge(g[["case_id", "stage1_accept"]], on="case_id", how="left")
        accept = route["stage1_accept"].fillna(0).astype(int).to_numpy().astype(bool)
        review = ~accept
        final = base_pred.copy()
        final_prob = df["p_upper"].astype(float).to_numpy().copy()
        fold_choices = []
        for fold in sorted(np.unique(folds)):
            test = (folds == fold) & review
            if not test.any():
                continue
            if train_scope == "all_train":
                train = folds != fold
            elif train_scope == "review_train":
                train = (folds != fold) & review
            else:
                raise ValueError(train_scope)
            if train.sum() < 15 or len(np.unique(y[train])) < 2:
                continue
            train_review = train & review
            inner = inner_probs(model, x[train], y[train], folds[train])
            train_indices = np.where(train)[0]
            inner_full = np.full(len(df), np.nan, dtype=float)
            inner_full[train_indices] = inner
            t_l2h, t_h2l, train_score = choose_flip_thresholds(
                y[train],
                base_pred[train],
                inner,
                review[train],
                objective=objective,
            )
            prob = predict_prob(model, x[train], y[train], x[test])
            final[test] = apply_flip(base_pred[test], prob, t_l2h, t_h2l)
            final_prob[test] = prob
            fold_choices.append(f"{int(fold)}:{t_l2h:.2f}/{t_h2l:.2f}/{train_score:.3f}")

        base_wrong = base_pred != y
        final_wrong = final != y
        rescued = base_wrong & (~final_wrong) & review
        hurt = (~base_wrong) & final_wrong & review
        hard = df["hard_core"].astype(int).to_numpy().astype(bool)
        upper_fn = df["upper_fn"].astype(int).to_numpy().astype(bool)
        final_metrics = metric_row(y, final)
        review_metrics = metric_row(y[review], final[review]) if review.any() else {}
        accept_metrics = metric_row(y[accept], base_pred[accept]) if accept.any() else {}
        rows.append(
            {
                "gate_score": score_name,
                "gate_target_accept_acc": float(target_acc),
                "selection_mode": selection_mode,
                "feature_set": feature_name,
                "model": model_name,
                "train_scope": train_scope,
                "objective": objective,
                "accept_n": int(accept.sum()),
                "accept_frac": float(accept.mean()),
                "review_n": int(review.sum()),
                "review_frac": float(review.mean()),
                "fold_thresholds": ";".join(fold_choices),
                "final_accuracy": final_metrics["accuracy"],
                "final_balanced_accuracy": final_metrics["balanced_accuracy"],
                "final_specificity_low": final_metrics["specificity_low"],
                "final_sensitivity_high": final_metrics["sensitivity_high"],
                "final_tn": final_metrics["tn"],
                "final_fp": final_metrics["fp"],
                "final_fn": final_metrics["fn"],
                "final_tp": final_metrics["tp"],
                "accept_accuracy": accept_metrics.get("accuracy", np.nan),
                "review_final_accuracy": review_metrics.get("accuracy", np.nan),
                "review_final_sensitivity_high": review_metrics.get("sensitivity_high", np.nan),
                "review_final_specificity_low": review_metrics.get("specificity_low", np.nan),
                "rescued_total": int(rescued.sum()),
                "hurt_total": int(hurt.sum()),
                "rescued_fn": int((rescued & upper_fn).sum()),
                "hardcore_final_accuracy": float(accuracy_score(y[hard], final[hard])) if hard.any() else np.nan,
                "hardcore_rescued": int((rescued & hard).sum()),
                "hardcore_hurt": int((hurt & hard).sum()),
            }
        )
        case_out = df[
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
        case_out["gate_score"] = score_name
        case_out["gate_target_accept_acc"] = float(target_acc)
        case_out["feature_set"] = feature_name
        case_out["model"] = model_name
        case_out["train_scope"] = train_scope
        case_out["objective"] = objective
        case_out["stage1_accept"] = accept.astype(int)
        case_out["stage2_final_pred"] = final
        case_out["stage2_prob_high"] = final_prob
        case_out["rescued"] = rescued.astype(int)
        case_out["hurt"] = hurt.astype(int)
        case_frames.append(case_out)
    return rows, pd.concat(case_frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.case_scores_csv, dtype={"case_id": str, "original_case_id": str})
    gate_scores = {x.strip() for x in args.gate_scores.split(",") if x.strip()}
    gate_targets = {float(x.strip()) for x in args.gate_targets.split(",") if x.strip()}
    gate_df = load_gate_options(Path(args.gate_decisions_csv), gate_scores, gate_targets)
    feature_sets = build_features(df, Path(args.registry_csv))
    models = make_models(args.seed)
    feature_keep = {x.strip() for x in args.feature_sets.split(",") if x.strip()}
    model_keep = {x.strip() for x in args.models.split(",") if x.strip()}
    train_scopes = [x.strip() for x in args.train_scopes.split(",") if x.strip()]
    feature_sets = {k: v for k, v in feature_sets.items() if k in feature_keep}
    models = {k: v for k, v in models.items() if k in model_keep}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    cases: list[pd.DataFrame] = []
    for feature_name, x_df in feature_sets.items():
        for model_name, model in models.items():
            for train_scope in train_scopes:
                for objective in ["accuracy", "balanced_accuracy"]:
                    r, c = run_combo(df, gate_df, feature_name, x_df, model_name, model, train_scope, objective)
                    rows.extend(r)
                    cases.append(c)
    summary = pd.DataFrame(rows).sort_values(["final_balanced_accuracy", "final_accuracy"], ascending=False)
    case_df = pd.concat(cases, ignore_index=True)
    summary.to_csv(out_dir / "stage2_selective_flipper_summary.csv", index=False)
    case_df.to_csv(out_dir / "stage2_selective_flipper_case_outputs.csv", index=False)
    print(summary.head(40).to_string(index=False))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
