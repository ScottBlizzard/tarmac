from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested route-budget selection for Task7 gross corrector.")
    parser.add_argument(
        "--signal-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/63_gross_hardcore_signal_fixed_20260521",
    )
    parser.add_argument("--output-name", default="nested_route_policy_summary.csv")
    return parser.parse_args()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def apply_threshold(base_pred: np.ndarray, corr_pred: np.ndarray, score: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    routed = score >= threshold
    pred = base_pred.copy()
    pred[routed] = corr_pred[routed]
    return pred, routed


def choose_threshold(
    y: np.ndarray,
    base_pred: np.ndarray,
    corr_pred: np.ndarray,
    score: np.ndarray,
    budgets: list[int],
    objective: str,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    n = len(y)
    for budget in budgets:
        if budget <= 0:
            threshold = float("inf")
        else:
            k = max(1, int(round(n * budget / 100.0)))
            threshold = float(np.sort(score)[-k])
        pred, routed = apply_threshold(base_pred, corr_pred, score, threshold)
        acc = float(accuracy_score(y, pred))
        bacc = float(balanced_accuracy_score(y, pred))
        pass_acc = float((pred[~routed] == y[~routed]).mean()) if (~routed).any() else 0.0
        routed_acc = float((pred[routed] == y[routed]).mean()) if routed.any() else 0.0
        if objective == "accuracy":
            main = acc
        elif objective == "pass90_accuracy":
            main = acc if pass_acc >= 0.90 else acc - 1.0
        elif objective == "balanced_accuracy":
            main = bacc
        else:
            raise ValueError(objective)
        row = {
            "budget_pct": int(budget),
            "threshold": threshold,
            "train_accuracy": acc,
            "train_balanced_accuracy": bacc,
            "train_pass_acc": pass_acc,
            "train_routed_acc": routed_acc,
            "train_routed_n": int(routed.sum()),
            "main": main,
        }
        key = (main, acc, bacc, pass_acc, -int(routed.sum()))
        if best is None or key > (
            float(best["main"]),
            float(best["train_accuracy"]),
            float(best["train_balanced_accuracy"]),
            float(best["train_pass_acc"]),
            -int(best["train_routed_n"]),
        ):
            best = row
    assert best is not None
    return best


def evaluate_pair(
    case_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    route_col: str,
    corrector_name: str,
    objective: str,
    budgets: list[int],
) -> tuple[dict[str, object], pd.DataFrame]:
    y = case_df["label_idx"].astype(int).to_numpy()
    base_pred = case_df["pred_best41"].astype(int).to_numpy()
    corr_pred = corr_df["corrector_pred"].astype(int).to_numpy()
    score = case_df[route_col].astype(float).to_numpy()
    folds = case_df["fold_id"].astype(int).to_numpy()
    final_pred = base_pred.copy()
    final_routed = np.zeros(len(case_df), dtype=bool)
    fold_rows = []
    for fold in sorted(set(folds)):
        train = folds != fold
        test = folds == fold
        choice = choose_threshold(y[train], base_pred[train], corr_pred[train], score[train], budgets, objective)
        pred_fold, routed_fold = apply_threshold(base_pred[test], corr_pred[test], score[test], float(choice["threshold"]))
        final_pred[test] = pred_fold
        final_routed[test] = routed_fold
        fold_rows.append(
            {
                "fold_id": int(fold),
                **choice,
                "test_n": int(test.sum()),
                "test_routed_n": int(routed_fold.sum()),
                "test_accuracy": float(accuracy_score(y[test], pred_fold)),
                "test_pass_acc": float((pred_fold[~routed_fold] == y[test][~routed_fold]).mean())
                if (~routed_fold).any()
                else float("nan"),
                "test_routed_acc": float((pred_fold[routed_fold] == y[test][routed_fold]).mean())
                if routed_fold.any()
                else float("nan"),
            }
        )
    out = metrics(y, final_pred)
    hard = case_df["hard_core"].astype(int).to_numpy()
    out.update(
        {
            "route_col": route_col,
            "corrector_name": corrector_name,
            "objective": objective,
            "routed_n": int(final_routed.sum()),
            "routed_pct": float(final_routed.mean()),
            "pass_n": int((~final_routed).sum()),
            "pass_acc": float((final_pred[~final_routed] == y[~final_routed]).mean()) if (~final_routed).any() else float("nan"),
            "routed_acc": float((final_pred[final_routed] == y[final_routed]).mean()) if final_routed.any() else float("nan"),
            "hard_core_routed": int(hard[final_routed].sum()),
            "hard_core_recall": float(hard[final_routed].sum() / max(hard.sum(), 1)),
            "rescue_n": int(((base_pred != y) & (final_pred == y) & final_routed).sum()),
            "hurt_n": int(((base_pred == y) & (final_pred != y) & final_routed).sum()),
        }
    )
    out["net_rescue"] = int(out["rescue_n"] - out["hurt_n"])
    return out, pd.DataFrame(fold_rows)


def main() -> None:
    args = parse_args()
    signal_dir = Path(args.signal_dir)
    case_df = pd.read_csv(signal_dir / "case_level_gross_signal_table.csv", dtype={"case_id": str})
    route_cols = [col for col in case_df.columns if col.startswith("route_score__")]
    correctors = {
        p.stem.replace("_oof", ""): pd.read_csv(p, dtype={"case_id": str})
        for p in signal_dir.glob("corrector_*_oof.csv")
    }
    budgets = [0, 5, 10, 15, 20, 25, 30, 35, 38, 40, 45, 50]
    rows = []
    for route_col in route_cols:
        for corr_name, corr_df in correctors.items():
            corr_df = case_df[["case_id"]].merge(corr_df, on="case_id", how="left")
            for objective in ["accuracy", "pass90_accuracy", "balanced_accuracy"]:
                row, fold_rows = evaluate_pair(case_df, corr_df, route_col, corr_name, objective, budgets)
                rows.append(row)
                safe_name = route_col.replace("route_score__", "").replace("/", "_")
                fold_rows.to_csv(
                    signal_dir / f"nested_folds__{safe_name}__{corr_name}__{objective}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
    summary = pd.DataFrame(rows).sort_values(["accuracy", "balanced_accuracy", "net_rescue"], ascending=False)
    summary.to_csv(signal_dir / args.output_name, index=False, encoding="utf-8-sig")
    (signal_dir / "nested_route_policy_best.json").write_text(
        json.dumps(summary.head(20).to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
