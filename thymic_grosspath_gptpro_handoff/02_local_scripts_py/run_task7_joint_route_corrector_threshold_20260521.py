from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jointly tune review-route and corrector thresholds for Task7 image-only reviewers.")
    parser.add_argument(
        "--run64-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/64_image_only_hardcore_reviewer_20260521",
    )
    parser.add_argument(
        "--route-case-table",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/63_gross_hardcore_signal_fixed_20260521/case_level_gross_signal_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/66_joint_route_corrector_threshold_20260521",
    )
    return parser.parse_args()


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if prob is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, prob))
    return out


def sanitize_route_name(route_col: str) -> str:
    return route_col.replace("route_score__", "").replace("__", "_")


def load_base(run64: Path, route_case_table: Path) -> pd.DataFrame:
    base = pd.read_csv(run64 / "model_non_easy_extra_oof.csv", dtype={"case_id": str})
    base = base[
        [
            "case_id",
            "original_case_id",
            "fold_id",
            "label_idx",
            "task_l6_label",
            "task_l7_label",
            "difficulty",
            "difficulty_fine",
            "hard_core",
            "p_best41",
            "pred_best41",
        ]
    ].copy()
    route = pd.read_csv(route_case_table, dtype={"case_id": str})
    route_cols = [c for c in route.columns if c.startswith("route_score__") and "model_visible" in c]
    base = base.merge(route[["case_id"] + route_cols], on="case_id", how="left")
    base["fold_id"] = base["fold_id"].astype(int)
    base["label_idx"] = base["label_idx"].astype(int)
    base["pred_best41"] = base["pred_best41"].astype(int)
    return base


def choose_joint(
    y: np.ndarray,
    base_pred: np.ndarray,
    corr_prob: np.ndarray,
    route_score: np.ndarray,
    train: np.ndarray,
    route_budgets: list[int],
    corr_thresholds: np.ndarray,
    objective: str,
) -> dict[str, object]:
    n_train = int(train.sum())
    best_key = (float("-inf"), float("-inf"), float("-inf"), 0.0)
    best: dict[str, object] | None = None
    for route_budget in route_budgets:
        if route_budget <= 0:
            route_threshold = float("inf")
        else:
            k = max(1, int(round(n_train * route_budget / 100.0)))
            route_threshold = float(np.sort(route_score[train])[-k])
        routed = train & (route_score >= route_threshold)
        for corr_threshold in corr_thresholds:
            corr_pred = (corr_prob >= corr_threshold).astype(int)
            pred = base_pred.copy()
            pred[routed] = corr_pred[routed]
            acc = float(accuracy_score(y[train], pred[train]))
            bacc = float(balanced_accuracy_score(y[train], pred[train]))
            f1 = float(f1_score(y[train], pred[train], zero_division=0))
            if objective == "accuracy":
                main = acc
            elif objective == "balanced_accuracy":
                main = bacc
            elif objective == "f1":
                main = f1
            else:
                raise ValueError(objective)
            key = (main, acc, bacc, -float(routed.sum()))
            if key > best_key:
                best_key = key
                best = {
                    "route_budget": int(route_budget),
                    "route_threshold": route_threshold,
                    "corr_threshold": float(corr_threshold),
                    "train_accuracy": acc,
                    "train_balanced_accuracy": bacc,
                    "train_f1": f1,
                    "train_routed_n": int(routed.sum()),
                }
    assert best is not None
    return best


def evaluate_candidate(
    base: pd.DataFrame,
    corr_prob: np.ndarray,
    corrector: str,
    route_col: str,
    objective: str,
    route_budgets: list[int],
    corr_thresholds: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    y = base["label_idx"].astype(int).to_numpy()
    folds = base["fold_id"].astype(int).to_numpy()
    base_pred = base["pred_best41"].astype(int).to_numpy()
    base_prob = base["p_best41"].astype(float).to_numpy()
    route_score = base[route_col].astype(float).to_numpy()
    final_pred = base_pred.copy()
    final_prob = base_prob.copy()
    routed_all = np.zeros(len(base), dtype=bool)
    fold_rows = []
    for fold in sorted(set(folds)):
        train = folds != fold
        test = folds == fold
        choice = choose_joint(y, base_pred, corr_prob, route_score, train, route_budgets, corr_thresholds, objective)
        routed = test & (route_score >= float(choice["route_threshold"]))
        corr_pred = (corr_prob >= float(choice["corr_threshold"])).astype(int)
        final_pred[routed] = corr_pred[routed]
        final_prob[routed] = corr_prob[routed]
        routed_all[routed] = True
        fold_rows.append(
            {
                "fold_id": int(fold),
                **choice,
                "test_routed_n": int(routed.sum()),
                "test_accuracy": float(accuracy_score(y[test], final_pred[test])),
            }
        )
    row = metrics(y, final_pred, final_prob)
    hard = base["hard_core"].astype(int).to_numpy()
    row.update(
        {
            "corrector": corrector,
            "route_col": route_col,
            "objective": objective,
            "routed_n": int(routed_all.sum()),
            "routed_pct": float(routed_all.mean()),
            "pass_n": int((~routed_all).sum()),
            "pass_acc": float((final_pred[~routed_all] == y[~routed_all]).mean()) if (~routed_all).any() else float("nan"),
            "routed_acc": float((final_pred[routed_all] == y[routed_all]).mean()) if routed_all.any() else float("nan"),
            "hard_core_routed": int(hard[routed_all].sum()),
            "hard_core_recall": float(hard[routed_all].sum() / max(hard.sum(), 1)),
            "rescue_n": int(((base_pred != y) & (final_pred == y) & routed_all).sum()),
            "hurt_n": int(((base_pred == y) & (final_pred != y) & routed_all).sum()),
        }
    )
    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
    case = base[["case_id", "original_case_id", "fold_id", "label_idx", "task_l6_label", "difficulty_fine", "p_best41", "pred_best41"]].copy()
    case["corrector_prob"] = corr_prob
    case["routed"] = routed_all
    case["final_prob"] = final_prob
    case["final_pred"] = final_pred
    case["final_correct"] = final_pred == y
    return row, pd.DataFrame(fold_rows), case


def main() -> None:
    args = parse_args()
    run64 = Path(args.run64_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_base(run64, Path(args.route_case_table))
    route_cols = [c for c in base.columns if c.startswith("route_score__")]

    nested = pd.read_csv(run64 / "nested_route_summary.csv")
    correctors: list[str] = []
    for name in nested["corrector"].tolist():
        name = str(name)
        if name not in correctors:
            correctors.append(name)
    # Keep enough breadth but avoid evaluating duplicate weak tails.
    correctors = correctors[:40]
    route_budgets = [0, 5, 10, 15, 20, 25, 30, 35, 38, 40, 45, 50]
    corr_thresholds = np.linspace(0.05, 0.95, 37)
    objectives = ["accuracy", "balanced_accuracy", "f1"]

    rows = []
    best_case = None
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    for corrector in correctors:
        path = run64 / f"{corrector}_oof.csv"
        if not path.exists():
            continue
        corr = pd.read_csv(path, dtype={"case_id": str})
        corr = base[["case_id"]].merge(corr[["case_id", "corrector_prob"]], on="case_id", how="left")
        corr_prob = corr["corrector_prob"].astype(float).to_numpy()
        for route_col in route_cols:
            for objective in objectives:
                row, folds, case = evaluate_candidate(base, corr_prob, corrector, route_col, objective, route_budgets, corr_thresholds)
                rows.append(row)
                safe = f"{corrector}__{sanitize_route_name(route_col)}__{objective}"
                folds.to_csv(output_dir / f"{safe}_fold_choices.csv", index=False, encoding="utf-8-sig")
                key = (float(row["accuracy"]), float(row["balanced_accuracy"]), float(row["f1"]))
                if key > best_key:
                    best_key = key
                    best_case = case.copy()
                    best_case["run_name"] = safe
                pd.DataFrame(rows).sort_values(["accuracy", "balanced_accuracy", "f1"], ascending=False).to_csv(
                    output_dir / "joint_threshold_summary.partial.csv", index=False, encoding="utf-8-sig"
                )
    summary = pd.DataFrame(rows).sort_values(["accuracy", "balanced_accuracy", "f1"], ascending=False)
    summary.to_csv(output_dir / "joint_threshold_summary.csv", index=False, encoding="utf-8-sig")
    if best_case is not None:
        best_case.to_csv(output_dir / "best_case_outputs.csv", index=False, encoding="utf-8-sig")
    report = {
        "best": summary.head(30).to_dict(orient="records"),
        "note": "No doctor text/pathology/case-id lookup. Jointly selects route budget and corrector probability threshold inside each training fold.",
    }
    (output_dir / "joint_threshold_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
