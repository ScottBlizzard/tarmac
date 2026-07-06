from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_task7_third_adaptation_probe_20260521 import (  # noqa: E402
    best_threshold,
    candidate_specs,
    fit_oof_and_external,
    make_model,
    metric_dict,
    read_old_features,
    read_third_features,
    repeat_train_rows,
    split_adapt_holdout,
)


PROFILE = {"AB": 24, "B1": 4, "B2": 22, "TC": 22}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused stronger-model run for third-batch high-risk adaptation.")
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
        default="outputs/batch1_batch2_task567_20260514/task7_adaptation_runs/04_adapt72_high_focus_focused_models_20260521",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    old, x_old = read_old_features(root, args.old_feature_dir)
    third, x_third = read_third_features(root, args.third_feature_dir)
    adapt_idx, hold_idx = split_adapt_holdout(third, PROFILE, args.seed)
    adapt = third.iloc[adapt_idx].reset_index(drop=True)
    hold = third.iloc[hold_idx].reset_index(drop=True)
    x_adapt = x_third[adapt_idx]
    x_hold = x_third[hold_idx]
    y_old = old["label_idx"].to_numpy(dtype=int)
    y_adapt = adapt["label_idx"].to_numpy(dtype=int)
    y_hold = hold["label_idx"].to_numpy(dtype=int)
    x_train = np.concatenate([x_old, x_adapt], axis=0)
    y_train = np.concatenate([y_old, y_adapt], axis=0)
    is_adapt = np.concatenate([np.zeros(len(old), dtype=bool), np.ones(len(adapt), dtype=bool)])

    rows = []
    selected_specs = [
        s
        for s in candidate_specs()
        if s.kind in {"logreg", "rf", "extra", "gb", "mlp"}
        and s.name
        not in {
            "mlp64_32_a003",
        }
    ]
    for repeat_adapt in [1, 2, 4, 8, 12]:
        for i, spec in enumerate(selected_specs):
            oof, p_hold = fit_oof_and_external(
                spec,
                x_train,
                y_train,
                is_adapt,
                x_hold,
                repeat_adapt,
                args.seed + 97 * i + repeat_adapt,
                5,
            )
            for objective in ["balanced_accuracy", "accuracy"]:
                t, _ = best_threshold(y_train, oof, objective)
                oof_pred = (oof >= t).astype(int)
                pred = (p_hold >= t).astype(int)
                oof_m = metric_dict(y_train, oof_pred, oof)
                hold_m = metric_dict(y_hold, pred, p_hold)
                row = {
                    "profile": "adapt72_high_focus",
                    "repeat_adapt": repeat_adapt,
                    "model": spec.name,
                    "kind": spec.kind,
                    "threshold_objective": objective,
                    "threshold": t,
                    "train_n_old": len(old),
                    "train_n_adapt": len(adapt),
                    "holdout_n": len(hold),
                }
                row.update({f"oof_{k}": v for k, v in oof_m.items()})
                row.update({f"holdout_{k}": v for k, v in hold_m.items()})
                rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["holdout_balanced_accuracy", "holdout_accuracy", "holdout_f1"], ascending=False)
    summary.to_csv(out / "focused_model_summary.csv", index=False, encoding="utf-8-sig")
    adapt[["case_id", "original_case_id", "task_l6_label", "label_idx", "image_name"]].to_csv(out / "adapt72_high_focus_adapt_cases.csv", index=False, encoding="utf-8-sig")
    hold[["case_id", "original_case_id", "task_l6_label", "label_idx", "image_name"]].to_csv(out / "adapt72_high_focus_holdout_cases.csv", index=False, encoding="utf-8-sig")

    top = summary.iloc[0].to_dict()
    spec = next(s for s in selected_specs if s.name == top["model"])
    x_full, y_full = repeat_train_rows(x_train, y_train, is_adapt, int(top["repeat_adapt"]))
    model = make_model(spec, args.seed + 7777)
    model.fit(x_full, y_full)
    p_hold = model.predict_proba(x_hold)[:, 1]
    pred = (p_hold >= float(top["threshold"])).astype(int)
    case_out = hold[["case_id", "original_case_id", "task_l6_label", "task_l7_label", "label_idx", "image_name", "image_path"]].copy()
    case_out["prob_high"] = p_hold
    case_out["pred_idx"] = pred
    case_out["correct"] = (pred == y_hold).astype(int)
    case_out.to_csv(out / "best_focused_holdout_case_predictions.csv", index=False, encoding="utf-8-sig")
    subtype = case_out.groupby("task_l6_label").agg(n=("case_id", "size"), correct=("correct", "sum"), accuracy=("correct", "mean")).reset_index()
    subtype.to_csv(out / "best_focused_holdout_metrics_by_subtype.csv", index=False, encoding="utf-8-sig")
    report = {"top": top, "subtype": subtype.to_dict("records")}
    (out / "focused_model_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
