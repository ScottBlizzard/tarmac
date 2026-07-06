from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "grosspath_rc_v2_20260526"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_v24_review_upper_bound_20260527"

MAIN_THRESHOLD = 0.595
ROBUST_THRESHOLD = 0.57


def prep(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["main_prob"] = out["prob_base162"].astype(float)
    out["main_pred"] = (out["main_prob"] >= MAIN_THRESHOLD).astype(int)
    out["robust_prob"] = out[["prob_base162", "prob103_vitl", "prob_mean_core"]].mean(axis=1).astype(float)
    out["robust_pred"] = (out["robust_prob"] >= ROBUST_THRESHOLD).astype(int)
    out["main_correct"] = (out["main_pred"] == out["label_idx"].astype(int)).astype(int)
    out["robust_correct"] = (out["robust_pred"] == out["label_idx"].astype(int)).astype(int)
    out["safety_trigger"] = ((out["main_pred"] == 0) & (out["robust_pred"] == 1)).astype(int)
    return out


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": int(len(y)),
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate(frame: pd.DataFrame, split: str) -> list[dict[str, object]]:
    y = frame["label_idx"].to_numpy(dtype=int)
    main = frame["main_pred"].to_numpy(dtype=int)
    robust = frame["robust_pred"].to_numpy(dtype=int)
    trigger = frame["safety_trigger"].to_numpy(dtype=bool)

    rows: list[dict[str, object]] = []
    policies = {
        "main_only": main,
        "robust_all": robust,
        "auto_switch_on_safety_trigger": np.where(trigger, robust, main),
        "human_review_on_safety_trigger_oracle": np.where(trigger, y, main),
        "oracle_choose_main_or_robust": np.where(frame["main_correct"].eq(1).to_numpy(), main, robust),
    }
    for name, pred in policies.items():
        row = {"split": split, "policy": name}
        row.update(metrics(y, pred))
        row["trigger_n"] = int(trigger.sum()) if "trigger" in name else 0
        row["trigger_rate"] = float(trigger.mean()) if "trigger" in name else 0.0
        rows.append(row)

    table = pd.crosstab(frame["main_correct"], frame["robust_correct"], rownames=["main_correct"], colnames=["robust_correct"])
    for main_correct in [0, 1]:
        for robust_correct in [0, 1]:
            rows.append(
                {
                    "split": split,
                    "policy": f"correctness_overlap_main{main_correct}_robust{robust_correct}",
                    "n": int(table.loc[main_correct, robust_correct]) if main_correct in table.index and robust_correct in table.columns else 0,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "sensitivity": np.nan,
                    "specificity": np.nan,
                    "tn": np.nan,
                    "fp": np.nan,
                    "fn": np.nan,
                    "tp": np.nan,
                    "trigger_n": np.nan,
                    "trigger_rate": np.nan,
                }
            )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = prep(pd.read_csv(V2_DIR / "v2_development_diagnostic_table.csv"))
    external = prep(pd.read_csv(V2_DIR / "v2_external_diagnostic_table.csv"))

    rows: list[dict[str, object]] = []
    for split, frame in [
        ("development", dev),
        ("development:old", dev[dev["domain"].eq("old")]),
        ("development:third", dev[dev["domain"].eq("third")]),
        ("external_strict", external),
    ]:
        rows.extend(evaluate(frame, split))
    pd.DataFrame(rows).to_csv(OUT_DIR / "v24_review_upper_bound_metrics.csv", index=False, encoding="utf-8-sig")

    for split, frame in [("development", dev), ("external_strict", external)]:
        cases = frame[frame["safety_trigger"].eq(1)].copy()
        cases["effect_if_auto"] = "other"
        cases.loc[cases["main_correct"].eq(0) & cases["robust_correct"].eq(1), "effect_if_auto"] = "rescued"
        cases.loc[cases["main_correct"].eq(1) & cases["robust_correct"].eq(0), "effect_if_auto"] = "harmed"
        keep = [
            "case_id",
            "original_case_id",
            "domain",
            "task_l6_label",
            "task_l7_label",
            "label_idx",
            "main_prob",
            "robust_prob",
            "prob103_vitl",
            "prob_mean_core",
            "main_pred",
            "robust_pred",
            "main_correct",
            "robust_correct",
            "effect_if_auto",
            "quality_group",
            "view_type_final",
            "analysis_image_path",
        ]
        cases[[col for col in keep if col in cases.columns]].to_csv(OUT_DIR / f"v24_safety_trigger_cases_{split.replace(':', '_')}.csv", index=False, encoding="utf-8-sig")

    print(f"[done] {OUT_DIR}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
