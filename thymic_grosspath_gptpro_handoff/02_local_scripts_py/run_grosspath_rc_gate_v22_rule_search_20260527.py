from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "grosspath_rc_v2_20260526"
OUT_DIR = ROOT / "outputs" / "grosspath_rc_gate_v22_20260527"

MAIN_THRESHOLD = 0.595
ROBUST_THRESHOLD = 0.57
ROBUST_SOURCES = ("prob_base162", "prob103_vitl", "prob_mean_core")


def prep(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["main_prob"] = out["prob_base162"].astype(float)
    out["main_pred"] = (out["main_prob"] >= MAIN_THRESHOLD).astype(int)
    out["robust_prob"] = out[list(ROBUST_SOURCES)].mean(axis=1).astype(float)
    out["robust_pred"] = (out["robust_prob"] >= ROBUST_THRESHOLD).astype(int)
    out["model_diff"] = (out["robust_prob"] - out["main_prob"]).astype(float)
    out["abs_diff"] = out["model_diff"].abs()
    out["main_margin_to_low"] = MAIN_THRESHOLD - out["main_prob"]
    out["robust_margin_to_high"] = out["robust_prob"] - ROBUST_THRESHOLD
    out["main_fn"] = ((out["label_idx"].astype(int) == 1) & (out["main_pred"] == 0)).astype(int)
    out["main_fp"] = ((out["label_idx"].astype(int) == 0) & (out["main_pred"] == 1)).astype(int)
    out["would_rescue"] = ((out["main_pred"] != out["label_idx"].astype(int)) & (out["robust_pred"] == out["label_idx"].astype(int))).astype(int)
    out["would_harm"] = ((out["main_pred"] == out["label_idx"].astype(int)) & (out["robust_pred"] != out["label_idx"].astype(int))).astype(int)
    return out


def counts(frame: pd.DataFrame, trigger: np.ndarray) -> dict[str, float | int]:
    trigger = np.asarray(trigger, dtype=bool)
    main_fn_total = int(frame["main_fn"].sum())
    main_fp_total = int(frame["main_fp"].sum())
    n = len(frame)
    trigger_n = int(trigger.sum())
    rescued = int((frame["would_rescue"].to_numpy(dtype=int)[trigger] == 1).sum())
    harmed = int((frame["would_harm"].to_numpy(dtype=int)[trigger] == 1).sum())
    caught_fn = int(((frame["main_fn"].to_numpy(dtype=int) == 1) & trigger).sum())
    caught_fp = int(((frame["main_fp"].to_numpy(dtype=int) == 1) & trigger).sum())
    return {
        "n": n,
        "trigger_n": trigger_n,
        "trigger_rate": trigger_n / n if n else 0.0,
        "main_fn_total": main_fn_total,
        "main_fp_total": main_fp_total,
        "caught_main_fn_n": caught_fn,
        "caught_main_fn_rate": caught_fn / main_fn_total if main_fn_total else 0.0,
        "caught_main_fp_n": caught_fp,
        "caught_main_fp_rate": caught_fp / main_fp_total if main_fp_total else 0.0,
        "rescued_if_auto_n": rescued,
        "harmed_if_auto_n": harmed,
        "trigger_precision_for_rescue": rescued / trigger_n if trigger_n else 0.0,
        "net_auto_gain": rescued - harmed,
    }


def make_trigger(frame: pd.DataFrame, params: dict[str, float | int | str]) -> np.ndarray:
    trigger = (frame["main_pred"].eq(0) & frame["robust_pred"].eq(1)).to_numpy()
    trigger &= frame["main_prob"].to_numpy(dtype=float) <= float(params["main_prob_max"])
    trigger &= frame["robust_prob"].to_numpy(dtype=float) >= float(params["robust_prob_min"])
    trigger &= frame["model_diff"].to_numpy(dtype=float) >= float(params["diff_min"])
    trigger &= frame["prob103_vitl"].to_numpy(dtype=float) >= float(params["vitl_min"])
    if params["require_mean_core_high"]:
        trigger &= frame["prob_mean_core"].to_numpy(dtype=float) >= ROBUST_THRESHOLD
    return trigger


def evaluate_rule(name: str, params: dict[str, float | int | str], dev: pd.DataFrame, external: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for split, frame in [("development", dev), ("development:old", dev[dev["domain"].eq("old")]), ("development:third", dev[dev["domain"].eq("third")]), ("external_strict", external)]:
        row = {"rule": name, "split": split, **params}
        row.update(counts(frame, make_trigger(frame, params)))
        rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = prep(pd.read_csv(V2_DIR / "v2_development_diagnostic_table.csv"))
    external = prep(pd.read_csv(V2_DIR / "v2_external_diagnostic_table.csv"))

    param_grid = []
    for main_prob_max, robust_prob_min, diff_min, vitl_min, require_mean_core_high in product(
        [0.40, 0.45, 0.50, 0.55, 0.595],
        [0.57, 0.59, 0.61, 0.63, 0.65, 0.68],
        [0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        [0.50, 0.60, 0.70, 0.80],
        [0, 1],
    ):
        param_grid.append(
            {
                "main_prob_max": main_prob_max,
                "robust_prob_min": robust_prob_min,
                "diff_min": diff_min,
                "vitl_min": vitl_min,
                "require_mean_core_high": require_mean_core_high,
            }
        )

    dev_rows = []
    for idx, params in enumerate(param_grid):
        trigger = make_trigger(dev, params)
        row = {"rule_id": f"grid_{idx:04d}", **params}
        row.update(counts(dev, trigger))
        dev_rows.append(row)
    dev_grid = pd.DataFrame(dev_rows)
    dev_grid.to_csv(OUT_DIR / "gate_v22_dev_grid_all.csv", index=False, encoding="utf-8-sig")

    # Strategies are selected on development only. External is evaluated after selection.
    candidates = []
    for name, subset in [
        ("balanced_review_5pct", dev_grid[(dev_grid["trigger_rate"] <= 0.06) & (dev_grid["trigger_n"] >= 5)]),
        ("sensitive_review_10pct", dev_grid[(dev_grid["trigger_rate"] <= 0.12) & (dev_grid["trigger_n"] >= 8)]),
        ("high_precision_warning", dev_grid[(dev_grid["trigger_precision_for_rescue"] >= 0.25) & (dev_grid["trigger_n"] >= 5)]),
        ("fn_capture_priority", dev_grid[(dev_grid["trigger_rate"] <= 0.15) & (dev_grid["trigger_n"] >= 8)]),
    ]:
        if subset.empty:
            continue
        if name == "high_precision_warning":
            key_cols = ["trigger_precision_for_rescue", "caught_main_fn_rate", "trigger_n"]
            selected = subset.sort_values(key_cols, ascending=[False, False, False]).iloc[0]
        elif name == "fn_capture_priority":
            selected = subset.sort_values(["caught_main_fn_rate", "trigger_precision_for_rescue", "trigger_n"], ascending=[False, False, True]).iloc[0]
        else:
            selected = subset.sort_values(["caught_main_fn_rate", "trigger_precision_for_rescue", "trigger_n"], ascending=[False, False, True]).iloc[0]
        candidates.append((name, selected.to_dict()))

    selected_rows: list[dict[str, object]] = []
    case_exports = []
    for name, selected in candidates:
        params = {key: selected[key] for key in ["main_prob_max", "robust_prob_min", "diff_min", "vitl_min", "require_mean_core_high"]}
        selected_rows.extend(evaluate_rule(name, params, dev, external))
        for split, frame in [("development", dev), ("external_strict", external)]:
            trigger = make_trigger(frame, params)
            cases = frame.loc[trigger].copy()
            cases["selected_rule"] = name
            cases["split"] = split
            cases["trigger_effect"] = "other"
            cases.loc[cases["would_rescue"].eq(1), "trigger_effect"] = "would_rescue"
            cases.loc[cases["would_harm"].eq(1), "trigger_effect"] = "would_harm"
            keep = [
                "selected_rule",
                "split",
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
                "model_diff",
                "main_fn",
                "would_rescue",
                "would_harm",
                "trigger_effect",
                "quality_group",
                "view_type_final",
                "analysis_image_path",
            ]
            case_exports.append(cases[[col for col in keep if col in cases.columns]])

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(OUT_DIR / "gate_v22_selected_rules_frozen_eval.csv", index=False, encoding="utf-8-sig")
    if case_exports:
        pd.concat(case_exports, ignore_index=True).to_csv(OUT_DIR / "gate_v22_selected_trigger_cases.csv", index=False, encoding="utf-8-sig")

    print(f"[done] {OUT_DIR}")
    print("\nSelected frozen evaluations:")
    print(selected_df.to_string(index=False))
    print("\nTop development grid:")
    print(
        dev_grid.sort_values(["caught_main_fn_rate", "trigger_precision_for_rescue", "trigger_n"], ascending=[False, False, True])
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
