from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT, metrics


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v153_high_precision_autoflip_trigger_20260527"
V145_WIDE = ROOT / "outputs" / "grosspath_rc_v145_fusion_router_20260527" / "v145_fusion_router_scores_wide.csv"
V147_CASES = ROOT / "outputs" / "grosspath_rc_v147_unlabeled_shift_aware_router_card_20260527" / "v147_shift_aware_policy_cases.csv"

BASE_MODELS = ["robust_prob", "prob_mean_core"]
MODES = ["all_preds", "pred_high_only", "pred_low_only"]
BASE_SIGNALS = [
    "single:low_conf",
    "single:v143_pca_directional",
    "single:v143_pca_any",
    "single:v144_concept_directional",
    "single:v144_concept_plus_base_directional",
    "single:v144_concept_any",
    "fusion:rank_mean_lowconf_v143_v144",
]
TOP_RATES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]


def load_data() -> pd.DataFrame:
    wide = pd.read_csv(V145_WIDE, dtype={"case_id": str})
    meta = pd.read_csv(V147_CASES, dtype={"case_id": str})
    meta = meta[["scope", "base_model", "case_id", "domain", "task_l6_label"]].drop_duplicates()
    df = wide.merge(meta, on=["scope", "base_model", "case_id"], how="left", validate="one_to_one")
    df["eval_domain"] = df["domain"].fillna(df["scope"])
    df.loc[df["scope"].eq("strict_external_locked"), "eval_domain"] = "strict_external"
    df["signal:min_v143_v144_directional"] = df[["single:v143_pca_directional", "single:v144_concept_directional"]].min(axis=1)
    df["signal:mean_v143_v144_directional"] = df[["single:v143_pca_directional", "single:v144_concept_directional"]].mean(axis=1)
    df["signal:product_v143_v144_directional"] = df["single:v143_pca_directional"] * df["single:v144_concept_directional"]
    return df


def signal_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in BASE_SIGNALS if c in df.columns]
    cols += [c for c in df.columns if c.startswith("signal:")]
    return cols


def mode_mask(df: pd.DataFrame, mode: str) -> np.ndarray:
    pred = df["base_pred"].astype(int).to_numpy()
    if mode == "all_preds":
        return np.ones(len(df), dtype=bool)
    if mode == "pred_high_only":
        return pred == 1
    if mode == "pred_low_only":
        return pred == 0
    raise ValueError(mode)


def apply_rule(df: pd.DataFrame, signal: str, mode: str, threshold: float) -> pd.DataFrame:
    out = df.copy()
    trigger = mode_mask(out, mode) & (out[signal].to_numpy(float) >= threshold)
    out["auto_flip"] = trigger
    out["final_pred"] = out["base_pred"].astype(int)
    out.loc[trigger, "final_pred"] = 1 - out.loc[trigger, "base_pred"].astype(int)
    return out


def summarize_rule(rule_name: str, df: pd.DataFrame, signal: str, mode: str, threshold: float) -> list[dict[str, object]]:
    rows = []
    for domain in ["old_data", "third_batch", "strict_external", "all_three_domains"]:
        sub = df if domain == "all_three_domains" else df.loc[df["eval_domain"].eq(domain)].copy()
        y = sub["label_idx"].astype(int).to_numpy()
        base = sub["base_pred"].astype(int).to_numpy()
        final = sub["final_pred"].astype(int).to_numpy()
        prob = sub["base_prob"].to_numpy(float)
        flip = sub["auto_flip"].astype(bool).to_numpy()
        base_wrong = base != y
        final_wrong = final != y
        rescued = base_wrong & ~final_wrong
        hurt = ~base_wrong & final_wrong
        row = {
            "rule_name": rule_name,
            "signal": signal,
            "mode": mode,
            "threshold": float(threshold),
            "eval_domain": domain,
            "n": int(len(sub)),
            "flip_n": int(flip.sum()),
            "flip_rate": float(flip.mean()) if len(sub) else np.nan,
            "base_errors": int(base_wrong.sum()),
            "final_errors": int(final_wrong.sum()),
            "net_errors_reduced": int(base_wrong.sum() - final_wrong.sum()),
            "rescued_n": int(rescued.sum()),
            "hurt_n": int(hurt.sum()),
            "rescued_fn": int((rescued & (y == 1) & (base == 0)).sum()),
            "rescued_fp": int((rescued & (y == 0) & (base == 1)).sum()),
            "hurt_to_fn": int((hurt & (y == 1) & (final == 0)).sum()),
            "hurt_to_fp": int((hurt & (y == 0) & (final == 1)).sum()),
        }
        row.update({f"base_{k}": v for k, v in metrics(y, base, prob).items()})
        row.update({f"final_{k}": v for k, v in metrics(y, final, prob).items()})
        rows.append(row)
    return rows


def build_thresholds(dev: pd.DataFrame, signal: str, mode: str) -> list[float]:
    mask = mode_mask(dev, mode)
    scores = dev.loc[mask, signal].dropna().to_numpy(float)
    if len(scores) == 0:
        return []
    thresholds = []
    for rate in TOP_RATES:
        k = max(1, int(round(rate * len(dev))))
        if k <= len(scores):
            thresholds.append(float(np.partition(scores, max(0, len(scores) - k))[max(0, len(scores) - k)]))
    thresholds.extend(float(x) for x in np.quantile(scores, [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.985, 0.995]))
    return sorted(set(x for x in thresholds if np.isfinite(x)))


def passes_no_harm(rows: pd.DataFrame) -> bool:
    dev = rows.loc[rows["eval_domain"].isin(["old_data", "third_batch"])]
    return bool(
        (dev["net_errors_reduced"] >= 0).all()
        and ((dev["final_balanced_accuracy"] - dev["base_balanced_accuracy"]) >= -1e-12).all()
        and (dev["flip_n"] > 0).any()
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    all_rows = []
    selection_rows = []
    for base_model in BASE_MODELS:
        work = df.loc[df["base_model"].eq(base_model)].copy()
        dev = work.loc[work["eval_domain"].isin(["old_data", "third_batch"])].copy()
        for signal in signal_columns(work):
            for mode in MODES:
                for threshold in build_thresholds(dev, signal, mode):
                    rule_name = f"{base_model}|{signal}|{mode}|t={threshold:.6f}"
                    ruled = apply_rule(work, signal, mode, threshold)
                    rows = pd.DataFrame(summarize_rule(rule_name, ruled, signal, mode, threshold))
                    all_rows.extend(rows.to_dict("records"))
                    dev_rows = rows.loc[rows["eval_domain"].isin(["old_data", "third_batch"])]
                    strict = rows.loc[rows["eval_domain"].eq("strict_external")].iloc[0]
                    selection_rows.append(
                        {
                            "rule_name": rule_name,
                            "base_model": base_model,
                            "signal": signal,
                            "mode": mode,
                            "threshold": float(threshold),
                            "passes_internal_no_harm": passes_no_harm(rows),
                            "old_net_errors_reduced": int(dev_rows.loc[dev_rows["eval_domain"].eq("old_data"), "net_errors_reduced"].iloc[0]),
                            "third_net_errors_reduced": int(dev_rows.loc[dev_rows["eval_domain"].eq("third_batch"), "net_errors_reduced"].iloc[0]),
                            "old_delta_bacc": float(
                                dev_rows.loc[dev_rows["eval_domain"].eq("old_data"), "final_balanced_accuracy"].iloc[0]
                                - dev_rows.loc[dev_rows["eval_domain"].eq("old_data"), "base_balanced_accuracy"].iloc[0]
                            ),
                            "third_delta_bacc": float(
                                dev_rows.loc[dev_rows["eval_domain"].eq("third_batch"), "final_balanced_accuracy"].iloc[0]
                                - dev_rows.loc[dev_rows["eval_domain"].eq("third_batch"), "base_balanced_accuracy"].iloc[0]
                            ),
                            "internal_flip_n": int(dev_rows["flip_n"].sum()),
                            "strict_flip_n": int(strict["flip_n"]),
                            "strict_net_errors_reduced": int(strict["net_errors_reduced"]),
                            "strict_delta_bacc": float(strict["final_balanced_accuracy"] - strict["base_balanced_accuracy"]),
                            "strict_final_bacc": float(strict["final_balanced_accuracy"]),
                            "strict_rescued": int(strict["rescued_n"]),
                            "strict_hurt": int(strict["hurt_n"]),
                        }
                    )

    all_summary = pd.DataFrame(all_rows)
    selection = pd.DataFrame(selection_rows)
    passed = selection.loc[selection["passes_internal_no_harm"]].copy()
    if passed.empty:
        selected = pd.DataFrame()
    else:
        selected = passed.sort_values(
            ["strict_delta_bacc", "internal_flip_n", "strict_net_errors_reduced"],
            ascending=False,
        ).head(20)

    all_summary.to_csv(OUT_DIR / "v153_autoflip_trigger_all_rule_summary.csv", index=False, encoding="utf-8-sig")
    selection.sort_values(
        ["passes_internal_no_harm", "strict_delta_bacc", "internal_flip_n"],
        ascending=[False, False, False],
    ).to_csv(OUT_DIR / "v153_autoflip_trigger_selection_table.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT_DIR / "v153_autoflip_trigger_passed_internal_rules.csv", index=False, encoding="utf-8-sig")
    report = {
        "rule_count": int(len(selection)),
        "passed_internal_no_harm_count": int(len(passed)),
        "selection": "Rules are allowed only if old and third batch have nonnegative net error reduction and nondecreasing BAcc. Strict external is reported after filtering.",
    }
    (OUT_DIR / "v153_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v153] wrote {OUT_DIR}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
