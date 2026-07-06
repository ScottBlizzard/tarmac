from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for item in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from run_task7_external_third_batch_64style_20260521 import (  # noqa: E402
    build_prob_features,
    candidate_specs,
    choose_route_threshold,
    corrector_specs,
    feature_matrix,
    load_old_frame,
    load_or_extract_third_features,
    metric_dict,
    oof_and_external_probs,
    route_specs,
    scope_mask,
    select_base_model,
)


LOW_LABEL = "low_risk_group"
HIGH_LABEL = "high_risk_group"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task7 external stress test on a free-form thymoma/carcinoma folder.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--images-root", default="datasets/external_thymoma_carcinoma_20260522")
    parser.add_argument(
        "--old-registry-csv",
        default="outputs/batch1_batch2_task567_20260514/frozen_inputs/combined_task567_registry_with_gross_findings_20260520.csv",
    )
    parser.add_argument(
        "--old-curriculum-csv",
        default="outputs/batch1_batch2_task567_20260514/task7_curriculum_runs/09_case_mlp_schemeB_m060_salvagehard_full5fold/curriculum_case_table.csv",
    )
    parser.add_argument(
        "--old-dino-feature-table",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521/case_dino_concat_feature_table.csv",
    )
    parser.add_argument(
        "--old-dino-feature-npy",
        default="outputs/batch1_batch2_task567_20260514/task7_gross_feature_runs/68_roi_whole_plus_crop_embedding_probe_20260521/case_dino_concat_features.npy",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch1_batch2_task567_20260514/task7_external_runs/20_external_thymoma_carcinoma_64style_wpc_20260522",
    )
    parser.add_argument("--repo-dir", default="third_party/round3/dinov2")
    parser.add_argument("--model-names", default="dinov2_vits14,dinov2_vitb14")
    parser.add_argument("--feature-mode", default="cls_patchmean", choices=("cls", "patch_mean", "cls_patchmean"))
    parser.add_argument("--input-variant", default="whole_plus_crop", choices=("whole", "crop", "whole_plus_crop"))
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def extract_original_case_id(name: str) -> str:
    stem = Path(name).stem
    match = re.search(r"--(\d+)", stem)
    if match:
        return match.group(1)
    match = re.search(r"(\d{6,8})", stem)
    return match.group(1) if match else re.sub(r"\W+", "_", stem)


def parse_task7_label(name: str) -> tuple[str, str, int, str]:
    text = Path(name).stem
    # High-risk rules first, because mixed labels such as B1-B2 contain a low-risk token.
    if any(key in text for key in ["鳞状细胞癌", "基底细胞样癌", "胸腺癌", "低分化"]):
        return "TC", HIGH_LABEL, 1, "carcinoma_high"
    if any(key in text for key in ["B2-B3", "B2型", "B3型", "B2--B3", "B1-B2", "B1--B2"]):
        if "B1-B2" in text or "B1--B2" in text:
            return "B1_B2_mixed", HIGH_LABEL, 1, "mixed_contains_B2_high"
        if "B2-B3" in text or "B2--B3" in text:
            return "B2_B3_mixed", HIGH_LABEL, 1, "mixed_contains_B2_B3_high"
        if "B3型" in text:
            return "B3", HIGH_LABEL, 1, "B3_high"
        return "B2", HIGH_LABEL, 1, "B2_high"
    if "大部分为B2" in text or "少部分为B3" in text:
        return "B2_B3_mixed", HIGH_LABEL, 1, "mixed_contains_B2_B3_high"
    if "微结节" in text:
        return "MNT_assumed_low", LOW_LABEL, 0, "nonstandard_assumed_low"
    if "AB型" in text or "AB胸腺瘤" in text or "AB型胸腺瘤" in text or "，AB型" in text:
        return "AB", LOW_LABEL, 0, "AB_low"
    if "A型" in text:
        return "A", LOW_LABEL, 0, "A_low"
    if "B1型" in text:
        return "B1", LOW_LABEL, 0, "B1_low"
    raise ValueError(f"Cannot parse Task7 label from filename: {name}")


def build_external_registry(images_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, image_path in enumerate(sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES), start=1):
        l6_label, l7_label, label_idx, label_rule = parse_task7_label(image_path.name)
        original_case_id = extract_original_case_id(image_path.name)
        case_id = f"doctor_ext_{idx:03d}_{original_case_id}"
        rows.append(
            {
                "case_id": case_id,
                "original_case_id": original_case_id,
                "source_dataset": "external_thymoma_carcinoma_20260522",
                "source_folder": image_path.parent.name,
                "image_count": 1,
                "image_filenames": image_path.name,
                "selected_original_image_name": image_path.name,
                "selected_original_image_relpath": str(image_path.relative_to(images_root)),
                "selected_original_image_path": str(image_path),
                "training_image_path": str(image_path),
                "image_name": image_path.name,
                "image_path": str(image_path),
                "selection_rule": "doctor_external_single_image",
                "task_l6_label": l6_label,
                "task_l7_label": l7_label,
                "label_idx": int(label_idx),
                "label_rule": label_rule,
                "strict_task7_eval": int(label_rule != "nonstandard_assumed_low"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No images found under {images_root}")
    return pd.DataFrame(rows).sort_values(["label_idx", "task_l6_label", "original_case_id", "image_name"]).reset_index(drop=True)


def summarize_metrics(frame: pd.DataFrame, pred_col: str, prob_col: str, prefix: str) -> dict[str, object]:
    y = frame["label_idx"].to_numpy(int)
    pred = frame[pred_col].to_numpy(int)
    prob = frame[prob_col].to_numpy(float)
    out = {f"{prefix}_{k}": v for k, v in metric_dict(y, pred, prob).items()}
    return out


def run_external(args: argparse.Namespace) -> dict[str, object]:
    project_root = Path(args.project_root).resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    old_frame, old_dino = load_old_frame(project_root, args)
    external_registry = build_external_registry(project_root / args.images_root)
    external_registry.to_csv(output_dir / "external_folder_task7_registry.csv", index=False, encoding="utf-8-sig")
    _, external_dino = load_or_extract_third_features(project_root, args, external_registry, output_dir)

    y_old = old_frame["label_idx"].astype(int).to_numpy()
    folds = old_frame["fold_id"].astype(int).to_numpy()
    y_external = external_registry["label_idx"].astype(int).to_numpy()

    print(f"[data] old={len(old_frame)} external_images={len(external_registry)} dino_dim={old_dino.shape[1]}", flush=True)

    cand_rows: list[dict[str, object]] = []
    cand_old_probs: list[np.ndarray] = []
    cand_external_probs: list[np.ndarray] = []
    cand_names: list[str] = []
    for idx, spec in enumerate(candidate_specs(), start=1):
        print(f"[candidate {idx:02d}] {spec.name}", flush=True)
        old_prob, external_prob = oof_and_external_probs(spec, old_dino, y_old, folds, external_dino, args.seed + idx * 17)
        # Threshold is selected only from old OOF by the base selector later; candidate table uses 0.50 for diagnostics.
        pred050 = (old_prob >= 0.5).astype(int)
        row = metric_dict(y_old, pred050, old_prob)
        row.update({"candidate": spec.name, "threshold": 0.5})
        cand_rows.append(row)
        cand_old_probs.append(old_prob)
        cand_external_probs.append(external_prob)
        cand_names.append(spec.name)

    cand_old = np.stack(cand_old_probs, axis=1)
    cand_external = np.stack(cand_external_probs, axis=1)
    pd.DataFrame(cand_rows).sort_values(["balanced_accuracy", "accuracy", "f1"], ascending=False).to_csv(
        output_dir / "old_oof_candidate_summary.csv", index=False, encoding="utf-8-sig"
    )

    base_summary, base_old_prob, base_old_pred, base_external_prob, base_external_pred = select_base_model(
        y_old, folds, cand_old, cand_external, cand_names, args.seed
    )
    base_external_metrics = metric_dict(y_external, base_external_pred, base_external_prob)

    old_model_feat = build_prob_features(
        cand_old,
        cand_names,
        image_count=pd.to_numeric(old_frame["image_count"], errors="coerce").fillna(1).to_numpy(),
        selection_rule=old_frame["selection_rule"],
    )
    external_model_feat = build_prob_features(
        cand_external,
        cand_names,
        image_count=np.ones(len(external_registry), dtype=float),
        selection_rule=external_registry["selection_rule"],
    )
    old_model_feat["p_selected_base"] = base_old_prob
    old_model_feat["pred_selected_base"] = base_old_pred.astype(float)
    old_model_feat["selected_base_margin"] = np.abs(base_old_prob - 0.5)
    external_model_feat["p_selected_base"] = base_external_prob
    external_model_feat["pred_selected_base"] = base_external_pred.astype(float)
    external_model_feat["selected_base_margin"] = np.abs(base_external_prob - 0.5)

    # Align columns without importing the helper to keep the feature namespace explicit.
    cols = sorted(set(old_model_feat.columns).union(external_model_feat.columns))
    old_model_feat = old_model_feat.reindex(columns=cols, fill_value=0.0)
    external_model_feat = external_model_feat.reindex(columns=cols, fill_value=0.0)

    route_scores_old: dict[str, np.ndarray] = {}
    route_scores_external: dict[str, np.ndarray] = {}
    route_targets = {
        "base_wrong": (base_old_pred != y_old).astype(int),
        "base_fn": ((y_old == 1) & (base_old_pred == 0)).astype(int),
        "hard_core": old_frame["hard_core"].astype(int).to_numpy(),
    }
    for feature_kind in ["model", "dino_model"]:
        x_old, x_external = feature_matrix(feature_kind, old_model_feat, external_model_feat, old_dino, external_dino)
        for target_name, target in route_targets.items():
            if len(np.unique(target)) < 2:
                continue
            for spec in route_specs():
                name = f"{target_name}_{feature_kind}_{spec.name}"
                print(f"[route] {name}", flush=True)
                old_score, external_score = oof_and_external_probs(
                    spec, x_old, target, folds, x_external, args.seed + 3000 + len(route_scores_old)
                )
                route_scores_old[name] = old_score
                route_scores_external[name] = external_score

    correctors: dict[str, dict[str, object]] = {}
    for feature_kind in ["model", "dino", "dino_model"]:
        x_old, x_external = feature_matrix(feature_kind, old_model_feat, external_model_feat, old_dino, external_dino)
        for scope in ["non_easy", "hard_all", "hard_core", "all"]:
            train_mask = scope_mask(old_frame, scope)
            for spec in corrector_specs():
                name = f"{feature_kind}_{scope}_{spec.name}"
                print(f"[corrector] {name}", flush=True)
                old_prob, external_prob = oof_and_external_probs(
                    spec,
                    x_old,
                    y_old,
                    folds,
                    x_external,
                    args.seed + 5000 + len(correctors),
                    train_mask=train_mask,
                )
                correctors[name] = {"old_prob": old_prob, "external_prob": external_prob}

    old_rows: list[dict[str, object]] = []
    budgets = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
    corrector_thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    for route_name, route_score in route_scores_old.items():
        for corr_name, corr in correctors.items():
            corr_prob = corr["old_prob"]
            for corr_t in corrector_thresholds:
                corr_pred = (corr_prob >= corr_t).astype(int)
                for budget in budgets:
                    route_t = choose_route_threshold(route_score, budget)
                    routed = route_score >= route_t
                    final_prob = base_old_prob.copy()
                    final_pred = base_old_pred.copy()
                    final_prob[routed] = corr_prob[routed]
                    final_pred[routed] = corr_pred[routed]
                    row = metric_dict(y_old, final_pred, final_prob)
                    row.update(
                        {
                            "route_name": route_name,
                            "route_threshold": route_t,
                            "budget_pct": int(budget),
                            "corrector": corr_name,
                            "corrector_threshold": float(corr_t),
                            "routed_n": int(routed.sum()),
                            "routed_pct": float(routed.mean()),
                            "rescue_n": int(((base_old_pred != y_old) & (final_pred == y_old) & routed).sum()),
                            "hurt_n": int(((base_old_pred == y_old) & (final_pred != y_old) & routed).sum()),
                        }
                    )
                    row["net_rescue"] = int(row["rescue_n"] - row["hurt_n"])
                    old_rows.append(row)

    old_summary = pd.DataFrame(old_rows).sort_values(
        ["accuracy", "balanced_accuracy", "f1", "net_rescue", "hurt_n", "routed_n"],
        ascending=[False, False, False, False, True, True],
    )
    old_summary.to_csv(output_dir / "old_oof_64style_selection_summary.csv", index=False, encoding="utf-8-sig")
    selected = old_summary.iloc[0].to_dict()

    selected_route = str(selected["route_name"])
    selected_corr = str(selected["corrector"])
    selected_route_t = float(selected["route_threshold"])
    selected_corr_t = float(selected["corrector_threshold"])
    external_route_score = route_scores_external[selected_route]
    external_corr_prob = correctors[selected_corr]["external_prob"]
    external_corr_pred = (external_corr_prob >= selected_corr_t).astype(int)
    external_routed = external_route_score >= selected_route_t
    external_final_prob = base_external_prob.copy()
    external_final_pred = base_external_pred.copy()
    external_final_prob[external_routed] = external_corr_prob[external_routed]
    external_final_pred[external_routed] = external_corr_pred[external_routed]
    external_metrics = metric_dict(y_external, external_final_pred, external_final_prob)

    case_out = external_registry[
        [
            "case_id",
            "original_case_id",
            "source_folder",
            "task_l6_label",
            "task_l7_label",
            "label_idx",
            "label_rule",
            "strict_task7_eval",
            "image_name",
            "image_path",
        ]
    ].copy()
    case_out["base_prob_high"] = base_external_prob
    case_out["base_pred_idx"] = base_external_pred
    case_out["route_score"] = external_route_score
    case_out["routed_to_reviewer"] = external_routed.astype(int)
    case_out["reviewer_prob_high"] = external_corr_prob
    case_out["reviewer_pred_idx"] = external_corr_pred
    case_out["final_prob_high"] = external_final_prob
    case_out["final_pred_idx"] = external_final_pred
    case_out["final_correct"] = (external_final_pred == y_external).astype(int)
    case_out["base_correct"] = (base_external_pred == y_external).astype(int)
    case_out["pred_label"] = np.where(case_out["final_pred_idx"].eq(1), HIGH_LABEL, LOW_LABEL)
    case_out.to_csv(output_dir / "external_folder_case_predictions.csv", index=False, encoding="utf-8-sig")

    subtype_rows: list[dict[str, object]] = []
    for subtype, group in case_out.groupby("task_l6_label"):
        idx = group.index.to_numpy()
        row = metric_dict(y_external[idx], external_final_pred[idx], external_final_prob[idx])
        row.update({"task_l6_label": subtype, "n": int(len(group))})
        subtype_rows.append(row)
    pd.DataFrame(subtype_rows).sort_values("task_l6_label").to_csv(
        output_dir / "external_folder_metrics_by_subtype.csv", index=False, encoding="utf-8-sig"
    )

    strict = case_out[case_out["strict_task7_eval"].eq(1)].copy()
    strict_metrics = metric_dict(
        strict["label_idx"].to_numpy(int),
        strict["final_pred_idx"].to_numpy(int),
        strict["final_prob_high"].to_numpy(float),
    )
    strict_base_metrics = metric_dict(
        strict["label_idx"].to_numpy(int),
        strict["base_pred_idx"].to_numpy(int),
        strict["base_prob_high"].to_numpy(float),
    )

    dup_original_ids = (
        case_out.groupby("original_case_id")
        .filter(lambda g: len(g) > 1)
        .sort_values(["original_case_id", "image_name"])
    )
    dup_original_ids.to_csv(output_dir / "external_folder_duplicate_original_case_ids.csv", index=False, encoding="utf-8-sig")

    report = {
        "boundary": {
            "external_folder": str(project_root / args.images_root),
            "training_or_tuning_on_this_folder": False,
            "model_family": "old-data-trained image-only 64-style whole+crop external inference",
            "label_source": "filename-derived Task7 labels; nonstandard micro-nodular thymoma is marked assumed low and excluded from strict metrics",
        },
        "data_n": int(len(case_out)),
        "strict_task7_n": int(len(strict)),
        "distribution": case_out["task_l6_label"].value_counts().sort_index().to_dict(),
        "old_oof_selected_base": base_summary,
        "old_oof_selected_policy": selected,
        "external_base_all_metrics": base_external_metrics,
        "external_final_all_metrics": external_metrics,
        "external_base_strict_metrics": strict_base_metrics,
        "external_final_strict_metrics": strict_metrics,
        "routed_n": int(external_routed.sum()),
        "routed_pct": float(external_routed.mean()),
        "duplicate_original_case_id_n": int(case_out["original_case_id"].duplicated(keep=False).sum()),
    }
    (output_dir / "external_folder_64style_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    args = parse_args()
    run_external(args)


if __name__ == "__main__":
    main()
