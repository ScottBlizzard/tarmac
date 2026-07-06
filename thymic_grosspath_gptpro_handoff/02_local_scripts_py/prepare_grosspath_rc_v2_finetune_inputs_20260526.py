from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "grosspath_rc_v2_20260526"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "batch1_batch2_task567_20260514"
    / "task7_adaptation_runs"
    / "75_grosspath_rc_v2_domain_robust_finetune_inputs_20260526"
)

WHO_BY_L6 = {
    "A": "thymoma_A",
    "AB": "thymoma_AB",
    "B1": "thymoma_B1",
    "B2": "thymoma_B2",
    "B3": "thymoma_B3",
    "TC": "thymic_carcinoma",
}


def read_case_set(filename: str) -> set[str]:
    path = V2_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return set(pd.read_csv(path, dtype={"case_id": str})["case_id"].astype(str))


def first_non_empty(*values: object) -> str:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def sample_weight(row: pd.Series) -> float:
    weight = 1.0
    if int(row["v2_hard_error"]) == 1:
        weight += 1.35
    if int(row["v2_domain_focus"]) == 1:
        weight += 0.60
    if str(row["domain"]) == "third":
        weight += 0.45
    if str(row["task_l7_label"]) == "high_risk_group":
        weight += 0.30
    if str(row["task_l6_label"]) == "B2":
        weight += 0.25
    if int(row["v2_safe_core"]) == 1:
        weight = max(0.90, weight - 0.15)
    return round(min(weight, 3.30), 3)


def sample_weight_soft(row: pd.Series) -> float:
    weight = 1.0
    if int(row["v2_hard_error"]) == 1:
        weight += 0.55
    if int(row["v2_domain_focus"]) == 1:
        weight += 0.22
    if str(row["domain"]) == "third":
        weight += 0.18
    if str(row["task_l7_label"]) == "high_risk_group":
        weight += 0.12
    if str(row["task_l6_label"]) == "B2":
        weight += 0.08
    if int(row["v2_safe_core"]) == 1:
        weight = max(0.95, weight - 0.05)
    return round(min(weight, 2.10), 3)


def build_inputs(output_dir: Path = DEFAULT_OUTPUT) -> None:
    diag = pd.read_csv(V2_DIR / "v2_development_diagnostic_table.csv", dtype={"case_id": str})
    safe_core = read_case_set("v2_training_safe_core_cases.csv")
    hard_error = read_case_set("v2_training_hard_error_cases.csv")
    domain_focus = read_case_set("v2_training_domain_focus_cases.csv")

    rows: list[dict[str, object]] = []
    for row in diag.to_dict(orient="records"):
        case_id = str(row["case_id"])
        domain = str(row["domain"])
        task_l6 = str(row["task_l6_label"])
        task_l7 = str(row["task_l7_label"])
        image_path = first_non_empty(row.get("selected_original_image_path"), row.get("image_path"))
        image_name = first_non_empty(row.get("selected_original_image_name"), row.get("image_name"), Path(image_path).name)
        source_folder = first_non_empty(row.get("source_case_folder"), row.get("source_folder"), domain)
        who_type = first_non_empty(row.get("who_type_raw"), WHO_BY_L6.get(task_l6, ""))
        source_dataset = "third_batch_306_20260521" if domain == "third" else "old_batch1_batch2_20260514"

        item = {
            "case_id": case_id,
            "patient_id": case_id,
            "original_case_id": str(row.get("original_case_id", case_id)),
            "include_main_study": 1,
            "source_case_folder": source_folder,
            "source_folder": source_folder,
            "source_dataset": source_dataset,
            "image_filenames": image_name,
            "selected_original_image_name": image_name,
            "selected_original_image_path": image_path,
            "training_image_path": image_path,
            "who_type_raw": who_type,
            "split_stratification_class": f"{domain}_{task_l6}",
            "task_l1_label": task_l7,
            "task_l2_label": task_l7,
            "task_l3_label": task_l7,
            "task_l4_label": task_l7,
            "task_l5_label": task_l6,
            "task_l6_label": task_l6,
            "task_l7_label": task_l7,
            "domain": domain,
            "third_split": str(row.get("third_split", "")),
            "view_type_final": str(row.get("view_type_final", "")),
            "quality_group": str(row.get("quality_group", "")),
            "multi_image_group": str(row.get("multi_image_group", "")),
            "v2_safe_core": 1 if case_id in safe_core else 0,
            "v2_hard_error": 1 if case_id in hard_error else 0,
            "v2_domain_focus": 1 if case_id in domain_focus else 0,
            "v2_main_wrong": int(float(row.get("main_wrong", 0) or 0)),
            "v2_error_type": str(row.get("error_type", "")),
        }
        item["v2_train_role"] = (
            "hard_error"
            if item["v2_hard_error"]
            else "domain_focus"
            if item["v2_domain_focus"]
            else "safe_core"
            if item["v2_safe_core"]
            else "standard"
        )
        rows.append(item)

    registry = pd.DataFrame(rows)
    registry["v2_sample_weight"] = registry.apply(sample_weight, axis=1)
    registry["v2_sample_weight_soft"] = registry.apply(sample_weight_soft, axis=1)
    split = diag[["case_id", "fold_id"]].rename(columns={"fold_id": "master_fold_id"}).copy()
    split["case_id"] = split["case_id"].astype(str)
    split["master_fold_id"] = split["master_fold_id"].astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_dir / "registry.csv", index=False, encoding="utf-8-sig")
    split.to_csv(output_dir / "split.csv", index=False, encoding="utf-8-sig")

    summary_rows = [
        {"name": "all", "n": len(registry), "mean_weight": registry["v2_sample_weight"].mean()},
        {"name": "old", "n": int((registry["domain"] == "old").sum()), "mean_weight": registry.loc[registry["domain"] == "old", "v2_sample_weight"].mean()},
        {"name": "third", "n": int((registry["domain"] == "third").sum()), "mean_weight": registry.loc[registry["domain"] == "third", "v2_sample_weight"].mean()},
        {"name": "safe_core", "n": int(registry["v2_safe_core"].sum()), "mean_weight": registry.loc[registry["v2_safe_core"] == 1, "v2_sample_weight"].mean()},
        {"name": "hard_error", "n": int(registry["v2_hard_error"].sum()), "mean_weight": registry.loc[registry["v2_hard_error"] == 1, "v2_sample_weight"].mean()},
        {"name": "domain_focus", "n": int(registry["v2_domain_focus"].sum()), "mean_weight": registry.loc[registry["v2_domain_focus"] == 1, "v2_sample_weight"].mean()},
    ]
    summary = pd.DataFrame(summary_rows)
    summary["mean_weight_soft"] = [
        registry["v2_sample_weight_soft"].mean(),
        registry.loc[registry["domain"] == "old", "v2_sample_weight_soft"].mean(),
        registry.loc[registry["domain"] == "third", "v2_sample_weight_soft"].mean(),
        registry.loc[registry["v2_safe_core"] == 1, "v2_sample_weight_soft"].mean(),
        registry.loc[registry["v2_hard_error"] == 1, "v2_sample_weight_soft"].mean(),
        registry.loc[registry["v2_domain_focus"] == 1, "v2_sample_weight_soft"].mean(),
    ]
    summary.to_csv(output_dir / "input_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[done] wrote {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    build_inputs()
