from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MANUAL_FIELDS = [
    "exp_manual_pale_uniform",
    "exp_manual_round_smooth",
    "exp_manual_microcystic",
    "exp_manual_multinodular",
    "exp_manual_hemonec",
    "exp_manual_irregularity",
    "exp_manual_confound_target",
    "exp_manual_view_limit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Task7 anti-shortcut auxiliary labels on frozen selected images.")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--manualstruct-csv", default="")
    parser.add_argument(
        "--weak-csvs",
        default="",
        help="Comma-separated experience-label CSVs with exp_round1_* fields, used as weak anti-shortcut labels.",
    )
    parser.add_argument("--salvage-failed-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def empty_label_row(case_id: str, image_name: str, source: str) -> dict[str, str]:
    row = {"case_id": case_id, "image_name": image_name, "antishortcut_source": source}
    for field in MANUAL_FIELDS:
        row[field] = ""
    return row


def salvage_review_labels(row: pd.Series) -> dict[str, str]:
    # These labels encode visual confounders, not diagnostic truth.
    label = int(row["label_idx"])
    pred = int(row["blend_pred"])
    out = {
        "exp_manual_pale_uniform": "no",
        "exp_manual_round_smooth": "no",
        "exp_manual_microcystic": "no",
        "exp_manual_multinodular": "yes",
        "exp_manual_hemonec": "mild",
        "exp_manual_irregularity": "high",
        "exp_manual_confound_target": "B2",
        "exp_manual_view_limit": "no",
    }
    if label == 0 and pred == 1:
        # Low-risk cases being pushed up by high-risk-looking nuisance signals.
        out["exp_manual_confound_target"] = "B2"
        if str(row.get("task_l6_label", "")) in {"A", "B1"}:
            out["exp_manual_pale_uniform"] = "yes"
        if str(row.get("original_case_id", "")) in {"2423743"}:
            out["exp_manual_hemonec"] = "marked"
        if str(row.get("original_case_id", "")) in {"2517760"}:
            out["exp_manual_view_limit"] = "yes"
    elif label == 1 and pred == 0:
        # High-risk cases being pushed down by bland/low-risk-looking signals.
        out.update(
            {
                "exp_manual_pale_uniform": "yes",
                "exp_manual_round_smooth": "yes",
                "exp_manual_multinodular": "no",
                "exp_manual_hemonec": "none",
                "exp_manual_irregularity": "low",
                "exp_manual_confound_target": "A_AB",
            }
        )
        if str(row.get("original_case_id", "")) in {"2202031", "2202513"}:
            out["exp_manual_multinodular"] = "yes"
            out["exp_manual_irregularity"] = "high"
            out["exp_manual_hemonec"] = "mild"
        if str(row.get("original_case_id", "")) in {"2202302", "2504617"}:
            out["exp_manual_view_limit"] = "yes"
    return out


def weak_round_labels(row: pd.Series) -> dict[str, str]:
    color = str(row.get("exp_round1_color_pattern", "")).strip()
    surface = str(row.get("exp_round1_surface_pattern", "")).strip()
    structure = str(row.get("exp_round1_structure_pattern", "")).strip()
    overall = str(row.get("exp_round1_overall_pattern", "")).strip()
    hemo = str(row.get("exp_round1_hemorrhage_necrosis", "")).strip()
    boundary = str(row.get("exp_round1_boundary_axis", "")).strip()
    text = " ".join(
        str(row.get(col, ""))
        for col in ["exp_round2_key_discriminative_clues", "exp_round2_confounding_clues", "ai_visual_summary"]
    ).lower()

    out = {
        "exp_manual_pale_uniform": "yes" if color in {"pale_pink", "tan_pink"} or "uniform pale" in text or "淡白" in text or "均质" in text else "no",
        "exp_manual_round_smooth": "yes" if surface == "smooth_glossy" or "smooth" in text or "光滑" in text or "圆钝" in text else "no",
        "exp_manual_microcystic": "yes"
        if structure in {"microcystic", "cystic_hemorrhagic"} or surface == "fine_bubbly" or "cystic" in text or "microcystic" in text or "囊" in text or "泡" in text
        else "no",
        "exp_manual_multinodular": "yes"
        if structure == "multinodular" or surface in {"fine_nodular", "coarse_nodular"} or "nodular" in text or "结节" in text or "分叶" in text
        else "no",
        "exp_manual_hemonec": "none",
        "exp_manual_irregularity": "high"
        if surface in {"fragmented_irregular", "coarse_nodular"}
        or structure in {"necrotic_heterogeneous", "cystic_hemorrhagic"}
        or overall in {"hemorrhagic_irregular", "coarse_heterogeneous"}
        or "fragment" in text
        or "irregular" in text
        or "坏死" in text
        or "碎片" in text
        else "low",
        "exp_manual_confound_target": "none",
        "exp_manual_view_limit": "yes"
        if "limited view" in text
        or "small field" in text
        or "too small" in text
        or "视野过小" in text
        or "标本太小" in text
        or "无切面" in text
        or "仅外观" in text
        else "no",
    }
    if hemo == "none":
        out["exp_manual_hemonec"] = "none"
    elif hemo in {"mild", "moderate"}:
        out["exp_manual_hemonec"] = "mild"
    elif hemo == "marked":
        out["exp_manual_hemonec"] = "marked"

    boundary_map = {
        "A_AB": "A_AB",
        "B1_B2": "B2",
        "B2_B3": "B3",
        "B2_TC": "TC",
        "TC_B_group": "TC",
        "none": "none",
    }
    out["exp_manual_confound_target"] = boundary_map.get(boundary, "none")
    if "toward b1" in text:
        out["exp_manual_confound_target"] = "B1"
    elif "toward b2" in text:
        out["exp_manual_confound_target"] = "B2"
    elif "toward b3" in text:
        out["exp_manual_confound_target"] = "B3"
    elif "toward tc" in text:
        out["exp_manual_confound_target"] = "TC"
    elif "toward ab" in text or "向ab" in text:
        out["exp_manual_confound_target"] = "A_AB"
    return out


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.registry_csv, dtype=str)
    manual = pd.read_csv(args.manualstruct_csv, dtype=str).fillna("") if args.manualstruct_csv else pd.DataFrame()
    salvage = pd.read_csv(args.salvage_failed_csv, dtype=str).fillna("")

    registry["selected_image_basename"] = registry["training_image_path"].map(lambda p: Path(str(p)).name)
    by_original = registry.set_index("original_case_id", drop=False)

    rows: dict[tuple[str, str], dict[str, str]] = {}
    weak_paths = [Path(item.strip()) for item in args.weak_csvs.split(",") if item.strip()]
    for weak_path in weak_paths:
        weak_df = pd.read_csv(weak_path, dtype=str).fillna("")
        for _, src in weak_df.iterrows():
            original = str(src["case_id"])
            if original not in by_original.index:
                continue
            reg = by_original.loc[original]
            if isinstance(reg, pd.DataFrame):
                reg = reg.iloc[0]
            if str(src["image_name"]) != str(reg["selected_original_image_name"]):
                continue
            out = empty_label_row(str(reg["case_id"]), str(reg["selected_image_basename"]), "weak_round")
            out.update(weak_round_labels(src))
            rows[(out["case_id"], out["image_name"])] = out

    for _, src in manual.iterrows():
        original = str(src["case_id"])
        if original not in by_original.index:
            continue
        reg = by_original.loc[original]
        if isinstance(reg, pd.DataFrame):
            reg = reg.iloc[0]
        if str(src["image_name"]) != str(reg["selected_original_image_name"]):
            continue
        out = empty_label_row(str(reg["case_id"]), str(reg["selected_image_basename"]), "manualstruct")
        for field in MANUAL_FIELDS:
            out[field] = str(src.get(field, ""))
        key = (out["case_id"], out["image_name"])
        if key in rows:
            merged_source = rows[key]["antishortcut_source"]
            rows[key].update(out)
            rows[key]["antishortcut_source"] = f"{merged_source}+manualstruct"
        else:
            rows[key] = out

    for _, src in salvage.iterrows():
        case_id = str(src["case_id"])
        reg = registry.loc[registry["case_id"] == case_id]
        if reg.empty:
            continue
        reg_row = reg.iloc[0]
        out = empty_label_row(case_id, str(reg_row["selected_image_basename"]), "salvage_review")
        out.update(salvage_review_labels(src))
        key = (out["case_id"], out["image_name"])
        if key in rows:
            merged_source = rows[key]["antishortcut_source"]
            rows[key].update(out)
            rows[key]["antishortcut_source"] = f"{merged_source}+salvage_review"
        else:
            rows[key] = out

    out_df = pd.DataFrame(rows.values()).sort_values(["case_id", "image_name"]).reset_index(drop=True)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {out_path} rows={len(out_df)}")
    print(out_df["antishortcut_source"].value_counts().to_string())
    for field in MANUAL_FIELDS:
        print(f"\n{field}")
        print(out_df[field].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
