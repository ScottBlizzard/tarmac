from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CASE_PATTERN = re.compile(r"(?<!\d)(\d{7})(?!\d)")
TASK7_SUBTYPES = {"A", "AB", "B1", "B2", "B3", "TC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blinded H10 visual audit set.")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--role-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def case_id_from_path(path: Path) -> str | None:
    for value in (path.name, path.parent.name, str(path)):
        match = CASE_PATTERN.search(value)
        if match:
            return match.group(1)
    return None


def directory_subtype_hint(path: Path) -> str:
    value = str(path.parent)
    if "AB" in value:
        return "AB"
    for subtype in ("B1", "B2", "B3"):
        if subtype in value:
            return subtype
    if "A型" in value:
        return "A"
    if "胸腺癌" in value:
        return "TC"
    return "OTHER"


def image_record(path: Path, image_root: Path) -> dict[str, object] | None:
    case_id = case_id_from_path(path)
    if case_id is None:
        return None
    with Image.open(path) as image:
        width, height = image.size
    return {
        "original_case_id": case_id,
        "image_path": str(path.resolve()),
        "relative_image_path": str(path.relative_to(image_root)),
        "width": int(width),
        "height": int(height),
        "pixel_area": int(width * height),
        "file_bytes": int(path.stat().st_size),
        "directory_subtype_hint": directory_subtype_hint(path),
    }


def inventory_images(image_root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            record = image_record(path, image_root)
            if record is not None:
                records.append(record)
    if not records:
        raise ValueError(f"No case images found under {image_root}")
    inventory = pd.DataFrame(records)
    inventory["representative_rank"] = inventory.groupby("original_case_id")[
        ["pixel_area", "file_bytes"]
    ].rank(method="first", ascending=False)["pixel_area"]
    return inventory


def normalize_case_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = CASE_PATTERN.search(str(value).split(".")[0])
    return match.group(1) if match else None


def load_workbook(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=object)
    required = {"病理号", "病理诊断", "肉眼所见"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Workbook missing columns: {sorted(required - set(frame.columns))}")
    frame = frame[["病理号", "病理诊断", "肉眼所见"]].copy()
    frame["original_case_id"] = frame["病理号"].map(normalize_case_id)
    frame = frame.dropna(subset=["original_case_id"])
    frame = frame.drop_duplicates("original_case_id", keep="first")
    return frame[["original_case_id", "病理诊断", "肉眼所见"]]


def deterministic_order(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def balanced_take(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    working = frame.copy()
    working["_order"] = working["original_case_id"].map(
        lambda case_id: deterministic_order(str(case_id), seed)
    )
    groups = {
        key: group.sort_values("_order").to_dict("records")
        for key, group in working.groupby(
            ["task_l6_label", "diagnostic_role"], sort=True, dropna=False
        )
    }
    selected = []
    keys = list(groups)
    while keys and len(selected) < count:
        next_keys = []
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop(0))
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return pd.DataFrame(selected).drop(columns=["_order"], errors="ignore")


def select_blinded_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    available = frame.dropna(subset=["image_path", "diagnostic_role"]).copy()
    available = available[available["task_l6_label"].isin(TASK7_SUBTYPES)]
    available["difficulty_tier"] = np.select(
        [
            available["model_correct_count"].eq(3),
            available["model_correct_count"].isin([1, 2]),
            available["model_correct_count"].eq(0),
        ],
        ["easy", "boundary", "hard"],
        default="invalid",
    )
    if (available["difficulty_tier"] == "invalid").any():
        raise ValueError("Unexpected model-correct count")

    target = {tier: sample_size // 3 for tier in ("easy", "boundary", "hard")}
    target["easy"] += sample_size - sum(target.values())
    pieces = []
    selected_ids: set[str] = set()
    for index, tier in enumerate(("hard", "boundary", "easy")):
        subset = available[available["difficulty_tier"] == tier]
        chosen = balanced_take(subset, min(target[tier], len(subset)), seed + index)
        pieces.append(chosen)
        selected_ids.update(chosen["original_case_id"].astype(str))

    selected = pd.concat(pieces, ignore_index=True) if pieces else available.iloc[0:0]
    remaining_n = min(sample_size, len(available)) - len(selected)
    if remaining_n > 0:
        remaining = available[~available["original_case_id"].astype(str).isin(selected_ids)]
        selected = pd.concat(
            [selected, balanced_take(remaining, remaining_n, seed + 100)],
            ignore_index=True,
        )
    selected["_blind_order"] = selected["original_case_id"].map(
        lambda case_id: deterministic_order(str(case_id), seed + 1000)
    )
    selected = selected.sort_values("_blind_order").reset_index(drop=True)
    selected["audit_code"] = [f"V{index:03d}" for index in range(1, len(selected) + 1)]
    return selected.drop(columns=["_blind_order"])


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_contact_sheets(sample: pd.DataFrame, output_dir: Path) -> None:
    columns, rows = 3, 3
    cell_width, cell_height = 700, 560
    image_height = 510
    font = load_font(28)
    per_page = columns * rows
    sheets_dir = output_dir / "blind_contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for page_index in range(math.ceil(len(sample) / per_page)):
        page = sample.iloc[page_index * per_page : (page_index + 1) * per_page]
        canvas = Image.new(
            "RGB", (columns * cell_width, rows * cell_height), color=(255, 255, 255)
        )
        draw = ImageDraw.Draw(canvas)
        for position, (_, record) in enumerate(page.iterrows()):
            column = position % columns
            row = position // columns
            left = column * cell_width
            top = row * cell_height
            with Image.open(record["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((cell_width - 20, image_height - 20), Image.Resampling.LANCZOS)
            x = left + (cell_width - image.width) // 2
            y = top + (image_height - image.height) // 2
            canvas.paste(image, (x, y))
            draw.rectangle(
                [left, top, left + cell_width - 1, top + cell_height - 1],
                outline=(120, 120, 120),
                width=2,
            )
            draw.text((left + 12, top + image_height + 5), record["audit_code"], fill=(0, 0, 0), font=font)
        canvas.save(sheets_dir / f"blind_page_{page_index + 1:02d}.jpg", quality=94)


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = inventory_images(image_root)
    representative = (
        inventory.sort_values(
            ["original_case_id", "pixel_area", "file_bytes"],
            ascending=[True, False, False],
        )
        .drop_duplicates("original_case_id")
        .drop(columns=["representative_rank"], errors="ignore")
    )
    workbook = load_workbook(Path(args.workbook))
    matched = representative.merge(workbook, on="original_case_id", how="left", validate="one_to_one")

    inventory.to_csv(output_dir / "all_local_images.csv", index=False, encoding="utf-8-sig")
    matched.to_csv(output_dir / "representative_local_cases.csv", index=False, encoding="utf-8-sig")
    (output_dir / "local_case_ids.txt").write_text(
        "\n".join(matched["original_case_id"].astype(str).sort_values()) + "\n",
        encoding="utf-8",
    )
    print(
        {
            "image_files": int(len(inventory)),
            "unique_image_cases": int(len(representative)),
            "workbook_matched": int(matched["肉眼所见"].notna().sum()),
        }
    )

    if not args.role_csv:
        return
    roles = pd.read_csv(
        args.role_csv,
        dtype={"case_id": str, "original_case_id": str},
        encoding="utf-8-sig",
    )
    if roles["original_case_id"].duplicated().any():
        raise ValueError("Role CSV contains duplicate original case IDs")
    combined = roles.merge(matched, on="original_case_id", how="inner", validate="one_to_one")
    combined.to_csv(output_dir / "matched_h10_cases_unblinded.csv", index=False, encoding="utf-8-sig")
    sample = select_blinded_sample(combined, args.sample_size, args.seed)
    sample.to_csv(output_dir / "blind_key_do_not_open.csv", index=False, encoding="utf-8-sig")
    observation = sample[["audit_code"]].copy()
    for column in (
        "image_evidence_sufficiency",
        "specimen_presentation",
        "cut_surface_exposed",
        "capsule_or_boundary_visible",
        "surface_or_cut_heterogeneity",
        "necrosis_or_hemorrhage_visible",
        "cystic_change_visible",
        "adjacent_fat_or_tissue_visible",
        "photo_quality",
        "visual_risk_impression",
        "visual_confidence",
        "blind_notes",
    ):
        observation[column] = ""
    observation.to_csv(
        output_dir / "blind_observation_template.csv", index=False, encoding="utf-8-sig"
    )
    render_contact_sheets(sample, output_dir)
    print(
        sample.groupby(["difficulty_tier", "task_l6_label"], sort=True)
        .size()
        .unstack(fill_value=0)
        .to_string()
    )


if __name__ == "__main__":
    main()
