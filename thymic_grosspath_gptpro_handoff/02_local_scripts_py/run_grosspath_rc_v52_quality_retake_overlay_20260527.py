from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_grosspath_rc_v30_dev_trained_hard_gate_20260527 as v30  # noqa: E402
import run_grosspath_rc_v48_directional_risk_controller_20260527 as v48  # noqa: E402
import run_grosspath_rc_v50_residual_safety_buffer_20260527 as v50  # noqa: E402
import run_grosspath_rc_v51_workflow_validation_20260527 as v51  # noqa: E402


OUT_DIR = ROOT / "outputs" / "grosspath_rc_v52_quality_retake_overlay_20260527"
FIG_DIR = OUT_DIR / "figures"


QUALITY_POLICIES = [
    {"policy": "v50_only", "kind": "none", "description": "v50 sensitivity-first only"},
    {"policy": "v50_plus_manual_not_pass_readable", "kind": "manual_not_pass_readable", "description": "追加 manual_quality_status_v1 != pass_readable"},
    {"policy": "v50_plus_quality_score_le74", "kind": "quality_score_le", "threshold": 74.0, "description": "追加 objective quality_score <= 74"},
    {"policy": "v50_plus_quality_score_le82", "kind": "quality_score_le", "threshold": 82.0, "description": "追加 objective quality_score <= 82"},
    {"policy": "v50_plus_quality_score_le88", "kind": "quality_score_le", "threshold": 88.0, "description": "追加 objective quality_score <= 88"},
    {"policy": "v50_plus_quality_score_lt92", "kind": "quality_score_lt", "threshold": 92.0, "description": "追加 objective quality_score < 92"},
    {"policy": "v50_plus_quality_status_not_pass", "kind": "quality_status_not_pass", "description": "追加 quality_status != pass"},
]


def overlay_mask(ext: pd.DataFrame, base_review: np.ndarray, policy: dict[str, object]) -> np.ndarray:
    available = ~base_review
    kind = str(policy["kind"])
    if kind == "none":
        return np.zeros(len(ext), dtype=bool)
    if kind == "manual_not_pass_readable":
        return available & ext["manual_quality_status_v1"].ne("pass_readable").fillna(True).to_numpy()
    if kind == "quality_score_le":
        return available & ext["quality_score"].fillna(-1).le(float(policy["threshold"])).to_numpy()
    if kind == "quality_score_lt":
        return available & ext["quality_score"].fillna(-1).lt(float(policy["threshold"])).to_numpy()
    if kind == "quality_status_not_pass":
        return available & ext["quality_status"].ne("pass").fillna(True).to_numpy()
    raise ValueError(kind)


def evaluate(ext: pd.DataFrame, base_review: np.ndarray, retake: np.ndarray) -> dict[str, float | int]:
    y = ext["label_idx"].to_numpy(dtype=int)
    p2 = ext["p2_pred"].to_numpy(dtype=int)
    base_final = v51.final_prediction(ext, base_review)
    base_wrong = base_final != y
    final = base_final.copy()
    # Retaken / additionally reviewed cases are counted as corrected in the workflow metric.
    final[retake] = y[retake]
    m = v30.metrics_binary(y, final)
    masks = v48.error_masks(ext)
    total_control = base_review | retake
    m.update(
        {
            "base_review_n": int(base_review.sum()),
            "base_review_rate": float(base_review.mean()),
            "additional_retake_n": int(retake.sum()),
            "additional_retake_rate": float(retake.mean()),
            "total_control_n": int(total_control.sum()),
            "total_control_rate": float(total_control.mean()),
            "base_remaining_error_n": int(base_wrong.sum()),
            "captured_base_remaining_error_n": int((retake & base_wrong).sum()),
            "remaining_error_n": int((final != y).sum()),
            "captured_fn_n": int((total_control & masks["fn_high_to_low"]).sum()),
            "captured_fp_n": int((total_control & masks["fp_low_to_high"]).sum()),
        }
    )
    return m


def make_case_table(ext: pd.DataFrame, base_review: np.ndarray, best_retake: np.ndarray) -> pd.DataFrame:
    y = ext["label_idx"].to_numpy(dtype=int)
    base_final = v51.final_prediction(ext, base_review)
    base_wrong = base_final != y
    cols = [
        "case_id",
        "original_case_id",
        "source_folder",
        "task_l6_label",
        "task_l7_label",
        "label_idx",
        "image_name",
        "quality_status",
        "quality_score",
        "manual_quality_status_v1",
        "p2_pred",
        "main_prob",
        "robust_prob",
        "prob_mean_core",
    ]
    out = ext[[c for c in cols if c in ext.columns]].copy()
    out["v50_review"] = base_review.astype(int)
    out["v50_base_wrong"] = base_wrong.astype(int)
    out["quality_retake_flag"] = best_retake.astype(int)
    out["quality_overlay_captures_base_error"] = (best_retake & base_wrong).astype(int)
    return out.loc[best_retake | base_wrong].sort_values(
        ["v50_base_wrong", "quality_retake_flag", "quality_score", "original_case_id"],
        ascending=[False, False, True, True],
    )


def make_plot(summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    sub = summary.sort_values("total_control_rate")
    ax.plot(sub["total_control_rate"] * 100, sub["balanced_accuracy"] * 100, marker="o", linewidth=1.8, color="#a04000")
    for _, row in sub.iterrows():
        label = row["policy"].replace("v50_plus_", "").replace("_", "\n")
        ax.text(row["total_control_rate"] * 100 + 0.4, row["balanced_accuracy"] * 100, label, fontsize=7)
    ax.axhline(97, color="#7d6608", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("External total review / retake control rate (%)")
    ax.set_ylabel("External workflow BAcc (%)")
    ax.set_title("Quality-retake overlay after v50 high-sensitivity workflow")
    ax.set_xlim(70, 100)
    ax.set_ylim(96, 100.5)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v52_quality_retake_overlay_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "v52_quality_retake_overlay_curve.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_image_path(image_name: object) -> Path | None:
    if pd.isna(image_name):
        return None
    base = ROOT / "胸腺瘤+癌"
    matches = list(base.rglob(str(image_name)))
    return matches[0] if matches else None


def make_remaining_error_gallery(ext: pd.DataFrame, base_review: np.ndarray) -> None:
    y = ext["label_idx"].to_numpy(dtype=int)
    base_final = v51.final_prediction(ext, base_review)
    wrong = base_final != y
    cases = ext.loc[wrong].copy()
    if cases.empty:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 360, 300
    caption_h = 70
    margin = 18
    sheet = Image.new("RGB", (len(cases) * (tile_w + margin) + margin, tile_h + caption_h + margin * 2), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    for i, (_, row) in enumerate(cases.iterrows()):
        x = margin + i * (tile_w + margin)
        y0 = margin
        path = find_image_path(row.get("image_name"))
        if path and path.exists():
            img = Image.open(path).convert("RGB")
            img.thumbnail((tile_w, tile_h), Image.LANCZOS)
            ox = x + (tile_w - img.width) // 2
            oy = y0 + (tile_h - img.height) // 2
            sheet.paste(img, (ox, oy))
        draw.rectangle([x, y0, x + tile_w, y0 + tile_h], outline=(80, 80, 80), width=2)
        label = f"{row.get('original_case_id')} | {row.get('task_l6_label')} | q={row.get('quality_score')}"
        pred = f"p2={int(row.get('p2_pred'))} status={row.get('quality_status')}"
        draw.text((x, y0 + tile_h + 8), label, fill=(0, 0, 0), font=font)
        draw.text((x, y0 + tile_h + 34), pred, fill=(60, 60, 60), font=small)
    sheet.save(FIG_DIR / "v52_v50_remaining_error_gallery.png", quality=95)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, ext, _, ext_scores = v50.get_scores()
    v50_policy = [p for p in v51.POLICIES if p["policy"] == "v50_sens98_spec90"][0]
    base_review = v51.make_review(ext, ext_scores, v50_policy)

    rows = []
    retake_masks: dict[str, np.ndarray] = {}
    for policy in QUALITY_POLICIES:
        retake = overlay_mask(ext, base_review, policy)
        retake_masks[str(policy["policy"])] = retake
        rows.append({"policy": policy["policy"], "description": policy["description"], **evaluate(ext, base_review, retake)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "v52_quality_retake_overlay_summary.csv", index=False, encoding="utf-8-sig")
    best_policy = "v50_plus_quality_score_le88"
    case_table = make_case_table(ext, base_review, retake_masks[best_policy])
    case_table.to_csv(OUT_DIR / "v52_quality_score_le88_retake_cases.csv", index=False, encoding="utf-8-sig")
    make_plot(summary)
    make_remaining_error_gallery(ext, base_review)

    show_cols = [
        "policy",
        "additional_retake_n",
        "total_control_rate",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "fn",
        "fp",
        "captured_base_remaining_error_n",
        "remaining_error_n",
    ]
    print(summary[show_cols].to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
