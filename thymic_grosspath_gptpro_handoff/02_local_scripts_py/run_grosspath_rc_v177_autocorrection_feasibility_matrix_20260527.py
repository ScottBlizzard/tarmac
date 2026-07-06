from __future__ import annotations

import shutil

from run_grosspath_rc_v143_image_feature_error_router_20260527 import ROOT
import run_grosspath_rc_v176_autocorrection_feasibility_matrix_20260527 as v176


v176.OUT_DIR = ROOT / "outputs" / "grosspath_rc_v177_autocorrection_feasibility_matrix_20260527"


if __name__ == "__main__":
    v176.main()
    rename_pairs = [
        ("v176_autocorrection_feasibility_matrix.csv", "v177_autocorrection_feasibility_matrix.csv"),
        ("v176_autocorrection_feasibility_matrix.md", "v177_autocorrection_feasibility_matrix.md"),
        ("v176_run_report.json", "v177_run_report.json"),
    ]
    for old, new in rename_pairs:
        src = v176.OUT_DIR / old
        if src.exists():
            shutil.copy2(src, v176.OUT_DIR / new)
    print(f"[v177] wrote {v176.OUT_DIR}")
