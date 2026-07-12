from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an existing blinded ROI template into independent reader forms."
    )
    parser.add_argument("--packet-root", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    packet_root = Path(args.packet_root)
    blinded_dir = packet_root / "blinded_packet"
    template_path = blinded_dir / "BLINDED_ANNOTATION_TEMPLATE.csv"
    manifest_path = blinded_dir / "BLINDED_IMAGE_MANIFEST.csv"
    template = pd.read_csv(
        template_path, dtype={"oracle_id": str, "reader_id": str}, encoding="utf-8-sig"
    )
    manifest = pd.read_csv(manifest_path, dtype={"oracle_id": str}, encoding="utf-8-sig")
    expected_ids = set(manifest["oracle_id"])
    if len(expected_ids) != 120 or manifest["oracle_id"].duplicated().any():
        raise ValueError("Blinded image manifest must contain 120 unique oracle IDs")
    if set(template["oracle_id"]) != expected_ids:
        raise ValueError("Combined annotation template does not match the image manifest")

    output_paths = []
    for reader_id in ("reader_1", "reader_2"):
        output_path = blinded_dir / f"{reader_id.upper()}_ANNOTATION.csv"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing reader form: {output_path}")
        form = template[template["reader_id"].astype(str) == reader_id].copy()
        if len(form) != 120 or set(form["oracle_id"]) != expected_ids:
            raise ValueError(f"Template rows are invalid for {reader_id}")
        form.to_csv(output_path, index=False, encoding="utf-8-sig")
        output_paths.append(output_path)

    instructions_path = blinded_dir / "READER_FORM_WORKFLOW.md"
    if instructions_path.exists():
        raise FileExistsError(f"Refusing to overwrite workflow: {instructions_path}")
    instructions_path.write_text(
        """# Independent Reader Workflow

1. Reader 1 receives `READER_1_ANNOTATION.csv`; reader 2 receives `READER_2_ANNOTATION.csv`.
2. Both readers use the common read-only `blinded_images` directory and must not see the other form.
3. Allowed categorical values:
   - `annotation_status`: `complete`
   - `image_quality`: `adequate`, `limited`, `nondiagnostic`
   - `image_sufficient_for_low_high_judgment`: `yes`, `no`, `uncertain`
   - `physician_risk_judgment`: `low`, `high`, `indeterminate`
   - `no_visually_diagnostic_roi`: `yes`, `no`
   - `recommended_additional_view`: `none`, `whole`, `cut_surface_closeup`, `capsule_interface_closeup`, `multiple_views`, `other`
4. Confidence is an integer from 1 to 5.
5. Every coordinate box is normalized to `[0,1]` and must satisfy `x1<x2`, `y1<y2`. Fill all four coordinates or leave all four blank.
6. Specimen extent is mandatory. If `no_visually_diagnostic_roi=no`, ROI1 coordinates and reason are mandatory. If `yes`, ROI1-3 must remain blank.
7. Do not reconcile disagreements before both forms have passed validation and their hashes have been locked.
""",
        encoding="utf-8",
    )
    output_paths.append(instructions_path)
    hash_path = blinded_dir / "READER_FORMS_SHA256.txt"
    hash_path.write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in output_paths) + "\n",
        encoding="utf-8",
    )
    print(f"[done] created {len(output_paths) - 1} reader forms in {blinded_dir}")


if __name__ == "__main__":
    main()
