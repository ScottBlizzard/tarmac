from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate cached Task7 old/third feature variants.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--old-output-dir", required=True)
    parser.add_argument("--third-output-dir", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="name|old_feature_dir|third_external_dir. Can be repeated.",
    )
    parser.add_argument("--l2-per-block", action="store_true")
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260521)
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"case_id": str})


def _load_features(table_path: Path, npy_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    table = _read_table(table_path)
    features = np.load(npy_path).astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    return table, features


def _align_features(order: pd.Series, table: pd.DataFrame, features: np.ndarray, l2: bool) -> np.ndarray:
    idx = order.to_frame("case_id").merge(table[["case_id", "feature_idx"]], on="case_id", how="left")["feature_idx"]
    if idx.isna().any():
        missing = order[idx.isna()].head(10).tolist()
        raise KeyError(f"Missing feature rows: {missing}")
    block = features[idx.astype(int).to_numpy()].astype(np.float32)
    if l2:
        norm = np.linalg.norm(block, axis=1, keepdims=True)
        block = block / np.maximum(norm, 1e-6)
    return block


def _parse_variant(item: str) -> tuple[str, Path, Path]:
    parts = item.split("|")
    if len(parts) != 3:
        raise ValueError("--variant must be name|old_feature_dir|third_external_dir")
    return parts[0], Path(parts[1]), Path(parts[2])


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    old_output = root / args.old_output_dir
    third_output = root / args.third_output_dir
    old_output.mkdir(parents=True, exist_ok=True)
    third_output.mkdir(parents=True, exist_ok=True)

    old_blocks: list[np.ndarray] = []
    third_blocks: list[np.ndarray] = []
    old_order: pd.Series | None = None
    third_order: pd.Series | None = None
    manifest_rows: list[dict[str, object]] = []

    for raw in args.variant:
        name, old_dir_rel, third_dir_rel = _parse_variant(raw)
        old_dir = root / old_dir_rel
        third_dir = root / third_dir_rel
        old_table, old_feat = _load_features(old_dir / "case_dino_concat_feature_table.csv", old_dir / "case_dino_concat_features.npy")
        third_table, third_feat = _load_features(
            third_dir / "third_batch_dino_concat_feature_table.csv",
            third_dir / "third_batch_dino_concat_features.npy",
        )
        if old_order is None:
            old_order = old_table["case_id"].astype(str).reset_index(drop=True)
        if third_order is None:
            third_order = third_table["case_id"].astype(str).reset_index(drop=True)

        old_block = _align_features(old_order, old_table, old_feat, args.l2_per_block)
        third_block = _align_features(third_order, third_table, third_feat, args.l2_per_block)
        old_blocks.append(old_block)
        third_blocks.append(third_block)
        manifest_rows.append(
            {
                "variant": name,
                "old_dir": str(old_dir_rel),
                "third_dir": str(third_dir_rel),
                "old_dim": int(old_block.shape[1]),
                "third_dim": int(third_block.shape[1]),
            }
        )
        print(f"[variant] {name} old={old_block.shape} third={third_block.shape}", flush=True)

    if old_order is None or third_order is None:
        raise RuntimeError("No variants loaded.")

    old_concat = np.concatenate(old_blocks, axis=1).astype(np.float32)
    third_concat = np.concatenate(third_blocks, axis=1).astype(np.float32)
    if args.pca_components > 0:
        n_components = min(args.pca_components, old_concat.shape[0] - 1, old_concat.shape[1])
        scaler = StandardScaler()
        old_scaled = scaler.fit_transform(old_concat)
        third_scaled = scaler.transform(third_concat)
        pca = PCA(n_components=n_components, random_state=args.seed, svd_solver="randomized")
        old_concat = pca.fit_transform(old_scaled).astype(np.float32)
        third_concat = pca.transform(third_scaled).astype(np.float32)
        manifest_rows.append(
            {
                "variant": "pca_after_concat",
                "old_dir": "",
                "third_dir": "",
                "old_dim": int(old_concat.shape[1]),
                "third_dim": int(third_concat.shape[1]),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            }
        )
        print(
            f"[pca] n_components={n_components} explained={pca.explained_variance_ratio_.sum():.4f}",
            flush=True,
        )
    old_table_out = pd.DataFrame({"case_id": old_order, "feature_idx": np.arange(len(old_order), dtype=int)})
    third_table_out = pd.DataFrame({"case_id": third_order, "feature_idx": np.arange(len(third_order), dtype=int)})

    old_table_out.to_csv(old_output / "case_dino_concat_feature_table.csv", index=False, encoding="utf-8-sig")
    np.save(old_output / "case_dino_concat_features.npy", old_concat)
    third_table_out.to_csv(third_output / "third_batch_dino_concat_feature_table.csv", index=False, encoding="utf-8-sig")
    np.save(third_output / "third_batch_dino_concat_features.npy", third_concat)
    pd.DataFrame(manifest_rows).to_csv(third_output / "feature_concat_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"[done] old={old_concat.shape} third={third_concat.shape}", flush=True)


if __name__ == "__main__":
    main()
