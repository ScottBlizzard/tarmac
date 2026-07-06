from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average two aligned Task7 feature caches.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--old-a-dir", required=True)
    parser.add_argument("--old-b-dir", required=True)
    parser.add_argument("--third-a-dir", required=True)
    parser.add_argument("--third-b-dir", required=True)
    parser.add_argument("--old-output-dir", required=True)
    parser.add_argument("--third-output-dir", required=True)
    return parser.parse_args()


def load_old(root: Path, feature_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / feature_dir
    table = pd.read_csv(d / "case_dino_concat_feature_table.csv", dtype={"case_id": str})
    feat = np.load(d / "case_dino_concat_features.npy").astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    return table, feat


def load_third(root: Path, feature_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = root / feature_dir
    table = pd.read_csv(d / "third_batch_dino_concat_feature_table.csv", dtype={"case_id": str})
    feat = np.load(d / "third_batch_dino_concat_features.npy").astype(np.float32)
    if "feature_idx" not in table.columns:
        table = table.copy()
        table["feature_idx"] = np.arange(len(table), dtype=int)
    return table, feat


def align(order: pd.Series, table: pd.DataFrame, feat: np.ndarray) -> np.ndarray:
    idx = order.to_frame("case_id").merge(table[["case_id", "feature_idx"]], on="case_id", how="left")["feature_idx"]
    if idx.isna().any():
        missing = order[idx.isna()].head(10).tolist()
        raise KeyError(f"Missing features: {missing}")
    return feat[idx.astype(int).to_numpy()].astype(np.float32)


def average_pair(
    root: Path,
    table_a: pd.DataFrame,
    feat_a: np.ndarray,
    table_b: pd.DataFrame,
    feat_b: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    order = table_a["case_id"].astype(str).reset_index(drop=True)
    a = align(order, table_a, feat_a)
    b = align(order, table_b, feat_b)
    if a.shape != b.shape:
        raise ValueError(f"Feature shape mismatch: {a.shape} vs {b.shape}")
    out_table = pd.DataFrame({"case_id": order, "feature_idx": np.arange(len(order), dtype=int)})
    return out_table, ((a + b) * 0.5).astype(np.float32)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    old_a_t, old_a_f = load_old(root, args.old_a_dir)
    old_b_t, old_b_f = load_old(root, args.old_b_dir)
    third_a_t, third_a_f = load_third(root, args.third_a_dir)
    third_b_t, third_b_f = load_third(root, args.third_b_dir)

    old_table, old_feat = average_pair(root, old_a_t, old_a_f, old_b_t, old_b_f)
    third_table, third_feat = average_pair(root, third_a_t, third_a_f, third_b_t, third_b_f)

    old_out = root / args.old_output_dir
    third_out = root / args.third_output_dir
    old_out.mkdir(parents=True, exist_ok=True)
    third_out.mkdir(parents=True, exist_ok=True)
    old_table.to_csv(old_out / "case_dino_concat_feature_table.csv", index=False, encoding="utf-8-sig")
    np.save(old_out / "case_dino_concat_features.npy", old_feat)
    third_table.to_csv(third_out / "third_batch_dino_concat_feature_table.csv", index=False, encoding="utf-8-sig")
    np.save(third_out / "third_batch_dino_concat_features.npy", third_feat)
    print(f"[done] old={old_feat.shape} third={third_feat.shape}", flush=True)


if __name__ == "__main__":
    main()
