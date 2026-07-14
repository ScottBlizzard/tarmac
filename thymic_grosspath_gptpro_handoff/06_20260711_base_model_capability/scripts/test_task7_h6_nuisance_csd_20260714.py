from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from build_task7_h6_integrity_manifest_20260714 import (
    source_tree_sha256,
    strict_bool,
    validate_bank_metadata,
)
from run_task7_h6_nuisance_csd_20260714 import (
    RankOneCSDHead,
    build_nuisance_environments,
    build_source_environments,
)


class H6CSDTest(unittest.TestCase):
    def test_source_tree_hash_ignores_runtime_bytecode(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("value = 1\n", encoding="utf-8")
            before = source_tree_sha256(root)
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "model.pyc").write_bytes(b"runtime-only")
            self.assertEqual(source_tree_sha256(root), before)
            (root / "model.py").write_text("value = 2\n", encoding="utf-8")
            self.assertNotEqual(source_tree_sha256(root), before)

    def test_bank_metadata_allows_deterministic_reordering(self) -> None:
        registry = pd.DataFrame(
            {
                "case_id": [f"case-{index:03d}" for index in range(591)],
                "domain": ["old_data"] * 591,
                "label_idx": np.arange(591) % 2,
                "image_count": np.arange(591) + 1,
                "is_frozen_external": [False] * 591,
            }
        )
        bank = registry.iloc[::-1].reset_index(drop=True).copy()
        bank.insert(0, "feature_row", np.arange(591))
        validate_bank_metadata(registry, bank)
        bank.loc[0, "label_idx"] = 1 - int(bank.loc[0, "label_idx"])
        with self.assertRaises(ValueError):
            validate_bank_metadata(registry, bank)

    def test_manifest_boolean_parsing_is_strict(self) -> None:
        parsed = strict_bool(
            pd.Series([False, True, "false", "TRUE", 0, 1]), "test"
        )
        self.assertEqual(parsed.tolist(), [False, True, False, True, False, True])
        with self.assertRaises(ValueError):
            strict_bool(pd.Series(["unknown"]), "test")

    def make_metadata(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rows = []
        for source_index, source in enumerate(("batch1", "batch2", "third_batch")):
            for risk in (0, 1):
                for item in range(12):
                    rows.append(
                        {
                            "case_id": f"{source_index}-{risk}-{item:02d}",
                            "source_dataset": source,
                            "label_idx": risk,
                        }
                    )
        metadata = pd.DataFrame(rows)
        train = np.flatnonzero(metadata["case_id"].str.endswith(tuple(f"{i:02d}" for i in range(8))))
        validation = np.setdiff1d(np.arange(len(metadata)), train)
        return metadata, train, validation

    def test_nuisance_construction_is_outer_training_only(self) -> None:
        metadata, train, validation = self.make_metadata()
        rng = np.random.default_rng(7)
        frequency = rng.normal(size=(len(metadata), 156))
        first = build_nuisance_environments(metadata, frequency, train, validation)
        changed = frequency.copy()
        changed[validation] += rng.normal(1000.0, 50.0, size=changed[validation].shape)
        second = build_nuisance_environments(metadata, changed, train, validation)
        np.testing.assert_array_equal(first.train_ids, second.train_ids)
        self.assertEqual(
            first.diagnostics["pca_loading_sha256"],
            second.diagnostics["pca_loading_sha256"],
        )
        self.assertEqual(
            first.diagnostics["bin_boundary_sha256"],
            second.diagnostics["bin_boundary_sha256"],
        )
        self.assertEqual(set(first.gamma_initialization), {-1.0, 1.0})
        self.assertTrue(
            all(
                row["low_risk_n"] >= 4 and row["high_risk_n"] >= 4
                for row in first.diagnostics["environment_counts"]
            )
        )

    def test_source_control_has_deterministic_centered_gamma(self) -> None:
        metadata, train, validation = self.make_metadata()
        assignment = build_source_environments(metadata, train, validation)
        self.assertEqual(assignment.names, ("batch1", "batch2", "third_batch"))
        np.testing.assert_allclose(assignment.gamma_initialization, [-1.0, 0.0, 1.0])

    def test_rank_one_readout_and_orthogonality(self) -> None:
        model = RankOneCSDHead(
            feature_dim=4,
            num_views=1,
            num_environments=2,
            gamma_initialization=np.asarray([-1.0, 1.0], dtype=np.float32),
        )
        model.eval()
        with torch.no_grad():
            model.common_weight.zero_()
            model.specific_weight.zero_()
            model.common_bias.copy_(torch.tensor([0.2, -0.1]))
            model.specific_bias.copy_(torch.tensor([0.3, 0.4]))
            model.common_weight[0, 0] = 1.0
            model.specific_weight[0, 1] = 1.0
            model.common_weight[1, 2] = 1.0
            model.specific_weight[1, 3] = 1.0
        self.assertAlmostEqual(float(model.orthogonality_loss()), 0.0, places=7)
        features = torch.randn(2, 1, 3, 4)
        mask = torch.ones(2, 1, 3, dtype=torch.bool)
        environments = torch.tensor([0, 1], dtype=torch.long)
        with torch.no_grad():
            embedding = model.embed(features, mask).float()
            common, specific = model(features, mask, environments)
            expected_common = F.linear(
                embedding, model.common_weight, model.common_bias
            )
            gamma = model.gamma[environments]
            expected_weight = model.common_weight.unsqueeze(0) + gamma.view(
                -1, 1, 1
            ) * model.specific_weight.unsqueeze(0)
            expected_bias = model.common_bias.unsqueeze(0) + gamma.view(
                -1, 1
            ) * model.specific_bias.unsqueeze(0)
            expected_specific = torch.einsum(
                "bch,bh->bc", expected_weight, embedding
            ) + expected_bias
        torch.testing.assert_close(common, expected_common)
        self.assertIsNotNone(specific)
        torch.testing.assert_close(specific, expected_specific)

    def test_locked_parameter_counts(self) -> None:
        four_environment_model = RankOneCSDHead(
            feature_dim=1024,
            num_views=6,
            num_environments=4,
            gamma_initialization=np.asarray([-1, 1, -1, 1], dtype=np.float32),
        )
        six_environment_model = RankOneCSDHead(
            feature_dim=1024,
            num_views=6,
            num_environments=6,
            gamma_initialization=np.asarray([-1, 1, -1, 1, -1, 1], dtype=np.float32),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in four_environment_model.parameters()),
            151369,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in six_environment_model.parameters()),
            151371,
        )


if __name__ == "__main__":
    unittest.main()
