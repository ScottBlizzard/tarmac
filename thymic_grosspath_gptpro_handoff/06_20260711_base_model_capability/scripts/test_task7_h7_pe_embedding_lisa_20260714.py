from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from run_task7_h7_pe_embedding_lisa_20260714 import (
    H3LISAHead,
    LISAPartnerSelector,
    MODE_CROSS_SOURCE_SAME_RISK,
    MODE_WITHIN_SOURCE_OPPOSITE_RISK,
    soft_cross_entropy,
    symmetric_lisa_mix,
)


class H7LISATest(unittest.TestCase):
    @staticmethod
    def metadata() -> pd.DataFrame:
        rows = []
        for source in ("batch1", "batch2", "third_batch"):
            for risk in (0, 1):
                for item in range(5):
                    rows.append(
                        {
                            "case_id": f"{source}-{risk}-{item}",
                            "source_dataset": source,
                            "label_idx": risk,
                        }
                    )
        return pd.DataFrame(rows)

    def test_partner_selector_obeys_both_constraints(self) -> None:
        metadata = self.metadata()
        indices = np.arange(len(metadata), dtype=int)
        selector = LISAPartnerSelector(metadata, indices, seed=17)
        anchors = np.arange(24, dtype=int)
        partners, modes, lambdas = selector.select(anchors)
        self.assertTrue(np.all((lambdas > 0.0) & (lambdas < 1.0)))
        for anchor, partner, mode in zip(anchors, partners, modes, strict=True):
            first = metadata.iloc[anchor]
            second = metadata.iloc[partner]
            if mode == MODE_CROSS_SOURCE_SAME_RISK:
                self.assertNotEqual(first.source_dataset, second.source_dataset)
                self.assertEqual(first.label_idx, second.label_idx)
            elif mode == MODE_WITHIN_SOURCE_OPPOSITE_RISK:
                self.assertEqual(first.source_dataset, second.source_dataset)
                self.assertNotEqual(first.label_idx, second.label_idx)
            else:
                self.fail(f"Unknown LISA mode {mode}")
        diagnostics = selector.diagnostics()
        self.assertAlmostEqual(
            diagnostics["actual_cross_source_fraction"], 0.5, places=12
        )

    def test_partner_selection_is_deterministic(self) -> None:
        metadata = self.metadata()
        indices = np.arange(len(metadata), dtype=int)
        first = LISAPartnerSelector(metadata, indices, seed=23)
        second = LISAPartnerSelector(metadata, indices, seed=23)
        anchors = np.arange(17, dtype=int)
        for left, right in zip(first.select(anchors), second.select(anchors), strict=True):
            np.testing.assert_array_equal(left, right)

    def test_selector_rejects_nontraining_anchor(self) -> None:
        metadata = self.metadata()
        selector = LISAPartnerSelector(metadata, np.arange(20), seed=31)
        with self.assertRaises(ValueError):
            selector.select(np.asarray([25]))

    def test_symmetric_mix_has_expected_embeddings_and_targets(self) -> None:
        first = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        second = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        first_labels = torch.tensor([0, 1])
        second_labels = torch.tensor([0, 0])
        lambdas = torch.tensor([0.25, 0.75])
        embeddings, targets = symmetric_lisa_mix(
            first, second, first_labels, second_labels, lambdas
        )
        torch.testing.assert_close(embeddings[0], torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(embeddings[2], torch.tensor([0.75, 0.25]))
        torch.testing.assert_close(targets[0], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(targets[1], torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(targets[3], torch.tensor([0.75, 0.25]))
        self.assertTrue(torch.isfinite(soft_cross_entropy(embeddings, targets)))

    def test_model_matches_locked_h3_capacity(self) -> None:
        model = H3LISAHead(feature_dim=1024, num_views=6)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 151107)
        features = torch.randn(2, 2, 3, 1024)
        mask = torch.ones(2, 2, 3, dtype=torch.bool)
        logits = model(features, mask)
        self.assertEqual(tuple(logits.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
