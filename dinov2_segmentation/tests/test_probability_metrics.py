from __future__ import annotations

import unittest

import torch

from dinov2_segmentation.probability_metrics import BinaryProbabilityMetrics


def _logits(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp(1e-6, 1 - 1e-6)
    two_class = torch.stack((1 - probabilities, probabilities), dim=1)
    return two_class.log()


class BinaryProbabilityMetricsTest(unittest.TestCase):
    def test_probability_diagnostics_and_threshold_search(self):
        probability = torch.tensor([[[0.1, 0.3], [0.7, 0.9]]])
        target = torch.tensor([[[0, 1], [0, 1]]])
        metrics = BinaryProbabilityMetrics(bins=100)
        metrics.update(_logits(probability), target)
        result = metrics.compute()
        self.assertAlmostEqual(result["mean_tumor_probability_on_tumor"], 0.6)
        self.assertAlmostEqual(result["mean_tumor_probability_on_background"], 0.4)
        self.assertGreater(result["approx_pr_auc"], 0.5)
        self.assertGreaterEqual(result["best_f2_threshold"], 0.29)
        self.assertLessEqual(result["best_f2_threshold"], 0.31)
        self.assertAlmostEqual(result["best_threshold_recall"], 1.0)

    def test_ignore_pixels_do_not_contribute(self):
        probability = torch.tensor([[[0.2, 0.8]]])
        target = torch.tensor([[[0, 255]]])
        metrics = BinaryProbabilityMetrics(bins=16)
        metrics.update(_logits(probability), target)
        result = metrics.compute()
        self.assertEqual(result["best_threshold_recall"], 0.0)
        self.assertAlmostEqual(result["mean_tumor_probability_on_background"], 0.2)


if __name__ == "__main__":
    unittest.main()
