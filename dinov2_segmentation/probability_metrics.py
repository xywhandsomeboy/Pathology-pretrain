"""Streaming binary probability diagnostics for very large WSI patch sets."""

from __future__ import annotations

import torch


def binary_confusion_metrics(confusion: torch.Tensor) -> dict[str, float]:
    if tuple(confusion.shape) != (2, 2):
        raise ValueError("binary confusion matrix must have shape [2,2]")
    true_negative = float(confusion[0, 0])
    false_positive = float(confusion[0, 1])
    false_negative = float(confusion[1, 0])
    true_positive = float(confusion[1, 1])
    denominator_dice = 2 * true_positive + false_positive + false_negative
    denominator_iou = true_positive + false_positive + false_negative
    pixel_count = max(float(confusion.sum()), 1.0)
    return {
        "tumor_dice": (
            2 * true_positive / denominator_dice if denominator_dice else 1.0
        ),
        "tumor_iou": (
            true_positive / denominator_iou if denominator_iou else 1.0
        ),
        "tumor_precision": true_positive
        / max(true_positive + false_positive, 1.0),
        "tumor_recall": true_positive
        / max(true_positive + false_negative, 1.0),
        "tumor_specificity": true_negative
        / max(true_negative + false_positive, 1.0),
        "tumor_f2": 5.0
        * true_positive
        / max(5.0 * true_positive + 4.0 * false_negative + false_positive, 1.0),
        "predicted_tumor_fraction": (true_positive + false_positive) / pixel_count,
        "true_tumor_fraction": (true_positive + false_negative) / pixel_count,
        "pixel_accuracy": float(confusion.diag().sum()) / pixel_count,
    }


class BinaryProbabilityMetrics:
    """Accumulate a bounded histogram instead of retaining billions of pixels."""

    def __init__(self, bins: int = 256) -> None:
        self.bins = int(bins)
        if self.bins < 16:
            raise ValueError("probability histogram requires at least 16 bins")
        self.positive_histogram = torch.zeros(self.bins, dtype=torch.int64)
        self.negative_histogram = torch.zeros(self.bins, dtype=torch.int64)
        self.positive_probability_sum = 0.0
        self.negative_probability_sum = 0.0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        ignore_index: int = 255,
    ) -> None:
        if logits.ndim != 4 or logits.shape[1] != 2 or target.ndim != 3:
            raise ValueError("binary metrics require logits [B,2,H,W] and target [B,H,W]")
        valid = target != ignore_index
        if not valid.any():
            return
        probability = logits.detach().float().softmax(dim=1)[:, 1][valid]
        truth = target[valid] == 1
        indices = torch.floor(probability * self.bins).long().clamp_(0, self.bins - 1)
        for positive, histogram_name, sum_name in (
            (True, "positive_histogram", "positive_probability_sum"),
            (False, "negative_histogram", "negative_probability_sum"),
        ):
            selected = probability[truth == positive]
            selected_indices = indices[truth == positive]
            if not selected_indices.numel():
                continue
            counts = torch.bincount(selected_indices, minlength=self.bins).cpu()
            getattr(self, histogram_name).add_(counts)
            setattr(self, sum_name, getattr(self, sum_name) + float(selected.sum()))

    def compute(self) -> dict[str, float | int]:
        positive = self.positive_histogram.to(torch.float64)
        negative = self.negative_histogram.to(torch.float64)
        positive_count = int(positive.sum())
        negative_count = int(negative.sum())
        if positive_count == 0:
            return {
                "probability_histogram_bins": self.bins,
                "mean_tumor_probability_on_tumor": 0.0,
                "mean_tumor_probability_on_background": (
                    self.negative_probability_sum / max(negative_count, 1)
                ),
                "approx_pr_auc": 0.0,
                "best_f2_threshold": 1.0,
                "best_threshold_f2": 0.0,
                "best_threshold_precision": 1.0,
                "best_threshold_recall": 0.0,
            }

        true_positive = positive.flip(0).cumsum(0).flip(0)
        false_positive = negative.flip(0).cumsum(0).flip(0)
        false_negative = positive_count - true_positive
        precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
        recall = true_positive / float(positive_count)
        f2 = 5.0 * true_positive / (
            5.0 * true_positive + 4.0 * false_negative + false_positive
        ).clamp_min(1.0)

        maximum_f2 = f2.max()
        # Prefer the highest threshold when several histogram bins tie.
        best_index = int(torch.nonzero(f2 == maximum_f2, as_tuple=False)[-1])
        thresholds = torch.arange(self.bins, dtype=torch.float64) / self.bins

        recall_curve = torch.cat(
            (torch.zeros(1, dtype=torch.float64), recall.flip(0))
        )
        precision_curve = torch.cat(
            (torch.ones(1, dtype=torch.float64), precision.flip(0))
        )
        approximate_pr_auc = torch.trapezoid(precision_curve, recall_curve)
        return {
            "probability_histogram_bins": self.bins,
            "mean_tumor_probability_on_tumor": (
                self.positive_probability_sum / positive_count
            ),
            "mean_tumor_probability_on_background": (
                self.negative_probability_sum / max(negative_count, 1)
            ),
            "approx_pr_auc": float(approximate_pr_auc),
            "best_f2_threshold": float(thresholds[best_index]),
            "best_threshold_f2": float(f2[best_index]),
            "best_threshold_precision": float(precision[best_index]),
            "best_threshold_recall": float(recall[best_index]),
        }
