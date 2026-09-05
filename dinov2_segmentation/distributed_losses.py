"""Global-batch segmentation objectives for synchronous DDP training.

Dice and Tversky are ratios of batch statistics, so averaging per-rank losses
would change the objective. This module sums their sufficient statistics and
the weighted cross-entropy numerator/denominator before evaluating the loss.

Every rank must call this function and backpropagate its returned loss once.
The autograd-aware SUM collective sums backward gradients too; DDP's parameter
gradient averaging cancels that world-size factor. Do not divide this loss by
world size again. With identical logits, this produces the serial global-batch
objective and parameter gradient, including uneven shard sizes.

Use only for synchronized training. Uneven-length distributed validation uses
the existing local loss and sample-weighted reporting, with no per-batch
collectives; its reported overlap loss is therefore batch-partition dependent.
"""

from __future__ import annotations

import torch
import torch.distributed as distributed
from torch.distributed.nn.functional import all_reduce
import torch.nn.functional as F

from dinov2_segmentation.losses import segmentation_loss


def segmentation_loss_distributed(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
    cross_entropy_weight: float = 1.0,
    dice_weight: float = 1.0,
    tumor_class_weight: float = 1.0,
    overlap_loss: str = "dice",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match :func:`segmentation_loss` over the union of DDP rank batches."""

    if not distributed.is_initialized() or distributed.get_world_size() == 1:
        return segmentation_loss(
            logits,
            target,
            ignore_index=ignore_index,
            cross_entropy_weight=cross_entropy_weight,
            dice_weight=dice_weight,
            tumor_class_weight=tumor_class_weight,
            overlap_loss=overlap_loss,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
        )
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("logits and target must be [B,C,H,W] and [B,H,W]")
    if logits.shape[1] < 2:
        raise ValueError("tumor_class_weight requires at least two output classes")
    if cross_entropy_weight < 0 or dice_weight < 0:
        raise ValueError("loss component weights must be non-negative")
    if tumor_class_weight <= 0:
        raise ValueError("tumor_class_weight must be positive")
    if overlap_loss not in {"dice", "foreground_tversky"}:
        raise ValueError(f"Unsupported overlap_loss: {overlap_loss!r}")
    if overlap_loss == "foreground_tversky":
        if logits.shape[1] != 2:
            raise ValueError("foreground Tversky requires two output classes")
        if tversky_alpha <= 0 or tversky_beta <= 0:
            raise ValueError("Tversky alpha and beta must be positive")

    logits = logits.float()
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    class_weights = logits.new_ones(logits.shape[1])
    class_weights[1] = tumor_class_weight
    cross_entropy_numerator = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="sum",
    )
    cross_entropy_denominator = (class_weights[safe_target] * valid).sum()
    probability = logits.softmax(dim=1) * valid[:, None]
    one_hot = F.one_hot(safe_target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    one_hot = one_hot.to(logits.dtype) * valid[:, None]
    dimensions = (0, 2, 3)

    # One unconditional collective per micro-batch, including ranks with only
    # background or ignored pixels. All branches below depend on global counts.
    if overlap_loss == "dice":
        local_statistics = torch.cat(
            (
                torch.stack((cross_entropy_numerator, cross_entropy_denominator)),
                (probability * one_hot).sum(dimensions),
                probability.sum(dimensions) + one_hot.sum(dimensions),
                one_hot.sum(dimensions),
            )
        )
    else:
        truth = one_hot[:, 1]
        foreground = probability[:, 1]
        local_statistics = torch.stack(
            (
                cross_entropy_numerator,
                cross_entropy_denominator,
                (foreground * truth).sum(),
                (foreground * (1.0 - truth) * valid).sum(),
                ((1.0 - foreground) * truth).sum(),
                truth.sum(),
            )
        )
    statistics = all_reduce(local_statistics, op=distributed.ReduceOp.SUM)
    zero = statistics.sum() * 0.0
    cross_entropy = statistics[0] / statistics[1] if statistics[1] > 0 else zero
    epsilon = 1e-6
    if overlap_loss == "dice":
        classes = logits.shape[1]
        intersection, denominator, truth_count = statistics[2:].split(classes)
        present = truth_count > 0
        overlap = (
            1.0 - ((2.0 * intersection[present] + epsilon) / (denominator[present] + epsilon)).mean()
            if present.any()
            else zero
        )
        overlap_name = "dice_loss"
    else:
        true_positive, false_positive, false_negative, truth_count = statistics[2:]
        overlap = (
            1.0
            - (true_positive + epsilon)
            / (
                true_positive
                + tversky_alpha * false_positive
                + tversky_beta * false_negative
                + epsilon
            )
            if truth_count > 0
            else zero
        )
        overlap_name = "tversky_loss"
    total = cross_entropy_weight * cross_entropy + dice_weight * overlap
    return total, {
        "cross_entropy": cross_entropy.detach(),
        overlap_name: overlap.detach(),
    }
