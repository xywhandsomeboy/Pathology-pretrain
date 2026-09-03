"""Segmentation losses used by the decoder training entry point."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("logits and target must be [B,C,H,W] and [B,H,W]")
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=logits.dtype)
    valid = valid[:, None].to(dtype=logits.dtype)
    probability = logits.softmax(dim=1) * valid
    one_hot = one_hot * valid
    reduce_dimensions = (0, 2, 3)
    intersection = (probability * one_hot).sum(reduce_dimensions)
    denominator = probability.sum(reduce_dimensions) + one_hot.sum(reduce_dimensions)
    present = one_hot.sum(reduce_dimensions) > 0
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice[present].mean() if present.any() else logits.sum() * 0.0


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
    cross_entropy_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if (target != ignore_index).any():
        cross_entropy = F.cross_entropy(logits.float(), target, ignore_index=ignore_index)
    else:
        cross_entropy = logits.sum() * 0.0
    dice = soft_dice_loss(logits.float(), target, ignore_index=ignore_index)
    total = cross_entropy_weight * cross_entropy + dice_weight * dice
    return total, {"cross_entropy": cross_entropy.detach(), "dice_loss": dice.detach()}
