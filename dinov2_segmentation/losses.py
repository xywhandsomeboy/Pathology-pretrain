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


def foreground_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
    alpha: float = 0.3,
    beta: float = 0.7,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Batch-reduced tumor-only Tversky loss.

    ``alpha`` weights false positives and ``beta`` weights false negatives.
    A batch without annotated tumor is left to cross entropy instead of turning
    the overlap term into another background objective.
    """

    if logits.ndim != 4 or logits.shape[1] != 2 or target.ndim != 3:
        raise ValueError(
            "foreground Tversky requires [B,2,H,W] logits and [B,H,W] target"
        )
    if alpha <= 0 or beta <= 0:
        raise ValueError("Tversky alpha and beta must be positive")
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    safe_target = target.masked_fill(~valid, 0)
    truth = (safe_target == 1).to(dtype=logits.dtype)
    valid_float = valid.to(dtype=logits.dtype)
    truth = truth * valid_float
    if not truth.any():
        return logits.sum() * 0.0
    probability = logits.softmax(dim=1)[:, 1] * valid_float
    true_positive = (probability * truth).sum()
    false_positive = (probability * (1.0 - truth) * valid_float).sum()
    false_negative = ((1.0 - probability) * truth).sum()
    index = (true_positive + epsilon) / (
        true_positive + alpha * false_positive + beta * false_negative + epsilon
    )
    return 1.0 - index


def segmentation_loss(
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
    if cross_entropy_weight < 0 or dice_weight < 0:
        raise ValueError("loss component weights must be non-negative")
    if tumor_class_weight <= 0:
        raise ValueError("tumor_class_weight must be positive")
    if logits.shape[1] < 2:
        raise ValueError("tumor_class_weight requires at least two output classes")
    if (target != ignore_index).any():
        class_weights = logits.new_ones(logits.shape[1], dtype=torch.float32)
        class_weights[1] = tumor_class_weight
        cross_entropy = F.cross_entropy(
            logits.float(),
            target,
            weight=class_weights,
            ignore_index=ignore_index,
        )
    else:
        cross_entropy = logits.sum() * 0.0
    if overlap_loss == "dice":
        overlap = soft_dice_loss(logits.float(), target, ignore_index=ignore_index)
        overlap_parts = {"dice_loss": overlap.detach()}
    elif overlap_loss == "foreground_tversky":
        overlap = foreground_tversky_loss(
            logits.float(),
            target,
            ignore_index=ignore_index,
            alpha=tversky_alpha,
            beta=tversky_beta,
        )
        overlap_parts = {"tversky_loss": overlap.detach()}
    else:
        raise ValueError(f"Unsupported overlap_loss: {overlap_loss!r}")
    total = cross_entropy_weight * cross_entropy + dice_weight * overlap
    return total, {"cross_entropy": cross_entropy.detach(), **overlap_parts}
