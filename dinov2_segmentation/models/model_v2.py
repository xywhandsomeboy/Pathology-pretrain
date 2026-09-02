"""End-to-end full-resolution residual-correction segmentation model V2."""

from __future__ import annotations

import torch
from torch import nn

from .v2_decoder import (
    FullResolutionDetailEncoder,
    FullResolutionGlobalUpsampler,
    FullResolutionResidualCorrectionDecoder,
)


class GlobalLocalSegmentationModelV2(nn.Module):
    """Use raw high-resolution evidence to repeatedly correct global semantics."""

    model_version = "v2_full_resolution_residual_correction"

    def __init__(
        self,
        num_classes: int,
        token_dim: int = 1024,
        context_dim: int = 1024,
        channels: int = 64,
        high_resolution_depth: int = 3,
        correction_depth: int = 4,
        channel_expansion: int = 2,
        attention_reduction: int = 16,
        residual_scale_init: float = 1e-2,
        max_upsample_stages: int = 6,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and be at least 2")
        self.detail_encoder = FullResolutionDetailEncoder(
            channels=channels,
            depth=high_resolution_depth,
            expansion=channel_expansion,
            attention_reduction=attention_reduction,
        )
        self.semantic_encoder = FullResolutionGlobalUpsampler(
            token_dim=token_dim,
            context_dim=context_dim,
            channels=channels,
            max_upsample_stages=max_upsample_stages,
        )
        self.decoder = FullResolutionResidualCorrectionDecoder(
            global_channels=channels,
            detail_channels=channels,
            num_classes=num_classes,
            depth=correction_depth,
            expansion=channel_expansion,
            attention_reduction=attention_reduction,
            residual_scale_init=residual_scale_init,
        )

    def forward(
        self,
        raw_patch: torch.Tensor,
        dense_tokens: torch.Tensor,
        global_context: torch.Tensor,
        return_features: bool = False,
    ):
        if raw_patch.ndim != 4 or raw_patch.shape[1] != 3:
            raise ValueError(
                f"raw_patch must be [B,3,H,W], got {raw_patch.shape}"
            )
        output_size = tuple(raw_patch.shape[-2:])
        detail = self.detail_encoder(raw_patch)
        global_initial = self.semantic_encoder(
            dense_tokens, global_context, target_size=output_size
        )
        if global_initial.shape[-2:] != output_size or detail.shape[-2:] != output_size:
            raise RuntimeError("Both V2 branches must reach the original patch resolution")

        if return_features:
            logits, global_refined, corrections = self.decoder(
                global_initial, detail, return_corrections=True
            )
            scales = torch.stack(
                [block.residual_scale for block in self.decoder.correction_blocks]
            )
            return {
                "logits": logits,
                "global_initial": global_initial,
                "detail": detail,
                "global_refined": global_refined,
                "corrections": corrections,
                "residual_scales": scales,
            }
        return self.decoder(global_initial, detail)
