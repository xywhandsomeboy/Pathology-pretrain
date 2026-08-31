"""End-to-end decoder for cached DINO/GNN features and raw image patches."""

from __future__ import annotations

from torch import nn

from .detail_encoder import HighResolutionDetailEncoder
from .fusion_decoder import AttentionConvFusionDecoder
from .semantic_encoder import GlobalSemanticUpsampler


class GlobalLocalSegmentationModel(nn.Module):
    """Two-branch patch decoder with whole-slide semantic conditioning.

    Branch 1 receives cached dense DINO tokens for one patch and its Stage-2
    whole-slide GNN context vector. Branch 2 receives exactly the raw image
    patch identified by the same slide/coordinate key.
    """

    def __init__(
        self,
        num_classes: int,
        token_dim: int = 1024,
        context_dim: int = 1024,
        channels: int = 128,
        detail_depth: int = 4,
        fusion_depth: int = 4,
        num_heads: int = 4,
        window_size: int = 7,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.detail_encoder = HighResolutionDetailEncoder(
            channels=channels,
            depth=detail_depth,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
        )
        self.semantic_encoder = GlobalSemanticUpsampler(
            token_dim=token_dim,
            context_dim=context_dim,
            channels=channels,
        )
        self.decoder = AttentionConvFusionDecoder(
            semantic_channels=channels,
            detail_channels=channels,
            channels=channels,
            num_classes=num_classes,
            depth=fusion_depth,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
        )

    def forward(
        self,
        raw_patch,
        dense_tokens,
        global_context,
        return_features: bool = False,
    ):
        if raw_patch.ndim != 4 or raw_patch.shape[1] != 3:
            raise ValueError(f"raw_patch must be [B,3,H,W], got {raw_patch.shape}")
        detail = self.detail_encoder(raw_patch)
        semantic = self.semantic_encoder(
            dense_tokens,
            global_context,
            target_size=detail.shape[-2:],
        )
        logits = self.decoder(semantic, detail, output_size=raw_patch.shape[-2:])
        if return_features:
            return {"logits": logits, "semantic": semantic, "detail": detail}
        return logits
