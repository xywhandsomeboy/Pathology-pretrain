"""Attention-convolution fusion and full-resolution mask head."""

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import (
    ChannelAttention2d,
    ConvNeXtV2Block,
    HRFormerBlock,
    LayerNorm2d,
    UpsampleRefine,
)


class AttentionConvFusionDecoder(nn.Module):
    def __init__(
        self,
        semantic_channels: int,
        detail_channels: int,
        channels: int = 128,
        num_classes: int = 2,
        depth: int = 4,
        num_heads: int = 4,
        window_size: int = 7,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("fusion depth must be at least 2")
        self.input_projection = nn.Sequential(
            nn.Conv2d(semantic_channels + detail_channels, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.channel_attention = ChannelAttention2d(channels)
        probabilities = [drop_path_rate * index / max(depth - 1, 1) for index in range(depth)]
        blocks = []
        for index, probability in enumerate(probabilities):
            if index % 2 == 0:
                blocks.append(
                    HRFormerBlock(
                        channels,
                        num_heads=num_heads,
                        window_size=window_size,
                        drop_path_probability=probability,
                    )
                )
            else:
                blocks.append(ConvNeXtV2Block(channels, drop_path_probability=probability))
        self.fusion_blocks = nn.Sequential(*blocks)
        half_channels = max(channels // 2, 32)
        quarter_channels = max(channels // 4, 32)
        self.upsample = nn.Sequential(
            UpsampleRefine(channels, half_channels),
            UpsampleRefine(half_channels, quarter_channels),
        )
        self.head = nn.Conv2d(quarter_channels, num_classes, 1)

    def forward(
        self,
        semantic: torch.Tensor,
        detail: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if semantic.shape[0] != detail.shape[0]:
            raise ValueError("semantic and detail features must have the same batch")
        if semantic.shape[-2:] != detail.shape[-2:]:
            semantic = F.interpolate(
                semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False
            )
        fused = self.input_projection(torch.cat((semantic, detail), dim=1))
        fused = self.fusion_blocks(self.channel_attention(fused))
        logits = self.head(self.upsample(fused))
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
