"""DINO dense-token branch conditioned by Stage-2 graph context."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConvNeXtV2Block, LayerNorm2d


class ContextFiLM(nn.Module):
    """Use the WSI graph vector to modulate every DINO spatial token."""

    def __init__(self, context_dim: int, channels: int):
        super().__init__()
        self.context_norm = nn.LayerNorm(context_dim)
        self.affine = nn.Sequential(
            nn.Linear(context_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, feature: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, bias = self.affine(self.context_norm(context)).chunk(2, dim=-1)
        return feature * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]


class GlobalSemanticUpsampler(nn.Module):
    """Turn patch-level DINO tokens + whole-slide GNN context into an H/4 map."""

    def __init__(
        self,
        token_dim: int = 1024,
        context_dim: int = 1024,
        channels: int = 128,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.token_projection = nn.Sequential(
            nn.Conv2d(token_dim, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.condition = ContextFiLM(context_dim, channels)
        self.pre_refine = ConvNeXtV2Block(channels)
        self.up_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            ConvNeXtV2Block(channels),
        )
        self.target_refine = ConvNeXtV2Block(channels)
        self.out_channels = channels

    def _as_map(self, dense_tokens: torch.Tensor) -> torch.Tensor:
        if dense_tokens.ndim == 4:
            if dense_tokens.shape[1] == self.token_dim:
                return dense_tokens
            if dense_tokens.shape[-1] == self.token_dim:
                return dense_tokens.permute(0, 3, 1, 2)
            raise ValueError(
                f"4-D dense_tokens must contain token_dim={self.token_dim}, got {dense_tokens.shape}"
            )
        if dense_tokens.ndim != 3 or dense_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"dense_tokens must be [B,N,{self.token_dim}] or a spatial map, got {dense_tokens.shape}"
            )
        side = math.isqrt(dense_tokens.shape[1])
        if side * side != dense_tokens.shape[1]:
            raise ValueError(
                f"DINO token count {dense_tokens.shape[1]} is not a square patch grid"
            )
        return dense_tokens.transpose(1, 2).reshape(-1, self.token_dim, side, side)

    def forward(
        self,
        dense_tokens: torch.Tensor,
        global_context: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        token_map = self._as_map(dense_tokens)
        if global_context.ndim != 2 or global_context.shape[0] != token_map.shape[0]:
            raise ValueError(
                "global_context must be [B,C] and have the same batch as dense_tokens"
            )
        feature = self.condition(self.token_projection(token_map), global_context)
        feature = self.pre_refine(feature)
        # A learned convolution follows each interpolation. The first x2 step
        # preserves the DINO grid structure before the exact H/4 alignment.
        doubled = tuple(min(target, current * 2) for target, current in zip(target_size, feature.shape[-2:]))
        if doubled != feature.shape[-2:]:
            feature = F.interpolate(feature, size=doubled, mode="bilinear", align_corners=False)
            feature = self.up_refine(feature)
        if feature.shape[-2:] != target_size:
            feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
        return self.target_refine(feature)
