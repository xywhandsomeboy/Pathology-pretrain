"""Full-resolution attention and residual-correction blocks for Decoder V2.

V2 deliberately never downsamples the raw-patch branch.  The global semantic
feature is first restored to the raw patch resolution; only then is it refined
by a sequence of independent high-resolution residual corrections.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ChannelAttention2d, ConvNeXtV2Block, LayerNorm2d
from .semantic_encoder import ContextFiLM


class SpatialAttention(nn.Module):
    """CBAM spatial attention without changing spatial resolution."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("CBAM spatial kernel size must be 3 or 7")
        self.projection = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        descriptor = torch.cat(
            (feature.mean(dim=1, keepdim=True), feature.amax(dim=1, keepdim=True)),
            dim=1,
        )
        return feature * torch.sigmoid(self.projection(descriptor))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel then spatial attention)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention2d(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(feature))


class HighResolutionAttentionConvBlock(nn.Module):
    """Attention plus 3x3 channel expansion at unchanged H x W.

    This is a feature extractor, not one of the global correction steps.  The
    single correction residual required by the design is applied later by
    :class:`ResidualCorrectionBlock`.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        attention_reduction: int = 16,
    ):
        super().__init__()
        if expansion < 1:
            raise ValueError("channel expansion must be positive")
        hidden_channels = channels * expansion
        self.norm = LayerNorm2d(channels)
        self.attention = CBAM(channels, reduction=attention_reduction)
        self.convolution = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.convolution(self.attention(self.norm(feature)))


class FullResolutionDetailEncoder(nn.Module):
    """Extract raw-patch correction evidence without any spatial downsampling."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 64,
        depth: int = 3,
        expansion: int = 2,
        attention_reduction: int = 16,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("high-resolution depth must be at least 1")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride=1, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                HighResolutionAttentionConvBlock(
                    channels,
                    expansion=expansion,
                    attention_reduction=attention_reduction,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = LayerNorm2d(channels)
        self.out_channels = channels
        self.output_stride = 1

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        input_size = image.shape[-2:]
        feature = self.output_norm(self.blocks(self.stem(image)))
        if feature.shape[-2:] != input_size:
            raise RuntimeError("V2 detail branch must preserve the original H x W")
        return feature


class FullResolutionGlobalUpsampler(nn.Module):
    """Restore DINO/GNN semantic features to the raw patch resolution."""

    def __init__(
        self,
        token_dim: int = 1024,
        context_dim: int = 1024,
        channels: int = 64,
        max_upsample_stages: int = 6,
    ):
        super().__init__()
        if max_upsample_stages < 1:
            raise ValueError("max_upsample_stages must be at least 1")
        self.token_dim = token_dim
        self.token_projection = nn.Sequential(
            nn.Conv2d(token_dim, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.condition = ContextFiLM(context_dim, channels)
        self.pre_refine = ConvNeXtV2Block(channels)
        self.upsample_stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    LayerNorm2d(channels),
                    nn.GELU(),
                    ConvNeXtV2Block(channels),
                )
                for _ in range(max_upsample_stages)
            ]
        )
        self.output_norm = LayerNorm2d(channels)
        self.out_channels = channels

    def _as_map(self, dense_tokens: torch.Tensor) -> torch.Tensor:
        if dense_tokens.ndim == 4:
            if dense_tokens.shape[1] == self.token_dim:
                return dense_tokens
            if dense_tokens.shape[-1] == self.token_dim:
                return dense_tokens.permute(0, 3, 1, 2)
            raise ValueError(
                f"4-D dense_tokens must contain token_dim={self.token_dim}, "
                f"got {dense_tokens.shape}"
            )
        if dense_tokens.ndim != 3 or dense_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"dense_tokens must be [B,N,{self.token_dim}] or a spatial map, "
                f"got {dense_tokens.shape}"
            )
        side = math.isqrt(dense_tokens.shape[1])
        if side * side != dense_tokens.shape[1]:
            raise ValueError(
                f"DINO token count {dense_tokens.shape[1]} is not a square patch grid"
            )
        return dense_tokens.transpose(1, 2).reshape(
            -1, self.token_dim, side, side
        )

    @staticmethod
    def _next_size(
        current_size: tuple[int, int], target_size: tuple[int, int]
    ) -> tuple[int, int]:
        return tuple(
            min(target, current * 2)
            for current, target in zip(current_size, target_size)
        )

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
        if any(target < current for target, current in zip(target_size, token_map.shape[-2:])):
            raise ValueError(
                "V2 global branch only restores resolution; target_size cannot be "
                f"smaller than the DINO token grid {token_map.shape[-2:]}"
            )

        feature = self.condition(self.token_projection(token_map), global_context)
        feature = self.pre_refine(feature)
        stage_index = 0
        while feature.shape[-2:] != target_size:
            if stage_index >= len(self.upsample_stages):
                raise ValueError(
                    f"Restoring {token_map.shape[-2:]} to {target_size} requires more "
                    f"than {len(self.upsample_stages)} upsample stages"
                )
            next_size = self._next_size(feature.shape[-2:], target_size)
            feature = F.interpolate(
                feature, size=next_size, mode="bilinear", align_corners=False
            )
            feature = self.upsample_stages[stage_index](feature)
            stage_index += 1
        return self.output_norm(feature)


class ResidualCorrectionBlock(nn.Module):
    """Apply exactly one scaled residual correction to a global feature map."""

    def __init__(
        self,
        global_channels: int,
        detail_channels: int,
        expansion: int = 2,
        attention_reduction: int = 16,
        residual_scale_init: float = 1e-2,
    ):
        super().__init__()
        if expansion < 1:
            raise ValueError("correction expansion must be positive")
        hidden_channels = global_channels * expansion
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                global_channels + detail_channels,
                global_channels,
                1,
                bias=False,
            ),
            LayerNorm2d(global_channels),
            nn.GELU(),
        )
        self.attention = CBAM(
            global_channels, reduction=attention_reduction
        )
        self.correction = nn.Sequential(
            nn.Conv2d(
                global_channels, hidden_channels, 3, padding=1, bias=False
            ),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels, global_channels, 3, padding=1, bias=False
            ),
            LayerNorm2d(global_channels),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

    def forward(
        self, global_feature: torch.Tensor, detail_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if global_feature.shape[0] != detail_feature.shape[0]:
            raise ValueError("global and detail features must have the same batch")
        if global_feature.shape[-2:] != detail_feature.shape[-2:]:
            raise ValueError(
                "V2 correction requires global and detail features at identical H x W"
            )
        fused = self.input_projection(
            torch.cat((global_feature, detail_feature), dim=1)
        )
        residual = self.correction(self.attention(fused))
        # This is the one and only residual addition performed by this
        # correction step: G_(k+1) = G_k + alpha_k * R_k.
        corrected = global_feature + self.residual_scale * residual
        return corrected, residual


class FullResolutionResidualCorrectionDecoder(nn.Module):
    """Repeated full-resolution corrections followed by a segmentation head."""

    def __init__(
        self,
        global_channels: int,
        detail_channels: int,
        num_classes: int,
        depth: int = 4,
        expansion: int = 2,
        attention_reduction: int = 16,
        residual_scale_init: float = 1e-2,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("correction depth must be at least 1")
        self.correction_blocks = nn.ModuleList(
            [
                ResidualCorrectionBlock(
                    global_channels=global_channels,
                    detail_channels=detail_channels,
                    expansion=expansion,
                    attention_reduction=attention_reduction,
                    residual_scale_init=residual_scale_init,
                )
                for _ in range(depth)
            ]
        )
        self.output_head = nn.Sequential(
            nn.Conv2d(
                global_channels, global_channels, 3, padding=1, bias=False
            ),
            LayerNorm2d(global_channels),
            nn.GELU(),
            nn.Conv2d(global_channels, num_classes, 1),
        )

    def forward(
        self,
        global_feature: torch.Tensor,
        detail_feature: torch.Tensor,
        return_corrections: bool = False,
    ):
        corrected = global_feature
        corrections = []
        for block in self.correction_blocks:
            corrected, residual = block(corrected, detail_feature)
            if return_corrections:
                corrections.append(residual)
        logits = self.output_head(corrected)
        if return_corrections:
            return logits, corrected, corrections
        return logits
