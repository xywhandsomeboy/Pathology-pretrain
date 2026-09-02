"""Reusable ConvNeXtV2 and HRFormer-style building blocks.

The decoder deliberately uses established components: ConvNeXtV2-style
depthwise convolution/GRN blocks and HRFormer-style local-window attention
with a convolutional feed-forward network.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    if probability == 0.0 or not training:
        return x
    keep_probability = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_probability + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x * random_tensor.floor() / keep_probability


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.probability, self.training)


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for an NCHW tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ChannelAttention2d(nn.Module):
    """CBAM-style channel attention for an NCHW feature map.

    Average and maximum spatial descriptors share the same projection so the
    module can recalibrate interactions between semantic/detail channels
    without changing the tensor shape or adding a residual connection.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if reduction < 1:
            raise ValueError("attention reduction must be positive")
        hidden_channels = max(channels // reduction, 4)
        self.projection = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        average = self.projection(F.adaptive_avg_pool2d(feature, 1))
        maximum = self.projection(F.adaptive_max_pool2d(feature, 1))
        return feature * torch.sigmoid(average + maximum)


class GlobalResponseNorm(nn.Module):
    """Global Response Normalization from ConvNeXt V2 (channels last)."""

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + 1e-6)
        return x + self.gamma * (x * normalized) + self.beta


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 residual block that preserves spatial resolution."""

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        drop_path_probability: float = 0.0,
    ):
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, hidden)
        self.activation = nn.GELU()
        self.grn = GlobalResponseNorm(hidden)
        self.project = nn.Linear(hidden, channels)
        self.drop_path = DropPath(drop_path_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.project(self.grn(self.activation(self.expand(self.norm(x)))))
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class WindowAttention(nn.Module):
    """2-D local-window multi-head attention with relative position bias."""

    def __init__(self, channels: int, num_heads: int, window_size: int):
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)

        relative_count = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_count, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coordinate = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(coordinate, coordinate, indexing="ij"))
        flat = grid.flatten(1)
        relative = flat[:, :, None] - flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=False)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        batch_windows, tokens, channels = windows.shape
        qkv = self.qkv(windows).reshape(
            batch_windows, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attention = (
            attention + relative_bias.to(dtype=attention.dtype).unsqueeze(0)
        ).softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.projection(output)


def _partition_windows(x: torch.Tensor, window_size: int):
    batch, channels, height, width = x.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    x = F.pad(x, (0, pad_width, 0, pad_height))
    padded_height, padded_width = height + pad_height, width + pad_width
    windows = x.reshape(
        batch,
        channels,
        padded_height // window_size,
        window_size,
        padded_width // window_size,
        window_size,
    ).permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, channels)
    return windows, (height, width, padded_height, padded_width)


def _reverse_windows(
    windows: torch.Tensor,
    window_size: int,
    shape: tuple[int, int, int, int],
    batch: int,
) -> torch.Tensor:
    height, width, padded_height, padded_width = shape
    channels = windows.shape[-1]
    x = windows.reshape(
        batch,
        padded_height // window_size,
        padded_width // window_size,
        window_size,
        window_size,
        channels,
    ).permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_height, padded_width)
    return x[:, :, :height, :width]


class ConvolutionalFeedForward(nn.Module):
    """HRFormer convolutional FFN: pointwise MLP plus depthwise 3x3 mixing."""

    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = channels * expansion
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HRFormerBlock(nn.Module):
    """Resolution-preserving HRFormer-style local attention block."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 7,
        expansion: int = 4,
        dropout: float = 0.0,
        drop_path_probability: float = 0.0,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.norm1 = LayerNorm2d(channels)
        self.attention = WindowAttention(channels, num_heads, self.window_size)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = ConvolutionalFeedForward(channels, expansion, dropout)
        self.drop_path = DropPath(drop_path_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        windows, padded_shape = _partition_windows(normalized, self.window_size)
        attended = self.attention(windows)
        attended = _reverse_windows(
            attended, self.window_size, padded_shape, batch=x.shape[0]
        )
        x = x + self.drop_path(attended)
        return x + self.drop_path(self.ffn(self.norm2(x)))


class UpsampleRefine(nn.Module):
    """CNN-only x2 upsampling and boundary refinement."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        self.refine = ConvNeXtV2Block(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.refine(self.projection(x))
