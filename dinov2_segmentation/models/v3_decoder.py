"""Independent V3 semantic-to-detail cross-attention building blocks.

This module does not import or alter the V2 decoder. Attention runs on bounded
spatial grids; full-resolution detail is also available to the final head.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import LayerNorm2d, ConvNeXtV2Block


class V3DetailEncoder(nn.Module):
    def __init__(self, channels: int, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError('detail depth must be positive')
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1),
                                  LayerNorm2d(channels), nn.GELU())
        self.blocks = nn.Sequential(*(ConvNeXtV2Block(channels) for _ in range(depth)))

    def forward(self, image):
        return self.blocks(self.stem(image))


class SemanticDetailCrossAttention(nn.Module):
    """Q=semantic, K/V=detail, with explicit normalized spatial coordinates.

    Attention and convolutional FFN have independent learnable residual scales.
    No attention matrix is returned or retained for visualization by default.
    """
    def __init__(self, channels: int = 64, heads: int = 4,
                 kv_grid_size: int = 28, residual_scale_init: float = 0.01):
        super().__init__()
        if channels < 1 or heads < 1 or channels % heads:
            raise ValueError('positive channels must be divisible by positive heads')
        if kv_grid_size < 1:
            raise ValueError('kv_grid_size must be positive')
        self.kv_grid_size = kv_grid_size
        self.query_norm = nn.LayerNorm(channels)
        self.detail_norm = nn.LayerNorm(channels)
        self.position = nn.Linear(2, channels, bias=False)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(LayerNorm2d(channels),
                                 nn.Conv2d(channels, channels * 2, 1), nn.GELU(),
                                 nn.Conv2d(channels * 2, channels * 2, 3,
                                           padding=1, groups=channels * 2), nn.GELU(),
                                 nn.Conv2d(channels * 2, channels, 1))
        self.attention_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _position(self, height, width, reference):
        y = (torch.arange(height, device=reference.device, dtype=reference.dtype) + .5) * (2 / height) - 1
        x = (torch.arange(width, device=reference.device, dtype=reference.dtype) + .5) * (2 / width) - 1
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return self.position(torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2))

    def forward(self, semantic, detail):
        if semantic.ndim != 4 or detail.ndim != 4 or semantic.shape[:2] != detail.shape[:2]:
            raise ValueError('semantic and detail must be NCHW with matching batch/channels')
        b, c, h, w = semantic.shape
        kh, kw = (min(self.kv_grid_size, size) for size in detail.shape[-2:])
        memory = F.adaptive_avg_pool2d(detail, (kh, kw)).flatten(2).transpose(1, 2)
        query = semantic.flatten(2).transpose(1, 2)
        query = self.query_norm(query)
        value = self.detail_norm(memory)
        correction, _ = self.attention(
            query + self._position(h, w, query),
            value + self._position(kh, kw, value), value, need_weights=False,
        )
        correction = correction.transpose(1, 2).reshape(b, c, h, w)
        refined = semantic + self.attention_scale * correction
        return refined + self.ffn_scale * self.ffn(refined)


class V3CrossAttentionDecoder(nn.Module):
    def __init__(self, channels=64, num_classes=2, depth=4, heads=4,
                 kv_grid_size=28, residual_scale_init=0.01):
        super().__init__()
        if depth < 1:
            raise ValueError('cross-attention depth must be positive')
        self.cross_blocks = nn.ModuleList([
            SemanticDetailCrossAttention(channels, heads, kv_grid_size, residual_scale_init)
            for _ in range(depth)
        ])
        self.up_refine = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                       LayerNorm2d(channels), nn.GELU())
        self.output_head = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, padding=1),
                                         LayerNorm2d(channels), nn.GELU(),
                                         nn.Conv2d(channels, num_classes, 1))

    def forward(self, semantic, detail, return_features=False):
        refined = semantic
        stages = []
        for block in self.cross_blocks:
            refined = block(refined, detail)
            if return_features:
                stages.append(refined)
        restored = refined
        target = detail.shape[-2:]
        while restored.shape[-2:] != target:
            size = tuple(min(t, s * 2) for s, t in zip(restored.shape[-2:], target))
            restored = self.up_refine(F.interpolate(restored, size=size, mode='bilinear', align_corners=False))
        logits = self.output_head(torch.cat((restored, detail), dim=1))
        if return_features:
            return {'logits': logits, 'semantic_initial': semantic,
                    'semantic_refined': refined, 'detail': detail,
                    'semantic_stages': stages,
                    'attention_scales': torch.stack([b.attention_scale for b in self.cross_blocks]),
                    'ffn_scales': torch.stack([b.ffn_scale for b in self.cross_blocks])}
        return logits
