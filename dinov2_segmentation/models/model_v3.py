"""Independent V3 cross-attention segmentation model, using shared V1 primitives."""
from __future__ import annotations

from torch import nn
from .semantic_encoder import GlobalSemanticUpsampler
from .v3_decoder import V3DetailEncoder, V3CrossAttentionDecoder


class GlobalLocalSegmentationModelV3(nn.Module):
    model_version = 'v3_semantic_detail_cross_attention'

    def __init__(self, num_classes, token_dim=1024, context_dim=1024, channels=64,
                 detail_depth=3, cross_attention_depth=4, heads=4,
                 query_grid_size=56, kv_grid_size=28, residual_scale_init=0.01):
        super().__init__()
        if num_classes < 2:
            raise ValueError('num_classes must include background and be at least 2')
        if query_grid_size < 1:
            raise ValueError('query_grid_size must be positive')
        self.query_grid_size = query_grid_size
        self.detail_encoder = V3DetailEncoder(channels, detail_depth)
        self.semantic_encoder = GlobalSemanticUpsampler(token_dim, context_dim, channels)
        self.decoder = V3CrossAttentionDecoder(channels, num_classes, cross_attention_depth,
                                               heads, kv_grid_size, residual_scale_init)

    def forward(self, raw_patch, dense_tokens, global_context, return_features=False):
        if raw_patch.ndim != 4 or raw_patch.shape[1] != 3:
            raise ValueError('raw_patch must be [B,3,H,W]')
        if min(raw_patch.shape[-2:]) < 1:
            raise ValueError('image dimensions must be positive')
        grid = tuple(min(self.query_grid_size, max(1, (s + 3) // 4)) for s in raw_patch.shape[-2:])
        detail = self.detail_encoder(raw_patch)
        semantic = self.semantic_encoder(dense_tokens, global_context, target_size=grid)
        return self.decoder(semantic, detail, return_features=return_features)
