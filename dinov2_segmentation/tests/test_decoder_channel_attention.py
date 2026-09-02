"""Channel-attention invariants shared by Decoder V1 and Decoder V2."""

import unittest

import torch

from dinov2_segmentation.models.blocks import ChannelAttention2d
from dinov2_segmentation.models.model import GlobalLocalSegmentationModel
from dinov2_segmentation.models.model_v2 import GlobalLocalSegmentationModelV2


class DecoderChannelAttentionTest(unittest.TestCase):
    def test_channel_attention_preserves_shape_and_backpropagates(self):
        attention = ChannelAttention2d(channels=8, reduction=4)
        feature = torch.randn(2, 8, 7, 9, requires_grad=True)
        output = attention(feature)
        self.assertEqual(output.shape, feature.shape)
        output.mean().backward()
        self.assertIsNotNone(attention.projection[0].weight.grad)

    def test_v1_fusion_has_explicit_channel_attention(self):
        model = GlobalLocalSegmentationModel(
            num_classes=3,
            token_dim=16,
            context_dim=12,
            channels=8,
            detail_depth=2,
            fusion_depth=2,
            num_heads=2,
            window_size=4,
            drop_path_rate=0.0,
        )
        self.assertIsInstance(
            model.decoder.channel_attention, ChannelAttention2d
        )
        logits = model(
            torch.randn(1, 3, 32, 32),
            torch.randn(1, 4, 16),
            torch.randn(1, 12),
        )
        self.assertEqual(logits.shape, (1, 3, 32, 32))
        logits.mean().backward()
        self.assertIsNotNone(
            model.decoder.channel_attention.projection[0].weight.grad
        )

    def test_v2_high_resolution_and_corrections_use_channel_attention(self):
        model = GlobalLocalSegmentationModelV2(
            num_classes=3,
            token_dim=16,
            context_dim=12,
            channels=8,
            high_resolution_depth=1,
            correction_depth=2,
            attention_reduction=4,
            max_upsample_stages=4,
        )
        self.assertIsInstance(
            model.detail_encoder.blocks[0].attention.channel,
            ChannelAttention2d,
        )
        for block in model.decoder.correction_blocks:
            self.assertIsInstance(block.attention.channel, ChannelAttention2d)


if __name__ == "__main__":
    unittest.main()
