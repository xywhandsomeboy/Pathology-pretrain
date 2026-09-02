"""CPU invariants for the independent full-resolution Decoder V2."""

import unittest

import torch

from dinov2_segmentation.models.model_v2 import GlobalLocalSegmentationModelV2


class DecoderV2Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GlobalLocalSegmentationModelV2(
            num_classes=3,
            token_dim=16,
            context_dim=12,
            channels=8,
            high_resolution_depth=1,
            correction_depth=3,
            channel_expansion=2,
            attention_reduction=4,
            residual_scale_init=1e-2,
            max_upsample_stages=4,
        )
        self.image = torch.randn(1, 3, 32, 32)
        self.tokens = torch.randn(1, 4, 16)
        self.context = torch.randn(1, 12)

    def test_full_resolution_shapes_and_correction_count(self):
        output = self.model(
            self.image, self.tokens, self.context, return_features=True
        )
        self.assertEqual(output["logits"].shape, (1, 3, 32, 32))
        self.assertEqual(output["detail"].shape, (1, 8, 32, 32))
        self.assertEqual(output["global_initial"].shape, (1, 8, 32, 32))
        self.assertEqual(output["global_refined"].shape, (1, 8, 32, 32))
        self.assertEqual(len(output["corrections"]), 3)
        self.assertEqual(output["residual_scales"].shape, (3,))

    def test_each_correction_has_one_independent_scale(self):
        scales = [
            block.residual_scale
            for block in self.model.decoder.correction_blocks
        ]
        self.assertEqual(len(scales), 3)
        self.assertEqual(len({id(scale) for scale in scales}), 3)

    def test_zero_scales_leave_global_feature_unchanged(self):
        for block in self.model.decoder.correction_blocks:
            block.residual_scale.data.zero_()
        output = self.model(
            self.image, self.tokens, self.context, return_features=True
        )
        torch.testing.assert_close(
            output["global_refined"], output["global_initial"]
        )

    def test_each_step_applies_exactly_one_scaled_residual(self):
        detail = self.model.detail_encoder(self.image)
        current = self.model.semantic_encoder(
            self.tokens, self.context, target_size=(32, 32)
        )
        for block in self.model.decoder.correction_blocks:
            previous = current
            current, residual = block(previous, detail)
            torch.testing.assert_close(
                current, previous + block.residual_scale * residual
            )

    def test_backward_reaches_both_branches_and_scales(self):
        logits = self.model(self.image, self.tokens, self.context)
        logits.square().mean().backward()
        self.assertIsNotNone(self.model.detail_encoder.stem[0].weight.grad)
        self.assertIsNotNone(
            self.model.semantic_encoder.token_projection[0].weight.grad
        )
        for block in self.model.decoder.correction_blocks:
            self.assertIsNotNone(block.residual_scale.grad)


if __name__ == "__main__":
    unittest.main()
