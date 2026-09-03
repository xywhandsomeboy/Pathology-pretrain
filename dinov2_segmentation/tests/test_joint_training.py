from __future__ import annotations

import csv
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch_geometric.data import Data

from dinov2_segmentation.data.joint_dataset import JointPatchSegmentationDataset
from dinov2_segmentation.joint_graph import JointGraphRepository
from dinov2_segmentation.joint_optim import WarmupCosineScheduler, _vit_blocks
from dinov2_segmentation.losses import foreground_tversky_loss, segmentation_loss
from dinov2_segmentation.probability_metrics import binary_confusion_metrics
from dinov2_segmentation.profiles import validate_experiment_profile


class _TinyGNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.projection = nn.Linear(channels, channels, bias=False)

    def forward(self, x, edge_index, edge_attr, use_edge_attr=None):
        del edge_index, edge_attr, use_edge_attr
        return x + self.projection(x)


class JointTrainingTest(unittest.TestCase):
    @staticmethod
    def _profile_args(profile: str = "ST") -> SimpleNamespace:
        return SimpleNamespace(
            experiment_profile=profile,
            sampling_mode="slide_stratified",
            sampling_positive_fraction=0.60,
            sampling_boundary_positive_fraction=0.50,
            sampling_slide_balance_power=0.50,
            sampling_max_patch_repeats=2,
            overlap_loss="dice" if profile == "S" else "foreground_tversky",
            cross_entropy_weight=1.0,
            dice_weight=1.0,
            tumor_class_weight=1.0,
            tversky_alpha=0.3,
            tversky_beta=0.7,
            color_augmentation="mild" if profile == "STA" else "none",
            probability_metric_bins=256,
        )

    def test_named_profiles_accept_only_their_fixed_ablation(self):
        for profile in ("S", "ST", "STA"):
            validate_experiment_profile(self._profile_args(profile))

    def test_named_profile_rejects_mislabeled_configuration(self):
        args = self._profile_args()
        args.sampling_mode = "uniform"
        with self.assertRaisesRegex(ValueError, "sampling_mode"):
            validate_experiment_profile(args)

    def test_confusion_metrics_expose_under_segmentation(self):
        confusion = torch.tensor([[80, 10], [20, 40]], dtype=torch.int64)
        result = binary_confusion_metrics(confusion)
        self.assertAlmostEqual(result["tumor_precision"], 0.8)
        self.assertAlmostEqual(result["tumor_recall"], 2 / 3)
        self.assertAlmostEqual(result["tumor_specificity"], 8 / 9)
        self.assertAlmostEqual(result["predicted_tumor_fraction"], 1 / 3)
        self.assertAlmostEqual(result["true_tumor_fraction"], 0.4)

    def test_foreground_tversky_emphasizes_false_negatives(self):
        target = torch.tensor([[[1, 0]]], dtype=torch.int64)
        missed_tumor = torch.tensor(
            [[[[0.0, 0.0]], [[-2.1972246, -2.1972246]]]], dtype=torch.float32
        )
        false_alarm = torch.tensor(
            [[[[0.0, 0.0]], [[2.1972246, 2.1972246]]]], dtype=torch.float32
        )
        fn_loss = foreground_tversky_loss(
            missed_tumor, target, alpha=0.3, beta=0.7
        )
        fp_loss = foreground_tversky_loss(
            false_alarm, target, alpha=0.3, beta=0.7
        )
        self.assertGreater(float(fn_loss), float(fp_loss))

    def test_foreground_tversky_defers_empty_masks_to_cross_entropy(self):
        logits = torch.randn(2, 2, 4, 4, requires_grad=True)
        target = torch.zeros(2, 4, 4, dtype=torch.int64)
        total, parts = segmentation_loss(
            logits, target, overlap_loss="foreground_tversky"
        )
        self.assertEqual(float(parts["tversky_loss"]), 0.0)
        self.assertGreater(float(parts["cross_entropy"]), 0.0)
        total.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_tumor_class_weight_only_reweights_cross_entropy(self):
        logits = torch.tensor(
            [[[[2.0, 2.0]], [[0.0, 0.0]]]], dtype=torch.float32
        )
        target = torch.tensor([[[0, 1]]], dtype=torch.int64)
        plain, plain_parts = segmentation_loss(logits, target)
        weighted, weighted_parts = segmentation_loss(
            logits, target, tumor_class_weight=1.5
        )
        self.assertGreater(float(weighted), float(plain))
        self.assertGreater(
            float(weighted_parts["cross_entropy"]),
            float(plain_parts["cross_entropy"]),
        )
        self.assertEqual(
            float(weighted_parts["dice_loss"]),
            float(plain_parts["dice_loss"]),
        )

    def test_raw_dataset_and_binary_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "patch.jpg"
            mask_path = root / "mask.png"
            Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(image_path)
            Image.fromarray(np.eye(8, dtype=np.uint8)).save(mask_path)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "slide_id",
                        "patch_id",
                        "image_path",
                        "mask_path",
                        "x",
                        "y",
                        "level",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "slide_id": "slide",
                        "patch_id": "patch",
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "x": 0,
                        "y": 0,
                        "level": 1,
                    }
                )
            sample = JointPatchSegmentationDataset(
                manifest, image_size=8, training=False
            )[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 8, 8))
            self.assertEqual(set(sample["mask"].unique().tolist()), {0, 1})

    def test_mild_color_augmentation_changes_only_the_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "patch.png"
            mask_path = root / "mask.png"
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[..., 0] = 180
            image[..., 1] = 90
            image[..., 2] = 45
            mask = np.eye(8, dtype=np.uint8)
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "slide_id",
                        "patch_id",
                        "image_path",
                        "mask_path",
                        "x",
                        "y",
                        "level",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "slide_id": "slide",
                        "patch_id": "patch",
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "x": 0,
                        "y": 0,
                        "level": 1,
                    }
                )
            plain = JointPatchSegmentationDataset(
                manifest, image_size=8, training=False
            )[0]
            random.seed(3)
            torch.manual_seed(3)
            augmented = JointPatchSegmentationDataset(
                manifest,
                image_size=8,
                training=True,
                horizontal_flip_probability=0,
                vertical_flip_probability=0,
                color_augmentation="mild",
            )[0]
            self.assertFalse(torch.equal(plain["image"], augmented["image"]))
            self.assertTrue(torch.equal(plain["mask"], augmented["mask"]))

    def test_context_keeps_online_stage1_and_stage2_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = Data(
                x=torch.randn(4, 3),
                edge_index=torch.tensor(
                    [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
                ),
            )
            graph.patch_ids = ["p0", "p1", "p2", "p3"]
            graph.slide_id = "slide"
            graph.edge_mode = "distance"
            torch.save(graph, root / "slide.pt")
            repository = JointGraphRepository(root, expected_edge_mode="distance")
            online = torch.randn(1, 3, requires_grad=True)
            gnn = _TinyGNN(3)
            context = repository.contextualize(
                gnn,
                online,
                ["slide"],
                ["p1"],
                num_hops=2,
                use_edge_attr=False,
                update_memory=True,
            )
            context.square().sum().backward()
            self.assertGreater(float(online.grad.abs().sum()), 0)
            self.assertGreater(float(gnn.projection.weight.grad.abs().sum()), 0)
            cached_graph, mapping = repository._get("slide")
            self.assertTrue(torch.equal(cached_graph.x[mapping["p1"]], online.detach()[0]))

    def test_chunk_padding_is_not_counted_as_vit_layers(self):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)]),
                        nn.ModuleList(
                            [nn.Identity(), nn.Identity(), nn.Linear(2, 2), nn.Linear(2, 2)]
                        ),
                    ]
                )

        self.assertEqual(len(_vit_blocks(Backbone())), 4)

    def test_warmup_cosine_preserves_parameter_group_ratios(self):
        left = nn.Parameter(torch.tensor(1.0))
        right = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW(
            [{"params": [left], "lr": 1e-4}, {"params": [right], "lr": 2e-4}]
        )
        scheduler = WarmupCosineScheduler(
            optimizer, total_steps=10, warmup_steps=2, min_ratio=0.1
        )
        initial = [group["lr"] for group in optimizer.param_groups]
        self.assertAlmostEqual(initial[1] / initial[0], 2.0)
        for _ in range(9):
            scheduler.step()
        final = [group["lr"] for group in optimizer.param_groups]
        self.assertAlmostEqual(final[0], 1e-5)
        self.assertAlmostEqual(final[1] / final[0], 2.0)

    def test_phase_scales_can_freeze_and_restore_groups(self):
        left = nn.Parameter(torch.tensor(1.0))
        right = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW(
            [
                {"params": [left], "lr": 1e-4, "phase_scale": 1.0},
                {"params": [right], "lr": 2e-4, "phase_scale": 1.0},
            ]
        )
        scheduler = WarmupCosineScheduler(
            optimizer, total_steps=10, warmup_steps=2, min_ratio=0.1
        )
        scheduler.set_phase_scales([0.0, 0.5])
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 5e-5)
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        scheduler.set_phase_scales([1.0, 1.0])
        self.assertAlmostEqual(
            optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
