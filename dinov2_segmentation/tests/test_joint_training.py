from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch_geometric.data import Data

from dinov2_segmentation.data.joint_dataset import JointPatchSegmentationDataset
from dinov2_segmentation.joint_graph import JointGraphRepository
from dinov2_segmentation.joint_optim import WarmupCosineScheduler, _vit_blocks
from dinov2_segmentation.losses import segmentation_loss


class _TinyGNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.projection = nn.Linear(channels, channels, bias=False)

    def forward(self, x, edge_index, edge_attr, use_edge_attr=None):
        del edge_index, edge_attr, use_edge_attr
        return x + self.projection(x)


class JointTrainingTest(unittest.TestCase):
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

    def test_tumor_ce_weight_emphasizes_tumor_error(self):
        logits = torch.tensor(
            [[[[4.0, 4.0]], [[-4.0, -4.0]]]], dtype=torch.float32
        )
        target = torch.tensor([[[0, 1]]], dtype=torch.long)
        _, unweighted = segmentation_loss(logits, target, tumor_ce_weight=1.0)
        _, weighted = segmentation_loss(logits, target, tumor_ce_weight=1.5)
        self.assertGreater(
            float(weighted["cross_entropy"]),
            float(unweighted["cross_entropy"]),
        )


if __name__ == "__main__":
    unittest.main()
