"""Exercise the shared trainer through real serial and two-rank CPU execution."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.data import Data

from dinov2_segmentation.distributed_execution import DistributedExecution

# Match the training launcher's explicit choice among the repository's DINO trees.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dinov2_stage2_2_FmH2ST"))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3)])
        self.norm = nn.LayerNorm(3)

    def forward(self, images):
        return self.norm(self.blocks[0](images.mean((2, 3))))


class TinyStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.spatial_agg = nn.Linear(3, 3)
        self.local_spatial_agg = nn.Linear(3, 3)
        self.local_crop_fusion = nn.Linear(3, 3)
        self.node_fusion = nn.Linear(9, 3)

    def forward(self, images):
        dense = self.backbone(images)
        nodes = self.node_fusion(torch.cat((
            dense, self.spatial_agg(dense),
            self.local_crop_fusion(self.local_spatial_agg(dense)),
        ), dim=1))
        return nodes, dense


class TinyGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.unused_edge_encoder = nn.Linear(1, 3)

    def forward(self, x, edge_index, edge_attr=None, use_edge_attr=False):
        neighbors = torch.zeros_like(x).index_add(0, edge_index[1], x[edge_index[0]])
        return self.projection(x + 0.1 * neighbors)


class TinySystem(nn.Module):
    model_version = "parallel_test_v1"
    decoder_version = "v1"

    def __init__(self, **kwargs):
        super().__init__()
        self.stage1 = TinyStage1()
        self.stage2 = TinyGNN()
        self.decoder = nn.Conv2d(9, 2, 1)
        self.stage2_runtime = SimpleNamespace(
            context_edge_mode="distance", num_layers=1, use_edge_attr=False,
        )

    def decode(self, images, dense, contexts):
        b, _, h, w = images.shape
        features = torch.cat((dense, contexts), dim=1)[:, :, None, None].expand(b, 6, h, w)
        return self.decoder(torch.cat((images, features), dim=1))


def _trainer_args(root, mode, output, extra=()):
    from dinov2_segmentation.train_joint_parallel import parse_args
    return parse_args([
        "--execution-mode", mode, "--device", "cpu", "--decoder-version", "v1",
        "--train-manifest", str(root / "train.csv"),
        "--val-manifest", str(root / "valid.csv"),
        "--graph-dir", str(root / "graphs"),
        "--stage1-config", str(root / "dummy"),
        "--stage1-checkpoint", str(root / "dummy"),
        "--stage2-config", str(root / "dummy"),
        "--stage2-checkpoint", str(root / "dummy"),
        "--output-dir", str(output), "--image-size", "8", "--epochs", "3",
        "--batch-size", "1", "--workers", "0", "--gradient-accumulation", "2",
        "--decoder-only-epochs", "1", "--stage1-top-unfreeze-epoch", "2",
        "--max-train-batches", "3", "--probability-metric-bins", "16", *extra,
    ])


def _prepare_sources(root):
    (root / "dummy").touch()
    graphs = root / "graphs"
    graphs.mkdir()
    for slide in ("training", "validation"):
        count = 6 if slide == "training" else 1
        edges = torch.cartesian_prod(torch.arange(count), torch.arange(count)).T.contiguous()
        torch.save(Data(
            x=torch.zeros(count, 3), edge_index=edges, slide_id=slide,
            patch_ids=[f"p{i}" for i in range(count)], edge_mode="distance",
        ), graphs / f"{slide}.pt")
    fields = ("slide_id", "patch_id", "image_path", "mask_path", "x", "y", "level")
    for name, slide, count in (("train", "training", 6), ("valid", "validation", 1)):
        with (root / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in range(count):
                image = root / f"{name}{i}.png"
                mask = root / f"{name}{i}_mask.png"
                pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
                Image.fromarray(pixels + i).save(image)
                Image.fromarray(np.indices((8, 8)).sum(0).astype(np.uint8) % 2).save(mask)
                writer.writerow(dict(slide_id=slide, patch_id=f"p{i}", image_path=image,
                                     mask_path=mask, x=i * 8, y=0, level=0))


def _worker(rank, root_string, rendezvous):
    torch.set_num_threads(1)
    root = Path(root_string)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    execution = DistributedExecution("cpu", rank=rank, world_size=2, owns_process_group=True)
    try:
        from dinov2_segmentation import train_joint
        train_joint.JointSegmentationSystem = TinySystem
        args = _trainer_args(root, "ddp", root / "ddp")
        train_joint.main(args=args, execution=execution)
        checkpoint = torch.load(root / "ddp" / "checkpoint_last.pt", map_location="cpu")
        assert checkpoint["gradient_audit"]["complete"]
        assert checkpoint["scheduler"]["current_step"] == 5
        assert checkpoint["execution"]["world_size"] == 2
        history = json.loads((root / "ddp" / "history.json").read_text())
        assert [row["training_phase"] for row in history] == [
            "decoder_only", "adapters_and_decoder", "top_backbone_joint",
        ]
        # Validation has just one sample: rank 1 participates only in epoch-end reductions.
        assert sum(sum(row) for row in history[-1]["val"]["confusion"]) == 64
    finally:
        execution.close()


def test_serial_trainer_checkpoint_and_world_size_resume_guard(tmp_path, monkeypatch):
    from dinov2_segmentation import train_joint
    torch.set_num_threads(1)
    monkeypatch.setattr(train_joint, "JointSegmentationSystem", TinySystem)
    _prepare_sources(tmp_path)
    with DistributedExecution("cpu") as execution:
        args = _trainer_args(tmp_path, "serial", tmp_path / "serial")
        train_joint.main(args=args, execution=execution)
    history = json.loads((tmp_path / "serial" / "history.json").read_text())
    assert len(history) == 3
    checkpoint = tmp_path / "serial" / "checkpoint_last.pt"
    with DistributedExecution("cpu") as execution:
        # A new output with identical execution can resume all optimizer/schedule state.
        resumed = _trainer_args(tmp_path, "serial", tmp_path / "resumed", ["--resume", str(checkpoint)])
        train_joint.main(args=resumed, execution=execution)
        assert (tmp_path / "resumed" / "complete").exists()
        assert (tmp_path / "resumed" / "checkpoint_last.pt").is_file()
        assert json.loads((tmp_path / "resumed" / "history.json").read_text()) == history
    with DistributedExecution("cpu") as execution:
        args = _trainer_args(tmp_path, "serial", tmp_path / "mismatch", ["--resume", str(checkpoint)])
        args.execution_mode = "ddp"
        with pytest.raises(ValueError, match="Resume execution"):
            train_joint.main(args=args, execution=execution)
    with DistributedExecution("cpu") as execution:
        args = _trainer_args(tmp_path, "serial", tmp_path / "warmstart", ["--init-checkpoint", str(checkpoint)])
        train_joint.main(args=args, execution=execution)
        assert (tmp_path / "warmstart" / "complete").exists()
    with DistributedExecution("cpu") as execution:
        args = _trainer_args(tmp_path, "serial", tmp_path / "warmstart_resume", [
            "--resume", str(tmp_path / "warmstart" / "checkpoint_last.pt"),
        ])
        train_joint.main(args=args, execution=execution)
        manifest = json.loads((tmp_path / "warmstart_resume" / "run_manifest.json").read_text())
        assert manifest["initialization_checkpoint"] == str(checkpoint)


def test_two_rank_trainer_all_phases_and_empty_validation_rank(tmp_path):
    _prepare_sources(tmp_path)
    mp.spawn(_worker, args=(str(tmp_path), (tmp_path / "rendezvous").as_uri()), nprocs=2, join=True)
