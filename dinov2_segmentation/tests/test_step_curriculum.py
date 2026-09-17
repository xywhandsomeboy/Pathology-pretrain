"""Regression checks for optimizer-step gradual unfreezing.

The production epoch contains millions of patches, so all boundaries in this
module are intentionally compressed to steps 1/2/3.  This exercises every
transition inside one epoch, including the DDP reducer rebuild that is easy to
miss when phase changes no longer happen at epoch boundaries.
"""

from __future__ import annotations

import json
from datetime import timedelta
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from dinov2_segmentation.distributed_execution import DistributedExecution
from dinov2_segmentation.data.joint_dataset import JointPatchSegmentationDataset
from dinov2_segmentation.joint_optim import WarmupCosineScheduler, build_joint_adamw
from dinov2_segmentation.tests.test_parallel_training import (
    TinyGNN,
    TinySystem,
    _prepare_sources,
)


class _FourBlockBackbone(nn.Module):
    """Small ViT-shaped backbone with enough blocks to distinguish top-2/top-4."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(4)])
        self.norm = nn.LayerNorm(3)

    def forward(self, images):
        values = images.mean((2, 3))
        for block in self.blocks:
            values = torch.tanh(block(values))
        return self.norm(values)


class _FourBlockStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FourBlockBackbone()
        self.spatial_agg = nn.Linear(3, 3)
        self.local_spatial_agg = nn.Linear(3, 3)
        self.local_crop_fusion = nn.Linear(3, 3)
        self.node_fusion = nn.Linear(9, 3)

    def forward(self, images):
        dense = self.backbone(images)
        nodes = self.node_fusion(
            torch.cat(
                (
                    dense,
                    self.spatial_agg(dense),
                    self.local_crop_fusion(self.local_spatial_agg(dense)),
                ),
                dim=1,
            )
        )
        return nodes, dense


class _CurriculumSystem(TinySystem):
    model_version = "step_curriculum_test_v1"

    def __init__(self, **kwargs):
        del kwargs
        nn.Module.__init__(self)
        self.stage1 = _FourBlockStage1()
        self.stage2 = TinyGNN()
        self.decoder = nn.Conv2d(9, 2, 1)
        self.stage2_runtime = SimpleNamespace(
            context_edge_mode="distance", num_layers=1, use_edge_attr=False
        )


class _VisitRecordingDataset(JointPatchSegmentationDataset):
    """Record only training-row reads to audit a cross-mode batch cursor."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        log_path = os.environ.get("DINOV2_TEST_VISIT_LOG")
        phase = os.environ.get("DINOV2_TEST_VISIT_PHASE")
        # The fixture has ten training rows and one validation row. Avoid
        # counting validation reads in the train-cursor assertion.
        if log_path and phase and len(self) > 1:
            with Path(log_path).open("a", encoding="utf-8") as stream:
                stream.write(f"{phase}:{int(index)}\n")
        return sample


def _trainer_args(root: Path, mode: str, output: Path, extra=()):
    from dinov2_segmentation.train_joint_parallel import parse_args

    return parse_args(
        [
            "--execution-mode",
            mode,
            "--device",
            "cpu",
            "--decoder-version",
            "v1",
            "--train-manifest",
            str(root / "train.csv"),
            "--val-manifest",
            str(root / "valid.csv"),
            "--graph-dir",
            str(root / "graphs"),
            "--stage1-config",
            str(root / "dummy"),
            "--stage1-checkpoint",
            str(root / "dummy"),
            "--stage2-config",
            str(root / "dummy"),
            "--stage2-checkpoint",
            str(root / "dummy"),
            "--output-dir",
            str(output),
            "--image-size",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--gradient-accumulation",
            "1",
            "--warmup-steps",
            "1",
            "--decoder-only-steps",
            "1",
            "--stage1-partial-unfreeze-step",
            "2",
            "--stage1-partial-unfreeze-blocks",
            "2",
            "--stage1-final-unfreeze-step",
            "3",
            "--stage1-final-unfreeze-blocks",
            "4",
            "--checkpoint-interval-steps",
            "2",
            "--max-train-batches",
            "5",
            "--probability-metric-bins",
            "16",
            *extra,
        ]
    )


def _optimizer_group_step(checkpoint: dict, group_prefix: str) -> int:
    """Return the Adam update count for a representative parameter group."""

    for group in checkpoint["optimizer"]["param_groups"]:
        if str(group["group_name"]).startswith(group_prefix):
            parameter_state = checkpoint["optimizer"]["state"].get(group["params"][0])
            if parameter_state is not None:
                return int(parameter_state["step"])
    return 0


def test_step_schedule_cli_and_four_exact_boundaries(tmp_path):
    from dinov2_segmentation.train_joint import _training_phase

    args = _trainer_args(tmp_path, "serial", tmp_path / "unused")
    assert args.warmup_steps == 1
    assert args.decoder_only_steps == 1
    assert args.stage1_partial_unfreeze_step == 2
    assert args.stage1_partial_unfreeze_blocks == 2
    assert args.stage1_final_unfreeze_step == 3
    assert args.stage1_final_unfreeze_blocks == 4
    assert args.checkpoint_interval_steps == 2

    system = _CurriculumSystem()
    optimizer, _ = build_joint_adamw(system)
    scheduler = WarmupCosineScheduler(
        optimizer, total_steps=5, warmup_steps=1, min_ratio=0.01
    )
    phases = [
        _training_phase(system, optimizer, scheduler, step, args, epoch=0)["name"]
        for step in range(4)
    ]
    assert phases == [
        "decoder_only",
        "adapters_and_decoder",
        "top2_backbone_joint",
        "top4_backbone_joint",
    ]

    # Step 2 enables only the two highest transformer blocks; step 3 enables
    # all four.  This guards against treating the two thresholds as aliases.
    _training_phase(system, optimizer, scheduler, 2, args, epoch=0)
    assert [any(parameter.requires_grad for parameter in block.parameters())
            for block in system.stage1.backbone.blocks] == [False, False, True, True]
    _training_phase(system, optimizer, scheduler, 3, args, epoch=0)
    assert all(
        parameter.requires_grad
        for block in system.stage1.backbone.blocks
        for parameter in block.parameters()
    )


class _StopAfterProgressCheckpoint(RuntimeError):
    pass


@pytest.mark.parametrize('resume_interval', [2, 1])
def test_progress_checkpoint_restores_curriculum_step_mid_epoch(tmp_path, monkeypatch, resume_interval):
    from dinov2_segmentation import train_joint

    torch.set_num_threads(1)
    _prepare_sources(tmp_path, train_count=5)
    monkeypatch.setattr(train_joint, "JointSegmentationSystem", _CurriculumSystem)
    args = _trainer_args(tmp_path, "serial", tmp_path / "serial")
    real_save = train_joint._atomic_torch_save

    def stop_after_first_progress_checkpoint(state, path):
        real_save(state, path)
        if Path(path).name == "checkpoint_progress.pt":
            raise _StopAfterProgressCheckpoint

    monkeypatch.setattr(
        train_joint, "_atomic_torch_save", stop_after_first_progress_checkpoint
    )
    with DistributedExecution("cpu") as execution:
        with pytest.raises(_StopAfterProgressCheckpoint):
            train_joint.main(args=args, execution=execution)

    progress_path = tmp_path / "serial" / "checkpoint_progress.pt"
    progress = torch.load(progress_path, map_location="cpu", weights_only=False)
    assert progress["epoch"] == 0
    assert progress["epoch_complete"] is False
    assert progress["next_batch_index"] == 2
    assert progress["curriculum_step"] == 2
    assert isinstance(progress["train_progress"], dict)

    monkeypatch.setattr(train_joint, "_atomic_torch_save", real_save)
    resumed = _trainer_args(
        tmp_path,
        "serial",
        tmp_path / "serial",
        ("--resume", str(progress_path)),
    )
    resumed.checkpoint_interval_steps = resume_interval
    resumed.retain_progress_checkpoints = True
    resumed.monitor_interval_steps = 2
    with DistributedExecution("cpu") as execution:
        train_joint.main(args=resumed, execution=execution)

    checkpoint = torch.load(
        tmp_path / "serial" / "checkpoint_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch_complete"] is True
    assert checkpoint["curriculum_step"] == 5
    history = json.loads((tmp_path / "serial" / "history.json").read_text())
    assert len(history) == 1
    assert history[0]["training_phase"] == "top4_backbone_joint"
    # No optimizer update is replayed or skipped across the mid-epoch resume.
    assert _optimizer_group_step(checkpoint, "decoder_v1_") == 5
    assert _optimizer_group_step(checkpoint, "stage2_gatv2_") == 4
    assert _optimizer_group_step(checkpoint, "stage1_backbone_layer_01_") == 2
    assert _optimizer_group_step(checkpoint, "stage1_backbone_layer_03_") == 3


def _ddp_curriculum_worker(rank: int, root_string: str, rendezvous: str) -> None:
    torch.set_num_threads(1)
    root = Path(root_string)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    execution = DistributedExecution(
        "cpu", rank=rank, world_size=2, owns_process_group=True
    )
    try:
        from dinov2_segmentation import train_joint

        train_joint.JointSegmentationSystem = _CurriculumSystem
        args = _trainer_args(root, "ddp", root / "ddp")
        train_joint.main(args=args, execution=execution)
    finally:
        execution.close()


def _ddp_progress_migration_worker(
    rank: int,
    root_string: str,
    rendezvous: str,
    source_checkpoint: str,
) -> None:
    torch.set_num_threads(1)
    root = Path(root_string)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    execution = DistributedExecution(
        "cpu", rank=rank, world_size=2, owns_process_group=True
    )
    try:
        from dinov2_segmentation import train_joint

        train_joint.JointSegmentationSystem = _CurriculumSystem
        train_joint.JointPatchSegmentationDataset = _VisitRecordingDataset
        args = _trainer_args(
            root,
            "ddp",
            root / "migrated_ddp",
            ("--migrate-resume", source_checkpoint),
        )
        train_joint.main(args=args, execution=execution)
    finally:
        execution.close()


def test_two_rank_ddp_rebuilds_reducer_for_all_phases_in_one_epoch(tmp_path):
    _prepare_sources(tmp_path, train_count=10)
    mp.spawn(
        _ddp_curriculum_worker,
        args=(str(tmp_path), (tmp_path / "rendezvous").as_uri()),
        nprocs=2,
        join=True,
    )
    checkpoint = torch.load(
        tmp_path / "ddp" / "checkpoint_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["curriculum_step"] == 5
    assert checkpoint["epoch_complete"] is True
    assert _optimizer_group_step(checkpoint, "decoder_v1_") == 5
    assert _optimizer_group_step(checkpoint, "stage2_gatv2_") == 4
    assert _optimizer_group_step(checkpoint, "stage1_backbone_layer_01_") == 2
    assert _optimizer_group_step(checkpoint, "stage1_backbone_layer_03_") == 3


def test_serial_progress_checkpoint_migrates_mid_epoch_to_two_rank_ddp(
    tmp_path, monkeypatch
):
    """A serial batch cursor remains exact after switching to two DDP ranks."""

    from dinov2_segmentation import train_joint

    torch.set_num_threads(1)
    _prepare_sources(tmp_path, train_count=10)
    visits = tmp_path / "training_visits.log"
    monkeypatch.setenv("DINOV2_TEST_VISIT_LOG", str(visits))
    monkeypatch.setenv("DINOV2_TEST_VISIT_PHASE", "serial")
    monkeypatch.setattr(train_joint, "JointSegmentationSystem", _CurriculumSystem)
    monkeypatch.setattr(
        train_joint, "JointPatchSegmentationDataset", _VisitRecordingDataset
    )
    serial_args = _trainer_args(
        tmp_path,
        "serial",
        tmp_path / "serial_progress",
        ("--batch-size", "2"),
    )
    real_save = train_joint._atomic_torch_save

    def stop_at_first_progress_checkpoint(state, path):
        real_save(state, path)
        if Path(path).name == "checkpoint_progress.pt":
            raise _StopAfterProgressCheckpoint

    monkeypatch.setattr(
        train_joint, "_atomic_torch_save", stop_at_first_progress_checkpoint
    )
    with DistributedExecution("cpu") as execution:
        with pytest.raises(_StopAfterProgressCheckpoint):
            train_joint.main(args=serial_args, execution=execution)

    source_path = tmp_path / "serial_progress" / "checkpoint_progress.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    assert source["epoch"] == 0
    assert source["epoch_complete"] is False
    assert source["next_batch_index"] == 2
    assert source["train_progress"]["samples"] == 4
    assert source["curriculum_step"] == 2
    assert source["scheduler"]["current_step"] == 2

    monkeypatch.setattr(train_joint, "_atomic_torch_save", real_save)
    monkeypatch.setenv("DINOV2_TEST_VISIT_PHASE", "ddp")
    mp.spawn(
        _ddp_progress_migration_worker,
        args=(
            str(tmp_path),
            (tmp_path / "migration_rendezvous").as_uri(),
            str(source_path),
        ),
        nprocs=2,
        join=True,
    )

    completed = torch.load(
        tmp_path / "migrated_ddp" / "checkpoint_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert completed["epoch"] == 0
    assert completed["epoch_complete"] is True
    assert completed["curriculum_step"] == 5
    # WarmupCosineScheduler stores its final valid zero-based position.
    assert completed["scheduler"]["current_step"] == 4
    assert _optimizer_group_step(completed, "decoder_v1_") == 5
    assert _optimizer_group_step(completed, "stage2_gatv2_") == 4
    assert _optimizer_group_step(completed, "stage1_backbone_layer_01_") == 2
    assert _optimizer_group_step(completed, "stage1_backbone_layer_03_") == 3

    history = json.loads(
        (tmp_path / "migrated_ddp" / "history.json").read_text()
    )
    assert len(history) == 1
    assert sum(sum(row) for row in history[0]["train"]["confusion"]) == 10 * 64
    assert [
        transition["to"] for transition in history[0]["phase_transitions"]
    ] == [
        "decoder_only",
        "adapters_and_decoder",
        "top2_backbone_joint",
        "top4_backbone_joint",
    ]
    migration = completed["run_manifest"]["migration"]
    assert migration["kind"] == "serial_to_ddp_step_checkpoint"
    assert migration["source_epoch_complete"] is False
    assert migration["source_next_batch_index"] == 2
    assert migration["scheduler_remap"]["policy"] == (
        "absolute_optimizer_update_position"
    )
    assert migration["curriculum_remap"] == {
        "policy": "preserved_from_step_checkpoint",
        "source": 2,
        "target": 2,
    }

    recorded = [line.split(":", 1) for line in visits.read_text().splitlines()]
    serial_indices = [int(index) for phase, index in recorded if phase == "serial"]
    ddp_indices = [int(index) for phase, index in recorded if phase == "ddp"]
    assert len(serial_indices) == 4
    assert len(ddp_indices) == 6
    assert sorted(serial_indices + ddp_indices) == list(range(10))
