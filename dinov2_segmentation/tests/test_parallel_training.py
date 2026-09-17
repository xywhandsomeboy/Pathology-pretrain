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
        "--warmup-steps", "2",
        "--decoder-only-steps", "1",
        "--stage1-partial-unfreeze-step", "2",
        "--stage1-partial-unfreeze-blocks", "1",
        "--stage1-final-unfreeze-step", "3",
        "--stage1-final-unfreeze-blocks", "1",
        "--checkpoint-interval-steps", "0",
        "--max-train-batches", "3", "--probability-metric-bins", "16", *extra,
    ])


def _prepare_sources(root, train_count=6):
    (root / "dummy").touch()
    graphs = root / "graphs"
    graphs.mkdir()
    for slide in ("training", "validation"):
        count = train_count if slide == "training" else 1
        edges = torch.cartesian_prod(torch.arange(count), torch.arange(count)).T.contiguous()
        torch.save(Data(
            x=torch.zeros(count, 3), edge_index=edges, slide_id=slide,
            patch_ids=[f"p{i}" for i in range(count)], edge_mode="distance",
        ), graphs / f"{slide}.pt")
    fields = ("slide_id", "patch_id", "image_path", "mask_path", "x", "y", "level")
    for name, slide, count in (
        ("train", "training", train_count),
        ("valid", "validation", 1),
    ):
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
        assert checkpoint["format_version"] == 2
        assert checkpoint["epoch_complete"] is True
        assert checkpoint["curriculum_step"] == 6
        assert checkpoint["execution"]["world_size"] == 2
        history = json.loads((root / "ddp" / "history.json").read_text())
        assert [row["training_phase"] for row in history] == [
            "adapters_and_decoder", "top4_backbone_joint", "top4_backbone_joint",
        ]
        assert [
            transition["to"]
            for row in history
            for transition in row["phase_transitions"]
        ] == [
            "decoder_only",
            "adapters_and_decoder",
            "top2_backbone_joint",
            "top4_backbone_joint",
        ]
        assert [(row["curriculum_step_start"], row["curriculum_step_end"])
                for row in history] == [(0, 2), (2, 4), (4, 6)]
        assert all("approx_pr_auc" not in row["train"] for row in history)
        assert all("approx_pr_auc" not in row["val"] for row in history[:-1])
        assert "approx_pr_auc" in history[-1]["val"]
        # Validation has just one sample: rank 1 participates only in epoch-end reductions.
        assert sum(sum(row) for row in history[-1]["val"]["confusion"]) == 64
    finally:
        execution.close()


_MIGRATION_OVERRIDES = (
    "--epochs", "3",
    "--gradient-accumulation", "2",
    "--max-train-batches", "0",
    "--warmup-steps", "3",
)


_LEGACY_SOURCE_PHASE_EMULATION = (
    # With three optimizer updates per epoch these boundaries reproduce the
    # retired 1-epoch decoder / 1-epoch adapter schedule before the checkpoint
    # is converted below into its historical schema.
    "--decoder-only-steps", "3",
    "--stage1-partial-unfreeze-step", "6",
    "--stage1-final-unfreeze-step", "8",
)


def _migration_worker(rank, root_string, rendezvous, source_checkpoint):
    torch.set_num_threads(1)
    root = Path(root_string)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    execution = DistributedExecution(
        "cpu", rank=rank, world_size=2, owns_process_group=True
    )
    try:
        from dinov2_segmentation import train_joint
        train_joint.JointSegmentationSystem = TinySystem
        args = _trainer_args(
            root,
            "ddp",
            root / "migrated",
            (
                "--batch-size", "1",
                *_MIGRATION_OVERRIDES,
                "--migrate-resume", source_checkpoint,
            ),
        )
        train_joint.main(args=args, execution=execution)
        resumed = _trainer_args(
            root,
            "ddp",
            root / "migrated_resume",
            (
                "--batch-size", "1",
                *_MIGRATION_OVERRIDES,
                "--resume", str(root / "migrated" / "checkpoint_last.pt"),
            ),
        )
        train_joint.main(args=resumed, execution=execution)
    finally:
        execution.close()


def _optimizer_step(checkpoint, group_name):
    group = next(
        group
        for group in checkpoint["optimizer"]["param_groups"]
        if group["group_name"] == group_name
    )
    state = checkpoint["optimizer"]["state"][group["params"][0]]
    return int(state["step"])


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


def test_migrate_resume_cli_guards(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="requires --execution-mode ddp"):
        _trainer_args(
            tmp_path,
            "serial",
            tmp_path / "serial",
            ("--migrate-resume", str(checkpoint)),
        )
    for other_flag in ("--resume", "--init-checkpoint"):
        with pytest.raises(ValueError, match="cannot be combined"):
            _trainer_args(
                tmp_path,
                "ddp",
                tmp_path / other_flag.removeprefix("--"),
                (
                    "--migrate-resume", str(checkpoint),
                    other_flag, str(checkpoint),
                ),
            )
    args = _trainer_args(
        tmp_path,
        "ddp",
        tmp_path / "valid",
        ("--migrate-resume", str(checkpoint)),
    )
    assert args.migrate_resume == checkpoint.resolve()


def test_migrate_serial_epoch_checkpoint_to_two_rank_ddp(tmp_path, monkeypatch):
    from dinov2_segmentation import train_joint

    class StopBeforeThirdEpoch(RuntimeError):
        pass

    torch.set_num_threads(1)
    monkeypatch.setattr(train_joint, "JointSegmentationSystem", TinySystem)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _prepare_sources(tmp_path, train_count=9)
    source_args = _trainer_args(
        tmp_path,
        "serial",
        tmp_path / "serial_source",
        (
            "--batch-size", "2",
            *_MIGRATION_OVERRIDES,
            *_LEGACY_SOURCE_PHASE_EMULATION,
        ),
    )
    original_run_epoch = train_joint._run_epoch
    training_epochs = 0

    def stop_after_two_epochs(*args, **kwargs):
        nonlocal training_epochs
        if kwargs.get("optimizer") is not None:
            if training_epochs == 2:
                raise StopBeforeThirdEpoch
            training_epochs += 1
        return original_run_epoch(*args, **kwargs)

    monkeypatch.setattr(train_joint, "_run_epoch", stop_after_two_epochs)
    with pytest.raises(StopBeforeThirdEpoch):
        # Deliberately exercise a legacy serial checkpoint without embedded
        # execution/history metadata, matching the original production jobs.
        train_joint.main(args=source_args)

    source_path = tmp_path / "serial_source" / "checkpoint_last.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_history = json.loads(
        (tmp_path / "serial_source" / "history.json").read_text()
    )
    # Turn the checkpoint produced by the shared modern test harness into the
    # exact schema of the already-running epoch-scheduled production jobs. The
    # emulated phase boundaries above ensure its optimizer state also matches
    # one decoder-only epoch followed by one adapter epoch.
    source["format_version"] = 1
    legacy_configuration = dict(source["configuration"])
    for name in (
        "warmup_steps",
        "decoder_only_steps",
        "stage1_partial_unfreeze_step",
        "stage1_partial_unfreeze_blocks",
        "stage1_final_unfreeze_step",
        "stage1_final_unfreeze_blocks",
        "checkpoint_interval_steps",
    ):
        legacy_configuration.pop(name)
    legacy_configuration.update(
        {
            "warmup_ratio": 0.34,
            "decoder_only_epochs": 1,
            "stage1_top_unfreeze_epoch": 2,
            "stage1_unfreeze_blocks": 1,
        }
    )
    source["configuration"] = legacy_configuration
    source["run_manifest"]["configuration"] = dict(legacy_configuration)
    for name in (
        "epoch_complete",
        "curriculum_step",
        "training_phase",
        "phase_transitions",
        "next_batch_index",
        "train_progress",
        "history",
        "execution",
    ):
        source.pop(name, None)
    torch.save(source, source_path)

    assert source["epoch"] == 1
    assert "execution" not in source
    assert "history" not in source
    assert source["scheduler"] == {
        **source["scheduler"],
        "total_steps": 9,
        "warmup_steps": 3,
        "current_step": 6,
    }
    assert _optimizer_step(source, "decoder_v1_decay") == 6
    assert _optimizer_step(source, "stage2_gatv2_decay") == 3
    assert source["gradient_audit"]["updates_observed"] == 3

    with DistributedExecution("cpu", world_size=2) as execution:
        bad_batch = _trainer_args(
            tmp_path,
            "ddp",
            tmp_path / "bad_batch",
            (
                "--batch-size", "2",
                *_MIGRATION_OVERRIDES,
                # The deliberately larger per-rank batch yields only three
                # target updates, so keep this invalid-geometry probe's four
                # phases inside that shortened schedule.
                "--decoder-only-steps", "0",
                "--stage1-partial-unfreeze-step", "1",
                "--stage1-final-unfreeze-step", "2",
                "--migrate-resume", str(source_path),
            ),
        )
        with pytest.raises(ValueError, match="effective batch sizes differ"):
            train_joint.main(args=bad_batch, execution=execution)
    with DistributedExecution("cpu", world_size=2) as execution:
        bad_seed = _trainer_args(
            tmp_path,
            "ddp",
            tmp_path / "bad_seed",
            (
                "--batch-size", "1",
                *_MIGRATION_OVERRIDES,
                "--seed", "43",
                "--migrate-resume", str(source_path),
            ),
        )
        with pytest.raises(ValueError, match="only permits batch geometry"):
            train_joint.main(args=bad_seed, execution=execution)

    mp.spawn(
        _migration_worker,
        args=(
            str(tmp_path),
            (tmp_path / "migration_rendezvous").as_uri(),
            str(source_path),
        ),
        nprocs=2,
        join=True,
    )
    migrated = torch.load(
        tmp_path / "migrated" / "checkpoint_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    history = json.loads((tmp_path / "migrated" / "history.json").read_text())
    assert migrated["epoch"] == 2
    assert migrated["format_version"] == 2
    assert migrated["epoch_complete"] is True
    assert migrated["curriculum_step"] == 4
    assert migrated["execution"]["mode"] == "ddp"
    assert migrated["execution"]["world_size"] == 2
    assert migrated["execution"]["effective_batch_size"] == 4
    assert migrated["execution"]["dropped_training_samples"] == 1
    assert history[:2] == source_history
    assert [row["training_phase"] for row in history] == [
        "decoder_only", "adapters_and_decoder", "top4_backbone_joint",
    ]
    assert migrated["scheduler"]["total_steps"] == 6
    assert migrated["scheduler"]["warmup_steps"] == 3
    assert migrated["scheduler"]["current_step"] == 5
    migration = migrated["run_manifest"]["migration"]
    assert migration["source_checkpoint"] == str(source_path.resolve())
    assert migration["source_execution"]["mode"] == "serial"
    assert migration["source_execution"]["world_size"] == 1
    assert migration["target_execution"]["world_size"] == 2
    assert migration["effective_batch_size"] == {"source": 4, "target": 4}
    assert migration["scheduler_remap"]["source"] == {
        "updates_per_epoch": 3,
        "total_steps": 9,
        "warmup_steps": 3,
        "current_step": 6,
    }
    assert migration["scheduler_remap"]["target"] == {
        "updates_per_epoch": 2,
        "total_steps": 6,
        "warmup_steps": 3,
        "current_step": 4,
    }
    assert migration["curriculum_remap"] == {
        "policy": "legacy_adapter_phase_completed",
        "source": None,
        "target": 2,
    }
    assert migration["learning_rate_discontinuity_expected"] is False
    assert _optimizer_step(migrated, "decoder_v1_decay") == 8
    assert _optimizer_step(migrated, "stage2_gatv2_decay") == 5
    assert migrated["gradient_audit"]["updates_observed"] == 5
    assert migrated["gradient_audit"]["complete"]
    assert torch.equal(
        migrated["model"]["stage2.unused_edge_encoder.weight"],
        source["model"]["stage2.unused_edge_encoder.weight"],
    )
    last_dice = history[-1]["val"]["tumor_dice"]
    assert migrated["best_dice"] == max(source["best_dice"], last_dice)
    assert migrated["early_stopping_best"] == max(
        source["early_stopping_best"], last_dice
    )
    resumed_manifest = json.loads(
        (tmp_path / "migrated_resume" / "run_manifest.json").read_text()
    )
    assert resumed_manifest["migration"] == migration
