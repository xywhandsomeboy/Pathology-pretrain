"""Real optimizer and DDP checks for the independently frozen feature producer."""
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from dinov2_segmentation import train_joint as t
from dinov2_segmentation.distributed_execution import DistributedExecution
from dinov2_segmentation.joint_optim import build_joint_adamw, WarmupCosineScheduler
from dinov2_segmentation.tests.test_step_curriculum import (
    _CurriculumSystem, _trainer_args, _optimizer_group_step, _prepare_sources,
    _StopAfterProgressCheckpoint,
)


def args_for(root, mode, output, extra=()):
    return _trainer_args(root, mode, output, (
        "--unfreeze-schedule", "separate_gnn_fusion", *extra,
    ))


def test_freeze_masks_modes_and_cache_policy(tmp_path):
    args = args_for(tmp_path, "serial", tmp_path / "out")
    system = _CurriculumSystem()
    # Verify frozen producer dropout is disabled as well as its gradients.
    system.stage1.dropout_probe = nn.Dropout(.5)
    optimizer, _ = build_joint_adamw(system)
    scheduler = WarmupCosineScheduler(optimizer, total_steps=5, warmup_steps=1)
    for step, name, fusion, dino, cache in [
        (0, "decoder_only", False, False, False),
        (1, "gnn_and_decoder", False, False, False),
        (2, "fusion_gnn_decoder", True, False, True),
        (3, "dino_joint", True, True, True),
    ]:
        phase = t._training_phase(system, optimizer, scheduler, step, args)
        t._set_runtime_modes(system, training=True, phase=phase)
        assert phase['name'] == name
        assert all(p.requires_grad == fusion for p in system.stage1.node_fusion.parameters())
        assert all(p.requires_grad == dino for p in system.stage1.backbone.parameters())
        assert all(p.requires_grad == (step >= 1) for p in system.stage2.parameters())
        assert system.stage1.dropout_probe.training == fusion
        assert system.stage1.backbone.blocks[-1].training == dino
        assert system.refresh_graph_memory == cache


def test_no_memory_write_before_fusion_and_refresh_after():
    graph = SimpleNamespace(x=torch.tensor([[1., 2.], [3., 4.]]))
    repository = SimpleNamespace(_get=lambda slide: (graph, {'a': 0, 'b': 1}))
    execution = DistributedExecution('cpu')
    features = torch.tensor([[9., 8.]])
    original = graph.x.clone()
    execution.synchronize_graph_memory(repository, features, ['s'], ['a'], refresh=False)
    assert torch.equal(graph.x, original)
    execution.synchronize_graph_memory(repository, features, ['s'], ['a'], refresh=True)
    assert torch.equal(graph.x[0], features[0])
    assert torch.equal(graph.x[1], original[1])


def assert_completed(path):
    c = torch.load(path, map_location='cpu', weights_only=False)
    assert c['curriculum_step'] == 5
    assert c['configuration']['unfreeze_schedule'] == 'separate_gnn_fusion'
    assert c['training_phase'] == 'dino_joint'
    for group, count in [('decoder_v1_', 5), ('stage2_gatv2_', 4),
                         ('stage1_node_fusion_', 3), ('stage1_backbone_layer_01_', 2)]:
        assert _optimizer_group_step(c, group) == count, group
    assert c['run_manifest']['gradual_unfreezing']['phases'] == [
        'decoder_only', 'gnn_and_decoder', 'fusion_gnn_decoder', 'dino_joint']
    return c


def test_serial_resume_preserves_new_phase_and_rejects_legacy(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    _prepare_sources(tmp_path, train_count=5)
    monkeypatch.setattr(t, 'JointSegmentationSystem', _CurriculumSystem)
    args = args_for(tmp_path, 'serial', tmp_path / 'serial')
    real_save = t._atomic_torch_save
    def stop(state, path):
        real_save(state, path)
        if Path(path).name == 'checkpoint_progress.pt':
            raise _StopAfterProgressCheckpoint
    monkeypatch.setattr(t, '_atomic_torch_save', stop)
    with DistributedExecution('cpu') as execution:
        with pytest.raises(_StopAfterProgressCheckpoint):
            t.main(args=args, execution=execution)
    monkeypatch.setattr(t, '_atomic_torch_save', real_save)
    checkpoint = tmp_path / 'serial/checkpoint_progress.pt'
    args.resume = checkpoint
    with DistributedExecution('cpu') as execution:
        t.main(args=args, execution=execution)
    c = assert_completed(tmp_path / 'serial/checkpoint_last.pt')
    from dinov2_segmentation.checkpoint_retention import validate_resume_configuration
    legacy = dict(c['configuration']); legacy.pop('unfreeze_schedule')
    with pytest.raises(ValueError):
        validate_resume_configuration(legacy, c['configuration'])


def worker(rank, root_string, rendezvous):
    torch.set_num_threads(1)
    root = Path(root_string)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    execution = DistributedExecution('cpu', rank=rank, world_size=2, owns_process_group=True)
    try:
        t.JointSegmentationSystem = _CurriculumSystem
        t.main(args=args_for(root, 'ddp', root / 'ddp'), execution=execution)
    finally:
        execution.close()


def test_ddp_reducer_transitions_through_separate_gnn_fusion(tmp_path):
    _prepare_sources(tmp_path, train_count=10)
    mp.spawn(worker, args=(str(tmp_path), (tmp_path / 'rendezvous').as_uri()), nprocs=2, join=True)
    assert_completed(tmp_path / 'ddp/checkpoint_last.pt')
