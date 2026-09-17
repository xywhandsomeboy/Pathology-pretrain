"""Numerical/gradient equivalence against the original full DINO path."""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from PIL import Image

from dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2_segmentation.joint_model import (
    TrainableStage1, SpatialPatchAggregator, LocalCropFusion, NodeFeatureFusion)
from dinov2_segmentation.joint_optim import _vit_blocks
from dinov2_segmentation.consistent_features import ConsistentFeatureProvider
from dinov2_segmentation.frozen_prefix import PrefixCachedFeatureProvider, FrozenDinoPrefix


def make_stage(root, *, chunks=2, registers=2, dropout=0.):
    torch.manual_seed(42)
    s = TrainableStage1.__new__(TrainableStage1)
    nn.Module.__init__(s)
    s.backbone = DinoVisionTransformer(img_size=16, patch_size=4, embed_dim=24,
        depth=4, num_heads=3, block_chunks=chunks, num_register_tokens=registers,
        drop_path_rate=dropout, drop_path_uniform=True)
    s.image_size = 16; s.local_size = 8; s.num_local_crops = 4; s.embed_dim = 24
    s.spatial_agg = SpatialPatchAggregator(24, 8, 24)
    s.local_spatial_agg = SpatialPatchAggregator(24, 8, 24)
    s.local_crop_fusion = LocalCropFusion(4, 24, 48)
    s.node_fusion = NodeFeatureFusion(24, 24, 48)
    s.checkpoint_path = str(root / "source.pt")
    if not Path(s.checkpoint_path).exists():
        Path(s.checkpoint_path).write_bytes(b"source")
    s.backbone.requires_grad_(False).eval()
    for b in _vit_blocks(s.backbone)[-2:]:
        b.requires_grad_(True).train()
    s.backbone.norm.requires_grad_(True)
    return s


def provider_pair(root, **kwargs):
    stage = make_stage(root, **kwargs)
    reference = copy.deepcopy(stage)
    folder = root / 'slides/s/images'; folder.mkdir(parents=True)
    for i, patch in enumerate(['a', 'b', 'c']):
        Image.new('RGB', (16, 16), (30+30*i, 70, 120)).save(folder/f'{patch}.jpg')
    args = dict(image_root=root/'slides', cache_root=root/'cache', chunk_size=2, stage='dino')
    return (stage, reference, PrefixCachedFeatureProvider(stage, trainable_blocks=2, min_free_gib=0, **args),
            ConsistentFeatureProvider(reference, **args))


@pytest.mark.parametrize('chunks,registers,dropout', [(0, 0, 0.), (2, 2, 0.), (2, 2, .3)])
@pytest.mark.parametrize('amp', [False, True])
def test_outputs_gradients_and_optimizer_updates_match_full_path(tmp_path, chunks, registers, dropout, amp):
    a, b, cached, full = provider_pair(tmp_path, chunks=chunks, registers=registers, dropout=dropout)
    opts = [torch.optim.AdamW(s.parameters(), lr=1e-3) for s in [a, b]]
    prefix_calls = []
    hook = _vit_blocks(a.backbone)[0].register_forward_hook(lambda *args: prefix_calls.append(1))
    for iteration in range(2):
        for opt in opts: opt.zero_grad(set_to_none=True)
        torch.manual_seed(50+iteration)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
            actual = cached.features('s', ['a', 'b', 'c'], 'cpu')
            loss = actual[1:].square().sum()  # Neighbor-only supervision must propagate.
        loss.backward()
        torch.manual_seed(50+iteration)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
            expected = full.features('s', ['a', 'b', 'c'], 'cpu')
            loss = expected[1:].square().sum()
        loss.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            if p.grad is None: assert q.grad is None, name
            else: torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
        for opt in opts: opt.step()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        if iteration == 0: calls_after_fill = len(prefix_calls)
        else: assert len(prefix_calls) == calls_after_fill
    assert cached.prefix_hits == 3 and cached.prefix_misses == 3
    hook.remove()


def test_cache_guards_and_new_weights_select_different_namespace(tmp_path):
    a, _, cached, _ = provider_pair(tmp_path)
    cached.features('s', ['a'], 'cpu')
    digest = cached.prefix.digest
    # Trainable suffix changes are allowed and must change the output.
    before = cached.features('s', ['a'], 'cpu').detach()
    with torch.no_grad(): a.node_fusion.norm.bias.add_(1)
    after = cached.features('s', ['a'], 'cpu').detach()
    assert not torch.equal(before, after)
    prefix_block = _vit_blocks(a.backbone)[0]
    prefix_block.requires_grad_(True)
    with pytest.raises(RuntimeError, match='trainable'): cached.features('s', ['a'], 'cpu')
    prefix_block.requires_grad_(False).train()
    with pytest.raises(RuntimeError, match='eval'): cached.features('s', ['a'], 'cpu')
    prefix_block.eval()
    with torch.no_grad(): next(prefix_block.parameters()).add_(.1)
    with pytest.raises(RuntimeError, match='changed'): cached.features('s', ['a'], 'cpu')
    assert FrozenDinoPrefix(a, 2).digest != digest


def test_eval_reordering_and_partial_hits_keep_correct_patch_identity(tmp_path):
    a, b, cached, full = provider_pair(tmp_path)
    a.eval(); b.eval()
    with torch.no_grad():
        cached.features('s', ['b'], 'cpu')
        actual = cached.features('s', ['c', 'b', 'a'], 'cpu')
        expected = full.features('s', ['c', 'b', 'a'], 'cpu')
        torch.testing.assert_close(actual, expected)
    assert cached.prefix_hits == 1 and cached.prefix_misses == 3


def test_prefix_cache_separates_precision_and_rejects_corruption(tmp_path):
    a, _, cached, _ = provider_pair(tmp_path)
    a.eval()
    with torch.no_grad(): cached.features('s', ['a'], 'cpu')
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        cached.features('s', ['a'], 'cpu')
    files = list((cached.root/'frozen_prefix_v1').rglob('a.pt'))
    assert len(files) == 2
    fp = next(p for p in files if 'float32' in p.parts)
    torch.save((torch.ones(2),), fp)
    with torch.no_grad(), pytest.raises(ValueError, match='Invalid'):
        cached.features('s', ['a'], 'cpu')


def test_deterministic_protocol_has_separate_cache(tmp_path):
    a, _, cached, _ = provider_pair(tmp_path)
    a.eval()
    original = torch.backends.cudnn.deterministic
    try:
        with torch.no_grad(): cached.features('s', ['a'], 'cpu')
        torch.backends.cudnn.deterministic = not original
        with torch.no_grad(): cached.features('s', ['a'], 'cpu')
        assert len(list((cached.root/'frozen_prefix_v1').rglob('a.pt'))) == 2
    finally:
        torch.backends.cudnn.deterministic = original


def test_opt_in_configuration_and_validation_provider(tmp_path):
    from types import SimpleNamespace
    from dinov2_segmentation.consistent_features import configure_repository
    a, _, cached, _ = provider_pair(tmp_path)
    cfg = dict(graph_feature_policy='staged_consistent', stage1_partial_unfreeze_step=60000,
        stage1_final_unfreeze_step=100000, stage1_final_unfreeze_blocks=2,
        node_image_root=tmp_path/'slides', raw_feature_cache=tmp_path/'cache', neighbor_chunk_size=2)
    repo = SimpleNamespace()
    configure_repository(repo, SimpleNamespace(stage1=a), cfg, step=100000)
    assert type(repo.feature_provider) is ConsistentFeatureProvider
    cfg['cache_frozen_dino_prefix'] = True
    configure_repository(repo, SimpleNamespace(stage1=a), cfg, step=100000)
    assert type(repo.feature_provider) is PrefixCachedFeatureProvider


def test_resume_only_allows_prefix_optimization_for_consistent_policy():
    from dinov2_segmentation.checkpoint_retention import validate_resume_configuration
    source = dict(graph_feature_policy='staged_consistent', batch_size=32)
    target = dict(source, cache_frozen_dino_prefix=True)
    changes = validate_resume_configuration(source, target)
    assert changes['cache_frozen_dino_prefix'] == {'source': False, 'target': True}
    with pytest.raises(ValueError):
        validate_resume_configuration(source, dict(target, batch_size=64))
    with pytest.raises(ValueError):
        validate_resume_configuration({'batch_size': 32}, {'batch_size': 32, 'cache_frozen_dino_prefix': True})


class PrefixForward(nn.Module):
    def __init__(self, stage, provider):
        super().__init__()
        self.stage = stage
        self.provider = provider

    def forward(self, patches):
        return self.provider.features('s', patches, 'cpu')


def prefix_ddp_worker(rank, root, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        stage = make_stage(root)
        options = dict(image_root=root/'slides', cache_root=root/'shared_cache',
                       chunk_size=2, stage='dino')
        cached = PrefixCachedFeatureProvider(stage, trainable_blocks=2, min_free_gib=0, **options)
        model = DistributedDataParallel(PrefixForward(stage, cached))
        # Both ranks request neighbor b, exercising concurrent atomic writes.
        patches = ['a', 'b'] if rank == 0 else ['b', 'c']
        reference = copy.deepcopy(stage)
        full = ConsistentFeatureProvider(reference, **options)
        opt = torch.optim.SGD(stage.parameters(), lr=.001)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            model(patches).square().mean().backward()
            # Average the two local objectives, including b twice.
            expected = (full.features('s', ['a', 'b'], 'cpu').square().mean()
                        + full.features('s', ['b', 'c'], 'cpu').square().mean())/2
            expected.backward()
            for (name, p), (_, q) in zip(stage.named_parameters(), reference.named_parameters()):
                if p.grad is None: assert q.grad is None
                else: torch.testing.assert_close(p.grad, q.grad, msg=name)
            opt.step()
            reference.load_state_dict(stage.state_dict())
        assert cached.prefix_hits >= 2
    finally:
        dist.destroy_process_group()


def test_prefix_two_rank_ddp_and_atomic_shared_cache(tmp_path):
    import torch.multiprocessing as mp
    provider_pair(tmp_path)
    mp.spawn(prefix_ddp_worker,
             args=(str(tmp_path), f'file://{tmp_path / "rdzv"}'), nprocs=2, join=True)


def test_low_space_recomputes_current_features_without_writing(tmp_path):
    a, b, cached, full = provider_pair(tmp_path)
    cached.minimum_free_bytes = 2**63
    for _ in range(2):
        torch.testing.assert_close(cached.features('s', ['a'], 'cpu'),
                                   full.features('s', ['a'], 'cpu'))
    assert cached.prefix_write_skips == 2 and cached.prefix_hits == 0
    assert not list((cached.root/'frozen_prefix_v1').rglob('*.pt'))
