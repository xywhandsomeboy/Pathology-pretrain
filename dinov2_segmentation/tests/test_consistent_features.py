from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from PIL import Image
from torch_geometric.data import Data

from dinov2_segmentation.consistent_features import ConsistentFeatureProvider
from dinov2_segmentation.joint_graph import JointGraphRepository
from dinov2_segmentation.distributed_execution import _JointForward
from dinov2_segmentation.distributed_execution import DistributedExecution
import torch.distributed as dist
import torch.multiprocessing as mp
from dinov2_segmentation.joint_model import TrainableStage1, SpatialPatchAggregator, LocalCropFusion, NodeFeatureFusion


class Stage(nn.Module):
    def __init__(self, root):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.fusion = nn.Linear(12, 4)
        self.checkpoint_path = str(root / "source.pt")
        Path(self.checkpoint_path).write_bytes(b"immutable-source")
        self.image_size = self.local_size = 4
        self.num_local_crops = 1
        self.calls = 0

    def extract_raw(self, images):
        self.calls += 1
        x = self.backbone(images.mean((2, 3)))
        return x, x[:, None], x[:, None, None]

    def fuse_raw(self, cls, dense, local):
        return self.fusion(torch.cat([cls, dense[:, 0], local[:, 0, 0]], 1))

    def forward(self, images):
        raw = self.extract_raw(images)
        return self.fuse_raw(*raw), raw[1]


def setup_provider(tmp_path):
    stage = Stage(tmp_path)
    stage.backbone.requires_grad_(False).eval()
    folder = tmp_path / "slides/s/images"
    folder.mkdir(parents=True)
    for i, p in enumerate(["a", "b", "c"]):
        Image.new("RGB", (4, 4), (20 + 30*i, 50, 90)).save(folder / f"{p}.jpg")
    provider = ConsistentFeatureProvider(stage, tmp_path / "slides", tmp_path / "cache", chunk_size=2, stage="fusion")
    return stage, provider


def test_fusion_uses_fixed_raw_but_current_weights_for_all_nodes(tmp_path):
    stage, provider = setup_provider(tmp_path)
    first = provider.features("s", ["a", "b", "c"], "cpu").detach()
    calls = stage.calls
    with torch.no_grad():
        stage.fusion.weight.add_(.2)
    updated = provider.features("s", ["a", "b", "c"], "cpu")
    assert stage.calls == calls
    assert not torch.allclose(first, updated)
    images = torch.stack([provider._image("s", p) for p in ["a", "b", "c"]])
    expected = stage(images)[0]
    torch.testing.assert_close(updated, expected)
    updated[1:].square().sum().backward()
    assert stage.fusion.weight.grad.abs().sum() > 0
    assert stage.backbone.weight.grad is None


def test_dino_updates_bypass_raw_cache_and_receive_neighbor_gradients(tmp_path):
    stage, provider = setup_provider(tmp_path)
    provider.features("s", ["a", "b"], "cpu")
    cache_files = {p: p.stat().st_mtime_ns for p in provider.root.rglob("*.pt")}
    provider.stage = "dino"
    stage.backbone.requires_grad_(True)
    first = provider.features("s", ["a", "b"], "cpu")
    with torch.no_grad():
        stage.backbone.weight.add_(.3)
    updated = provider.features("s", ["a", "b"], "cpu")
    assert not torch.allclose(first, updated)
    updated[1].square().sum().backward()
    assert stage.backbone.weight.grad.abs().sum() > 0
    assert {p: p.stat().st_mtime_ns for p in provider.root.rglob("*.pt")} == cache_files


class GNN(nn.Module):
    def forward(self, x, edges, edge_attr, **kwargs):
        return x + torch.zeros_like(x).index_add(0, edges[1], x[edges[0]])


def repository(tmp_path, provider):
    graphs = tmp_path / "graphs"; graphs.mkdir()
    torch.save(Data(x=torch.arange(12).reshape(3, 4).float(),
                    edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                    patch_ids=["a", "b", "c"], slide_id="s", edge_mode="distance"), graphs / "s.pt")
    repo = JointGraphRepository(graphs, expected_edge_mode="distance")
    repo.feature_provider = provider
    return repo


@pytest.mark.parametrize("phase", ["frozen", "fusion", "dino"])
def test_context_is_independent_of_online_batch_neighbors(tmp_path, phase):
    stage, provider = setup_provider(tmp_path)
    provider.stage = phase
    repo = repository(tmp_path, provider)
    original = repo._get("s")[0].x.clone()
    def context(patches, online):
        return repo.contextualize(GNN(), online, ["s"]*len(patches), patches,
                                 num_hops=1, use_edge_attr=False, update_memory=True)
    one = context(["a"], torch.randn(1, 4))
    both = context(["a", "b"], torch.randn(2, 4)*100)
    torch.testing.assert_close(one[0], both[0])
    torch.testing.assert_close(original, repo._get("s")[0].x)


def test_ddp_forward_does_not_gather_or_overwrite_consistent_features(tmp_path):
    stage, provider = setup_provider(tmp_path)
    repo = repository(tmp_path, provider)
    class System(nn.Module):
        def __init__(self):
            super().__init__(); self.stage1 = stage; self.stage2 = GNN()
            self.stage2_runtime = SimpleNamespace(num_layers=1, use_edge_attr=False)
        def decode(self, images, dense, context): return context
    forward = _JointForward(System(), SimpleNamespace(distributed=True))
    value = forward(torch.ones(1, 3, 4, 4), repo, ["s"], ["a"], True)
    value.square().sum().backward()
    assert stage.fusion.weight.grad.abs().sum() > 0


def test_split_raw_and_fusion_preserves_original_stage1_calculation():
    torch.manual_seed(1)
    class Backbone(nn.Module):
        def forward(self, images, **kwargs):
            tokens = images.flatten(2).transpose(1, 2)
            return {"x_norm_clstoken": tokens.mean(1), "x_norm_patchtokens": tokens}
    stage = TrainableStage1.__new__(TrainableStage1)
    nn.Module.__init__(stage)
    stage.backbone = Backbone()
    stage.image_size = 4; stage.local_size = 2
    stage.num_local_crops = 5; stage.embed_dim = 3
    stage.spatial_agg = SpatialPatchAggregator(3, 4, 3)
    stage.local_spatial_agg = SpatialPatchAggregator(3, 4, 3)
    stage.local_crop_fusion = LocalCropFusion(5, 3, 6)
    stage.node_fusion = NodeFeatureFusion(3, 3, 6)
    images = torch.randn(2, 3, 4, 4)
    raw = stage.backbone(images)
    local = stage.backbone(stage._local_crops(images))
    expected = stage.node_fusion(raw["x_norm_clstoken"],
        stage.spatial_agg(raw["x_norm_patchtokens"]),
        stage.local_crop_fusion(stage.local_spatial_agg(local["x_norm_patchtokens"]).view(5, 2, -1)))
    actual, dense = stage(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(dense, raw["x_norm_patchtokens"], rtol=0, atol=0)


class DistributedSystem(nn.Module):
    def __init__(self, stage):
        super().__init__(); self.stage1 = stage; self.stage2 = GNN()
        self.stage2_runtime = SimpleNamespace(num_layers=1, use_edge_attr=False)
    def decode(self, images, dense, context):
        return context


def consistent_worker(rank, root_string, rendezvous):
    torch.set_num_threads(1)
    root = Path(root_string)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    execution = DistributedExecution("cpu", rank=rank, world_size=2, owns_process_group=True)
    try:
        torch.manual_seed(42)
        stage = Stage(root)
        stage.backbone.requires_grad_(False).eval()
        dist.barrier()
        provider = ConsistentFeatureProvider(stage, root / "slides", root / "cache", stage="fusion")
        repo = JointGraphRepository(root / "graphs", expected_edge_mode="distance")
        repo.feature_provider = provider
        system = DistributedSystem(stage)
        execution.prepare_model(system)
        patches = ["a"] if rank == 0 else ["c"]
        value = execution.forward(system, repo, torch.ones(1, 3, 4, 4), ["s"], patches, training=True)
        value.square().mean().backward()
        # Compare ordinary DDP parameter reduction against the same global
        # objective computed in one process, including a shared neighbor b.
        expected_stage = Stage(root)
        expected_stage.load_state_dict(stage.state_dict())
        expected_stage.backbone.requires_grad_(False).eval()
        expected_repo = JointGraphRepository(root / "graphs", expected_edge_mode="distance")
        expected_repo.feature_provider = ConsistentFeatureProvider(
            expected_stage, root / "slides", root / f"reference_cache_{rank}", stage="fusion")
        expected = _JointForward(DistributedSystem(expected_stage), SimpleNamespace(distributed=False))(
            torch.ones(2, 3, 4, 4), expected_repo, ["s", "s"], ["a", "c"], True)
        expected.square().mean().backward()
        torch.testing.assert_close(stage.fusion.weight.grad, expected_stage.fusion.weight.grad)
    finally:
        execution.close()


def test_consistent_neighbors_two_rank_ddp_gradient_matches_global_objective(tmp_path):
    _, provider = setup_provider(tmp_path)
    repository(tmp_path, provider)
    mp.spawn(consistent_worker, args=(str(tmp_path), f"file://{tmp_path / 'rendezvous'}"), nprocs=2, join=True)
