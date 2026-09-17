"""CPU-only checks of rank sharding, gradients, metrics, and graph memory."""

from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from dinov2_segmentation.distributed_execution import (
    DistributedExecution,
    EpochRandomSampler,
    WholeBatchSampler,
    WholeBatchShardSampler,
)
from dinov2_segmentation.probability_metrics import BinaryProbabilityMetrics


def test_whole_batches_equal_training_steps_and_no_added_repeats():
    shards = [
        WholeBatchShardSampler(range(23), 3, rank=rank, world_size=2, training=True)
        for rank in range(2)
    ]
    assert [len(shard) for shard in shards] == [3, 3]
    assert [shard.dropped_samples for shard in shards] == [5, 5]
    batches = [list(shard) for shard in shards]
    assert batches[0] == [[0, 1, 2], [6, 7, 8], [12, 13, 14]]
    assert batches[1] == [[3, 4, 5], [9, 10, 11], [15, 16, 17]]
    assert sorted(value for rank in batches for batch in rank for value in batch) == list(range(18))


def test_validation_visits_every_sample_once_including_partial_and_empty_ranks():
    shards = [
        WholeBatchShardSampler(range(5), 2, rank=rank, world_size=4, training=False)
        for rank in range(4)
    ]
    assert [list(shard) for shard in shards] == [[[0, 1]], [[2, 3]], [[4]], []]
    assert [len(shard) for shard in shards] == [1, 1, 1, 0]
    assert all(shard.dropped_samples == 0 for shard in shards)
    serial = WholeBatchShardSampler(range(5), 2, rank=0, world_size=1, training=True)
    assert list(serial) == [[0, 1], [2, 3], [4]]


def test_global_epoch_random_order_is_reproducible_and_shuffled():
    source = EpochRandomSampler(range(40), seed=13)
    shard = WholeBatchShardSampler(source, 2, rank=0, world_size=2, training=True)
    before = list(shard)
    shard.set_epoch(7)
    after = list(shard)
    assert after != before
    shard.set_epoch(7)
    assert list(shard) == after


def test_sharded_batch_resume_uses_rank_local_offset_and_epoch_resets_it():
    shards = [
        WholeBatchShardSampler(range(23), 3, rank=rank, world_size=2, training=True)
        for rank in range(2)
    ]
    for shard in shards:
        shard.set_start_batch(1)
    assert [len(shard) for shard in shards] == [2, 2]
    assert list(shards[0]) == [[6, 7, 8], [12, 13, 14]]
    assert list(shards[1]) == [[9, 10, 11], [15, 16, 17]]

    shards[0].set_start_batch(3)
    assert len(shards[0]) == 0
    assert list(shards[0]) == []
    with pytest.raises(ValueError, match="start_batch"):
        shards[0].set_start_batch(4)
    with pytest.raises(ValueError, match="start_batch"):
        shards[0].set_start_batch(-1)

    shards[0].set_epoch(2)
    assert len(shards[0]) == 3
    assert list(shards[0]) == [[0, 1, 2], [6, 7, 8], [12, 13, 14]]


class _CountingDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.visited: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        self.visited.append(int(index))
        return int(index)


def test_serial_whole_batch_resume_skips_dataset_io_and_keeps_partial_tail():
    dataset = _CountingDataset(10)
    batches = WholeBatchSampler(range(len(dataset)), batch_size=4)
    assert batches.dropped_samples == 0
    assert len(batches) == 3
    assert list(batches) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]

    batches.set_start_batch(1)
    assert len(batches) == 2
    assert [batch.tolist() for batch in DataLoader(dataset, batch_sampler=batches)] == [
        [4, 5, 6, 7],
        [8, 9],
    ]
    assert dataset.visited == [4, 5, 6, 7, 8, 9]

    batches.set_start_batch(3)
    assert len(batches) == 0
    assert list(batches) == []
    with pytest.raises(ValueError, match="start_batch"):
        batches.set_start_batch(4)

    batches.set_epoch(9)
    assert len(batches) == 3


def test_too_small_training_cohort_fails_instead_of_empty_training():
    with pytest.raises(ValueError, match="at least"):
        WholeBatchShardSampler(range(3), 2, rank=0, world_size=2, training=True)


def test_serial_cannot_accidentally_run_duplicate_torchrun_jobs(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="Serial execution"):
        DistributedExecution.from_environment("serial", "cpu")
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="at least two"):
        DistributedExecution.from_environment("ddp", "cpu")


class _TinyStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)

    def forward(self, images):
        values = self.projection(images)
        return values, values


class _TinySystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = _TinyStage1()
        self.stage2 = nn.Linear(2, 2, bias=False)
        self.decoder = nn.Linear(4, 1, bias=False)
        self.stage2_runtime = SimpleNamespace(num_layers=1, use_edge_attr=False)

    def decode(self, images, dense, contexts):
        return self.decoder(torch.cat([dense, contexts], dim=1))


class _TinyRepository:
    """Exercise the same _get/contextualize contract as JointGraphRepository."""

    def __init__(self, cache_size=1):
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def _get(self, slide):
        value = self._cache.pop(slide, None)
        if value is None:
            value = (SimpleNamespace(x=torch.zeros(2, 2)), {"p0": 0, "p1": 1})
        self._cache[slide] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def contextualize(self, gnn, features, slides, patches, *, update_memory, **kwargs):
        result = [None] * len(slides)
        for slide in dict.fromkeys(slides):
            positions = [i for i, candidate in enumerate(slides) if candidate == slide]
            graph, mapping = self._get(slide)
            indices = torch.tensor([mapping[patches[i]] for i in positions])
            values = graph.x.index_copy(0, indices, features[positions])
            # A target explicitly depends on all nodes, including a remote
            # rank's live feature when both ranks contain this slide.
            context = gnn(values + values.mean(dim=0, keepdim=True))[indices]
            for offset, position in enumerate(positions):
                result[position] = context[offset]
            if update_memory:
                graph.x.index_copy_(0, indices, features[positions].detach())
        return torch.stack(result)


def _cpu_rank_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(directory, "rendezvous").as_uri(), rank=rank,
        world_size=2, timeout=timedelta(seconds=60),
    )
    execution = DistributedExecution("cpu", rank=rank, world_size=2, owns_process_group=True)
    try:
        # An objective on rank zero must reach the node owned by rank one.
        node = torch.tensor([[float(rank + 1)]], requires_grad=True)
        global_nodes, _, _, _ = execution.gather_graph_inputs(node, ["a"], [f"p{rank}"])
        remote_objective = global_nodes[1].sum() * 3 if rank == 0 else global_nodes[0].sum() * 0
        remote_objective.backward()
        assert float(node.grad) == (0.0 if rank == 0 else 3.0)

        torch.manual_seed(7)
        system = _TinySystem()
        reference = _TinySystem()
        reference.load_state_dict(system.state_dict())
        optimizer = torch.optim.SGD(system.parameters(), lr=0.03)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
        repository = _TinyRepository(cache_size=1)
        reference_repository = _TinyRepository(cache_size=1)
        serial = DistributedExecution("cpu")

        # Compare a global serial SGD update with DDP, then change the active
        # parameter set twice to exercise reducer rebuilding after unfreezing.
        all_images = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        all_targets = torch.tensor([[0.3], [-0.1]])
        for phase in ("joint", "decoder_only", "joint"):
            for model in (system, reference):
                for stage in (model.stage1, model.stage2):
                    for parameter in stage.parameters():
                        parameter.requires_grad_(phase == "joint")
            execution.prepare_model(system, phase)
            serial.prepare_model(reference, phase)
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            actual = execution.forward(
                system, repository, all_images[rank:rank + 1], ["shared"], [f"p{rank}"],
                training=True,
            )
            expected = serial.forward(
                reference, reference_repository, all_images, ["shared", "shared"], ["p0", "p1"],
                training=True,
            )
            (actual - all_targets[rank:rank + 1]).square().mean().backward()
            (expected - all_targets).square().mean().backward()
            optimizer.step()
            reference_optimizer.step()
            for left, right in zip(system.parameters(), reference.parameters()):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

        # Global batches spanning more slides than the cache still leave the
        # same LRU keys and values on both ranks, including decoder-only mode.
        for stage in (system.stage1, system.stage2):
            for parameter in stage.parameters():
                parameter.requires_grad_(False)
        execution.prepare_model(system, "decoder_only")
        output = execution.forward(
            system, repository, all_images[rank:rank + 1], [f"slide{rank}"], ["p0"], training=True,
        )
        output.square().mean().backward()
        cache_keys = list(repository._cache)
        assert cache_keys == ["slide1"]
        memory = repository._cache["slide1"][0].x.clone()

        # Only rank zero validates: the other rank immediately proceeds to
        # reductions. No DDP forward collective may hang this uneven epoch.
        execution.sync_buffers(system)
        if rank == 0:
            with torch.no_grad():
                execution.forward(system, _TinyRepository(), all_images[:1], ["eval"], ["p0"], training=False)
        histograms = BinaryProbabilityMetrics(16)
        if rank == 0:
            histograms.positive_histogram[2] = 3
            histograms.negative_histogram[1] = 4
            histograms.positive_probability_sum.fill_(0.6)
            histograms.negative_probability_sum.fill_(0.4)
        reduced = execution.reduce_metrics(
            {"loss": torch.tensor(6.0 if rank == 0 else 0.0)},
            3 if rank == 0 else 0,
            torch.tensor([[4, 0], [0, 3]]) if rank == 0 else torch.zeros(2, 2, dtype=torch.int64),
            histograms,
        )
        assert reduced[0] == {"loss": 6.0} and reduced[1] == 3
        assert reduced[2].tolist() == [[4, 0], [0, 3]]
        assert reduced[3].positive_histogram.sum() == 3
        assert reduced[3].negative_histogram.sum() == 4
        assert float(reduced[3].positive_probability_sum) == pytest.approx(0.6)
        assert not execution.all_finite(torch.tensor(float("nan") if rank else 1.0))
        assert execution.reduce_max_values({"gradient": rank + 1.0}) == {"gradient": 2.0}
        torch.save({"memory": memory, "parameters": system.state_dict()}, Path(directory, f"rank{rank}.pt"))
    finally:
        execution.close()


def test_two_cpu_ranks_preserve_gradients_memory_and_raw_metrics(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo process groups are unavailable")
    mp.spawn(_cpu_rank_worker, args=(str(tmp_path),), nprocs=2, join=True)
    states = [torch.load(tmp_path / f"rank{rank}.pt", map_location="cpu", weights_only=True) for rank in (0, 1)]
    torch.testing.assert_close(states[0]["memory"], states[1]["memory"])
    for name in states[0]["parameters"]:
        torch.testing.assert_close(states[0]["parameters"][name], states[1]["parameters"][name])
