"""Optional data-parallel execution for the joint segmentation trainer.

The ordinary trainer remains usable without a process group. Distributed
training shards *whole batches* from one deterministic global sample stream;
it drops only the final incomplete world-sized group of batches. Validation
never pads or repeats samples. Each rank owns a replica of the graph memory,
updated in identical order from detached online features on every train batch.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import timedelta
import math
import os
from typing import Iterator

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.utils.data import Sampler


class EpochRandomSampler(Sampler[int]):
    """A common random permutation on all ranks, independent of worker RNGs."""

    def __init__(self, data_source, seed: int = 0) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self), generator=generator).tolist()


class WholeBatchShardSampler(Sampler[list[int]]):
    """Shard complete global batches without introducing duplicate samples.

    Existing intentional repeats in a stratified source sampler are retained.
    For training with multiple ranks, ``dropped_samples`` reports the tail
    omitted to give every rank equal steps and equal local batch sizes. The
    deterministic source should change its ordering with ``set_epoch``.

    ``set_start_batch`` resumes at a rank-local batch boundary. Advancing the
    underlying sampler only visits integer indices; skipped samples are never
    yielded to the ``DataLoader`` and therefore never trigger dataset I/O.
    Calling ``set_epoch`` starts a new epoch and resets this offset to zero.
    """

    def __init__(
        self,
        sampler,
        batch_size: int,
        *,
        rank: int,
        world_size: int,
        training: bool,
    ) -> None:
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.training = bool(training)
        self.start_batch = 0
        if self.batch_size < 1 or self.world_size < 1:
            raise ValueError("batch_size and world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.training and self.world_size > 1 and self._total_batches() == 0:
            raise ValueError(
                "Training needs at least world_size * batch_size samples; "
                "reduce batch size or the number of ranks"
            )

    @property
    def dropped_samples(self) -> int:
        if self.training and self.world_size > 1:
            return len(self.sampler) % (self.world_size * self.batch_size)
        return 0

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.sampler, "set_epoch", None)
        if setter is not None:
            setter(int(epoch))
        self.start_batch = 0

    def set_start_batch(self, start_batch: int) -> None:
        """Skip completed local batches when resuming inside an epoch."""
        start_batch = int(start_batch)
        total_batches = self._total_batches()
        if not 0 <= start_batch <= total_batches:
            raise ValueError(
                f"start_batch must be in [0, {total_batches}], got {start_batch}"
            )
        self.start_batch = start_batch

    def _total_batches(self) -> int:
        """Return this rank's full-epoch batch count, ignoring resume offset."""
        if self.training and self.world_size > 1:
            return len(self.sampler) // (self.batch_size * self.world_size)
        global_batches = math.ceil(len(self.sampler) / self.batch_size)
        return max(0, (global_batches - self.rank + self.world_size - 1) // self.world_size)

    def __len__(self) -> int:
        return self._total_batches() - self.start_batch

    def __iter__(self) -> Iterator[list[int]]:
        keep_samples = len(self.sampler) - self.dropped_samples
        batch: list[int] = []
        global_batch_index = 0
        local_batch_index = 0
        for sample_index, value in enumerate(self.sampler):
            if sample_index >= keep_samples:
                break
            batch.append(int(value))
            if len(batch) == self.batch_size:
                if global_batch_index % self.world_size == self.rank:
                    if local_batch_index >= self.start_batch:
                        yield batch
                    local_batch_index += 1
                batch = []
                global_batch_index += 1
        if batch and global_batch_index % self.world_size == self.rank:
            if local_batch_index >= self.start_batch:
                yield batch


class WholeBatchSampler(Sampler[list[int]]):
    """Build complete serial batches with restartable epoch-local offsets.

    The source sampler owns ordering and may implement ``set_epoch``. No tail
    samples are dropped. As with :class:`WholeBatchShardSampler`, setting a new
    epoch clears the resume offset; callers then apply a saved offset, if any.
    """

    def __init__(self, sampler, batch_size: int) -> None:
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.start_batch = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    @property
    def dropped_samples(self) -> int:
        return 0

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.sampler, "set_epoch", None)
        if setter is not None:
            setter(int(epoch))
        self.start_batch = 0

    def set_start_batch(self, start_batch: int) -> None:
        start_batch = int(start_batch)
        total_batches = self._total_batches()
        if not 0 <= start_batch <= total_batches:
            raise ValueError(
                f"start_batch must be in [0, {total_batches}], got {start_batch}"
            )
        self.start_batch = start_batch

    def _total_batches(self) -> int:
        return math.ceil(len(self.sampler) / self.batch_size)

    def __len__(self) -> int:
        return self._total_batches() - self.start_batch

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        batch_index = 0
        for value in self.sampler:
            batch.append(int(value))
            if len(batch) == self.batch_size:
                if batch_index >= self.start_batch:
                    yield batch
                batch = []
                batch_index += 1
        if batch and batch_index >= self.start_batch:
            yield batch


class _PinnedRepository:
    """Use preloaded graph references without changing the common LRU order.

    JointGraphRepository.contextualize only needs ``_get`` from the repository.
    Pinning also works when a global batch spans more slides than cache_size:
    evicted graph objects remain live for this forward, without any rank
    reloading stale disk features during its local context computation.
    """

    def __init__(self, repository, graphs: dict) -> None:
        self.repository = repository
        self.graphs = graphs

    def _get(self, slide_id: str):
        return self.graphs[str(slide_id)]

    def contextualize(self, *args, **kwargs):
        return type(self.repository).contextualize(self, *args, **kwargs)


class _JointForward(nn.Module):
    """Put all differentiable stages inside the DDP forward boundary."""

    def __init__(self, system: nn.Module, execution: "DistributedExecution") -> None:
        super().__init__()
        self.system = system
        self.execution = execution

    def forward(self, images, repository, slide_ids, patch_ids, training: bool):
        node_features, dense_tokens = self.system.stage1(images)
        context_repository = repository
        local_context_slice = slice(None)
        if training and self.execution.distributed and getattr(repository, "feature_provider", None) is None:
            node_features, slide_ids, patch_ids, local_context_slice = (
                self.execution.gather_graph_inputs(node_features, slide_ids, patch_ids)
            )
            context_repository = self.execution.synchronize_graph_memory(
                repository, node_features, slide_ids, patch_ids,
                refresh=getattr(self.system, "refresh_graph_memory", True),
            )
        contexts = context_repository.contextualize(
            self.system.stage2,
            node_features,
            slide_ids,
            patch_ids,
            num_hops=self.system.stage2_runtime.num_layers,
            use_edge_attr=self.system.stage2_runtime.use_edge_attr,
            update_memory=bool(training and not self.execution.distributed
                               and getattr(self.system, "refresh_graph_memory", True)),
        )
        return self.system.decode(images, dense_tokens, contexts[local_context_slice])


class DistributedExecution:
    """Small execution adapter; construction does not initialize a group."""

    def __init__(
        self,
        device: torch.device | str = "cpu",
        *,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        owns_process_group: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self._owns_process_group = bool(owns_process_group)
        self._forward_module: _JointForward | None = None
        self._ddp: DistributedDataParallel | None = None
        self._trainable_signature: tuple | None = None

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(
        cls, mode: str = "auto", device: str | torch.device | None = None
    ) -> "DistributedExecution":
        if mode not in ("auto", "serial", "ddp"):
            raise ValueError("execution mode must be auto, serial, or ddp")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if mode == "serial" and world_size != 1:
            raise ValueError("Serial execution cannot run inside multi-rank torchrun")
        if mode == "ddp" and world_size < 2:
            raise ValueError("DDP requires torchrun with at least two processes")
        if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
            raise ValueError("Invalid torchrun rank environment")
        selected = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if selected.type not in ("cpu", "cuda"):
            raise ValueError("Joint distributed execution supports CPU and CUDA")
        if selected.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA execution requested but CUDA is unavailable")
            # torchrun's LOCAL_RANK indexes the visible CUDA devices. Reject
            # pinning all ranks to one explicit device instead of oversubscribing.
            if world_size > 1 and selected.index not in (None, local_rank):
                raise ValueError("DDP CUDA device must match LOCAL_RANK")
            selected = torch.device("cuda", local_rank if world_size > 1 else (selected.index or 0))
            torch.cuda.set_device(selected)
        owns_group = False
        if world_size > 1:
            backend = "nccl" if selected.type == "cuda" else "gloo"
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=backend, init_method="env://", timeout=timedelta(minutes=30)
                )
                owns_group = True
            elif dist.get_rank() != rank or dist.get_world_size() != world_size:
                raise ValueError("Existing process group disagrees with torchrun environment")
        return cls(
            selected, rank=rank, world_size=world_size, local_rank=local_rank,
            owns_process_group=owns_group,
        )

    def shard_batches(self, sampler, batch_size: int, *, training: bool):
        return WholeBatchShardSampler(
            sampler, batch_size, rank=self.rank, world_size=self.world_size,
            training=training,
        )

    def prepare_model(self, system: nn.Module, phase_signature=None):
        """Rebuild the reducer after gradual unfreezing changes parameters."""
        del phase_signature  # Actual requires_grad state is authoritative.
        signature = (id(system), tuple(id(p) for p in system.parameters() if p.requires_grad))
        if signature != self._trainable_signature:
            self._ddp = None
            self._forward_module = _JointForward(system, self)
            if self.distributed:
                self._ddp = DistributedDataParallel(
                    self._forward_module,
                    device_ids=[self.device.index] if self.device.type == "cuda" else None,
                    output_device=self.device.index if self.device.type == "cuda" else None,
                    broadcast_buffers=False,
                    find_unused_parameters=True,
                )
            self._trainable_signature = signature
        return self._ddp if self._ddp is not None else self._forward_module

    def forward(self, system, repository, images, slide_ids, patch_ids, *, training: bool):
        if self._forward_module is None or self._forward_module.system is not system:
            raise RuntimeError("Call prepare_model after setting the training phase")
        # Uneven validation ranks must bypass DDP (including forward collectives).
        model = self._ddp if training and self._ddp is not None else self._forward_module
        return model(images, repository, list(slide_ids), list(patch_ids), training)

    def gather_graph_inputs(self, node_features, slide_ids, patch_ids):
        """Gather live global node features, preserving cross-rank gradients.

        Only ViT images and dense tokens remain local. Graph contextualization
        is replicated for the global target batch so every rank differentiates
        through remote current-batch neighbors as well as its own targets.
        """
        payload = (
            list(map(str, slide_ids)), list(map(str, patch_ids)),
            tuple(node_features.shape), bool(torch.isfinite(node_features.detach()).all()),
        )
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, payload)
        global_slides, global_patches = [], []
        shape = tuple(node_features.shape)
        for slides, patches, feature_shape, finite in gathered:
            if len(feature_shape) != 2 or len(slides) != len(patches) or len(slides) != feature_shape[0]:
                raise ValueError("Gathered graph identities and node features disagree")
            if not finite:
                raise FloatingPointError("Non-finite online Stage1 graph features on a rank")
            if feature_shape != shape:
                raise ValueError("Ranks must produce equal-sized training node feature batches")
            global_slides.extend(slides)
            global_patches.extend(patches)
        identities = list(zip(global_slides, global_patches))
        if len(set(identities)) != len(identities):
            raise ValueError("A global training batch repeats a graph node; use global-batch sampling")
        features = torch.cat(differentiable_all_gather(node_features.contiguous()), dim=0)
        offset = self.rank * shape[0]
        return features, global_slides, global_patches, slice(offset, offset + shape[0])

    @torch.no_grad()
    def synchronize_graph_memory(self, repository, node_features, slide_ids, patch_ids, *, refresh=True):
        """Refresh detached global memories, then pin a common graph view.

        Inputs already follow rank order from ``gather_graph_inputs``. Every
        rank touches each global slide once, in that same order, including
        decoder-only phases. Local contextualize never mutates this LRU.
        """
        features = node_features.detach().float().cpu()
        by_slide: OrderedDict[str, list[tuple[str, torch.Tensor]]] = OrderedDict()
        for slide, patch, feature in zip(slide_ids, patch_ids, features):
            by_slide.setdefault(slide, []).append((patch, feature))
        pinned = {}
        for slide, entries in by_slide.items():
            graph, patch_to_index = repository._get(slide)
            replacements = OrderedDict()
            for patch, feature in entries:
                replacements[patch_to_index[patch]] = feature
            indices = torch.tensor(list(replacements), dtype=torch.long, device=graph.x.device)
            values = torch.stack(list(replacements.values())).to(graph.x)
            if refresh:
                graph.x.index_copy_(0, indices, values)
            pinned[slide] = (graph, patch_to_index)
        return _PinnedRepository(repository, pinned)

    def _reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        if not self.distributed:
            return tensor.clone()
        # NCCL cannot reduce CPU tensors; metric storage stays on CPU outside
        # the collective so billion-pixel counts retain integer precision.
        target_device = self.device if dist.get_backend() == "nccl" else torch.device("cpu")
        result = tensor.to(device=target_device).clone()
        dist.all_reduce(result, op=op)
        return result.to(device=tensor.device)

    def all_finite(self, value: torch.Tensor) -> bool:
        flag = torch.isfinite(value.detach()).all().to(dtype=torch.int64)
        return bool(self._reduce(flag, dist.ReduceOp.MIN).item())

    def reduce_max_values(self, values: dict[str, float]) -> dict[str, float]:
        keys = sorted(values)
        reduced = self._reduce(
            torch.tensor([float(values[key]) for key in keys], dtype=torch.float64),
            dist.ReduceOp.MAX,
        )
        return dict(zip(keys, reduced.tolist()))

    def reduce_metrics(self, totals: dict, samples: int, confusion, probability_metrics=None):
        keys = sorted(totals)
        raw_totals = self._reduce(
            torch.stack(
                [
                    totals[key].detach().to(device=self.device, dtype=torch.float64)
                    if torch.is_tensor(totals[key])
                    else torch.tensor(
                        totals[key], device=self.device, dtype=torch.float64
                    )
                    for key in keys
                ]
            )
        )
        count = self._reduce(torch.tensor(samples, dtype=torch.int64))
        confusion = self._reduce(confusion)
        if probability_metrics is not None:
            for name in ("positive_histogram", "negative_histogram"):
                setattr(
                    probability_metrics,
                    name,
                    self._reduce(getattr(probability_metrics, name)),
                )
            sums = self._reduce(
                torch.stack(
                    (
                        probability_metrics.positive_probability_sum,
                        probability_metrics.negative_probability_sum,
                    )
                )
            )
            probability_metrics.positive_probability_sum = sums[0]
            probability_metrics.negative_probability_sum = sums[1]
        return dict(zip(keys, raw_totals.tolist())), int(count), confusion, probability_metrics

    @torch.no_grad()
    def sync_buffers(self, system: nn.Module) -> None:
        """Call once before uneven validation, never inside its batch loop."""
        if self.distributed:
            for buffer in system.buffers():
                dist.broadcast(buffer, src=0)

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def close(self) -> None:
        self._ddp = None
        self._forward_module = None
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_process_group = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
