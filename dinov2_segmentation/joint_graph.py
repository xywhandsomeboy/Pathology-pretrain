"""Differentiable target-node context over cached WSI graph topology."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from torch_geometric.utils import k_hop_subgraph


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class JointGraphRepository:
    """Cache graphs and run an exact receptive-field subgraph for target nodes.

    ``graph.x`` initializes a feature memory from Stage1B. The current batch's
    online Stage1 node features replace its target entries before the trainable
    GATv2 forward. Detached online values then update the in-memory graph, so
    later batches see progressively refreshed neighbors without severing the
    current target's gradient path.
    """

    def __init__(
        self,
        graph_dir: str | Path,
        *,
        expected_edge_mode: str,
        cache_size: int = 64,
    ) -> None:
        self.graph_dir = Path(graph_dir).expanduser().resolve()
        self.expected_edge_mode = str(expected_edge_mode)
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive")
        self._cache: OrderedDict[str, tuple[object, dict[str, int]]] = OrderedDict()

    def _get(self, slide_id: str):
        slide_id = str(slide_id)
        cached = self._cache.pop(slide_id, None)
        if cached is not None:
            self._cache[slide_id] = cached
            return cached
        path = self.graph_dir / f"{slide_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing joint-training graph: {path}")
        graph = _load(path)
        for name in ("x", "edge_index", "patch_ids", "slide_id"):
            if not hasattr(graph, name):
                raise ValueError(f"Graph {path} lacks {name}")
        if str(graph.slide_id) != slide_id:
            raise ValueError(f"Graph slide_id mismatch in {path}")
        edge_mode = str(getattr(graph, "edge_mode", ""))
        if edge_mode != self.expected_edge_mode:
            raise ValueError(
                f"Expected {self.expected_edge_mode!r} graph for {slide_id}, got {edge_mode!r}"
            )
        patch_ids = list(map(str, graph.patch_ids))
        if len(patch_ids) != graph.x.size(0) or len(set(patch_ids)) != len(patch_ids):
            raise ValueError(f"Invalid patch identities in {path}")
        value = (graph, {patch_id: index for index, patch_id in enumerate(patch_ids)})
        self._cache[slide_id] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def contextualize(
        self,
        gnn,
        online_node_features: torch.Tensor,
        slide_ids: list[str] | tuple[str, ...],
        patch_ids: list[str] | tuple[str, ...],
        *,
        num_hops: int,
        use_edge_attr: bool,
        update_memory: bool,
    ) -> torch.Tensor:
        if online_node_features.ndim != 2:
            raise ValueError("online_node_features must be [B,C]")
        if len(slide_ids) != len(patch_ids) or len(slide_ids) != len(online_node_features):
            raise ValueError("Batch identities and online features have different lengths")
        grouped: dict[str, list[int]] = defaultdict(list)
        for batch_index, slide_id in enumerate(slide_ids):
            grouped[str(slide_id)].append(batch_index)
        contexts: list[torch.Tensor | None] = [None] * len(slide_ids)

        for slide_id, batch_indices in grouped.items():
            graph, patch_to_index = self._get(slide_id)
            try:
                node_indices = torch.tensor(
                    [patch_to_index[str(patch_ids[i])] for i in batch_indices],
                    dtype=torch.long,
                )
            except KeyError as error:
                raise KeyError(
                    f"Patch {error.args[0]!r} is absent from graph {slide_id}"
                ) from error
            if node_indices.unique().numel() != node_indices.numel():
                raise ValueError(f"A joint batch repeats a graph node in slide {slide_id}")

            subset, sub_edges, mapping, edge_mask = k_hop_subgraph(
                node_indices,
                int(num_hops),
                graph.edge_index,
                relabel_nodes=True,
                num_nodes=graph.x.size(0),
                flow="source_to_target",
                directed=False,
            )
            device = online_node_features.device
            x = graph.x.index_select(0, subset).to(
                device=device, dtype=online_node_features.dtype, non_blocking=True
            )
            batch_tensor = torch.tensor(batch_indices, device=device, dtype=torch.long)
            local_targets = mapping.to(device=device, dtype=torch.long)
            x = x.index_copy(
                0,
                local_targets,
                online_node_features.index_select(0, batch_tensor),
            )
            sub_edges = sub_edges.to(device=device, non_blocking=True)
            edge_attr = None
            if use_edge_attr:
                source_attr = getattr(graph, "edge_attr", None)
                if source_attr is None:
                    raise ValueError(f"Weighted context graph {slide_id} lacks edge_attr")
                edge_attr = source_attr[edge_mask].to(
                    device=device, dtype=online_node_features.dtype, non_blocking=True
                )
            contextualized = gnn(
                x,
                sub_edges,
                edge_attr,
                use_edge_attr=use_edge_attr,
            )
            selected = contextualized.index_select(0, local_targets)
            for offset, batch_index in enumerate(batch_indices):
                contexts[batch_index] = selected[offset]

            if update_memory:
                graph.x.index_copy_(
                    0,
                    node_indices,
                    online_node_features.index_select(0, batch_tensor)
                    .detach()
                    .float()
                    .cpu(),
                )

        if any(value is None for value in contexts):
            raise RuntimeError("Failed to create context for every batch item")
        return torch.stack([value for value in contexts if value is not None], dim=0)
