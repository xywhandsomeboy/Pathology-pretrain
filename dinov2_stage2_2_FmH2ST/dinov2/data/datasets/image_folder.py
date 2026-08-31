"""Lean PyG graph dataset for Stage-2 self-supervised pretraining."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


logger = logging.getLogger("gnn")


def _load_graph(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _lean_training_graph(graph: Data, max_nodes: int) -> Data:
    """Keep graph-training tensors and intentionally drop decoder metadata."""
    count = int(graph.num_nodes)
    node_indices = (
        torch.randperm(count)[:max_nodes]
        if max_nodes > 0 and count > max_nodes
        else torch.arange(count)
    )
    edge_index, edge_attr = subgraph(
        node_indices,
        graph.edge_index,
        getattr(graph, "edge_attr", None),
        relabel_nodes=True,
        num_nodes=count,
    )
    return Data(
        x=graph.x[node_indices],
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=(graph.pos[node_indices] if getattr(graph, "pos", None) is not None else None),
    )


class ImageFolder(Dataset):
    """Load one prebuilt WSI graph per sample.

    Masking and noisy-view construction belong to ``GCNMetaArch``. The dataset
    only filters invalid graphs and optionally samples a training subgraph.
    Dense DINO tokens and patch identity metadata are omitted from the training
    batch so that downstream decoder caches cannot accidentally consume VRAM.
    """

    def __init__(
        self,
        *,
        root: str,
        transform: Optional[Callable] = None,
        min_nodes: int | str | None = None,
        min_edges: int | str | None = None,
        max_nodes: int | str = 5000,
        edge_dim: int | str = 2,
    ) -> None:
        del transform
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Stage-2 graph directory does not exist: {self.root}")
        self.max_nodes = int(max_nodes)
        self.edge_dim = int(edge_dim)
        if self.max_nodes < 0:
            raise ValueError("max_nodes must be >= 0; use 0 for complete graphs")
        if self.edge_dim < 0:
            raise ValueError("edge_dim must be >= 0; use 0 for an unweighted graph")

        candidates = sorted(self.root.glob("*.pt"))
        valid: list[tuple[Path, int, int]] = []
        for path in candidates:
            try:
                graph = _load_graph(path)
            except Exception as error:
                logger.warning("Skipping unreadable graph %s: %s", path.name, error)
                continue
            if not isinstance(graph, Data) or graph.x is None or graph.edge_index is None:
                logger.warning("Skipping non-PyG or incomplete graph %s", path.name)
                continue
            edge_attr = getattr(graph, "edge_attr", None)
            actual_edge_dim = (
                0
                if edge_attr is None
                else (1 if edge_attr.ndim == 1 else edge_attr.size(-1))
            )
            if actual_edge_dim != self.edge_dim:
                logger.warning(
                    "Skipping %s: edge_dim=%d, expected %d. Rebuild it with build_graphs.py.",
                    path.name,
                    actual_edge_dim,
                    self.edge_dim,
                )
                continue
            valid.append((path, int(graph.num_nodes), int(graph.edge_index.size(1))))
        if not valid:
            raise ValueError(f"No valid PyG .pt graphs found in {self.root}")

        node_counts = np.asarray([item[1] for item in valid])
        edge_counts = np.asarray([item[2] for item in valid])
        resolved_min_nodes = (
            int(min_nodes)
            if min_nodes is not None
            else min(5, max(1, int(np.quantile(node_counts, 0.01))))
        )
        resolved_min_edges = (
            int(min_edges)
            if min_edges is not None
            else min(10, max(1, int(np.quantile(edge_counts, 0.01))))
        )
        self.samples = [
            path
            for path, nodes, edges in valid
            if nodes >= resolved_min_nodes and edges >= resolved_min_edges
        ]
        if not self.samples:
            raise ValueError(
                f"Every graph was filtered out (min_nodes={resolved_min_nodes}, "
                f"min_edges={resolved_min_edges}, edge_dim={self.edge_dim})"
            )
        logger.info(
            "Loaded %d/%d graphs (min_nodes=%d, min_edges=%d, max_nodes=%d)",
            len(self.samples),
            len(valid),
            resolved_min_nodes,
            resolved_min_edges,
            self.max_nodes,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path = self.samples[index]
        graph = _lean_training_graph(_load_graph(path), self.max_nodes)
        return {"original_graph": graph, "slide_name": path.stem}
