"""Build WSI patch graphs for Stage 2.

Edges are spatial KNN candidates filtered by a maximum physical distance.
The two edge channels are kept separate: [spatial_weight, semantic_weight].
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


def infer_max_distance(coords, multiplier=3.0):
    """Estimate a slide-specific cutoff from the median nearest-patch spacing."""
    coords = np.asarray(coords, dtype=np.float32)
    if len(coords) < 2:
        return 0.0
    neighbors = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(coords)
    distances, _ = neighbors.kneighbors(coords)
    positive = distances[:, 1][distances[:, 1] > 0]
    if positive.size == 0:
        return 0.0
    return float(np.median(positive) * multiplier)


def build_graph(coords, features, k=8, max_distance=None, distance_multiplier=3.0, sigma_d=None):
    """Return directed edges and independent spatial/semantic edge attributes."""
    coords = np.asarray(coords, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if len(coords) != len(features):
        raise ValueError("The numbers of coordinates and features must match")
    if len(coords) < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.float32)

    k_actual = min(int(k), len(coords) - 1)
    neighbors = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="ball_tree").fit(coords)
    distances, indices = neighbors.kneighbors(coords)
    if max_distance is None:
        max_distance = infer_max_distance(coords, distance_multiplier)
    if sigma_d is None:
        sigma_d = max(float(max_distance) / 2.0, np.finfo(np.float32).eps)

    semantic_similarity = (cosine_similarity(features) + 1.0) / 2.0
    edges, attributes = [], []
    for source in range(len(coords)):
        for target, distance in zip(indices[source, 1:], distances[source, 1:]):
            if distance > max_distance:
                continue
            spatial_weight = np.exp(-(distance ** 2) / (2.0 * sigma_d ** 2))
            edges.append((source, int(target)))
            attributes.append((spatial_weight, semantic_similarity[source, target]))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2), np.asarray(attributes, dtype=np.float32).reshape(-1, 2)


def to_pyg_data(features, coords, edges, edge_attr, labels=None):
    return Data(
        x=torch.as_tensor(features, dtype=torch.float32),
        edge_index=torch.as_tensor(edges.T, dtype=torch.long),
        edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32),
        pos=torch.as_tensor(coords, dtype=torch.float32),
        y=torch.as_tensor(labels, dtype=torch.long) if labels is not None else None,
    )


def build_pyg_graph(coords, features, k=8, max_distance=None, distance_multiplier=3.0, labels=None):
    edges, edge_attr = build_graph(
        coords,
        features,
        k=k,
        max_distance=max_distance,
        distance_multiplier=distance_multiplier,
    )
    return to_pyg_data(features, coords, edges, edge_attr, labels)
