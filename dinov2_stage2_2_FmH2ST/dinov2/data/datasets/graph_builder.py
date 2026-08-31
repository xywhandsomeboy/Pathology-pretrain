"""Build WSI patch graphs for Stage 2.

Edges are spatial KNN candidates filtered by a maximum physical distance. Each
edge stores spatial and semantic weights in separate channels.
"""

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:  # Small tests may use the exact torch fallback.
    NearestNeighbors = None


GRAPH_FORMAT_VERSION = 1


def _knn(coords, k, chunk_size=1024):
    """Exact spatial KNN, using BallTree for production-size WSI graphs."""
    if NearestNeighbors is not None:
        model = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
        distances, indices = model.kneighbors(coords)
        filtered_distances = np.empty((len(coords), k), dtype=distances.dtype)
        filtered_indices = np.empty((len(coords), k), dtype=indices.dtype)
        for row in range(len(coords)):
            keep = indices[row] != row
            filtered_distances[row] = distances[row][keep][:k]
            filtered_indices[row] = indices[row][keep][:k]
        return filtered_distances, filtered_indices
    if len(coords) > 10000:
        raise ImportError(
            "Building graphs with more than 10,000 nodes requires scikit-learn "
            "for the BallTree KNN implementation"
        )
    points = torch.as_tensor(coords, dtype=torch.float32)
    all_distances, all_indices = [], []
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        distances = torch.cdist(points[start:stop], points)
        local_rows = torch.arange(stop - start)
        distances[local_rows, torch.arange(start, stop)] = torch.inf
        values, indices = torch.topk(distances, k=k, dim=1, largest=False, sorted=True)
        all_distances.append(values)
        all_indices.append(indices)
    return (
        torch.cat(all_distances).numpy(),
        torch.cat(all_indices).numpy(),
    )


def infer_max_distance(coords, multiplier=3.0):
    """Estimate a slide-specific cutoff from the median nearest-patch spacing."""
    coords = np.asarray(coords, dtype=np.float32)
    if len(coords) < 2:
        return 0.0
    distances, _ = _knn(coords, 1)
    positive = distances[:, 0][distances[:, 0] > 0]
    if positive.size == 0:
        return 0.0
    return float(np.median(positive) * multiplier)


def build_graph(
    coords,
    features,
    k=8,
    max_distance=None,
    distance_multiplier=3.0,
    sigma_d=None,
):
    """Return directed edges with [spatial_weight, semantic_weight]."""
    coords = np.asarray(coords, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape [N,2], got {coords.shape}")
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N,C], got {features.shape}")
    if len(coords) != len(features):
        raise ValueError("The numbers of coordinates and features must match")
    if not np.isfinite(coords).all() or not np.isfinite(features).all():
        raise ValueError("coords and features must contain only finite values")
    if int(k) < 1:
        raise ValueError("k must be positive")
    if float(distance_multiplier) <= 0:
        raise ValueError("distance_multiplier must be positive")
    if max_distance is not None and float(max_distance) < 0:
        raise ValueError("max_distance must be non-negative")
    if len(coords) < 2:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
        )

    k_actual = min(int(k), len(coords) - 1)
    distances, indices = _knn(coords, k_actual)
    if max_distance is None:
        positive = distances[:, 0][distances[:, 0] > 0]
        max_distance = (
            float(np.median(positive) * distance_multiplier) if positive.size else 0.0
        )
    if sigma_d is None:
        sigma_d = max(float(max_distance) / 2.0, np.finfo(np.float32).eps)

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / np.maximum(norms, np.finfo(np.float32).eps)
    edges, attributes = [], []
    for source in range(len(coords)):
        for target, distance in zip(indices[source], distances[source]):
            if distance > max_distance:
                continue
            spatial_weight = np.exp(-(distance ** 2) / (2.0 * sigma_d ** 2))
            edges.append((source, int(target)))
            semantic_weight = (
                float(np.dot(normalized_features[source], normalized_features[target]))
                + 1.0
            ) / 2.0
            attributes.append((spatial_weight, semantic_weight))
    return (
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        np.asarray(attributes, dtype=np.float32).reshape(-1, 2),
    )


def to_pyg_data(
    features,
    coords,
    edges,
    edge_attr,
    labels=None,
    patch_ids=None,
    dense_tokens=None,
    levels=None,
    slide_id=None,
):
    count = len(features)
    if patch_ids is not None and len(patch_ids) != count:
        raise ValueError("patch_ids must have the same node order and length as features")
    if dense_tokens is not None and len(dense_tokens) != count:
        raise ValueError("dense_tokens must have the same node order and length as features")
    if levels is not None and len(levels) != count:
        raise ValueError("levels must have the same node order and length as features")
    data = Data(
        x=torch.as_tensor(features, dtype=torch.float32),
        edge_index=torch.as_tensor(edges.T, dtype=torch.long),
        edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32),
        pos=torch.as_tensor(coords, dtype=torch.float32),
        y=torch.as_tensor(labels, dtype=torch.long) if labels is not None else None,
    )
    data.graph_format_version = GRAPH_FORMAT_VERSION
    data.node_ids = torch.arange(count, dtype=torch.long)
    # These fields preserve the exact Stage-1 -> graph -> raw-patch identity
    # needed by the downstream segmentation decoder.
    if patch_ids is not None:
        if len(set(map(str, patch_ids))) != count:
            raise ValueError("patch_ids must be unique within a slide")
        data.patch_ids = list(map(str, patch_ids))
    if dense_tokens is not None:
        data.dense_tokens = torch.as_tensor(dense_tokens)
    data.levels = torch.as_tensor(
        levels if levels is not None else np.zeros(count), dtype=torch.long
    )
    if slide_id is not None:
        data.slide_id = str(slide_id)
    return data


def validate_graph_schema(data, *, require_decoder_metadata=False):
    """Validate tensors shared by Stage 2 and the segmentation cache builder."""
    if not isinstance(data, Data):
        raise TypeError(f"Expected torch_geometric.data.Data, got {type(data)}")
    if data.x is None or data.x.ndim != 2:
        raise ValueError("graph.x must be [N,C]")
    count = data.x.size(0)
    if data.edge_index is None or data.edge_index.shape[0] != 2:
        raise ValueError("graph.edge_index must be [2,E]")
    if data.edge_attr is None or data.edge_attr.ndim != 2 or data.edge_attr.size(1) != 2:
        raise ValueError("graph.edge_attr must be [E,2]")
    if data.edge_attr.size(0) != data.edge_index.size(1):
        raise ValueError("graph.edge_attr and graph.edge_index contain different edge counts")
    if data.pos is None or data.pos.shape != (count, 2):
        raise ValueError("graph.pos must be [N,2]")
    if require_decoder_metadata:
        missing = [
            name for name in ("patch_ids", "levels", "slide_id") if not hasattr(data, name)
        ]
        if missing:
            raise ValueError(f"Graph lacks decoder metadata: {missing}")
        if len(data.patch_ids) != count:
            raise ValueError("Patch metadata and graph node count differ")
        if len(set(map(str, data.patch_ids))) != count:
            raise ValueError("patch_ids must be unique inside one slide")
        if len(data.levels) != count:
            raise ValueError("levels and graph node count differ")
        coordinate_ids = [
            (int(level), int(round(x)), int(round(y)))
            for level, (x, y) in zip(data.levels.tolist(), data.pos.tolist())
        ]
        if len(set(coordinate_ids)) != count:
            raise ValueError("(level,x,y) coordinates must be unique inside one slide")
    return data


def build_pyg_graph(
    coords,
    features,
    k=8,
    max_distance=None,
    distance_multiplier=3.0,
    labels=None,
    patch_ids=None,
    dense_tokens=None,
    levels=None,
    slide_id=None,
):
    edges, edge_attr = build_graph(
        coords,
        features,
        k=k,
        max_distance=max_distance,
        distance_multiplier=distance_multiplier,
    )
    return validate_graph_schema(
        to_pyg_data(
            features,
            coords,
            edges,
            edge_attr,
            labels,
            patch_ids=patch_ids,
            dense_tokens=dense_tokens,
            levels=levels,
            slide_id=slide_id,
        ),
    )
