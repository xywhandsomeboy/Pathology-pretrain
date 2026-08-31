"""Create a strict decoder feature store from a metadata-rich Stage-2 graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dinov2_segmentation.data.feature_store import save_slide_feature_store


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _context_tensor(payload):
    if isinstance(payload, torch.Tensor):
        return payload, None, None
    if not isinstance(payload, dict):
        raise TypeError("Context file must contain a tensor or dictionary")
    for key in ("global_context", "context", "features"):
        if key in payload:
            return payload[key], payload.get("patch_ids"), payload.get("slide_id")
    raise ValueError("Context dictionary needs global_context, context, or features")


def _stage1_dense_inputs(graph, metadata_path: Path | None, dense_path: Path | None):
    patch_ids = list(map(str, graph.patch_ids))
    graph_levels = getattr(graph, "levels", torch.zeros(len(patch_ids), dtype=torch.long))
    if metadata_path is None and dense_path is None and hasattr(graph, "dense_tokens"):
        # Compatibility with the first metadata-rich graph format. New graphs
        # deliberately do not duplicate dense tokens.
        return graph.dense_tokens, graph_levels, {}
    if metadata_path is None or dense_path is None:
        raise ValueError(
            "New lightweight graphs require both --stage1-metadata and --dense-tokens"
        )
    metadata = _load(metadata_path)
    if not isinstance(metadata, dict):
        raise TypeError("Stage-1 metadata must be a dictionary")
    required = {"slide_id", "patch_ids", "levels"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Stage-1 metadata is missing fields: {missing}")
    if str(metadata["slide_id"]) != str(graph.slide_id):
        raise ValueError("Stage-1 metadata slide_id does not match graph.slide_id")
    if list(map(str, metadata["patch_ids"])) != patch_ids:
        raise ValueError("Stage-1 metadata patch_ids do not exactly match graph node order")
    levels = torch.as_tensor(metadata["levels"], dtype=torch.long)
    if not torch.equal(levels, torch.as_tensor(graph_levels, dtype=torch.long)):
        raise ValueError("Stage-1 metadata levels do not match graph levels")
    dense_tokens = (
        np.load(dense_path, mmap_mode="r", allow_pickle=False)
        if dense_path.suffix == ".npy"
        else _load(dense_path)
    )
    if isinstance(dense_tokens, dict):
        dense_tokens = dense_tokens.get("dense_tokens")
    if not isinstance(dense_tokens, (torch.Tensor, np.ndarray)) or len(dense_tokens) != len(patch_ids):
        raise ValueError("Dense-token tensor count does not match graph node count")
    return dense_tokens, levels, {
        "stage1_metadata_path": str(metadata_path.resolve()),
        "dense_tokens_path": str(dense_path.resolve()),
    }


def prepare_store(
    graph_path: Path,
    context_path: Path,
    output_path: Path,
    *,
    stage1_metadata_path: Path | None = None,
    dense_tokens_path: Path | None = None,
    allow_unkeyed_context: bool = False,
) -> Path:
    graph = _load(graph_path)
    context, context_patch_ids, context_slide_id = _context_tensor(_load(context_path))
    required = ("patch_ids", "pos", "slide_id")
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise ValueError(
            f"Graph lacks decoder metadata {missing}. Rebuild it with the extended graph_builder."
        )
    patch_ids = list(map(str, graph.patch_ids))
    dense_tokens, levels, stage1_provenance = _stage1_dense_inputs(
        graph, stage1_metadata_path, dense_tokens_path
    )
    if context_slide_id is not None and str(context_slide_id) != str(graph.slide_id):
        raise ValueError("Context slide_id does not match graph.slide_id")
    if context_patch_ids is None and not allow_unkeyed_context:
        raise ValueError(
            "Context file lacks patch_ids. Re-export context with node identities, "
            "or explicitly pass --allow-unkeyed-context for a verified legacy file."
        )
    if context_patch_ids is not None and list(map(str, context_patch_ids)) != patch_ids:
        raise ValueError("Context patch_ids do not exactly match graph node order")
    if context.ndim != 2 or len(context) != len(patch_ids):
        raise ValueError("Context count does not match graph node count")
    return save_slide_feature_store(
        output_path,
        slide_id=str(graph.slide_id),
        patch_ids=patch_ids,
        coords=graph.pos.round().to(torch.int64),
        levels=levels,
        dense_tokens=dense_tokens,
        global_context=context,
        node_features=graph.x,
        metadata={
            "graph_path": str(graph_path.resolve()),
            "context_path": str(context_path.resolve()),
            **stage1_provenance,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--stage1-metadata", type=Path)
    parser.add_argument("--dense-tokens", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-unkeyed-context",
        action="store_true",
        help="Accept a legacy context tensor after manual node-order verification",
    )
    args = parser.parse_args()
    output = prepare_store(
        args.graph,
        args.context,
        args.output,
        stage1_metadata_path=args.stage1_metadata,
        dense_tokens_path=args.dense_tokens,
        allow_unkeyed_context=args.allow_unkeyed_context,
    )
    print(f"Saved aligned slide feature store: {output}")


if __name__ == "__main__":
    main()
