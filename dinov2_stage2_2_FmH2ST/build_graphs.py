"""Build canonical Stage-2 graphs from paired ``*_features.npy``/``*_coords.npy`` files.

The historical notebook is intentionally not imported. Optional
``<slide>_metadata.pt`` dictionaries provide lightweight ``patch_ids`` and
``levels`` when the graphs will later feed the segmentation decoder. Dense
tokens deliberately remain outside PyG graph files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dinov2.data.datasets.graph_builder import build_pyg_graph, validate_graph_schema


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_all(args) -> tuple[int, int]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    built = skipped = 0
    for feature_path in sorted(args.embeddings_dir.glob("*_features.npy")):
        slide_id = feature_path.name[: -len("_features.npy")]
        coords_path = args.embeddings_dir / f"{slide_id}_coords.npy"
        if not coords_path.is_file():
            raise FileNotFoundError(f"Missing coordinate pair for {feature_path.name}")
        output_path = args.output_dir / f"{slide_id}.pt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        metadata = {}
        if args.metadata_dir is not None:
            metadata_path = args.metadata_dir / f"{slide_id}_metadata.pt"
            if metadata_path.is_file():
                metadata = _load(metadata_path)
                if not isinstance(metadata, dict):
                    raise TypeError(f"Metadata must be a dictionary: {metadata_path}")
        graph = build_pyg_graph(
            np.load(coords_path),
            np.load(feature_path),
            k=args.k,
            max_distance=args.max_distance,
            distance_multiplier=args.distance_multiplier,
            patch_ids=metadata.get("patch_ids"),
            levels=metadata.get("levels"),
            slide_id=slide_id,
            edge_mode=args.edge_mode,
        )
        validate_graph_schema(
            graph,
            expected_edge_dim=2 if args.edge_mode == "dual" else 0,
            require_decoder_metadata=args.require_decoder_metadata,
        )
        temporary_path = output_path.with_suffix(".pt.tmp")
        torch.save(graph, temporary_path)
        temporary_path.replace(output_path)
        built += 1
    return built, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--distance-multiplier", type=float, default=3.0)
    parser.add_argument("--edge-mode", choices=("dual", "distance"), default="dual")
    parser.add_argument("--require-decoder-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.output_dir / "stage2_graph_manifest.json"
    requested_manifest = {
        "format_version": 1,
        "edge_mode": args.edge_mode,
        "edge_dim": 2 if args.edge_mode == "dual" else 0,
        "edge_attr_present": args.edge_mode == "dual",
        "embeddings_dir": str(args.embeddings_dir.expanduser().resolve()),
        "metadata_dir": (
            str(args.metadata_dir.expanduser().resolve())
            if args.metadata_dir is not None
            else None
        ),
        "k": args.k,
        "max_distance": args.max_distance,
        "distance_multiplier": args.distance_multiplier,
        "require_decoder_metadata": args.require_decoder_metadata,
    }
    if manifest_path.is_file() and not args.overwrite:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (existing_manifest.get(key), value)
            for key, value in requested_manifest.items()
            if existing_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Graph output configuration differs from {manifest_path}: {mismatches}. "
                "Use a new output directory or explicitly pass --overwrite."
            )
    elif (
        args.output_dir.is_dir()
        and next(args.output_dir.glob("*.pt"), None) is not None
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Existing graphs in {args.output_dir} lack a manifest; use a new "
            "directory or explicitly pass --overwrite."
        )

    built, skipped = build_all(args)
    requested_manifest.update(built=built, skipped=skipped)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(requested_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(f"Built {built} graphs; skipped {skipped} existing graphs")


if __name__ == "__main__":
    main()
