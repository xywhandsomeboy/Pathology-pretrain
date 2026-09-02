"""Build all aligned decoder feature stores for one Stage-2 variant."""

from __future__ import annotations

import argparse
from pathlib import Path

from dinov2_segmentation.data.feature_store import SlideFeatureStore
from dinov2_segmentation.prepare_slide_store import prepare_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph_paths = sorted(args.graph_dir.glob("*.pt"))
    if not graph_paths:
        raise FileNotFoundError(f"No graphs found in {args.graph_dir}")
    built = skipped = 0
    for graph_path in graph_paths:
        slide_id = graph_path.stem
        context_path = args.context_dir / f"{slide_id}.pt"
        metadata_path = args.stage1_dir / f"{slide_id}_metadata.pt"
        dense_path = args.stage1_dir / f"{slide_id}_dense_tokens.npy"
        for required in (context_path, metadata_path, dense_path):
            if not required.is_file():
                raise FileNotFoundError(f"Missing aligned feature input: {required}")
        output_path = args.output_dir / f"{slide_id}.pt"
        if output_path.is_file():
            store = SlideFeatureStore(output_path)
            if store.slide_id != slide_id:
                raise ValueError(f"Feature-store slide mismatch: {output_path}")
            skipped += 1
            continue
        prepare_store(
            graph_path,
            context_path,
            output_path,
            stage1_metadata_path=metadata_path,
            dense_tokens_path=dense_path,
        )
        built += 1
    print(f"Built {built} feature stores; skipped {skipped} existing stores")


if __name__ == "__main__":
    main()
