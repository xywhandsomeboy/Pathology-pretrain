#!/usr/bin/env python3
"""Read-only structural checks for the maintained CerviPath pipeline."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_FILES = (
    "dinov2_stage1_Extract2s2/dinov2/data/collate.py",
    "dinov2_stage1_Extract2s2/dinov2/data/datasets/image_folder.py",
    "dinov2_stage1_Extract2s2/dinov2/fsdp/__init__.py",
    "dinov2_stage1_Extract2s2/dinov2/train/gcn_meta_arch.py",
    "dinov2_stage1_Extract2s2/dinov2/train/train.py",
    "dinov2_stage1_Extract2s2/organize_features.py",
    "dinov2_stage2_2_FmH2ST/build_graphs.py",
    "dinov2_stage2_2_FmH2ST/export_context.py",
    "dinov2_stage2_2_FmH2ST/dinov2/data/datasets/graph_builder.py",
    "dinov2_stage2_2_FmH2ST/dinov2/data/datasets/image_folder.py",
    "dinov2_stage2_2_FmH2ST/dinov2/train/gcn_meta_arch.py",
    "dinov2_segmentation/data/feature_store.py",
    "dinov2_segmentation/data/manifest_dataset.py",
    "dinov2_segmentation/build_manifest.py",
    "dinov2_segmentation/prepare_slide_store.py",
    "dinov2_segmentation/models/model.py",
)
ACTIVE_SCRIPTS = (
    "dinov2_stage1_Extract2s2/pretrain.sh",
    "dinov2_stage1_Extract2s2/pretrain_imgnet22k.sh",
    "dinov2_stage1_Extract2s2/pretrain_stage1a.sh",
    "dinov2_stage2_2_FmH2ST/pretrain.sh",
)
LEGACY_PROJECTS = (
    "dinov2_finetune",
    "dinov2_stage1_Extract2s2_local",
    "dinov2_stage2_2_FmH2ST_finetune",
)


def main() -> int:
    failures = []
    for relative in ACTIVE_FILES:
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing active file: {relative}")
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            failures.append(f"syntax error in {relative}: {error}")
    for relative in ACTIVE_SCRIPTS:
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing active script: {relative}")
            continue
        if "/home/li_yu" in path.read_text(encoding="utf-8"):
            failures.append(f"machine-specific legacy path in active script: {relative}")

    stage2_config = (ROOT / "dinov2_stage2_2_FmH2ST/dinov2/configs/ssl_default_config.yaml").read_text(
        encoding="utf-8"
    )
    stage2_gnn = (ROOT / "dinov2_stage2_2_FmH2ST/dinov2/models/gcn.py").read_text(
        encoding="utf-8"
    )
    if 'gnn_type: "gatv2"' not in stage2_config:
        failures.append("maintained Stage 2 configuration must select gatv2")
    if "class GATv2Conv" not in stage2_gnn:
        failures.append("maintained Stage 2 model lacks GATv2Conv")
    if "softmax(alpha, edge_index_i, num_nodes=size_i)" not in stage2_gnn:
        failures.append("GAT attention must normalize incoming edges by target node")

    for project in LEGACY_PROJECTS:
        legacy_path = ROOT / "legacy" / project
        old_root_path = ROOT / project
        if not legacy_path.is_dir():
            failures.append(f"missing legacy project: legacy/{project}")
        if old_root_path.exists():
            failures.append(f"legacy project leaked into repository root: {project}")

    embeddings = ROOT / "Graph/embeddings-1024"
    if embeddings.is_dir():
        features = {p.name[: -len("_features.npy")] for p in embeddings.glob("*_features.npy")}
        coords = {p.name[: -len("_coords.npy")] for p in embeddings.glob("*_coords.npy")}
        for slide in sorted(features ^ coords)[:10]:
            failures.append(f"unpaired feature/coordinate array: {slide}")

    if failures:
        print("CerviPath structural check FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("CerviPath structural check OK")
    print(f"- parsed {len(ACTIVE_FILES)} active Python files")
    print(f"- checked {len(ACTIVE_SCRIPTS)} maintained entry scripts")
    print(f"- isolated {len(LEGACY_PROJECTS)} legacy projects under legacy/")
    if embeddings.is_dir():
        print(f"- paired graph arrays: {len(features)} slides")
    current_graphs = ROOT / "Graph/graphs-current"
    if current_graphs.is_dir():
        print(f"- current two-channel graphs: {len(list(current_graphs.glob('*.pt')))}")
    else:
        print("- current two-channel graphs: not generated yet (legacy graphs were preserved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
