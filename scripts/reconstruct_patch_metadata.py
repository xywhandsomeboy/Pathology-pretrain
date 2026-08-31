#!/usr/bin/env python3
"""Reconstruct the complete Stage-1 patch manifest from immutable identities.

The archived ``ImageNet22k`` index retains every patch filename, while the
reference embedding directory retains the complete per-slide coordinate order.
It does not retain filenames.  This command joins the two without reusing the
reference node features:

* patch filenames are ordered by their numeric ``_patch_<index>`` suffix;
* reference nodes are ordered by ``(y, x)``;
* each reference slide keeps its original ``_<split>`` suffix.

The ordering contract can be checked independently against a surviving patch
CSV via ``--validation-csv``.  The generated CSV is accepted directly by
``dinov2_stage1_Extract2s2/organize_features.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PATCH_PATTERN = re.compile(r"^(?P<base>.+)_patch_(?P<index>\d+)(?P<suffix>\.[^.]+)$")
COORDS_SUFFIX = "_coords.npy"
FEATURES_SUFFIX = "_features.npy"


def _patch_identity(filename: str) -> tuple[str, int, str]:
    basename = Path(filename).name
    match = PATCH_PATTERN.fullmatch(basename)
    if match is None:
        raise ValueError(
            f"Patch filename does not follow '<slide>_patch_<index>.<ext>': {filename}"
        )
    return match.group("base"), int(match.group("index")), basename


def _validate_known_csv(path: Path) -> tuple[int, int]:
    """Prove that numeric patch order is the same as spatial raster order."""
    groups: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"filepath", "x", "y"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Validation CSV is missing columns: {missing}")
        for row in reader:
            base, patch_index, _ = _patch_identity(row["filepath"])
            groups[base].append((patch_index, int(float(row["x"])), int(float(row["y"]))))
    if not groups:
        raise ValueError(f"Validation CSV contains no patches: {path}")

    patch_count = 0
    for base, records in groups.items():
        by_patch = sorted(records, key=lambda item: item[0])
        by_position = sorted(records, key=lambda item: (item[2], item[1]))
        if by_patch != by_position:
            raise ValueError(
                f"Patch-index order differs from (y, x) order in validation slide {base}"
            )
        if len({item[0] for item in records}) != len(records):
            raise ValueError(f"Duplicate patch index in validation slide {base}")
        if len({item[1:] for item in records}) != len(records):
            raise ValueError(f"Duplicate coordinate in validation slide {base}")
        patch_count += len(records)
    return len(groups), patch_count


def _load_entry_groups(entries_path: Path) -> dict[str, list[tuple[int, str]]]:
    entries = np.load(entries_path, mmap_mode="r", allow_pickle=False)
    if entries.dtype.names is None or "filename" not in entries.dtype.names:
        raise ValueError(f"entries.npy has no structured 'filename' field: {entries_path}")
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in entries:
        base, patch_index, basename = _patch_identity(str(row["filename"]))
        groups[base].append((patch_index, basename))
    if not groups:
        raise ValueError(f"Image index contains no patch entries: {entries_path}")
    return groups


def _load_reference_groups(
    reference_dir: Path,
) -> dict[str, list[tuple[str, Path, Path]]]:
    groups: dict[str, list[tuple[str, Path, Path]]] = defaultdict(list)
    coord_paths = sorted(reference_dir.glob(f"*{COORDS_SUFFIX}"))
    if not coord_paths:
        raise FileNotFoundError(f"No *{COORDS_SUFFIX} files found in {reference_dir}")
    for coords_path in coord_paths:
        slide_id = coords_path.name[: -len(COORDS_SUFFIX)]
        try:
            base, split = slide_id.rsplit("_", 1)
            int(split)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Reference slide id must end in a numeric split suffix: {slide_id}"
            ) from error
        features_path = reference_dir / f"{slide_id}{FEATURES_SUFFIX}"
        if not features_path.is_file():
            raise FileNotFoundError(f"Missing paired reference features: {features_path}")
        groups[base].append((slide_id, coords_path, features_path))
    return groups


def reconstruct(args: argparse.Namespace) -> dict[str, int | str]:
    validation_slides = validation_patches = 0
    if args.validation_csv is not None:
        validation_slides, validation_patches = _validate_known_csv(args.validation_csv)

    entry_groups = _load_entry_groups(args.entries_npy)
    reference_groups = _load_reference_groups(args.reference_embeddings_dir)
    missing_reference = sorted(set(entry_groups).difference(reference_groups))
    missing_entries = sorted(set(reference_groups).difference(entry_groups))
    if missing_reference or missing_entries:
        raise ValueError(
            "Entry/reference slide identities differ: "
            f"missing_reference={missing_reference[:5]}, missing_entries={missing_entries[:5]}"
        )

    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {args.output_csv}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_csv.with_suffix(args.output_csv.suffix + ".tmp")

    total_patches = 0
    total_splits = 0
    fieldnames = ("slide_name_split", "filepath", "x", "y", "level", "patch_id")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for base in sorted(entry_groups):
                patches = sorted(entry_groups[base], key=lambda item: item[0])
                if len({index for index, _ in patches}) != len(patches):
                    raise ValueError(f"Duplicate numeric patch index for slide {base}")
                if len({filename for _, filename in patches}) != len(patches):
                    raise ValueError(f"Duplicate patch filename for slide {base}")

                spatial_nodes = []
                for slide_id, coords_path, features_path in sorted(reference_groups[base]):
                    coords = np.load(coords_path, mmap_mode="r", allow_pickle=False)
                    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
                    if coords.ndim != 2 or coords.shape[1] != 2:
                        raise ValueError(f"Expected [N,2] coordinates in {coords_path}")
                    if features.ndim != 2 or len(features) != len(coords):
                        raise ValueError(
                            f"Reference feature/coordinate count mismatch for {slide_id}"
                        )
                    expected_order = np.lexsort((coords[:, 0], coords[:, 1]))
                    if not np.array_equal(expected_order, np.arange(len(coords))):
                        raise ValueError(f"Reference coordinates are not (y,x) ordered: {coords_path}")
                    for node_index, (x, y) in enumerate(coords):
                        spatial_nodes.append(
                            (int(y), int(x), slide_id, node_index)
                        )
                    total_splits += 1

                spatial_nodes.sort(key=lambda item: (item[0], item[1]))
                if len({(y, x) for y, x, _, _ in spatial_nodes}) != len(spatial_nodes):
                    raise ValueError(f"Duplicate coordinates across reference splits for {base}")
                if len(patches) != len(spatial_nodes):
                    raise ValueError(
                        f"Patch/reference count mismatch for {base}: "
                        f"{len(patches)} != {len(spatial_nodes)}"
                    )

                # Both sequences describe the same raster traversal. Re-sort the
                # joined records by their original per-split node index before
                # writing so each Stage-2 graph keeps its historical node order.
                joined_by_split: dict[
                    str, list[tuple[int, int, str, int, int]]
                ] = defaultdict(list)
                for (patch_index, filename), (y, x, slide_id, node_index) in zip(
                    patches, spatial_nodes
                ):
                    joined_by_split[slide_id].append((node_index, patch_index, filename, x, y))
                for slide_id in sorted(joined_by_split):
                    records = sorted(joined_by_split[slide_id], key=lambda item: item[0])
                    if [record[0] for record in records] != list(range(len(records))):
                        raise ValueError(f"Incomplete node order reconstructed for {slide_id}")
                    for _, patch_index, filename, x, y in records:
                        writer.writerow(
                            {
                                "slide_name_split": slide_id,
                                "filepath": filename,
                                "x": x,
                                "y": y,
                                "level": 0,
                                "patch_id": Path(filename).stem,
                            }
                        )
                        total_patches += 1
        temporary.replace(args.output_csv)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    summary = {
        "entries_npy": str(args.entries_npy.resolve()),
        "reference_embeddings_dir": str(args.reference_embeddings_dir.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "slides": len(entry_groups),
        "slide_splits": total_splits,
        "patches": total_patches,
        "validation_slides": validation_slides,
        "validation_patches": validation_patches,
    }
    summary_path = args.output_csv.with_suffix(args.output_csv.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries-npy", type=Path, required=True)
    parser.add_argument("--reference-embeddings-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = reconstruct(args)
    print(
        "Reconstructed {patches} patches across {slides} slides / "
        "{slide_splits} slide splits".format(**summary)
    )
    print(f"Saved manifest: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
