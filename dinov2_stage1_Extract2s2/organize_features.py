"""Join Stage-1 shards with patch CSV metadata and create per-slide arrays.

Outputs are consumed by Stage-2 ``build_graphs.py``. Dense tokens are written
in a second streaming pass to ``<slide>_dense_tokens.npy`` so large WSIs do not
accumulate all token tensors in process memory.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _first_npz_array(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        if not payload.files:
            raise ValueError(f"Empty npz file: {path}")
        return payload[payload.files[0]]


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _metadata_indices(path: Path, args):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        required = {args.slide_column, args.path_column, "x", "y"}
        missing = sorted(required.difference(columns))
        if missing:
            raise ValueError(f"Patch CSV is missing columns: {missing}")
        rows = [dict(row) for row in reader]
    exact, basename = {}, defaultdict(list)
    for row in rows:
        raw = str(row[args.path_column])
        exact[raw] = row
        exact[str(Path(raw).expanduser())] = row
        basename[Path(raw).name].append(row)
    return exact, basename


def _find_row(filename: str, exact, basename):
    for candidate in (filename, str(Path(filename).expanduser())):
        if candidate in exact:
            return exact[candidate]
    matches = basename.get(Path(filename).name, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous patch basename in metadata: {Path(filename).name}")
    return None


def _group_factory():
    return {
        "features": [],
        "coords": [],
        "levels": [],
        "patch_ids": [],
        "filenames": [],
        "has_dense": None,
        "dense_shape": None,
        "dense_dtype": None,
    }


def organize(args) -> tuple[int, int]:
    exact, basename = _metadata_indices(args.patch_csv, args)
    groups = defaultdict(_group_factory)
    shard_assignments = defaultdict(list)
    matched = unmatched = 0
    feature_shards = sorted(args.embeddings_dir.glob("features_pretrained_s1_*.npz"))
    if not feature_shards:
        raise FileNotFoundError(f"No Stage-1 feature shards found in {args.embeddings_dir}")

    # First pass: join identities/features and remember where each dense token
    # should be copied. Only one decoder shard is resident at a time.
    for feature_path in feature_shards:
        suffix = feature_path.name[len("features_pretrained_s1_") : -len(".npz")]
        filename_path = args.embeddings_dir / f"filenames_pretrained_s1_{suffix}.npz"
        if not filename_path.is_file():
            raise FileNotFoundError(f"Missing filename shard for {feature_path.name}")
        features = _first_npz_array(feature_path)
        filenames = [str(value) for value in _first_npz_array(filename_path).tolist()]
        if len(features) != len(filenames):
            raise ValueError(f"Feature/filename count mismatch in shard {suffix}")

        decoder_path = args.embeddings_dir / f"decoder_features_s1_{suffix}.pt"
        decoder = _load(decoder_path) if decoder_path.is_file() else None
        if args.require_dense_tokens and decoder is None:
            raise FileNotFoundError(f"Dense-token shard is required but missing: {decoder_path}")
        if decoder is not None:
            if list(map(str, decoder.get("filenames", []))) != filenames:
                raise ValueError(f"Decoder filenames do not match npz shard {suffix}")
            dense_tokens = decoder["dense_tokens"]
            if len(dense_tokens) != len(filenames):
                raise ValueError(f"Dense-token count mismatch in shard {suffix}")
            dense_shape = tuple(dense_tokens.shape[1:])
            dense_dtype = dense_tokens.numpy().dtype
            exported_patch_ids = list(map(str, decoder["patch_ids"]))
        else:
            dense_tokens = None
            dense_shape = dense_dtype = None
            exported_patch_ids = [Path(filename).stem for filename in filenames]

        for source_index, (filename, feature, patch_id) in enumerate(
            zip(filenames, features, exported_patch_ids)
        ):
            row = _find_row(filename, exact, basename)
            if row is None:
                unmatched += 1
                if not args.skip_unmatched:
                    raise KeyError(f"Stage-1 patch is absent from metadata CSV: {filename}")
                continue
            slide_id = str(row[args.slide_column])
            group = groups[slide_id]
            has_dense = dense_tokens is not None
            if group["has_dense"] is not None and group["has_dense"] != has_dense:
                raise ValueError(f"Slide {slide_id} mixes shards with and without dense tokens")
            if has_dense and group["dense_shape"] not in (None, dense_shape):
                raise ValueError(f"Slide {slide_id} mixes incompatible dense-token shapes")
            if has_dense and group["dense_dtype"] not in (None, dense_dtype):
                raise ValueError(f"Slide {slide_id} mixes incompatible dense-token dtypes")
            destination_index = len(group["features"])
            group.update(
                has_dense=has_dense,
                dense_shape=dense_shape if has_dense else None,
                dense_dtype=dense_dtype if has_dense else None,
            )
            group["features"].append(np.asarray(feature, dtype=np.float32))
            group["coords"].append((int(float(row["x"])), int(float(row["y"]))))
            group["levels"].append(int(row.get(args.level_column) or 0))
            group["patch_ids"].append(patch_id)
            group["filenames"].append(filename)
            if has_dense:
                shard_assignments[suffix].append(
                    (slide_id, destination_index, source_index)
                )
            matched += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dense_maps = {}
    for slide_id, group in groups.items():
        if len(set(group["patch_ids"])) != len(group["patch_ids"]):
            raise ValueError(f"Duplicate patch_ids in slide {slide_id}")
        features_path = args.output_dir / f"{slide_id}_features.npy"
        coords_path = args.output_dir / f"{slide_id}_coords.npy"
        metadata_path = args.output_dir / f"{slide_id}_metadata.pt"
        dense_path = args.output_dir / f"{slide_id}_dense_tokens.npy"
        outputs = [features_path, coords_path, metadata_path]
        if group["has_dense"]:
            outputs.append(dense_path)
        if not args.overwrite and any(path.exists() for path in outputs):
            raise FileExistsError(f"Refusing to overwrite existing outputs for slide {slide_id}")
        np.save(features_path, np.stack(group["features"]))
        np.save(coords_path, np.asarray(group["coords"], dtype=np.int64))
        torch.save(
            {
                "format_version": 1,
                "slide_id": slide_id,
                "patch_ids": group["patch_ids"],
                "filenames": group["filenames"],
                "levels": torch.as_tensor(group["levels"], dtype=torch.int64),
            },
            metadata_path,
        )
        if group["has_dense"]:
            dense_maps[slide_id] = np.lib.format.open_memmap(
                dense_path,
                mode="w+",
                dtype=group["dense_dtype"],
                shape=(len(group["patch_ids"]), *group["dense_shape"]),
            )

    # Second pass: stream each decoder shard into its per-slide memory map.
    for suffix, assignments in shard_assignments.items():
        decoder_path = args.embeddings_dir / f"decoder_features_s1_{suffix}.pt"
        dense_tokens = _load(decoder_path)["dense_tokens"]
        for slide_id, destination_index, source_index in assignments:
            dense_maps[slide_id][destination_index] = dense_tokens[source_index].numpy()
    for dense_map in dense_maps.values():
        dense_map.flush()
    return matched, unmatched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--patch-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slide-column", default="slide_name_split")
    parser.add_argument("--path-column", default="filepath")
    parser.add_argument("--level-column", default="level")
    parser.add_argument("--require-dense-tokens", action="store_true")
    parser.add_argument("--skip-unmatched", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    matched, unmatched = organize(args)
    print(f"Organized {matched} patches; skipped {unmatched} unmatched patches")


if __name__ == "__main__":
    main()
